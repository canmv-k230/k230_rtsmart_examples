#include "libogg.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "mpi_vb_api.h"
#include "mpi_sys_api.h"
#include "k_acodec_comm.h"
#include "mpi_ai_api.h"
#include "mpi_ao_api.h"
#include "mpi_aenc_api.h"
#include "mpi_adec_api.h"

#define AUDIO_PERSEC_DIV_NUM 25
#define SAMPLE_RATE 16000
#define MAX_AUDIO_STREAM_SIZE 1000

// Ogg muxer/demuxer instances
static kd_ogg_muxer g_ogg_muxer = NULL;
static kd_ogg_demuxer g_ogg_demuxer = NULL;
static k_u32 g_audio_stream_pool_id = VB_INVALID_HANDLE;
static k_audio_stream g_audio_stream;

// Create a VB pool for audio data
static k_u32 audio_data_vb_create_pool()
{
    k_u32 private_pool_id;
    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = 1;
    pool_config.blk_size = MAX_AUDIO_STREAM_SIZE;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    private_pool_id = kd_mpi_vb_create_pool(&pool_config);
    printf("%s poolid %d\n", __func__, private_pool_id);

    return private_pool_id;
}

// Initialize VB (Video Buffer) for audio samples
static k_s32 audio_sample_vb_init() {
    k_s32 ret;
    k_vb_config config;
    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed, ret: %d\n", ret);
    } else {
        printf("vb_set_config succeeded\n");
    }
    ret = kd_mpi_vb_init();
    if (ret)
        printf("vb_init failed, ret: %d\n", ret);

    g_audio_stream_pool_id = audio_data_vb_create_pool();

    k_vb_blk_handle handle = kd_mpi_vb_get_block(g_audio_stream_pool_id, MAX_AUDIO_STREAM_SIZE, NULL);
    if (handle == VB_INVALID_HANDLE)
    {
        printf("%s get vb block error\n", __func__);
        return K_FAILED;
    }

    k_audio_stream* audio_stream = &g_audio_stream;
    audio_stream->len = MAX_AUDIO_STREAM_SIZE;
    audio_stream->phys_addr = kd_mpi_vb_handle_to_phyaddr(handle);
    audio_stream->stream = kd_mpi_sys_mmap(audio_stream->phys_addr, MAX_AUDIO_STREAM_SIZE);
    return ret;
}

// Initialize Ogg muxer
static k_s32 init_ogg_muxer()
{
    // Initialize using stream mode
    kd_ogg_muxer_params muxer_params;
    memset(muxer_params.filename, 0, sizeof(muxer_params.filename));
    muxer_params.sample_rate = SAMPLE_RATE;
    muxer_params.channels = 1;
    muxer_params.serial_no = 0;

    int ret = kd_ogg_muxer_init(&g_ogg_muxer, &muxer_params);
    if (ret != 0) {
        cerr << "Failed to initialize Ogg stream muxer: " << ret << endl;
        return -1;
    }
    cout << "Ogg stream muxer initialized successfully." << endl;

    return 0;
}

// Initialize Ogg demuxer
static k_s32 init_ogg_demuxer()
{
    kd_ogg_demuxer_params params;
    memset(&params, 0, sizeof(params));

    int ret = kd_ogg_demuxer_init(&g_ogg_demuxer, &params);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize demuxer: %d\n", ret);
        return -1;
    }

    return 0;
}

// Process Opus stream data
static uint8_t g_data_ogg[1000];
static uint32_t g_data_ogg_size = 0;
static uint8_t g_data_opus[1000];
static uint32_t g_data_opus_size = 0;

static void do_opus_stream(k_audio_stream *stream) {
    if (!g_ogg_muxer || !stream)
        return;

    k_u8 *raw_data = (k_u8 *)kd_mpi_sys_mmap(stream->phys_addr, stream->len);

    //muxer ogg
    kd_ogg_frame_params_ex frame_params;
    frame_params.data = raw_data;
    frame_params.len = stream->len;
    frame_params.frame_samples = SAMPLE_RATE / AUDIO_PERSEC_DIV_NUM;
    frame_params.out_page = g_data_ogg;
    frame_params.out_page_size = &g_data_ogg_size;

    int ret = kd_ogg_write_frame_ex(g_ogg_muxer, &frame_params);
    if (ret != 0) {
        printf("========kd_ogg_write_frame_ex failed\n");
    }
    kd_mpi_sys_munmap(raw_data, stream->len);

    //demuxer ogg to opus
    kd_ogg_page_params_ex page_parames;
    page_parames.page_data = g_data_ogg;
    page_parames.page_size = *frame_params.out_page_size;
    page_parames.out_frame = g_data_opus;
    page_parames.out_frame_size = &g_data_opus_size;

    ret = kd_ogg_demuxer_feed_page_ex(g_ogg_demuxer, &page_parames);
    if (ret != 0) {
        printf("========kd_ogg_demuxer_feed_page_ex failed,ret:%d\n",ret);
    }

    //send opus to adec
    g_audio_stream.len = *page_parames.out_frame_size;
    memcpy(g_audio_stream.stream, page_parames.out_frame, *page_parames.out_frame_size);
    if (K_SUCCESS != kd_mpi_adec_send_stream(0, &g_audio_stream, K_FALSE))
    {
        printf("========kd_mpi_adec_send_stream failed\n");
    }

}

// Audio sample processing: AI -> AENC -> ADEC -> AO with Opus
static k_s32 audio_sample_ogg(k_audio_dev ai_dev, k_ai_chn ai_chn, k_audio_dev ao_dev, k_ao_chn ao_chn,
                                              k_aenc_chn aenc_chn, k_adec_chn adec_chn, k_u32 samplerate,
                                              k_audio_bit_width bit_width, k_payload_type type, k_u32 enable_audio3a)
{
    bit_width = KD_AUDIO_BIT_WIDTH_16;
    k_u32 sample_rate = samplerate;
    k_i2s_work_mode i2s_work_mode = K_STANDARD_MODE;
    printf("Force the sampling precision to be set to 16, use inner codec\n");

    k_aio_dev_attr ai_dev_attr;
    ai_dev_attr.audio_type = KD_AUDIO_INPUT_TYPE_I2S;
    ai_dev_attr.kd_audio_attr.i2s_attr.sample_rate = sample_rate;
    ai_dev_attr.kd_audio_attr.i2s_attr.bit_width = bit_width;
    ai_dev_attr.kd_audio_attr.i2s_attr.chn_cnt = 2;
    ai_dev_attr.kd_audio_attr.i2s_attr.i2s_mode = i2s_work_mode;
    ai_dev_attr.kd_audio_attr.i2s_attr.snd_mode = KD_AUDIO_SOUND_MODE_MONO;
    ai_dev_attr.kd_audio_attr.i2s_attr.frame_num = AUDIO_PERSEC_DIV_NUM;
    ai_dev_attr.kd_audio_attr.i2s_attr.point_num_per_frame = ai_dev_attr.kd_audio_attr.i2s_attr.sample_rate / ai_dev_attr.kd_audio_attr.i2s_attr.frame_num;
    ai_dev_attr.kd_audio_attr.i2s_attr.i2s_type = K_AIO_I2STYPE_INNERCODEC ;
    if (K_SUCCESS != kd_mpi_ai_set_pub_attr(ai_dev, &ai_dev_attr))
    {
        printf("kd_mpi_ai_set_pub_attr failed\n");
        return K_FAILED;
    }

    k_aio_dev_attr ao_dev_attr;
    memset(&ao_dev_attr, 0, sizeof(ao_dev_attr));
    ao_dev_attr.audio_type = KD_AUDIO_OUTPUT_TYPE_I2S;
    ao_dev_attr.kd_audio_attr.i2s_attr.sample_rate = sample_rate;
    ao_dev_attr.kd_audio_attr.i2s_attr.bit_width = bit_width;
    ao_dev_attr.kd_audio_attr.i2s_attr.chn_cnt = 2;
    ao_dev_attr.kd_audio_attr.i2s_attr.i2s_mode = i2s_work_mode;
    ao_dev_attr.kd_audio_attr.i2s_attr.snd_mode = KD_AUDIO_SOUND_MODE_MONO;
    ao_dev_attr.kd_audio_attr.i2s_attr.frame_num = AUDIO_PERSEC_DIV_NUM;
    ao_dev_attr.kd_audio_attr.i2s_attr.point_num_per_frame = ao_dev_attr.kd_audio_attr.i2s_attr.sample_rate / ao_dev_attr.kd_audio_attr.i2s_attr.frame_num;
    ao_dev_attr.kd_audio_attr.i2s_attr.i2s_type = K_AIO_I2STYPE_INNERCODEC;

    if (K_SUCCESS != kd_mpi_ao_set_pub_attr(ao_dev, &ao_dev_attr))
    {
        printf("kd_mpi_ao_set_pub_attr failed\n");
        return K_FAILED;
    }

    k_aenc_chn_attr aenc_chn_attr;
    aenc_chn_attr.type = type;
    aenc_chn_attr.buf_size = AUDIO_PERSEC_DIV_NUM;
    aenc_chn_attr.sample_rate = sample_rate;
    aenc_chn_attr.channels = 1;
    aenc_chn_attr.bitrate = 16000;
    aenc_chn_attr.point_num_per_frame = sample_rate / aenc_chn_attr.buf_size;

    if (0 != kd_mpi_aenc_create_chn(aenc_chn, &aenc_chn_attr))
    {
        printf("kd_mpi_aenc_create_chn failed\n");
        return K_FAILED;
    }

    k_adec_chn_attr adec_chn_attr;
    adec_chn_attr.type = type;
    adec_chn_attr.sample_rate = sample_rate;
    adec_chn_attr.channels = 1;
    adec_chn_attr.buf_size = AUDIO_PERSEC_DIV_NUM;
    adec_chn_attr.point_num_per_frame = sample_rate / adec_chn_attr.buf_size;

    if (0 != kd_mpi_adec_create_chn(adec_chn, &adec_chn_attr))
    {
        printf("kd_mpi_adec_create_chn failed\n");
        return K_FAILED;
    }

    init_ogg_demuxer();
    init_ogg_muxer();

    kd_mpi_ai_enable(ai_dev);
    kd_mpi_ai_enable_chn(ai_dev, ai_chn);

    k_mpp_chn ai_mpp_chn;
    k_mpp_chn aenc_mpp_chn;

    ai_mpp_chn.mod_id = K_ID_AI;
    ai_mpp_chn.dev_id = ai_dev;
    ai_mpp_chn.chn_id = ai_chn;
    aenc_mpp_chn.mod_id = K_ID_AENC;
    aenc_mpp_chn.dev_id = 0;
    aenc_mpp_chn.chn_id = aenc_chn;

    kd_mpi_sys_bind(&ai_mpp_chn, &aenc_mpp_chn);

    kd_mpi_ao_enable(ao_dev);
    kd_mpi_ao_enable_chn(ao_dev, ao_chn);

    k_mpp_chn ao_mpp_chn;
    k_mpp_chn adec_mpp_chn;

    adec_mpp_chn.mod_id = K_ID_ADEC;
    adec_mpp_chn.dev_id = 0;
    adec_mpp_chn.chn_id = adec_chn;
    ao_mpp_chn.mod_id = K_ID_AO;
    ao_mpp_chn.dev_id = ao_dev;
    ao_mpp_chn.chn_id = ao_chn;

    kd_mpi_sys_bind(&adec_mpp_chn, &ao_mpp_chn);

    k_audio_stream audio_stream;
    while (1)
    {
        if (K_SUCCESS != kd_mpi_aenc_get_stream(aenc_chn, &audio_stream, 100))
        {
            printf("========kd_mpi_aenc_get_stream failed\n");
            continue;
        }

        do_opus_stream(&audio_stream);
        kd_mpi_aenc_release_stream(aenc_chn, &audio_stream);
    }

    kd_mpi_sys_unbind(&ai_mpp_chn, &aenc_mpp_chn);
    kd_mpi_sys_unbind(&adec_mpp_chn, &ao_mpp_chn);

    kd_mpi_ai_disable_chn(ai_dev, ai_chn);
    kd_mpi_ai_disable(ai_dev);
    kd_mpi_ao_disable_chn(ao_dev, ao_chn);
    kd_mpi_ao_disable(ao_dev);
    kd_mpi_aenc_destroy_chn(aenc_chn);
    kd_mpi_adec_destroy_chn(adec_chn);

    return K_SUCCESS;
}

int main() {
    audio_sample_vb_init();
    audio_sample_ogg(0, 0, 0, 0, 0, 0, SAMPLE_RATE, KD_AUDIO_BIT_WIDTH_16, K_PT_OPUS, K_FALSE);

    return 0;
}