#include "media.h"
#include <fcntl.h>
#include <cstring>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic>
#include "k_sensor_comm.h"
#include "mpi_sys_api.h"
#include "mpi_ai_api.h"
#include "mpi_aenc_api.h"
#include "mpi_ao_api.h"
#include "mpi_adec_api.h"
#include "mpi_vvi_api.h"
#include "mpi_venc_api.h"
#include "mpi_vicap_api.h"
#include "mpi_sensor_api.h"
#include "k_vicap_comm.h"
#include "k_vb_comm.h"
#include "mpi_vb_api.h"
#include "mpi_vdec_api.h"
#include "mpi_vo_api.h"
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "scoped_timing.hpp"

static void sleep_time(char*pfunc_name,int time)
{
    printf("%s sleep %ds\n",pfunc_name,time);
    sleep(time);
}

// Memory alignment macros
#define MEM_ALIGN_1K 0x400
#define MEM_ALIGN_4K 0x1000
#define MEM_ALIGN_UP(addr, size)	(((addr)+((size)-1U))&(~((size)-1U)))
#define VICAP_CHN_MIN_FRAME_COUNT 3

typedef struct
{
    k_u64 layer_phy_addr;
    k_pixel_format format;
    k_vo_point offset;
    k_vo_size act_size;
    k_u32 size;
    k_u32 stride;
    k_u8 global_alptha;
    //only layer0、layer1
    k_u32 func;
    // only layer0
    k_vo_scaler_attr attr;
} layer_info;

typedef struct
{
    k_u64 osd_phy_addr;
    void *osd_virt_addr;
    k_pixel_format format;
    k_vo_point offset;
    k_vo_size act_size;
    k_u32 size;
    k_u32 stride;
    k_u8 global_alptha;
} osd_info;

// ISP input dimensions
#define ISP_INPUT_WIDTH (1920)
#define ISP_INPUT_HEIGHT (1080)

static k_s32 g_mmap_fd_tmp = 0;
static std::mutex mmap_mutex_;

static void *_sys_mmap(k_u64 phys_addr, k_u32 size)
{
    void *virt_addr = NULL;
    void *mmap_addr = NULL;
    k_u32 page_size = sysconf(_SC_PAGESIZE);
    k_u64 page_mask = (page_size - 1);
    k_u32 mmap_size = ((size) + (phys_addr & page_mask) + page_mask) & ~(page_mask);

    std::unique_lock<std::mutex> lck(mmap_mutex_);
    if (g_mmap_fd_tmp == 0)
    {
        g_mmap_fd_tmp = open("/dev/mem", O_RDWR | O_SYNC);
    }

    mmap_addr = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_mmap_fd_tmp, phys_addr & ~page_mask);
    if (mmap_addr != (void *)(-1))
        virt_addr = (void *)((char *)mmap_addr + (phys_addr & page_mask));
    else
        printf("**** sys_mmap failed\n");

    return virt_addr;
}

static k_s32 _sys_munmap(k_u64 phys_addr, void *virt_addr, k_u32 size)
{
    std::unique_lock<std::mutex> lck(mmap_mutex_);
    if (g_mmap_fd_tmp == 0)
    {
        return -1;
    }
    k_u32 page_size = sysconf(_SC_PAGESIZE);
    k_u64 page_mask = page_size - 1;
    k_u32 mmap_size = ((size) + (phys_addr & page_mask) + page_mask) & ~(page_mask);
    void *mmap_addr = (void *)((char *)virt_addr - (phys_addr & page_mask));
    if (munmap(mmap_addr, mmap_size) < 0)
    {
        printf("**** munmap failed\n");
    }
    return 0;
}

static k_s32 kd_sample_vi_bind_venc(k_s32 src_dev, k_s32 src_chn, k_s32 chn_num)
{
    if (chn_num >= VENC_MAX_CHN_NUMS)
    {
        printf("kd_mpi_venc_bind_vi chn_num:%d error\n", chn_num);
        return -1;
    }

    k_mpp_chn vi_mpp_chn;
    k_mpp_chn venc_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = src_dev;
    vi_mpp_chn.chn_id = src_chn;
    venc_mpp_chn.mod_id = K_ID_VENC;
    venc_mpp_chn.dev_id = 0;
    venc_mpp_chn.chn_id = chn_num;
    k_s32 ret = kd_mpi_sys_bind(&vi_mpp_chn, &venc_mpp_chn);

    return ret;
}

static k_s32 kd_sample_vi_unbind_venc(k_s32 src_dev, k_s32 src_chn, k_s32 chn_num)
{
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn venc_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = src_dev;
    vi_mpp_chn.chn_id = src_chn;

    venc_mpp_chn.mod_id = K_ID_VENC;
    venc_mpp_chn.dev_id = 0;
    venc_mpp_chn.chn_id = chn_num;
    k_s32 ret = kd_mpi_sys_unbind(&vi_mpp_chn, &venc_mpp_chn);

    return ret;
}

static k_s32 kd_sample_vi_bind_vo(k_s32 src_dev, k_s32 src_chn, k_s32 dst_dev, k_s32 dst_chn)
{
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn vo_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = src_dev;
    vi_mpp_chn.chn_id = src_chn;

    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = dst_dev;
    vo_mpp_chn.chn_id = dst_chn;
    k_s32 ret = kd_mpi_sys_bind(&vi_mpp_chn, &vo_mpp_chn);

    return ret;
}

static k_s32 kd_sample_vi_unbind_vo(k_s32 src_dev, k_s32 src_chn, k_s32 dst_dev, k_s32 dst_chn)
{
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn vo_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = src_dev;
    vi_mpp_chn.chn_id = src_chn;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = dst_dev;
    vo_mpp_chn.chn_id = dst_chn;
    k_s32 ret = kd_mpi_sys_unbind(&vi_mpp_chn, &vo_mpp_chn);

    return ret;
}

static k_vicap_sensor_info sensor_info[VICAP_DEV_ID_MAX];

static k_s32 kd_sample_vicap_set_dev_attr(k_vicap_dev_set_info dev_info)
{
    k_s32 ret = 0;

    /* dev attr */
    if (dev_info.vicap_dev >= VICAP_DEV_ID_MAX || dev_info.vicap_dev < VICAP_DEV_ID_0)
    {
        printf("kd_mpi_vicap_set_dev_attr failed, dev_num %d out of range\n", dev_info.vicap_dev);
        return K_FAILED;
    }

    if (SENSOR_TYPE_MAX <= dev_info.sensor_type)
    {
        printf("kd_mpi_vicap_set_dev_attr failed, sensor_type %d out of range\n", dev_info.sensor_type);
        return K_FAILED;
    }

    k_vicap_dev_attr dev_attr;

    memset(&dev_attr, 0, sizeof(k_vicap_dev_attr));
    memset(&sensor_info[dev_info.vicap_dev], 0, sizeof(k_vicap_sensor_info));

    sensor_info[dev_info.vicap_dev].sensor_type = dev_info.sensor_type;
    ret = kd_mpi_vicap_get_sensor_info(sensor_info[dev_info.vicap_dev].sensor_type, &sensor_info[dev_info.vicap_dev]);
    if (ret)
    {
        printf("kd_mpi_vicap_get_sensor_info failed:0x%x\n", ret);
        return K_FAILED;
    }
    dev_attr.dw_enable = dev_info.dw_en;

    dev_attr.acq_win.h_start = 0;
    dev_attr.acq_win.v_start = 0;
    dev_attr.acq_win.width = sensor_info[dev_info.vicap_dev].width;
    dev_attr.acq_win.height = sensor_info[dev_info.vicap_dev].height;
    // vicap work mode process
    if ((dev_info.mode == VICAP_WORK_OFFLINE_MODE) || (dev_info.mode == VICAP_WORK_LOAD_IMAGE_MODE) || (dev_info.mode == VICAP_WORK_ONLY_MCM_MODE))
    {
        dev_attr.mode = dev_info.mode;
        dev_attr.buffer_num = dev_info.buffer_num;
        dev_attr.buffer_size = dev_info.buffer_size;
    }

    dev_attr.pipe_ctrl.data = dev_info.pipe_ctrl.data;
    // af need disable
    dev_attr.pipe_ctrl.bits.af_enable = 0;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.pipe_ctrl.bits.dnr3_enable = 0;
    dev_attr.cpature_frame = 0;
    memcpy(&dev_attr.sensor_info, &sensor_info[dev_info.vicap_dev], sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_set_dev_attr(dev_info.vicap_dev, dev_attr);
    if (ret)
    {
        printf("kd_mpi_vicap_set_dev_attr failed:0x%x\n", ret);
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 kd_sample_vicap_set_chn_attr(k_vicap_chn_set_info chn_info)
{
    k_s32 ret = 0;
    if (chn_info.vicap_dev >= VICAP_DEV_ID_MAX || chn_info.vicap_dev < VICAP_DEV_ID_0)
    {
        printf("kd_mpi_vicap_set_dev_attr failed, dev_num %d out of range\n", chn_info.vicap_dev);
        return K_FAILED;
    }

    if (chn_info.vicap_chn >= VICAP_CHN_ID_MAX || chn_info.vicap_chn < VICAP_CHN_ID_0)
    {
        printf("kd_mpi_vicap_set_attr failed, chn_num %d out of range\n", chn_info.vicap_chn);
        return K_FAILED;
    }

    k_vicap_chn_attr chn_attr;
    memset(&chn_attr, 0, sizeof(k_vicap_chn_attr));

    /* chn attr */
    if (!chn_info.out_height && chn_info.out_width)
    {
        printf("kd_mpi_vicap_set_attr, failed, out_width: %d, out_height: %d error\n", chn_info.out_width, chn_info.out_height);
        return K_FAILED;
    }

    if (chn_info.pixel_format > PIXEL_FORMAT_BUTT || chn_info.pixel_format < PIXEL_FORMAT_RGB_444)
    {
        printf("kd_mpi_vicap_set_attr, failed, pixel_formatr: %d out of range\n", chn_info.pixel_format);
        return K_FAILED;
    }

    if (chn_info.pixel_format == PIXEL_FORMAT_RGB_BAYER_10BPP || chn_info.pixel_format == PIXEL_FORMAT_RGB_BAYER_12BPP)
    {
        chn_attr.out_win.h_start = 0;
        chn_attr.out_win.v_start = 0;
        chn_attr.out_win.width = sensor_info[chn_info.vicap_dev].width;
        chn_attr.out_win.height = sensor_info[chn_info.vicap_dev].height;
    }
    else
    {
        chn_attr.out_win.width = chn_info.out_width;
        chn_attr.out_win.height = chn_info.out_height;
    }

    if (chn_info.crop_en)
    {
        chn_attr.crop_win.h_start = chn_info.crop_h_start;
        chn_attr.crop_win.v_start = chn_info.crop_v_start;
        chn_attr.crop_win.width = chn_info.out_width;
        chn_attr.crop_win.height = chn_info.out_height;
    }
    else
    {
        chn_attr.crop_win.h_start = 0;
        chn_attr.crop_win.v_start = 0;
        chn_attr.crop_win.width = sensor_info[chn_info.vicap_dev].width;
        chn_attr.crop_win.height = sensor_info[chn_info.vicap_dev].height;
    }

    chn_attr.scale_win.h_start = 0;
    chn_attr.scale_win.v_start = 0;
    chn_attr.scale_win.width = sensor_info[chn_info.vicap_dev].width;
    chn_attr.scale_win.height = sensor_info[chn_info.vicap_dev].height;

    chn_attr.crop_enable = chn_info.crop_en;
    chn_attr.scale_enable = chn_info.scale_en;
    chn_attr.chn_enable = chn_info.chn_en;
    chn_attr.pix_format = chn_info.pixel_format;
    chn_attr.buffer_num = chn_info.buffer_num;
    chn_attr.buffer_size = chn_info.buf_size;
    chn_attr.alignment = chn_info.alignment;
    chn_attr.fps = chn_info.fps;
    ret = kd_mpi_vicap_set_chn_attr(chn_info.vicap_dev, chn_info.vicap_chn, chn_attr);
    if (ret)
    {
        printf("kd_mpi_vicap_set_chn_attr failed:0x%x\n", ret);
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 kd_sample_vicap_start(k_vicap_dev vicap_dev)
{
    k_s32 ret = 0;
    if (vicap_dev > VICAP_DEV_ID_MAX || vicap_dev < VICAP_DEV_ID_0)
    {
        printf("kd_mpi_vicap_start failed, dev_num %d out of range\n", vicap_dev);
        return K_FAILED;
    }
    ret = kd_mpi_vicap_init(vicap_dev);
    if (ret)
    {
        printf("kd_mpi_vicap_init, vicap dev(%d) init failed.\n", vicap_dev);
        kd_mpi_vicap_deinit(vicap_dev);
        return K_FAILED;
    }

    ret = kd_mpi_vicap_start_stream(vicap_dev);
    if (ret)
    {
        printf("kd_mpi_vicap_start_stream, vicap dev(%d) start stream failed.\n", vicap_dev);
        kd_mpi_vicap_deinit(vicap_dev);
        kd_mpi_vicap_stop_stream(vicap_dev);
        return K_FAILED;
    }

    return K_SUCCESS;
}

static k_s32 kd_sample_vicap_stop(k_vicap_dev vicap_dev)
{
    k_s32 ret = 0;
    if (vicap_dev > VICAP_DEV_ID_MAX || vicap_dev < VICAP_DEV_ID_0)
    {
        printf("kd_mpi_vicap_stop failed, dev_num %d out of range\n", vicap_dev);
        return K_FAILED;
    }
    ret = kd_mpi_vicap_stop_stream(vicap_dev);
    if (ret)
    {
        printf("kd_mpi_vicap_stop_stream failed\n");
    }
    ret = kd_mpi_vicap_deinit(vicap_dev);
    if (ret)
    {
        printf("kd_mpi_vicap_deinit failed\n");
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 kd_sample_aenc_bind_ai(k_handle ai_dev, k_handle ai_chn, k_handle aenc_hdl)
{
    k_mpp_chn ai_mpp_chn;
    k_mpp_chn aenc_mpp_chn;

    ai_mpp_chn.mod_id = K_ID_AI;
    ai_mpp_chn.dev_id = ai_dev;
    ai_mpp_chn.chn_id = ai_chn;
    aenc_mpp_chn.mod_id = K_ID_AENC;
    aenc_mpp_chn.dev_id = 0;
    aenc_mpp_chn.chn_id = aenc_hdl;

    return kd_mpi_sys_bind(&ai_mpp_chn, &aenc_mpp_chn);
}

static k_s32 kd_sample_aenc_unbind_ai(k_handle ai_dev, k_handle ai_chn, k_handle aenc_hdl)
{

    k_mpp_chn ai_mpp_chn;
    k_mpp_chn aenc_mpp_chn;

    ai_mpp_chn.mod_id = K_ID_AI;
    ai_mpp_chn.dev_id = ai_dev;
    ai_mpp_chn.chn_id = ai_chn;
    aenc_mpp_chn.mod_id = K_ID_AENC;
    aenc_mpp_chn.dev_id = 0;
    aenc_mpp_chn.chn_id = aenc_hdl;

    return kd_mpi_sys_unbind(&ai_mpp_chn, &aenc_mpp_chn);
}

static k_s32 kd_sample_sensor_auto_detect(k_vicap_sensor_type* sensor_type)
{
    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_info sensor_info;

    probe_cfg.csi_num = CONFIG_MPP_SENSOR_DEFAULT_CSI + 1;
    probe_cfg.width = ISP_INPUT_WIDTH;
    probe_cfg.height = ISP_INPUT_HEIGHT;
    probe_cfg.fps = 30;

    if(0x00 != kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info)) {
        printf("sample_vicap, can't probe sensor on %d, output %dx%d@%d\n", probe_cfg.csi_num, probe_cfg.width, probe_cfg.height, probe_cfg.fps);

        return -1;
    }

    *sensor_type = sensor_info.sensor_type;
    printf("detect sensor type: %d\n", sensor_info.sensor_type);

    return 0;
}

int KdMedia::configure_media_features(const KdMediaInputConfig &input_config, const KdMediaFeatureConfig &feature_config)
{
    input_config_ = input_config;
    feature_config_ = feature_config;

    if (input_config.sensor_type == SENSOR_TYPE_MAX)
    {
        k_vicap_sensor_type sensor_type;
        if (0 != kd_sample_sensor_auto_detect(&sensor_type))
        {
            printf("kd_sample_sensor_auto_detect failed\n");
            return -1;
        }
        input_config_.sensor_type = sensor_type;
    }

    _init_vb_pool();

    return 0;
}

// int KdMedia::enable_media_features()
// {
//     //init vicap
//     _init_vi_cap();

//     //init vo
//     _init_vo_layer_osd();

//     //init venc
//     _init_venc();

//     //init ai aenc
//     _init_ai_aenc();

//     //start vicap venc
//     _start_venc();

//     _start_vi_cap();

//     //start dump frame for ai analysis
//     _start_dump_frame_for_ai_analysis();

//     //start ai and aenc
//     _start_ai_aenc();

//     return 0;
// }

int KdMedia::enable_media_features()
{
    // Audio codec initialization takes a long time, so it is executed in a separate thread
    pthread_create(&start_ai_aenc_tid_, NULL, start_ai_aenc_thread, this);

    //init vicap
    _init_vi_cap();

    //init vo
    {
        ScopedTiming st = ScopedTiming("@@@@@@_init_vo_layer_osd", 1);
        _init_vo_layer_osd();
    }

    // start vicap
    {
        ScopedTiming st = ScopedTiming("@@@@@@_start_vi_cap", 1);
        _start_vi_cap();
    }

    //start dump frame for ai analysis
    _start_dump_frame_for_ai_analysis();

    //init venc
    _init_venc();

    //start vicap venc
    _start_venc();

    return 0;
}

int KdMedia::disable_media_features()
{
    //stop dump frame for ai analysis
    _stop_dump_frame_for_ai_analysis();

    //stop venc
    _stop_venc();
    _deinit_venc();

    //stop vo
    _deinit_vo_layer_osd();

    //stop ai,aenc
    _stop_ai_aenc();
    _deinit_ai_aenc();

    //stop vi cap
    _stop_vi_cap();
    _deinit_vi_cap();

    return 0;
}

int KdMedia::destroy_media_features()
{
    return _deinit_vb_pool();
}

int KdMedia::_init_vb_pool()
{
    k_s32 ret = 0;
    k_vb_config vb_config;
    k_s32 pool_index = -1;

    memset(&vb_config, 0, sizeof(vb_config));
    vb_config.max_pool_cnt = 64;

    // vb for vicap
    {
        //vb for venc input
        if (feature_config_.enable_video_encoder)
        {
            pool_index ++;
            //VB vicap for YUV420SP input to venc
            vb_config.comm_pool[pool_index].blk_cnt = VICAP_CHN_MIN_FRAME_COUNT;
            vb_config.comm_pool[pool_index].mode = VB_REMAP_MODE_NOCACHE;
            vb_config.comm_pool[pool_index].blk_size = MEM_ALIGN_UP(input_config_.venc_width*input_config_.venc_height * 3 / 2, MEM_ALIGN_4K);//must align 4k
        }

        //vb for vo
        if (feature_config_.enable_render)
        {
            pool_index ++;
            //VB vicap for YUV420SP output to vo
            vb_config.comm_pool[pool_index].blk_cnt = VICAP_CHN_MIN_FRAME_COUNT;
            vb_config.comm_pool[pool_index].mode = VB_REMAP_MODE_NOCACHE;
            vb_config.comm_pool[pool_index].blk_size = MEM_ALIGN_UP(input_config_.vo_width * input_config_.vo_height * 3 / 2, MEM_ALIGN_1K);
        }

        //vb for ai anysis
        if (feature_config_.enable_ai_analysis)
        {
            pool_index ++;
            //VB vicap for RGB888 output to ai
            vb_config.comm_pool[pool_index].blk_cnt = VICAP_CHN_MIN_FRAME_COUNT;
            vb_config.comm_pool[pool_index].mode = VB_REMAP_MODE_NOCACHE;
            vb_config.comm_pool[pool_index].blk_size = MEM_ALIGN_UP(input_config_.ai_width * input_config_.ai_height * 3 , MEM_ALIGN_1K);
        }

        //vb for mcm
        // {
        //     pool_index ++;
        //     vb_config.comm_pool[pool_index].blk_cnt = VICAP_CHN_MIN_FRAME_COUNT;
        //     vb_config.comm_pool[pool_index].mode = VB_REMAP_MODE_NOCACHE;
        //     vb_config.comm_pool[pool_index].blk_size = MEM_ALIGN_UP((ISP_INPUT_WIDTH * ISP_INPUT_HEIGHT * 2 ), MEM_ALIGN_1K);
        // }

    }

    //vb for audio ai and aenc
    if (feature_config_.enable_audio_encoder)
    {
        pool_index ++;
        vb_config.comm_pool[pool_index].blk_cnt = audio_frame_divisor_ * 2;
        vb_config.comm_pool[pool_index].blk_size = input_config_.audio_samplerate * 2 * 4 / audio_frame_divisor_;
        vb_config.comm_pool[pool_index].mode = VB_REMAP_MODE_NOCACHE;
    }

    // vb for venc
    if (feature_config_.enable_video_encoder)
    {
        //venc input hold 2 vb
        pool_index ++;
        vb_config.comm_pool[pool_index].blk_cnt = 2;
        vb_config.comm_pool[pool_index].mode = VB_REMAP_MODE_NOCACHE;
        vb_config.comm_pool[pool_index].blk_size = MEM_ALIGN_UP(input_config_.venc_width*input_config_.venc_height * 3 / 2, MEM_ALIGN_4K);//must align 4k

        // vb for venc output
        pool_index ++;
        vb_config.comm_pool[pool_index].blk_cnt = 2;
        vb_config.comm_pool[pool_index].blk_size = MEM_ALIGN_UP(input_config_.venc_width*input_config_.venc_height/2,MEM_ALIGN_4K);
        vb_config.comm_pool[pool_index].mode = VB_REMAP_MODE_NOCACHE;
    }

    // set vb config
    ret = kd_mpi_vb_set_config(&vb_config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        return ret;
    }

    //init vb
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        return ret;
    }

    //vb for osd
    if (feature_config_.enable_ai_analysis)
    {
        k_vb_pool_config pool_config;
        memset(&pool_config, 0, sizeof(pool_config));
        pool_config.blk_size = MEM_ALIGN_UP((input_config_.osd_width * input_config_.osd_height * 4), MEM_ALIGN_4K);
        pool_config.blk_cnt = 1;
        pool_config.mode = VB_REMAP_MODE_NOCACHE;
        osd_pool_id_ = kd_mpi_vb_create_pool(&pool_config);
    }

    return 0;
}

int KdMedia::_deinit_vb_pool()
{
    kd_mpi_vb_exit();
    return 0;
}

int KdMedia::_init_vi_cap()
{
    k_s32 ret = 0;
    k_vicap_sensor_info sensor_info;
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_set_info vi_chn_attr_info;
    k_s32 vicap_chn_index = -1;
    k_vicap_dev vicap_dev = vi_dev_id_;

    memset(&vcap_dev_info_, 0, sizeof(vcap_dev_info_));
    vcap_dev_info_.dw_en = K_FALSE;
    vcap_dev_info_.pipe_ctrl.data = 0xFFFFFFFF;
    vcap_dev_info_.sensor_type = input_config_.sensor_type;
    vcap_dev_info_.vicap_dev = vi_dev_id_;

    kd_sample_vicap_set_dev_attr(vcap_dev_info_);

    //set chn0 output yuv420sp for vo render
    if (feature_config_.enable_render)
    {
        vicap_chn_index ++;
        vi_chn_render_id_ = (k_vicap_chn)vicap_chn_index;

        memset(&vi_chn_attr_info, 0, sizeof(vi_chn_attr_info));
        vi_chn_attr_info.crop_en = K_FALSE;
        vi_chn_attr_info.scale_en = K_FALSE;
        vi_chn_attr_info.chn_en = K_TRUE;
        vi_chn_attr_info.crop_h_start = 0;
        vi_chn_attr_info.crop_v_start = 0;

        vi_chn_attr_info.out_width = MEM_ALIGN_UP(input_config_.vo_width, 16);
        vi_chn_attr_info.out_height = input_config_.vo_height;
        vi_chn_attr_info.pixel_format = vi_chn_render_pixel_format_;// PIXEL_FORMAT_YUV_SEMIPLANAR_420
        vi_chn_attr_info.vicap_dev = vi_dev_id_;
        vi_chn_attr_info.buffer_num = VICAP_CHN_MIN_FRAME_COUNT;
        vi_chn_attr_info.vicap_chn = vi_chn_render_id_;
        if (!vcap_dev_info_.dw_en)
            vi_chn_attr_info.buf_size = MEM_ALIGN_UP(input_config_.vo_width * input_config_.vo_height * 3 / 2, MEM_ALIGN_1K);
        else
            vi_chn_attr_info.buf_size = MEM_ALIGN_UP(input_config_.vo_width * input_config_.vo_height * 3 / 2, MEM_ALIGN_4K);

        ret = kd_sample_vicap_set_chn_attr(vi_chn_attr_info);
        if (ret != K_SUCCESS)
        {
            printf("vicap chn %d set attr failed, %x.\n", vi_chn_render_id_, ret);
            return -1;
        }
        else
        {
            printf("vicap chn %d set attr success.\n", vi_chn_render_id_);
        }

    }

    //set chn1 output rgb888p for ai analysis
    if (feature_config_.enable_ai_analysis)
    {
        vicap_chn_index++;
        vi_chn_ai_id_ = (k_vicap_chn)vicap_chn_index;

        memset(&vi_chn_attr_info, 0, sizeof(vi_chn_attr_info));
        vi_chn_attr_info.crop_en = K_FALSE;
        vi_chn_attr_info.scale_en = K_FALSE;
        vi_chn_attr_info.chn_en = K_TRUE;
        vi_chn_attr_info.crop_h_start = 0;
        vi_chn_attr_info.crop_v_start = 0;

        vi_chn_attr_info.out_width = MEM_ALIGN_UP(input_config_.ai_width, 16);
        vi_chn_attr_info.out_height = input_config_.ai_height;
        vi_chn_attr_info.pixel_format = vi_chn_ai_pixel_format_;
        vi_chn_attr_info.vicap_dev = vi_dev_id_;
        vi_chn_attr_info.buffer_num = VICAP_CHN_MIN_FRAME_COUNT;
        vi_chn_attr_info.vicap_chn = vi_chn_ai_id_;
        if (!vcap_dev_info_.dw_en)
            vi_chn_attr_info.buf_size = MEM_ALIGN_UP(input_config_.ai_width * input_config_.ai_height * 3, MEM_ALIGN_1K);
        else
            vi_chn_attr_info.buf_size = MEM_ALIGN_UP(input_config_.ai_width * input_config_.ai_height * 3, MEM_ALIGN_4K);

        ret = kd_sample_vicap_set_chn_attr(vi_chn_attr_info);
        if (ret != K_SUCCESS)
        {
            printf("vicap chn %d set attr failed, %x.\n", vi_chn_ai_id_, ret);
            return -1;
        }
        else
        {
            printf("vicap chn %d set attr success.\n", vi_chn_ai_id_);
        }
    }

    //set chn2 output yuv420sp for venc
    if (feature_config_.enable_video_encoder)
    {
        vicap_chn_index ++;
        vi_chn_venc_id_ = (k_vicap_chn)vicap_chn_index;

        memset(&vi_chn_attr_info, 0, sizeof(vi_chn_attr_info));
        vi_chn_attr_info.crop_en = K_FALSE;
        vi_chn_attr_info.scale_en = K_FALSE;
        vi_chn_attr_info.chn_en = K_TRUE;
        vi_chn_attr_info.crop_h_start = 0;
        vi_chn_attr_info.crop_v_start = 0;

        vi_chn_attr_info.out_width = MEM_ALIGN_UP(input_config_.venc_width, 16);
        vi_chn_attr_info.out_height = input_config_.venc_height;
        vi_chn_attr_info.pixel_format = vi_chn_venc_pixel_format_;
        vi_chn_attr_info.vicap_dev = vi_dev_id_;
        vi_chn_attr_info.buffer_num = VICAP_CHN_MIN_FRAME_COUNT;
        vi_chn_attr_info.alignment = 12;//must align 4k
        vi_chn_attr_info.vicap_chn = vi_chn_venc_id_;
        if (!vcap_dev_info_.dw_en)
            vi_chn_attr_info.buf_size = MEM_ALIGN_UP(input_config_.venc_width * input_config_.venc_height * 3 / 2, MEM_ALIGN_1K);
        else
            vi_chn_attr_info.buf_size = MEM_ALIGN_UP(input_config_.venc_width * input_config_.venc_height * 3 / 2, MEM_ALIGN_4K);

        ret = kd_sample_vicap_set_chn_attr(vi_chn_attr_info);
        if (ret != K_SUCCESS)
        {
            printf("vicap chn %d set attr failed, %x.\n", vi_chn_venc_id_, ret);
            return -1;
        }
        else
        {
            printf("vicap chn %d set attr success.\n", vi_chn_venc_id_);
        }

    }

    return 0;
}

int KdMedia::_deinit_vi_cap()
{

    return 0;
}

int KdMedia::_init_connector()
{
    k_u32 ret = 0;
    k_s32 connector_fd;
    k_connector_info connector_info;
    k_connector_type connector_type;

    connector_type = input_config_.vo_connect_type;
    memset(&connector_info, 0, sizeof(k_connector_info));
    //connector get sensor info
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("%s, the sensor type not supported!\n",__func__);
        return ret;
    }

    connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return K_ERR_VO_NOTREADY;
    }

    // set connect power
    kd_mpi_connector_power_set(connector_fd, K_TRUE);
    // connector init
    kd_mpi_connector_init(connector_fd, connector_info);

    return 0;
}

int KdMedia::_init_layer(k_vo_layer chn_id)
{
    layer_info info;
    k_vo_video_layer_attr attr;

    memset(&info, 0, sizeof(info));
    if (input_config_.vo_connect_type == LT9611_MIPI_4LAN_1920X1080_30FPS)
    {
        info.act_size.width = input_config_.vo_width;
        info.act_size.height = input_config_.vo_height;
        info.format = vi_chn_render_pixel_format_;
        info.func = K_ROTATION_0;
    }
    else if (input_config_.vo_connect_type == ST7701_V1_MIPI_2LAN_480X800_30FPS)
    {
        info.act_size.width = input_config_.vo_height;
        info.act_size.height = input_config_.vo_width;
        info.format = vi_chn_render_pixel_format_;
        info.func = K_ROTATION_90;
    }
    info.global_alptha = 0xff;
    info.offset.x = 0;//(1080-w)/2,
    info.offset.y = 0;//(1920-h)/2;

    memset(&attr, 0, sizeof(attr));
    // set offset
    attr.display_rect = info.offset;
    // set act
    attr.img_size = info.act_size;
    // sget size
    info.size = info.act_size.height * info.act_size.width * 3 / 2;
    //set pixel format
    attr.pixel_format = info.format;
    if (info.format != PIXEL_FORMAT_YVU_PLANAR_420)
    {
        printf("input pix format failed \n");
        return -1;
    }

     // set stride
    attr.stride = (info.act_size.width / 8 - 1) + ((info.act_size.height - 1) << 16);
    // set function
    attr.func = info.func;
    // set scaler attr
    attr.scaler_attr = info.attr;

    // set video layer atrr
    kd_mpi_vo_set_video_layer_attr(chn_id, &attr);

    // enable layer
    kd_mpi_vo_enable_video_layer(chn_id);

    return 0;
}

int KdMedia::_deinit_layer(k_vo_layer chn_id)
{
    return kd_mpi_vo_disable_video_layer(vo_layer_chn_id_);
}

int KdMedia::_init_osd(k_vo_osd osd_id)
{
    osd_info osd;
    osd_info *info = &osd;

    osd.act_size.width = input_config_.osd_width ;
    osd.act_size.height = input_config_.osd_height;
    osd.offset.x = 0;
    osd.offset.y = 0;
    osd.global_alptha = 0xff;
    // osd.global_alptha = 0x32;
    osd.format = osd_format_;

    k_vo_video_osd_attr attr;

    // set attr
    attr.global_alptha = info->global_alptha;

    if (info->format == PIXEL_FORMAT_ABGR_8888 || info->format == PIXEL_FORMAT_ARGB_8888)
    {
        info->size = info->act_size.width  * info->act_size.height * 4;
        info->stride  = info->act_size.width * 4 / 8;
    }
    else if (info->format == PIXEL_FORMAT_RGB_565 || info->format == PIXEL_FORMAT_BGR_565)
    {
        info->size = info->act_size.width  * info->act_size.height * 2;
        info->stride  = info->act_size.width * 2 / 8;
    }
    else if (info->format == PIXEL_FORMAT_RGB_888 || info->format == PIXEL_FORMAT_BGR_888)
    {
        info->size = info->act_size.width  * info->act_size.height * 3;
        info->stride  = info->act_size.width * 3 / 8;
    }
    else if(info->format == PIXEL_FORMAT_ARGB_4444 || info->format == PIXEL_FORMAT_ABGR_4444)
    {
        info->size = info->act_size.width  * info->act_size.height * 2;
        info->stride  = info->act_size.width * 2 / 8;
    }
    else if(info->format == PIXEL_FORMAT_ARGB_1555 || info->format == PIXEL_FORMAT_ABGR_1555)
    {
        info->size = info->act_size.width  * info->act_size.height * 2;
        info->stride  = info->act_size.width * 2 / 8;
    }
    else
    {
        printf("set osd pixel format failed  \n");
    }

    attr.stride = info->stride;
    attr.pixel_format = info->format;
    attr.display_rect = info->offset;
    attr.img_size = info->act_size;
    kd_mpi_vo_set_video_osd_attr(osd_id, &attr);

    kd_mpi_vo_osd_enable(osd_id);

    return 0;
}

int KdMedia::_deinit_osd(k_vo_osd osd_id)
{
    if (osd_vb_handle_ != VB_INVALID_HANDLE)
    {
        kd_mpi_vo_osd_disable(osd_id_);
        kd_mpi_vb_release_block(osd_vb_handle_);
        osd_vb_handle_ = VB_INVALID_HANDLE;
    }
    return 0;
}

int KdMedia::_init_vo_layer_osd()
{
    if (!feature_config_.enable_render)
    {
        return 0;
    }

    k_s32 ret = 0;
    //init connector
    if (0 != _init_connector())
    {
        printf("init connector failed\n");
        return -1;
    }

    //init layer
    if (0 != _init_layer(vo_layer_chn_id_))
    {
        printf("init layer failed\n");
        return -1;
    }

    //init osd
    if (0 != _init_osd(osd_id_))
    {
        printf("init osd failed\n");
        return -1;
    }

    //vi bind vo
    ret = kd_sample_vi_bind_vo(vi_dev_id_, vi_chn_render_id_, K_VO_DISPLAY_DEV_ID, vo_layer_chn_id_);

    return ret;
}

int KdMedia::_deinit_vo_layer_osd()
{
    k_s32 ret = 0;
    //vi unbind vo
    if (0 != kd_sample_vi_unbind_vo(vi_dev_id_, vi_chn_render_id_, K_VO_DISPLAY_DEV_ID, vo_layer_chn_id_))
    {
        printf("vi unbind vo failed\n");
        return -1;
    }

    //deinit osd
    if (0 != _deinit_osd(osd_id_))
    {
        printf("deinit osd failed\n");
        return -1;
    }

    //deinit layer
    if (0 != _deinit_layer(vo_layer_chn_id_))
    {
        printf("deinit layer failed\n");
        return -1;
    }

    return 0;
}

int KdMedia::_init_venc()
{
    if (!feature_config_.enable_video_encoder)
    {
        return 0;
    }

    k_s32 ret;
    k_venc_chn_attr chn_attr;
    k_u32 venc_chn_id = venc_chn_id_;

    memset(&chn_attr, 0, sizeof(chn_attr));
    k_u64 stream_size = input_config_.venc_width * input_config_.venc_height / 2;
    chn_attr.venc_attr.pic_width = input_config_.venc_width;
    chn_attr.venc_attr.pic_height = input_config_.venc_height;
    chn_attr.venc_attr.stream_buf_size = ((stream_size + 0xfff) & ~0xfff);
    chn_attr.venc_attr.stream_buf_cnt = 2;
    chn_attr.rc_attr.rc_mode = K_VENC_RC_MODE_CBR;
    chn_attr.rc_attr.cbr.src_frame_rate = 30;
    chn_attr.rc_attr.cbr.dst_frame_rate = 30;
    chn_attr.rc_attr.cbr.bit_rate = input_config_.bitrate_kbps;
    if (input_config_.video_type == KdMediaVideoType::kVideoTypeH264)
    {
        chn_attr.venc_attr.type = K_PT_H264;
        chn_attr.venc_attr.profile = VENC_PROFILE_H264_HIGH;
    }
    else if (input_config_.video_type == KdMediaVideoType::kVideoTypeH265)
    {
        chn_attr.venc_attr.type = K_PT_H265;
        chn_attr.venc_attr.profile = VENC_PROFILE_H265_MAIN;
    }
    else if (input_config_.video_type == KdMediaVideoType::kVideoTypeMjpeg)
    {
        chn_attr.venc_attr.type = K_PT_JPEG;
        chn_attr.rc_attr.rc_mode = K_VENC_RC_MODE_MJPEG_FIXQP;
        chn_attr.rc_attr.mjpeg_fixqp.src_frame_rate = 30;
        chn_attr.rc_attr.mjpeg_fixqp.dst_frame_rate = 30;
        chn_attr.rc_attr.mjpeg_fixqp.q_factor = 45;
    }

    ret = kd_mpi_venc_create_chn(venc_chn_id, &chn_attr);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_venc_create_chn failed:0x%x\n", ret);
        return -1;
    }

    if (input_config_.video_type != KdMediaVideoType::kVideoTypeMjpeg)
    {
        ret = kd_mpi_venc_enable_idr(venc_chn_id, K_TRUE);
        if (ret != K_SUCCESS)
        {
            printf("kd_mpi_venc_enable_idr failed:0x%x\n", ret);
            return -1;
        }
    }

    return 0;
}

int KdMedia::_deinit_venc()
{
    k_s32 ret = kd_mpi_venc_destroy_chn(venc_chn_id_);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_venc_destroy_chn failed:0x%x\n", ret);
        return -1;
    }

    return 0;
}

int KdMedia::_init_ai_aenc()
{
    if (!feature_config_.enable_audio_encoder)
    {
        return 0;
    }

    if (ai_initialized_)
        return 0;
    k_aio_dev_attr aio_dev_attr;
    memset(&aio_dev_attr, 0, sizeof(aio_dev_attr));
    aio_dev_attr.audio_type = KD_AUDIO_INPUT_TYPE_I2S;
    aio_dev_attr.kd_audio_attr.i2s_attr.sample_rate = input_config_.audio_samplerate;
    aio_dev_attr.kd_audio_attr.i2s_attr.bit_width = KD_AUDIO_BIT_WIDTH_16;
    aio_dev_attr.kd_audio_attr.i2s_attr.chn_cnt = 2;
    aio_dev_attr.kd_audio_attr.i2s_attr.i2s_mode = K_STANDARD_MODE;
    aio_dev_attr.kd_audio_attr.i2s_attr.snd_mode = (input_config_.audio_channel_cnt == 1) ? KD_AUDIO_SOUND_MODE_MONO : KD_AUDIO_SOUND_MODE_STEREO;
    aio_dev_attr.kd_audio_attr.i2s_attr.frame_num = audio_frame_divisor_;
    aio_dev_attr.kd_audio_attr.i2s_attr.point_num_per_frame = input_config_.audio_samplerate / aio_dev_attr.kd_audio_attr.i2s_attr.frame_num;
    aio_dev_attr.kd_audio_attr.i2s_attr.i2s_type = K_AIO_I2STYPE_INNERCODEC;
    if (K_SUCCESS != kd_mpi_ai_set_pub_attr(ai_dev_, &aio_dev_attr))
    {
        printf("kd_mpi_ai_set_pub_attr failed.\n");
        return -1;
    }

    k_aenc_chn_attr aenc_chn_attr;
    memset(&aenc_chn_attr, 0, sizeof(aenc_chn_attr));
    aenc_chn_attr.buf_size = audio_frame_divisor_;
    aenc_chn_attr.point_num_per_frame = input_config_.audio_samplerate / aenc_chn_attr.buf_size;
    aenc_chn_attr.type = K_PT_G711U;
    if (K_SUCCESS != kd_mpi_aenc_create_chn(aenc_handle_, &aenc_chn_attr))
    {
        printf("kd_mpi_aenc_create_chn failed.\n");
        return -1;
    }

    ai_started_ = false;
    ai_initialized_ = true;
    return 0;
}

int KdMedia::_deinit_ai_aenc()
{
    k_s32 ret;
    if (!feature_config_.enable_audio_encoder)
    {
        return 0;
    }

    if (ai_started_)
    {
        printf("KdMedia::DestroyAiAEnc called, FAILED, stop first!!!\n");
        return -1;
    }

    if (ai_initialized_)
    {
        ret = kd_mpi_aenc_destroy_chn(aenc_handle_);
        if (ret != K_SUCCESS)
        {
            printf("kd_mpi_aenc_destroy_chn failed:0x%x\n", ret);
            ai_initialized_ = false;
            return K_FAILED;
        }
    }

    ai_initialized_ = false;

    return 0;
}

int KdMedia::_start_vi_cap()
{
    return kd_sample_vicap_start(vi_dev_id_);
}

int KdMedia::_stop_vi_cap()
{
    kd_sample_vicap_stop(vi_dev_id_);
    return 0;
}

int KdMedia::_start_venc()
{
    if (!feature_config_.enable_video_encoder)
    {
        return 0;
    }

    kd_mpi_venc_start_chn(venc_chn_id_);
    if (!start_get_video_stream_)
    {
        start_get_video_stream_ = true;
        pthread_create(&venc_tid_, NULL, venc_stream_thread, this);
    }

    //from vicap chn to venc chn，without ai analysis
    kd_sample_vi_bind_venc(vi_dev_id_, vi_chn_venc_id_, venc_chn_id_);

    return 0;
}

int KdMedia::_stop_venc()
{
    if (!feature_config_.enable_video_encoder)
    {
        return 0;
    }

    if (start_get_video_stream_)
    {
        start_get_video_stream_ = false;
        pthread_join(venc_tid_, NULL);
    }
    // unbind vi to venc
    kd_sample_vi_unbind_venc(vi_dev_id_, vi_chn_venc_id_, venc_chn_id_);
    // stop  encoder channel
    kd_mpi_venc_stop_chn(venc_chn_id_);
    return 0;
}

int KdMedia::_start_ai_aenc()
{
    k_s32 ret;
    if (!feature_config_.enable_audio_encoder)
    {
        return 0;
    }

    if (ai_started_)
        return 0;
    //eanble ai dev
    ret = kd_mpi_ai_enable(ai_dev_);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_ai_enable failed:0x%x\n", ret);
        return K_FAILED;
    }

    // enable ans
    k_ai_vqe_enable vqe_enable;
    memset(&vqe_enable, 0, sizeof(vqe_enable));
    vqe_enable.ans_enable = K_TRUE;
    ret = kd_mpi_ai_set_vqe_attr(ai_dev_, ai_chn_, vqe_enable);

    //enable ai chn
    ret = kd_mpi_ai_enable_chn(ai_dev_, ai_chn_);
    //bind ai and aenc
    kd_sample_aenc_bind_ai(ai_dev_, ai_chn_, aenc_handle_);

    //start get aenc data
    if (!start_get_audio_stream_)
    {
        start_get_audio_stream_ = K_TRUE;
    }
    else
    {
        printf("aenc handle:%d already start\n", aenc_handle_);
        return K_FAILED;
    }

    if (get_audio_stream_tid_ != 0)
    {
        start_get_audio_stream_ = K_FALSE;
        pthread_join(get_audio_stream_tid_, NULL);
        get_audio_stream_tid_ = 0;
    }
    pthread_create(&get_audio_stream_tid_, NULL, aenc_chn_get_stream_thread, this);

    ai_started_ = true;
    return 0;
}

int KdMedia::_stop_ai_aenc()
{
    k_s32 ret = 0;
    if (ai_started_)
    {
        kd_sample_aenc_unbind_ai(ai_dev_, ai_chn_, aenc_handle_);
        ret = kd_mpi_ai_disable_chn(ai_dev_, ai_chn_);
        if (ret != 0)
        {
            printf("kd_mpi_ai_disable_chn failed:0x%x\n", ret);
            return -1;
        }
        ret = kd_mpi_ai_disable(ai_dev_);
        if (ret != 0)
        {
            printf("kd_mpi_ai_disable failed:0x%x\n", ret);
            return -1;
        }

        if (start_get_audio_stream_)
        {
            start_get_audio_stream_ = K_FALSE;
        }
        else
        {
            printf("aenc handle:%d already stop\n", aenc_handle_);
        }

        if (get_audio_stream_tid_ != 0)
        {
            start_get_audio_stream_ = K_FALSE;
            pthread_join(get_audio_stream_tid_, NULL);
            get_audio_stream_tid_ = 0;
        }

        ai_started_ = false;

        if (start_ai_aenc_tid_ != 0)
        {
            pthread_join(start_ai_aenc_tid_, NULL);
            start_ai_aenc_tid_ = 0;
        }
    }

    return 0;
}

int KdMedia::_start_dump_frame_for_ai_analysis()
{
    if (!feature_config_.enable_ai_analysis)
    {
        return 0;
    }

    //start get aenc data
    if (!start_dump_ai_analysis_frame_)
    {
        start_dump_ai_analysis_frame_ = K_TRUE;
    }
    else
    {
        printf("ai analysis frame already start\n");
        return K_FAILED;
    }

    if (ai_analysis_frame_tid_ != 0)
    {
        pthread_join(ai_analysis_frame_tid_, NULL);
        ai_analysis_frame_tid_ = 0;
    }
    pthread_create(&ai_analysis_frame_tid_, NULL, ai_analysis_frame_thread, this);
    return 0;
}

int KdMedia::_stop_dump_frame_for_ai_analysis()
{
    if (start_dump_ai_analysis_frame_)
    {
        start_dump_ai_analysis_frame_ = false;
        pthread_join(ai_analysis_frame_tid_, NULL);
    }
    return 0;
}

void *KdMedia::aenc_chn_get_stream_thread(void *arg)
{
    KdMedia *pthis = (KdMedia*)arg;
    k_handle aenc_hdl = pthis->aenc_handle_;
    k_audio_stream audio_stream;
    k_u8 *pData;

    while (pthis->start_get_audio_stream_)
    {
        if (0 != kd_mpi_aenc_get_stream(aenc_hdl, &audio_stream, 500))
        {
            // printf("kd_mpi_aenc_get_stream failed\n");
            continue;
        }
        else
        {
            if (pthis->feature_config_.on_aenc_data)
            {
                pData = (k_u8 *)kd_mpi_sys_mmap(audio_stream.phys_addr, audio_stream.len);
                pthis->feature_config_.on_aenc_data->OnAEncData(aenc_hdl, pData, audio_stream.len, audio_stream.time_stamp);
                kd_mpi_sys_munmap(pData, audio_stream.len);
            }
        }
        kd_mpi_aenc_release_stream(aenc_hdl, &audio_stream);
    }

    return NULL;
}

void *KdMedia::venc_stream_thread(void *arg)
{
    KdMedia *pthis = (KdMedia*)arg;
    k_u32 venc_chn = pthis->venc_chn_id_;

    k_venc_stream output;
    k_venc_chn_status status;
    k_u8 *pData;
    while (pthis->start_get_video_stream_)
    {
        kd_mpi_venc_query_status(venc_chn, &status);

        if (status.cur_packs > 0)
            output.pack_cnt = status.cur_packs;
        else
            output.pack_cnt = 1;

        output.pack = (k_venc_pack *)malloc(sizeof(k_venc_pack) * output.pack_cnt);

        kd_mpi_venc_get_stream(venc_chn, &output, -1);

        for (int i = 0; i < output.pack_cnt; i++)
        {
            pData = (k_u8 *)kd_mpi_sys_mmap(output.pack[i].phys_addr, output.pack[i].len);

            if (pthis->feature_config_.on_venc_data != nullptr)
            {
                pthis->feature_config_.on_venc_data->OnVEncData(venc_chn, pData, output.pack[i].len, output.pack[i].type, output.pack[i].pts);
            }
            // printf("=====vend data size:%d\n",output.pack[i].len);

            kd_mpi_sys_munmap(pData, output.pack[i].len);
        }

        kd_mpi_venc_release_stream(venc_chn, &output);

        free(output.pack);
    }

    return NULL;
}

void *KdMedia::ai_analysis_frame_thread(void *arg)
{
    k_s32 ret = 0;
    KdMedia *pthis = (KdMedia*)arg;
    k_video_frame_info dump_info;
    k_vicap_dev vicap_dev = pthis->vi_dev_id_;
    k_vicap_chn vi_chn = pthis->vi_chn_ai_id_;

    while (pthis->start_dump_ai_analysis_frame_)
    {
        memset(&dump_info, 0, sizeof(k_video_frame_info));
        ret = kd_mpi_vicap_dump_frame(vicap_dev, vi_chn, VICAP_DUMP_YUV, &dump_info, 1000);
        if (ret)
        {
            printf("%s kd_mpi_vicap_dump_frame failed.\n",__func__);
            continue;
        }

        if (pthis->feature_config_.on_ai_frame_data)
        {
            pthis->feature_config_.on_ai_frame_data->OnAIFrameData(vi_chn,&dump_info);
        }

        kd_mpi_vicap_dump_release(vicap_dev, vi_chn, &dump_info);

    }

    return NULL;
}

void *KdMedia::start_ai_aenc_thread(void *arg)
{
    KdMedia *pthis = (KdMedia*)arg;
    pthis->_init_ai_aenc();
    pthis->_start_ai_aenc();
    return NULL;
}

int KdMedia::osd_alloc_frame(void **osd_vaddr)
{
    k_u64 phys_addr = 0;
    k_u32 *virt_addr;
    k_vb_blk_handle handle;
    k_s32 size;

    memset(&osd_vf_info_, 0, sizeof(osd_vf_info_));
    osd_vf_info_.v_frame.width = input_config_.osd_width;
    osd_vf_info_.v_frame.height = input_config_.osd_height;
    osd_vf_info_.v_frame.stride[0] = input_config_.osd_width;
    osd_vf_info_.v_frame.pixel_format = osd_format_;

    if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_ABGR_8888 || osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_ARGB_8888)
        size = osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.width * 4;
    else if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_RGB_565 || osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_BGR_565)
        size = osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.width * 2;
    else if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_ABGR_4444 || osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_ARGB_4444)
        size = osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.width * 2;
    else if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_RGB_888 || osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_BGR_888)
        size = osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.width * 3;
    else if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_ARGB_1555 || osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_ABGR_1555)
        size = osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.width * 2;
    else if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_YVU_PLANAR_420)
        size = osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.width * 3 / 2;

    size = MEM_ALIGN_UP(size, MEM_ALIGN_4K);
    printf("vb block size is %x \n", size);

    handle = kd_mpi_vb_get_block(osd_pool_id_, size, NULL);
    if (handle == VB_INVALID_HANDLE)
    {
        printf("%s get vb block error\n", __func__);
        return -1;
    }

    phys_addr = kd_mpi_vb_handle_to_phyaddr(handle);
    if (phys_addr == 0)
    {
        printf("%s get phys addr error\n", __func__);
        return -1;
    }

    virt_addr = (k_u32 *)kd_mpi_sys_mmap(phys_addr, size);
    // virt_addr = (k_u32 *)kd_mpi_sys_mmap_cached(phys_addr, size);

    if (virt_addr == NULL)
    {
        printf("%s mmap error\n", __func__);
        return -1;
    }

    osd_vf_info_.mod_id = K_ID_VO;
    osd_vf_info_.pool_id = osd_pool_id_;
    osd_vf_info_.v_frame.phys_addr[0] = phys_addr;
    if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_YVU_PLANAR_420)
        osd_vf_info_.v_frame.phys_addr[1] = phys_addr + (osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.stride[0]);
    *osd_vaddr = virt_addr;

    printf("phys_addr is %lx osd_pool_id is %d \n", phys_addr, osd_pool_id_);

    osd_vb_handle_ = handle;
    return 0;
}

int KdMedia::osd_draw_frame()
{
    return kd_mpi_vo_chn_insert_frame(osd_id_+3, &osd_vf_info_);
}

int KdMedia::osd_send_venc_frame()
{
    return kd_mpi_venc_send_frame(venc_chn_id_, &osd_vf_info_,-1);//-1 block send
}