/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>

#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "mpi_vb_api.h"
#include "mpi_nonai_2d_api.h"
#include "mpi_venc_api.h"
#include "mpi_sys_api.h"
#include "mpi_sensor_api.h"
#include "k_nonai_2d_comm.h"
#include "k_venc_comm.h"
#include "mpi_vicap_api.h"
#include "mpi_vo_api.h"
#include "kd_display.h"

#define MAX_WIDTH 1920
#define MAX_HEIGHT 1080
#define VICAP_BUF_CNT 4
#define NONAI_2D_BUF_CNT 16
#define VENC_BUF_CNT  6
#define OSD_BUF_CNT   1
#define ISP_CHN0_WIDTH  (1280)
#define ISP_CHN0_HEIGHT (720)
#define OSD_MAX_WIDTH 100
#define OSD_MAX_HEIGHT 100
#define OSD_BUF_SIZE OSD_MAX_WIDTH*OSD_MAX_HEIGHT*4
#define NONAI_2D_BIND_CH     0
#define NONAI_2D_RGB888_CH   1
#define NONAI_2D_RGB565_CH   2
#define CSC_WARMUP_FRAMES    30

typedef struct
{
    k_u32 width;
    k_u32 height;
    k_vicap_sensor_type sensor_type;
    k_connector_type vo;
    pthread_t dump_tid;
    pthread_t output_tid;
    char out_filename[50];
    FILE *output_file;
    k_u64 osd_phys_addr;
    k_vb_blk_handle osd_blk_handle;
    k_bool auto_dump;
} nonai_2d_conf_t;

static nonai_2d_conf_t g_nonai_2d_conf;
extern const unsigned int osd_data;
extern const int osd_data_size;
static k_u8 *dump_buf_rgb888=NULL;
static k_u8 *dump_buf_rgb565=NULL;
static k_bool vicap_started=K_FALSE;
static k_u32 venc_attach_pool_id = 0;
static k_u32 nonai_2d_attach_pool_id[3] = {0};
static k_u32 osd_attach_pool_id = 0;
static volatile k_bool g_output_stop = K_FALSE;
static volatile k_bool g_dump_stop = K_FALSE;
static k_bool g_display_initialized = K_FALSE;
static volatile k_bool g_rgb888_ready = K_FALSE;
static volatile k_bool g_rgb565_ready = K_FALSE;

static inline void CHECK_RET(k_s32 ret, const char *func, const int line)
{
    if (ret)
        printf("error ret %d, func %s line %d\n", ret, func, line);
}

static k_u64 sample_fnv1a64(const k_u8 *data, size_t size)
{
    k_u64 hash = 14695981039346656037ULL;

    for (size_t i = 0; i < size; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }

    return hash;
}

static void sample_print_rgb888_stats(const k_u8 *data)
{
    const size_t pixels = g_nonai_2d_conf.width * g_nonai_2d_conf.height;
    k_u64 sum[3] = {0};
    k_u8 min[3] = {255, 255, 255};
    k_u8 max[3] = {0};

    for (size_t channel = 0; channel < 3; channel++) {
        const k_u8 *plane = data + channel * pixels;

        for (size_t i = 0; i < pixels; i++) {
            k_u8 value = plane[i];
            sum[channel] += value;
            if (value < min[channel])
                min[channel] = value;
            if (value > max[channel])
                max[channel] = value;
        }
    }

    printf("RGB888_PLANAR fnv1a64=%016lx R=%u/%lu/%u G=%u/%lu/%u B=%u/%lu/%u\n",
           (unsigned long)sample_fnv1a64(data, pixels * 3),
           min[0], (unsigned long)(sum[0] / pixels), max[0],
           min[1], (unsigned long)(sum[1] / pixels), max[1],
           min[2], (unsigned long)(sum[2] / pixels), max[2]);
}

static void sample_print_rgb565_stats(const k_u8 *data)
{
    const size_t pixels = g_nonai_2d_conf.width * g_nonai_2d_conf.height;
    k_u64 sum[3] = {0};
    k_u8 min[3] = {255, 255, 255};
    k_u8 max[3] = {0};

    for (size_t i = 0; i < pixels; i++) {
        k_u16 pixel = ((k_u16)data[i * 2 + 1] << 8) | data[i * 2];
        k_u8 value[3] = {
            (k_u8)(((pixel >> 11) & 0x1f) * 255 / 31),
            (k_u8)(((pixel >> 5) & 0x3f) * 255 / 63),
            (k_u8)((pixel & 0x1f) * 255 / 31),
        };

        for (size_t channel = 0; channel < 3; channel++) {
            sum[channel] += value[channel];
            if (value[channel] < min[channel])
                min[channel] = value[channel];
            if (value[channel] > max[channel])
                max[channel] = value[channel];
        }
    }

    printf("RGB565 fnv1a64=%016lx R=%u/%lu/%u G=%u/%lu/%u B=%u/%lu/%u\n",
           (unsigned long)sample_fnv1a64(data, pixels * 2),
           min[0], (unsigned long)(sum[0] / pixels), max[0],
           min[1], (unsigned long)(sum[1] / pixels), max[1],
           min[2], (unsigned long)(sum[2] / pixels), max[2]);
}

static k_s32 sample_compare_rgb_outputs(const k_u8 *rgb888, const k_u8 *rgb565)
{
    const size_t pixels = g_nonai_2d_conf.width * g_nonai_2d_conf.height;
    static const k_u8 permutations[][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
        {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
    };
    static const char *permutation_names[] = {
        "RGB", "RBG", "GRB", "GBR", "BRG", "BGR",
    };
    k_u64 best_sum[3] = {0};
    k_u8 best_max[3] = {0};
    k_u64 best_total = ~(k_u64)0;
    size_t best_permutation = 0;
    k_bool best_little_endian = K_FALSE;
    k_bool best_bgr_bits = K_FALSE;

    for (size_t little_endian = 0; little_endian < 2; little_endian++) {
        for (size_t bgr_bits = 0; bgr_bits < 2; bgr_bits++) {
            for (size_t permutation = 0;
                 permutation < sizeof(permutations) / sizeof(permutations[0]);
                 permutation++) {
                k_u64 error_sum[3] = {0};
                k_u8 max_error[3] = {0};

                for (size_t i = 0; i < pixels; i++) {
                    k_u16 pixel = little_endian ?
                        ((k_u16)rgb565[i * 2 + 1] << 8) | rgb565[i * 2] :
                        ((k_u16)rgb565[i * 2] << 8) | rgb565[i * 2 + 1];
                    k_u8 value[3] = {
                        (k_u8)(((pixel >> 11) & 0x1f) * 255 / 31),
                        (k_u8)(((pixel >> 5) & 0x3f) * 255 / 63),
                        (k_u8)((pixel & 0x1f) * 255 / 31),
                    };

                    if (bgr_bits) {
                        k_u8 swap = value[0];
                        value[0] = value[2];
                        value[2] = swap;
                    }

                    for (size_t channel = 0; channel < 3; channel++) {
                        k_u8 reference = rgb888[
                            permutations[permutation][channel] * pixels + i];
                        k_u8 error = reference > value[channel] ?
                            reference - value[channel] : value[channel] - reference;

                        error_sum[channel] += error;
                        if (error > max_error[channel])
                            max_error[channel] = error;
                    }
                }

                k_u64 total = error_sum[0] + error_sum[1] + error_sum[2];
                if (total < best_total) {
                    best_total = total;
                    memcpy(best_sum, error_sum, sizeof(best_sum));
                    memcpy(best_max, max_error, sizeof(best_max));
                    best_permutation = permutation;
                    best_little_endian = little_endian;
                    best_bgr_bits = bgr_bits;
                }
            }
        }
    }

    printf("RGB888/RGB565 cross-check pixels=%lu layout=%s-endian/%s-bits/planes-%s "
           "MAE=%lu/%lu/%lu max=%u/%u/%u\n",
           (unsigned long)pixels,
           best_little_endian ? "little" : "big",
           best_bgr_bits ? "BGR" : "RGB",
           permutation_names[best_permutation],
           (unsigned long)(best_sum[0] / pixels),
           (unsigned long)(best_sum[1] / pixels),
           (unsigned long)(best_sum[2] / pixels),
           best_max[0], best_max[1], best_max[2]);

    if (best_max[0] > 8 || best_max[1] > 4 || best_max[2] > 8) {
        printf("ERROR: RGB output formats disagree on color or pixel geometry\n");
        return K_FAILED;
    }

    printf("RGB output color and geometry cross-check passed\n");
    return K_SUCCESS;
}

static k_s32 sample_write_file(const char *path, const k_u8 *data, size_t size)
{
    FILE *file = fopen(path, "wb");

    if (!file) {
        printf("ERROR: cannot open %s\n", path);
        return K_FAILED;
    }
    if (fwrite(data, 1, size, file) != size) {
        printf("ERROR: short write to %s\n", path);
        fclose(file);
        return K_FAILED;
    }
    if (fclose(file) != 0) {
        printf("ERROR: cannot close %s\n", path);
        return K_FAILED;
    }

    printf("wrote %s (%lu bytes)\n", path, (unsigned long)size);
    return K_SUCCESS;
}

static void sample_vicap_config()
{
    k_s32 ret;
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_vicap_sensor_info sensor_info;

    memset(&dev_attr, 0, sizeof(k_vicap_dev_attr));
    memset(&chn_attr, 0, sizeof(k_vicap_chn_attr));
    memset(&sensor_info, 0, sizeof(k_vicap_sensor_info));

    sensor_info.sensor_type = g_nonai_2d_conf.sensor_type;
    ret = kd_mpi_vicap_get_sensor_info(sensor_info.sensor_type, &sensor_info);
    CHECK_RET(ret, __func__, __LINE__);

    dev_attr.acq_win.width = sensor_info.width;
    dev_attr.acq_win.height = sensor_info.height;
    dev_attr.mode = VICAP_WORK_ONLINE_MODE;
    dev_attr.pipe_ctrl.bits.ae_enable = 1;
    dev_attr.pipe_ctrl.bits.awb_enable = 1;

    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    CHECK_RET(ret, __func__, __LINE__);

    chn_attr.out_win.width = g_nonai_2d_conf.width;
    chn_attr.out_win.height = g_nonai_2d_conf.height;

    chn_attr.crop_win = chn_attr.out_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_FALSE;
    chn_attr.scale_enable = K_FALSE;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.alignment = 12;

    chn_attr.pix_format = PIXEL_FORMAT_RGB_888;
    chn_attr.buffer_num = VICAP_BUF_CNT;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;
    chn_attr.buffer_size = (g_nonai_2d_conf.width * g_nonai_2d_conf.height * 3 + 0xfff) & ~ 0xfff;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    CHECK_RET(ret, __func__, __LINE__);

    chn_attr.out_win.width = g_nonai_2d_conf.width;
    chn_attr.out_win.height = g_nonai_2d_conf.height;

    chn_attr.crop_win = chn_attr.out_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_FALSE;
    chn_attr.scale_enable = K_FALSE;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.alignment = 12;

    chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num = VICAP_BUF_CNT;
    chn_attr.buffer_size = (g_nonai_2d_conf.width * g_nonai_2d_conf.height * 3 / 2 + 0xfff) & ~ 0xfff;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_1, chn_attr);
    CHECK_RET(ret, __func__, __LINE__);

    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    CHECK_RET(ret, __func__, __LINE__);

}

typedef struct {
    k_connector_type type;
    k_vo_layer_id layer_id;
    int width;
    int height;
    int ratation_90;
} sample_vo_info;

static k_s32 sample_vo_init_ex(sample_vo_info* vo_info)
{
    k_s32 ret;
    k_u32 display_width;
    k_u32 display_height;
    k_gdma_rotation_e rotate = vo_info->ratation_90 ? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;

    ret = kd_display_init(vo_info->type, 0, 0, rotate);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_display_init failed, ret=%d\n", ret);
        return ret;
    }
    g_display_initialized = K_TRUE;

    ret = kd_display_get_resolution(&display_width, &display_height);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_display_get_resolution failed, ret=%d\n", ret);
        kd_display_deinit();
        g_display_initialized = K_FALSE;
        return ret;
    }

    if (g_nonai_2d_conf.width > display_width ||
        g_nonai_2d_conf.height > display_height) {
        printf("CSC frame resized from %ux%u to display %ux%u\n",
               g_nonai_2d_conf.width, g_nonai_2d_conf.height,
               display_width, display_height);
        g_nonai_2d_conf.width = display_width;
        g_nonai_2d_conf.height = display_height;
        vo_info->width = display_width;
        vo_info->height = display_height;
    }

    printf("%s>vo init width %d height %d\n", __func__, vo_info->width, vo_info->height);

    ret = kd_display_layer_configure_ex(vo_info->layer_id, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                        vo_info->width, vo_info->height, 0, 0, 0xff, rotate,
                                        vo_info->ratation_90 ? 2 : 0, 0);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_display_layer_configure failed, ret=%d\n", ret);
        kd_display_deinit();
        g_display_initialized = K_FALSE;
        return ret;
    }

    ret = kd_display_layer_enable(vo_info->layer_id);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_display_layer_enable failed, ret=%d\n", ret);
        kd_display_deinit();
        g_display_initialized = K_FALSE;
        return ret;
    }

    printf("enable vo layer:%d\n", vo_info->layer_id);
    return 0;
}

static k_s32 sample_vo_init(k_connector_type type)
{
    int width = g_nonai_2d_conf.width;
    int height = g_nonai_2d_conf.height;
    int ratation_90 = 0;
    if(type == HX8377_V2_MIPI_4LAN_1080X1920_30FPS)
    {
        width = ISP_CHN0_WIDTH ;//1080;//640;//1080;
        height = ISP_CHN0_HEIGHT;//1920;//480;//1920;
        ratation_90 = 1;
    }
    else if (type == LT9611_MIPI_4LAN_1920X1080_30FPS)
    {
        width = ISP_CHN0_WIDTH;//1080;//640;//1080;
        height = ISP_CHN0_HEIGHT;//1920;//480;//1920;
        ratation_90 = 0;
    }
    else if (type == ST7701_V1_MIPI_2LAN_480X800_30FPS)
    {
        width = ISP_CHN0_WIDTH;//480;//640;//1080;
        height = ISP_CHN0_HEIGHT;//800;//1920;//1920;
        ratation_90 = 1;
    }


    sample_vo_info vo_info;
    vo_info.type = type;
    vo_info.layer_id = K_VO_LAYER1;
    vo_info.width = width;
    vo_info.height = height;
    vo_info.ratation_90 = ratation_90;

    return sample_vo_init_ex(&vo_info);
}

void sample_vicap_start()
{
    k_s32 ret;

    ret = kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);
    CHECK_RET(ret, __func__, __LINE__);

    vicap_started = K_TRUE;
}

void sample_vicap_stop()
{
    k_s32 ret;

    vicap_started = K_FALSE;

    ret = kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
    CHECK_RET(ret, __func__, __LINE__);
    ret = kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
    CHECK_RET(ret, __func__, __LINE__);
}

static k_u32 venc_vb_create_pool()
{
    k_u32 private_pool_id;
    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = VENC_BUF_CNT;
    pool_config.blk_size = VICAP_ALIGN_UP(g_nonai_2d_conf.width*g_nonai_2d_conf.height/2, 0x1000);
    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    private_pool_id = kd_mpi_vb_create_pool(&pool_config);
    printf("%s poolid %d\n", __func__,private_pool_id);

    return private_pool_id;
}

static k_u32 nonai_2d_vb_create_pool(int chn)
{
    k_u32 private_pool_id;
    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));


    if (chn == 0){
        pool_config.blk_cnt =  NONAI_2D_BUF_CNT;
        pool_config.blk_size = VICAP_ALIGN_UP(g_nonai_2d_conf.width*g_nonai_2d_conf.height*3/2, 0x1000);
    }
    else if (chn == 1){
        pool_config.blk_cnt =  NONAI_2D_BUF_CNT;
        pool_config.blk_size = VICAP_ALIGN_UP(g_nonai_2d_conf.width*g_nonai_2d_conf.height*3, 0x1000);
    }
    else if (chn == 2){
        pool_config.blk_cnt =  NONAI_2D_BUF_CNT;
        pool_config.blk_size = VICAP_ALIGN_UP(g_nonai_2d_conf.width*g_nonai_2d_conf.height*2, 0x1000);
    }
    else{
        return -1;
    }

    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    private_pool_id = kd_mpi_vb_create_pool(&pool_config);
    printf("%s nonai_2d chn:%d, poolid %d\n", __func__,chn,private_pool_id);

    return private_pool_id;
}

static k_u32 osd_vb_create_pool()
{
    k_u32 private_pool_id;
    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = OSD_BUF_CNT;
    pool_config.blk_size = OSD_BUF_SIZE;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    private_pool_id = kd_mpi_vb_create_pool(&pool_config);
    printf("%s poolid %d\n", __func__,private_pool_id);

    return private_pool_id;
}

static k_s32 vb_destory_pool(k_u32 pool_id)
{
    k_s32 ret;

    ret = kd_mpi_vb_destory_pool(pool_id);
    CHECK_RET(ret, __func__, __LINE__);
    return ret;
}

static k_s32 sample_vb_init()
{
    k_s32 ret;
    k_vb_config config;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;

    ret = kd_mpi_vb_set_config(&config);

    if (ret)
        printf("vb_set_config failed ret:%d\n", ret);

    ret = kd_mpi_vb_init();
    if (ret)
        printf("vb_init failed ret:%d\n", ret);

    return ret;
}

static k_s32 sample_vb_exit(void)
{
    k_s32 ret;
    ret = kd_mpi_vb_exit();
    if (ret)
        printf("vb_exit failed ret:%d\n", ret);
    return ret;
}

static k_s32 prepare_osd()
{
    k_vb_blk_handle handle;
    k_s32 pool_id = 0;
    k_u64 phys_addr = 0;
    k_u8 *virt_addr_osd;

    handle = kd_mpi_vb_get_block(osd_attach_pool_id, OSD_BUF_SIZE, NULL);

    if (handle == VB_INVALID_HANDLE)
    {
        printf("%s osd get vb block error\n", __func__);
    }

    pool_id = kd_mpi_vb_handle_to_pool_id(handle);
    if (pool_id == VB_INVALID_POOLID)
    {
        printf("%s osd get pool id error\n", __func__);
    }

    phys_addr = kd_mpi_vb_handle_to_phyaddr(handle);
    if (phys_addr == 0)
    {
        printf("%s osd get phys addr error\n", __func__);
    }

    printf("%s>phys_addr 0x%lx, blk_size %d\n", __func__, phys_addr, OSD_BUF_SIZE);
    virt_addr_osd = (k_u8 *)kd_mpi_sys_mmap_cached(phys_addr, OSD_BUF_SIZE);

    if (virt_addr_osd == NULL)
    {
        printf("%s osd mmap error\n", __func__);
    }

    memcpy(virt_addr_osd, &osd_data, osd_data_size);
    kd_mpi_sys_mmz_flush_cache(phys_addr, virt_addr_osd, OSD_BUF_SIZE);

    g_nonai_2d_conf.osd_phys_addr = phys_addr;
    g_nonai_2d_conf.osd_blk_handle = handle;

    return K_SUCCESS;
}

static void *output_thread(void *arg)
{
    k_venc_stream output;
    int out_cnt, out_frames;
    k_s32 ret;
    int i;
    k_u32 total_len = 0;

    out_cnt = 0;
    out_frames = 0;

    while (!g_output_stop)
    {
        k_venc_chn_status status;

        ret = kd_mpi_venc_query_status(0, &status);
        CHECK_RET(ret, __func__, __LINE__);

        if (status.cur_packs > 0)
            output.pack_cnt = status.cur_packs;
        else
            output.pack_cnt = 1;

        output.pack = malloc(sizeof(k_venc_pack) * output.pack_cnt);

        ret = kd_mpi_venc_get_stream(0, &output, 1000);
        if(ret)
        {
            free(output.pack);
            continue;
        }

        out_cnt += output.pack_cnt;
        for (i = 0; i < output.pack_cnt; i++)
        {
            if (output.pack[i].type != K_VENC_HEADER)
            {
                out_frames++;
            }

            k_u8 *pData;
            pData = (k_u8 *)kd_mpi_sys_mmap(output.pack[i].phys_addr, output.pack[i].len);
            if (g_nonai_2d_conf.output_file)
                fwrite(pData, 1, output.pack[i].len, g_nonai_2d_conf.output_file);
            kd_mpi_sys_munmap(pData, output.pack[i].len);

            total_len += output.pack[i].len;
        }

        ret = kd_mpi_venc_release_stream(0, &output);
        CHECK_RET(ret, __func__, __LINE__);

        free(output.pack);

        //insert a new osd and border
        if(out_frames >= 90)
        {
            int ch = 0;
            k_venc_2d_osd_attr venc_2d_osd_attr;
            k_venc_2d_border_attr venc_2d_border_attr;
            int index;

            index = 1;
            venc_2d_osd_attr.width  = 40;
            venc_2d_osd_attr.height = 40;
            venc_2d_osd_attr.startx = 80;
            venc_2d_osd_attr.starty = 80;
            venc_2d_osd_attr.phys_addr[0] = g_nonai_2d_conf.osd_phys_addr;
            venc_2d_osd_attr.phys_addr[1] = 0;
            venc_2d_osd_attr.phys_addr[2] = 0;
            venc_2d_osd_attr.bg_alpha = 200;
            venc_2d_osd_attr.osd_alpha = 100;
            venc_2d_osd_attr.video_alpha = 200;
            venc_2d_osd_attr.add_order = K_VENC_2D_ADD_ORDER_VIDEO_OSD;
            venc_2d_osd_attr.bg_color = (200 << 16) | (128 << 8) | (128 << 0);
            venc_2d_osd_attr.fmt = K_VENC_2D_OSD_FMT_ARGB8888;

            ret = kd_mpi_venc_set_2d_osd_param(ch, index, &venc_2d_osd_attr);
            CHECK_RET(ret, __func__, __LINE__);

            venc_2d_border_attr.width = 40;
            venc_2d_border_attr.height = 40;
            venc_2d_border_attr.line_width = 2;
            venc_2d_border_attr.color = (100 << 16) | (200 << 8) | (70 << 0);
            venc_2d_border_attr.startx = 130;
            venc_2d_border_attr.starty = 80;
            ret = kd_mpi_venc_set_2d_border_param(ch, index, &venc_2d_border_attr);
            CHECK_RET(ret, __func__, __LINE__);
        }
    }

    /* drain: take all remaining encoded data before exit */
    while (1)
    {
        k_venc_chn_status status;

        ret = kd_mpi_venc_query_status(0, &status);
        if (ret || status.cur_packs == 0)
            break;

        output.pack_cnt = status.cur_packs;
        output.pack = malloc(sizeof(k_venc_pack) * output.pack_cnt);

        ret = kd_mpi_venc_get_stream(0, &output, 500);
        if (ret)
        {
            free(output.pack);
            break;
        }

        for (i = 0; i < output.pack_cnt; i++)
        {
            if (output.pack[i].type != K_VENC_HEADER)
                out_frames++;

            k_u8 *pData;
            pData = (k_u8 *)kd_mpi_sys_mmap(output.pack[i].phys_addr, output.pack[i].len);
            if (g_nonai_2d_conf.output_file)
                fwrite(pData, 1, output.pack[i].len, g_nonai_2d_conf.output_file);
            kd_mpi_sys_munmap(pData, output.pack[i].len);

            total_len += output.pack[i].len;
        }

        ret = kd_mpi_venc_release_stream(0, &output);
        CHECK_RET(ret, __func__, __LINE__);

        free(output.pack);
    }

    printf("%s>done, out_frames %d, size %d bits\n", __func__, out_frames, total_len * 8);
    return arg;
}

static void *dump_thread(void *arg)
{
    k_s32 ret;
    k_video_frame_info dumm_vf_info;
    k_video_frame_info rgb888_vf_info;
    k_video_frame_info rgb565_vf_info;
    k_u32 dump_cnt=0;
    k_u32 rgb888_size = g_nonai_2d_conf.width * g_nonai_2d_conf.height * 3;
    k_u32 rgb565_size = g_nonai_2d_conf.width * g_nonai_2d_conf.height * 2;

    dump_buf_rgb888 = malloc(rgb888_size);
    dump_buf_rgb565 = malloc(rgb565_size);
    if (!dump_buf_rgb888 || !dump_buf_rgb565) {
        printf("ERROR: cannot allocate CSC dump buffers\n");
        g_dump_stop = K_TRUE;
        return arg;
    }

    while (!g_dump_stop)
    {
        if(!vicap_started)
        {
            usleep(30000);
            continue;
        }

        ret = kd_mpi_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dumm_vf_info, 1000);
        if (ret) {
            printf("sample_vicap, chn(1) dump frame failed.\n");
            continue;
        }

        ret = kd_mpi_nonai_2d_send_frame(NONAI_2D_RGB888_CH, &dumm_vf_info, 1000);
        if (ret) {
            printf("kd_mpi_nonai_2d_send_frame failed. %d\n", ret);
            kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_1, &dumm_vf_info);
            continue;
        }

        ret = kd_mpi_nonai_2d_get_frame(NONAI_2D_RGB888_CH, &rgb888_vf_info, 1000);
        if (ret) {
            printf("kd_mpi_nonai_2d_get_frame failed. %d\n", ret);
            kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_1, &dumm_vf_info);
            continue;
        }

        ret = kd_mpi_nonai_2d_send_frame(NONAI_2D_RGB565_CH, &dumm_vf_info, 1000);
        if (ret) {
            printf("kd_mpi_nonai_2d_send_frame failed. %d\n", ret);
            kd_mpi_nonai_2d_release_frame(NONAI_2D_RGB888_CH, &rgb888_vf_info);
            kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_1, &dumm_vf_info);
            continue;
        }

        ret = kd_mpi_nonai_2d_get_frame(NONAI_2D_RGB565_CH, &rgb565_vf_info, 1000);
        if (ret) {
            printf("kd_mpi_nonai_2d_get_frame failed. %d\n", ret);
            kd_mpi_nonai_2d_release_frame(NONAI_2D_RGB888_CH, &rgb888_vf_info);
            kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_1, &dumm_vf_info);
            continue;
        }

        if (dump_cnt == CSC_WARMUP_FRAMES) {
            k_u8 *rgb565 = (k_u8 *)kd_mpi_sys_mmap(
                rgb565_vf_info.v_frame.phys_addr[0], rgb565_size);

            for (size_t plane = 0; plane < 3; plane++) {
                k_u8 *rgb888 = (k_u8 *)kd_mpi_sys_mmap(
                    rgb888_vf_info.v_frame.phys_addr[plane],
                    g_nonai_2d_conf.width * g_nonai_2d_conf.height);

                memcpy(dump_buf_rgb888 +
                       plane * g_nonai_2d_conf.width * g_nonai_2d_conf.height,
                       rgb888,
                       g_nonai_2d_conf.width * g_nonai_2d_conf.height);
                kd_mpi_sys_munmap(
                    rgb888, g_nonai_2d_conf.width * g_nonai_2d_conf.height);
            }
            memcpy(dump_buf_rgb565, rgb565, rgb565_size);
            kd_mpi_sys_munmap(rgb565, rgb565_size);
            g_rgb888_ready = K_TRUE;
            g_rgb565_ready = K_TRUE;
            printf("dump 2D rgb888 output done\n");
            printf("dump 2D rgb565 output done\n");
        }

        ret = kd_mpi_nonai_2d_release_frame(NONAI_2D_RGB565_CH, &rgb565_vf_info);
        if (ret) {
            printf("kd_mpi_nonai_2d_release_frame failed. %d\n", ret);
        }
        ret = kd_mpi_nonai_2d_release_frame(NONAI_2D_RGB888_CH, &rgb888_vf_info);
        if (ret) {
            printf("kd_mpi_nonai_2d_release_frame failed. %d\n", ret);
        }
        kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_1, &dumm_vf_info);

        usleep(30000);
        dump_cnt++;
    }

    printf("%s>done\n", __func__);
    return arg;
}

static void sample_bind()
{
    k_s32 ret;
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn nonai_2d_mpp_chn;
    k_mpp_chn venc_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = 0;
    vi_mpp_chn.chn_id = 0;
    nonai_2d_mpp_chn.mod_id = K_ID_NONAI_2D;
    nonai_2d_mpp_chn.dev_id = 0;
    nonai_2d_mpp_chn.chn_id = 0;
    venc_mpp_chn.mod_id = K_ID_VENC;
    venc_mpp_chn.dev_id = 0;
    venc_mpp_chn.chn_id = 0;


    ret = kd_mpi_sys_bind(&vi_mpp_chn, &nonai_2d_mpp_chn);
    CHECK_RET(ret, __func__, __LINE__);

    if (g_nonai_2d_conf.output_file) {
        ret = kd_mpi_sys_bind(&nonai_2d_mpp_chn, &venc_mpp_chn);
        CHECK_RET(ret, __func__, __LINE__);
    }

    k_mpp_chn vo_mpp_chn;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = 0;
    vo_mpp_chn.chn_id = 1;
    ret = kd_mpi_sys_bind(&nonai_2d_mpp_chn, &vo_mpp_chn);
    CHECK_RET(ret, __func__, __LINE__);

    return;
}

static void sample_unbind()
{
    k_s32 ret;
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn nonai_2d_mpp_chn;
    k_mpp_chn venc_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = 0;
    vi_mpp_chn.chn_id = 0;
    nonai_2d_mpp_chn.mod_id = K_ID_NONAI_2D;
    nonai_2d_mpp_chn.dev_id = 0;
    nonai_2d_mpp_chn.chn_id = 0;
    venc_mpp_chn.mod_id = K_ID_VENC;
    venc_mpp_chn.dev_id = 0;
    venc_mpp_chn.chn_id = 0;

    ret = kd_mpi_sys_unbind(&vi_mpp_chn, &nonai_2d_mpp_chn);
    CHECK_RET(ret, __func__, __LINE__);

    if (g_nonai_2d_conf.output_file) {
        ret = kd_mpi_sys_unbind(&nonai_2d_mpp_chn, &venc_mpp_chn);
        CHECK_RET(ret, __func__, __LINE__);
    }

    k_mpp_chn vo_mpp_chn;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = 0;
    vo_mpp_chn.chn_id = 1;
    ret = kd_mpi_sys_unbind(&nonai_2d_mpp_chn, &vo_mpp_chn);
    CHECK_RET(ret, __func__, __LINE__);
    return;
}


k_s32 sample_exit()
{
    int ch = 0;
    int ret = 0;
    int i;

    g_dump_stop = K_TRUE;
    sample_vicap_stop();
    sample_unbind();

    pthread_join(g_nonai_2d_conf.dump_tid, NULL);
    if (g_nonai_2d_conf.output_file) {
        g_output_stop = K_TRUE;
        pthread_join(g_nonai_2d_conf.output_tid, NULL);
    }

    kd_display_layer_disable(K_VO_LAYER1);

    kd_mpi_nonai_2d_stop_chn(NONAI_2D_RGB888_CH);
    kd_mpi_nonai_2d_destroy_chn(NONAI_2D_RGB888_CH);
    kd_mpi_nonai_2d_detach_vb_pool(NONAI_2D_RGB888_CH);
    kd_mpi_nonai_2d_stop_chn(NONAI_2D_BIND_CH);
    kd_mpi_nonai_2d_destroy_chn(NONAI_2D_BIND_CH);
    kd_mpi_nonai_2d_detach_vb_pool(NONAI_2D_BIND_CH);
    kd_mpi_nonai_2d_stop_chn(NONAI_2D_RGB565_CH);
    kd_mpi_nonai_2d_destroy_chn(NONAI_2D_RGB565_CH);
    kd_mpi_nonai_2d_detach_vb_pool(NONAI_2D_RGB565_CH);

    if (g_nonai_2d_conf.output_file) {
        kd_mpi_venc_detach_2d(ch);
        kd_mpi_venc_stop_chn(ch);
        kd_mpi_venc_destroy_chn(ch);
        kd_mpi_venc_detach_vb_pool(ch);

        ret = kd_mpi_venc_close_fd();
        CHECK_RET(ret, __func__, __LINE__);
    }

    ret = kd_mpi_nonai_2d_close();
    CHECK_RET(ret, __func__, __LINE__);

    if (g_nonai_2d_conf.output_file) {
        ret = kd_mpi_vb_release_block(g_nonai_2d_conf.osd_blk_handle);
        CHECK_RET(ret, __func__, __LINE__);
        vb_destory_pool(venc_attach_pool_id);
        vb_destory_pool(osd_attach_pool_id);
    }

    for (i = 0; i < 3; i++)
        vb_destory_pool(nonai_2d_attach_pool_id[i]);

    if (g_display_initialized) {
        ret = kd_display_deinit();
        CHECK_RET(ret, __func__, __LINE__);
        g_display_initialized = K_FALSE;
    }

    sample_vb_exit();

    if (g_nonai_2d_conf.output_file)
        fclose(g_nonai_2d_conf.output_file);

    return K_SUCCESS;
}

static k_s32 sample_venc_osd_h265()
{
    int ch = 0;
    k_u32 bitrate   = 4000;   //kbps
    int width       = g_nonai_2d_conf.width;
    int height      = g_nonai_2d_conf.height;
    k_venc_rc_mode rc_mode  = K_VENC_RC_MODE_CBR;
    k_payload_type type     = K_PT_H265;
    k_venc_profile profile  = VENC_PROFILE_H265_MAIN;
    int ret = 0;
    k_venc_chn_attr attr;
    k_venc_2d_osd_attr venc_2d_osd_attr;
    k_venc_2d_border_attr venc_2d_border_attr;
    int index;

    venc_attach_pool_id = venc_vb_create_pool();
    kd_mpi_venc_attach_vb_pool(ch,venc_attach_pool_id);

    memset(&attr, 0, sizeof(attr));
    attr.venc_attr.pic_width = width;
    attr.venc_attr.pic_height = height;

    attr.rc_attr.rc_mode = rc_mode;
    attr.rc_attr.cbr.src_frame_rate = 30;
    attr.rc_attr.cbr.dst_frame_rate = 30;
    attr.rc_attr.cbr.bit_rate = bitrate;

    attr.venc_attr.type = type;
    attr.venc_attr.profile = profile;
    printf("payload type is H265\n");

    ret = kd_mpi_venc_create_chn(ch, &attr);
    CHECK_RET(ret, __func__, __LINE__);

    ret = kd_mpi_venc_start_chn(ch);
    CHECK_RET(ret, __func__, __LINE__);

    kd_mpi_venc_attach_2d(ch);
    CHECK_RET(ret, __func__, __LINE__);

    ret = kd_mpi_venc_set_2d_mode(ch, K_VENC_2D_CALC_MODE_OSD_BORDER);
    CHECK_RET(ret, __func__, __LINE__);

    osd_attach_pool_id = osd_vb_create_pool();
    prepare_osd();

    index = 0;
    venc_2d_osd_attr.width  = 40;
    venc_2d_osd_attr.height = 40;
    venc_2d_osd_attr.startx = 4;
    venc_2d_osd_attr.starty = 4;
    venc_2d_osd_attr.phys_addr[0] = g_nonai_2d_conf.osd_phys_addr;
    venc_2d_osd_attr.phys_addr[1] = 0;
    venc_2d_osd_attr.phys_addr[2] = 0;
    venc_2d_osd_attr.bg_alpha = 200;
    venc_2d_osd_attr.osd_alpha = 200;
    venc_2d_osd_attr.video_alpha = 200;
    venc_2d_osd_attr.add_order = K_VENC_2D_ADD_ORDER_VIDEO_OSD;
    venc_2d_osd_attr.bg_color = (200 << 16) | (128 << 8) | (128 << 0);
    venc_2d_osd_attr.fmt = K_VENC_2D_OSD_FMT_ARGB8888;
    ret = kd_mpi_venc_set_2d_osd_param(ch, index, &venc_2d_osd_attr);
    CHECK_RET(ret, __func__, __LINE__);

    venc_2d_border_attr.width = 40;
    venc_2d_border_attr.height = 40;
    venc_2d_border_attr.line_width = 2;
    venc_2d_border_attr.color = (100 << 16) | (200 << 8) | (70 << 0);
    venc_2d_border_attr.startx = 50;
    venc_2d_border_attr.starty = 4;
    ret = kd_mpi_venc_set_2d_border_param(ch, index, &venc_2d_border_attr);
    CHECK_RET(ret, __func__, __LINE__);

    pthread_create(&g_nonai_2d_conf.output_tid, NULL, output_thread, NULL);

    return K_SUCCESS;
}

static void sample_nonai_2d()
{
    k_s32 ret;
    k_nonai_2d_chn_attr attr_2d;

    nonai_2d_attach_pool_id[NONAI_2D_BIND_CH] = nonai_2d_vb_create_pool(NONAI_2D_BIND_CH);
    nonai_2d_attach_pool_id[NONAI_2D_RGB888_CH] = nonai_2d_vb_create_pool(NONAI_2D_RGB888_CH);
    nonai_2d_attach_pool_id[NONAI_2D_RGB565_CH] = nonai_2d_vb_create_pool(NONAI_2D_RGB565_CH);

    kd_mpi_nonai_2d_attach_vb_pool(NONAI_2D_BIND_CH,nonai_2d_attach_pool_id[NONAI_2D_BIND_CH]);
    attr_2d.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d.dst_fmt = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ret = kd_mpi_nonai_2d_create_chn(NONAI_2D_BIND_CH, &attr_2d);
    CHECK_RET(ret, __func__, __LINE__);

    ret = kd_mpi_nonai_2d_start_chn(NONAI_2D_BIND_CH);
    CHECK_RET(ret, __func__, __LINE__);

    kd_mpi_nonai_2d_attach_vb_pool(NONAI_2D_RGB888_CH,nonai_2d_attach_pool_id[NONAI_2D_RGB888_CH]);
    attr_2d.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d.dst_fmt = PIXEL_FORMAT_RGB_888_PLANAR;
    ret = kd_mpi_nonai_2d_create_chn(NONAI_2D_RGB888_CH, &attr_2d);
    CHECK_RET(ret, __func__, __LINE__);

    ret = kd_mpi_nonai_2d_start_chn(NONAI_2D_RGB888_CH);
    CHECK_RET(ret, __func__, __LINE__);


    kd_mpi_nonai_2d_attach_vb_pool(NONAI_2D_RGB565_CH,nonai_2d_attach_pool_id[NONAI_2D_RGB565_CH]);
    attr_2d.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d.dst_fmt = PIXEL_FORMAT_RGB_565;
    ret = kd_mpi_nonai_2d_create_chn(NONAI_2D_RGB565_CH, &attr_2d);
    CHECK_RET(ret, __func__, __LINE__);

    ret = kd_mpi_nonai_2d_start_chn(NONAI_2D_RGB565_CH);
    CHECK_RET(ret, __func__, __LINE__);

    pthread_create(&g_nonai_2d_conf.dump_tid, NULL, dump_thread, NULL);
}

static void sample_csc_usage()
{
    printf("Usage : [-c 0|1|2] [-sensor sensor_index] [-o filename] [-vo display_type] [-L 2|4] [-dump 1]\n");
    printf("-c selects the CSI index to probe (default: board configuration)\n");
    printf("-L selects the MIPI lane preference (default: ANY)\n");
    printf("-dump 1 exits after capturing one frame from each CSC output\n");
    printf("\n");
    printf("sensor_index:\n");
    printf("\t  see vicap doc\n");
    return;
}

int main(int argc, char *argv[])
{
    k_vicap_sensor_type sensor_type = SENSOR_TYPE_MAX;
    k_vicap_mipi_lane_pref lane_pref = VICAP_MIPI_LANE_PREF_ANY;
    k_u32 csi_num = CONFIG_MPP_SENSOR_DEFAULT_CSI;
    k_bool csi_set = K_FALSE;
    k_s32 result = K_SUCCESS;

    memset(&g_nonai_2d_conf, 0, sizeof(nonai_2d_conf_t));
    g_nonai_2d_conf.width = 1280;
    g_nonai_2d_conf.height = 720;
    g_nonai_2d_conf.vo = LT9611_MIPI_4LAN_1920X1080_30FPS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
            sample_csc_usage();
            return K_SUCCESS;
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 >= argc) {
                printf("ERROR: -c requires 0, 1, or 2\n");
                return K_FAILED;
            }
            csi_num = atoi(argv[++i]);
            if (csi_num > 2) {
                printf("ERROR: -c must be 0, 1, or 2\n");
                return K_FAILED;
            }
            csi_set = K_TRUE;
        } else if (strcmp(argv[i], "-sensor") == 0) {
            if (i + 1 >= argc) {
                printf("ERROR: -sensor requires a sensor index\n");
                return K_FAILED;
            }
            sensor_type = (k_vicap_sensor_type)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) {
                printf("ERROR: -o requires a filename\n");
                return K_FAILED;
            }
            if (snprintf(g_nonai_2d_conf.out_filename,
                         sizeof(g_nonai_2d_conf.out_filename), "%s", argv[++i]) >=
                (int)sizeof(g_nonai_2d_conf.out_filename)) {
                printf("ERROR: output filename is too long\n");
                return K_FAILED;
            }
        } else if (strcmp(argv[i], "-vo") == 0) {
            if (i + 1 >= argc) {
                printf("ERROR: -vo requires a display type\n");
                return K_FAILED;
            }
            g_nonai_2d_conf.vo = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-dump") == 0) {
            if (i + 1 >= argc || atoi(argv[++i]) != 1) {
                printf("ERROR: -dump currently supports only 1\n");
                return K_FAILED;
            }
            g_nonai_2d_conf.auto_dump = K_TRUE;
        } else if (strcmp(argv[i], "-L") == 0) {
            if (i + 1 >= argc) {
                printf("ERROR: -L requires 2 or 4\n");
                return K_FAILED;
            }
            int lane = atoi(argv[++i]);
            if (lane == 2)
                lane_pref = VICAP_MIPI_LANE_PREF_2LANE;
            else if (lane == 4)
                lane_pref = VICAP_MIPI_LANE_PREF_4LANE;
            else {
                printf("ERROR: -L must be 2 or 4\n");
                return K_FAILED;
            }
        } else {
            printf("ERROR: unknown option %s\n", argv[i]);
            sample_csc_usage();
            return K_FAILED;
        }
    }

    if(SENSOR_TYPE_MAX == sensor_type) {
        k_vicap_probe_config probe_cfg;
        k_vicap_sensor_info sensor_info;

        memset(&sensor_info, 0, sizeof(sensor_info));
        probe_cfg.csi_num = csi_num;
        probe_cfg.width = 1920;
        probe_cfg.height = 1080;
        probe_cfg.fps = 30;

        if(0x00 != kd_mpi_sensor_adapt_get_ex(&probe_cfg, &sensor_info, lane_pref)) {
            printf("sample_vicap, can't probe sensor on %d, output %dx%d@%d\n", probe_cfg.csi_num, probe_cfg.width, probe_cfg.height, probe_cfg.fps);

            return -1;
        }

        sensor_type = sensor_info.sensor_type;
    } else if (csi_set) {
        k_vicap_sensor_info sensor_info;

        memset(&sensor_info, 0, sizeof(sensor_info));
        if (kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info) != K_SUCCESS) {
            printf("ERROR: invalid sensor type %d\n", sensor_type);
            return K_FAILED;
        }
        if (sensor_info.csi_num != (k_vicap_csi_num)(csi_num + VICAP_CSI0)) {
            printf("ERROR: sensor type %d is not on CSI%u\n", sensor_type, csi_num);
            return K_FAILED;
        }
    }

    g_nonai_2d_conf.sensor_type = sensor_type;
    if (g_nonai_2d_conf.out_filename[0]) {
        g_nonai_2d_conf.output_file = fopen(g_nonai_2d_conf.out_filename, "wb");
        if (!g_nonai_2d_conf.output_file) {
            printf("ERROR: cannot open output file %s\n", g_nonai_2d_conf.out_filename);
            return K_FAILED;
        }
        printf("out_filename: %s\n", g_nonai_2d_conf.out_filename);
    }

    printf("-----------csc sample test---------------------\n");

    if (sample_vb_init() != K_SUCCESS)
    {
        return -1;
    }

    if (sample_vo_init(g_nonai_2d_conf.vo) != K_SUCCESS)
    {
        printf("sample_vo_init failed\n");
        sample_vb_exit();
        return -1;
    }

    sample_nonai_2d();
    if (g_nonai_2d_conf.output_file)
        sample_venc_osd_h265();
    sample_vicap_config();
    sample_bind();
    sample_vicap_start();

    if (g_nonai_2d_conf.auto_dump) {
        int timeout = 300;

        while (timeout-- > 0 && (!g_rgb888_ready || !g_rgb565_ready))
            usleep(50000);
        if (!g_rgb888_ready || !g_rgb565_ready) {
            printf("ERROR: timed out waiting for CSC output frames\n");
            result = K_FAILED;
        }
    } else {
        printf("press 'q' to exit application!!\n");
        while (getchar() != 'q')
            usleep(50000);
    }
    sample_exit();

    if (g_rgb888_ready && g_rgb565_ready &&
        sample_compare_rgb_outputs(dump_buf_rgb888, dump_buf_rgb565) != K_SUCCESS)
        result = K_FAILED;

    if (dump_buf_rgb888) {
        size_t size = g_nonai_2d_conf.width * g_nonai_2d_conf.height * 3;

        if (g_rgb888_ready) {
            sample_print_rgb888_stats(dump_buf_rgb888);
            if (sample_write_file("/data/out_2d_rgb888.rgb", dump_buf_rgb888, size) != K_SUCCESS)
                result = K_FAILED;
        } else {
            printf("ERROR: RGB888 conversion produced no frame\n");
            result = K_FAILED;
        }
        free(dump_buf_rgb888);
    }

    if (dump_buf_rgb565) {
        size_t size = g_nonai_2d_conf.width * g_nonai_2d_conf.height * 2;

        if (g_rgb565_ready) {
            sample_print_rgb565_stats(dump_buf_rgb565);
            if (sample_write_file("/data/out_2d_rgb565.rgb", dump_buf_rgb565, size) != K_SUCCESS)
                result = K_FAILED;
        } else {
            printf("ERROR: RGB565 conversion produced no frame\n");
            result = K_FAILED;
        }
        free(dump_buf_rgb565);
    }

    printf("sample csc done!\n");

    return result;
}
