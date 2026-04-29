/* Copyright (c) 2025, Canaan Bright Sight Co., Ltd
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

/*
 * VICAP sensor preview with dump support.
 *
 * 链路：
 *   Sensor → VICAP DEV0
 *        ├─ CHN0 → Dump (支持 YUV/RGB/RAW 格式)
 *        └─ CHN1 → VO layer → Connector
 *
 * 命令行参数：
 *   sample_vicap_sensor -c <connector_type> [options]
 *
 * 例如：
 *   sample_vicap_sensor -c 20                    # 默认预览
 *   sample_vicap_sensor -c 20 -ofmt 0            # CHN0 dump YUV 格式，CHN1 预览
 *   sample_vicap_sensor -c 20 -width 1280 -height 720 -fps 60
 *
 * 交互命令（运行后输入）：
 *   d      - Dump 一帧
 *   d 5    - Dump 5 帧
 *   q      - 退出
 *
 * 注意：
 *   - CHN0 用于 dump，支持多种格式
 *   - CHN1 用于预览，固定 YUV420SP
 *   - 宽度会自动调整为 8 的倍数（驱动硬件要求）
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include "k_module.h"
#include "k_sys_comm.h"
#include "k_vb_comm.h"
#include "k_vicap_comm.h"
#include "k_vo_comm.h"

#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"

#include "kd_display.h"

// ---------------------------------------------------------------------------
// Global Defines and State
// ---------------------------------------------------------------------------


/* VICAP device */
static k_vicap_dev g_vicap_dev_id = VICAP_DEV_ID_2;

/* Ctrl+C 退出标志 */
static volatile bool g_app_run = true;

/* Sensor resolution (obtained dynamically) */
static k_u32 g_sensor_width = 1920;
static k_s32 g_sensor_fd = -1;
static k_u32 g_sensor_height = 1080;
static k_vicap_sensor_info g_sensor_info;  // Save full sensor info

/* Channel 0 output format for dump: 0=YUV, 1=RGB888, 2=RGB888P, 3=RAW */
static k_u32 g_ch0_format = 0;

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------

static void handle_signal(int sig)
{
    if (sig == SIGINT) {
        printf("Caught SIGINT, exiting preview loop...\n");
        g_app_run = false;
    }
}

// Dump frame to file
static void sample_vicap_dump_frame(k_vicap_dev dev, k_vicap_chn chn, k_u32 *dump_count)
{
    k_video_frame_info dump_info;
    k_s32 ret;
    
    memset(&dump_info, 0, sizeof(dump_info));
    
    ret = kd_mpi_vicap_dump_frame(dev, chn, VICAP_DUMP_YUV, &dump_info, 1000);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_dump_frame failed, ret=%d\n", ret);
        return;
    }
    
    k_char *suffix;
    k_u32 data_size = 0;
    k_u8 *virt_addr = NULL;
    k_char filename[256];
    
    if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_420) {
        suffix = "yuv420sp";
        data_size = dump_info.v_frame.width * dump_info.v_frame.height * 3 / 2;
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_888) {
        suffix = "rgb888";
        data_size = dump_info.v_frame.width * dump_info.v_frame.height * 3;
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_888_PLANAR) {
        suffix = "rgb888p";
        data_size = dump_info.v_frame.width * dump_info.v_frame.height * 3;
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_BAYER_10BPP) {
        suffix = "raw10";
        data_size = dump_info.v_frame.width * dump_info.v_frame.height * 2;
    } else {
        suffix = "yuv420sp";
        data_size = dump_info.v_frame.width * dump_info.v_frame.height * 3 / 2;
    }
    
    virt_addr = kd_mpi_sys_mmap(dump_info.v_frame.phys_addr[0], data_size);
    if (virt_addr) {
        memset(filename, 0, sizeof(filename));
        snprintf(filename, sizeof(filename), "vicap_dev%d_chn%d_%dx%d_%04d.%s",
                 dev, chn, dump_info.v_frame.width, dump_info.v_frame.height, *dump_count, suffix);
        
        printf("Saving dump data to %s...\n", filename);
        FILE *file = fopen(filename, "wb");
        if (file) {
            fwrite(virt_addr, 1, data_size, file);
            fclose(file);
            printf("Dump saved: %s (%u bytes)\n", filename, data_size);
        } else {
            printf("ERROR: Failed to open dump file\n");
        }
        kd_mpi_sys_munmap(virt_addr, data_size);
    } else {
        printf("ERROR: Failed to mmap dump address\n");
    }
    
    ret = kd_mpi_vicap_dump_release(dev, chn, &dump_info);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_dump_release failed, ret=%d\n", ret);
    }
    
    (*dump_count)++;
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -c <type>   Connector type (e.g., 20) [REQUIRED]\n");
    printf("  -r <degree> Rotation (0, 90, 180, 270) [default: 0]\n");
    printf("  -s <csi>    CSI index (0-2) [default: 2]\n");
    printf("  -ae <0|1>   AE status (0: disable, 1: enable) [default: 1]\n");
    printf("  -exp <value> Manual exposure time in microseconds (must set -ae 0 when using)\n");
    printf("  -again <value> Manual analog gain (must set -ae 0 when using)\n");
    printf("  -awb <0|1>  AWB status (0: disable, 1: enable) [default: 1]\n");
    printf("  -hdr <0|1>  HDR status (0: disable, 1: enable) [default: 0]\n");
    printf("  -dw <0|1>   Dewarp status (0: disable, 1: enable) [default: 0]\n");
    printf("  -dnr3 <0|1> DNR3 status (0: disable, 1: enable) [default: 1]\n");
    printf("  -width <value> Sensor width [default: 1920]\n");
    printf("  -height <value> Sensor height [default: 1080]\n");
    printf("  -fps <value> Sensor FPS [default: 30]\n");
    printf("  -ofmt <0|1|2|3> Channel 0 format [0:yuv, 1:rgb888, 2:rgb888p, 3:raw][default: 0]\n");
    printf("  -scene_name <name>  Scene name (e.g., \"day\", \"night\")\n");
    printf("  -scene_path <path>  Config path (must end with /, e.g., \"/etc/vicap/day/\")\n");
    printf("\nNote: CHN0 for dump, CHN1 for preview.\n");
    printf("\nExample:\n");
    printf("  %s -c 20 -ae 1 -awb 1 -hdr 0\n", prog);
    printf("  %s -c 20 -ae 0 -exp 10000  # 10000 us = 10ms\n", prog);
    printf("  %s -c 20 -scene_name day -scene_path /etc/vicap/day/\n", prog);
}

// ---------------------------------------------------------------------------
// Parameter parsing
// ---------------------------------------------------------------------------

typedef struct {
    k_connector_type connector_type;
    int rot_val;
    int csi_idx;
    k_bool ae_enable;
    k_bool awb_enable;
    k_bool hdr_enable;
    k_bool dw_enable;
    k_bool dnr3_enable;
    k_bool exp_set;
    k_u32 exp_value_us;
    k_u32 sensor_width;
    k_u32 sensor_height;
    k_u32 sensor_fps;
    k_u32 ch0_format;
    k_bool again_set;
    float again_value;
    bool c_set;
    char scene_name[32];
    char scene_path[256];
    k_bool scene_set;
} sample_params_t;

static k_s32 parse_parameters(int argc, char **argv, sample_params_t *params)
{
    memset(params, 0, sizeof(sample_params_t));
    
    // Set defaults
    params->csi_idx = 2;
    params->ae_enable = K_TRUE;
    params->awb_enable = K_TRUE;
    params->hdr_enable = K_FALSE;
    params->dw_enable = K_FALSE;
    params->dnr3_enable = K_TRUE;  // Default enable DNR3
    params->sensor_width = 1920;
    params->sensor_height = 1080;
    params->sensor_fps = 30;
    params->ch0_format = 0;
    params->again_set = K_FALSE;
    params->again_value = 1.0f;
    params->scene_set = K_FALSE;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            params->connector_type = (k_connector_type)atoi(argv[++i]);
            params->c_set = true;
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            params->rot_val = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            params->csi_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ae") == 0 && i + 1 < argc) {
            params->ae_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-awb") == 0 && i + 1 < argc) {
            params->awb_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-hdr") == 0 && i + 1 < argc) {
            params->hdr_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-dw") == 0 && i + 1 < argc) {
            params->dw_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-exp") == 0 && i + 1 < argc) {
            params->exp_set = K_TRUE;
            params->exp_value_us = (k_u32)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-width") == 0 && i + 1 < argc) {
            params->sensor_width = (k_u32)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-height") == 0 && i + 1 < argc) {
            params->sensor_height = (k_u32)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-fps") == 0 && i + 1 < argc) {
            params->sensor_fps = (k_u32)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ofmt") == 0 && i + 1 < argc) {
            params->ch0_format = (k_u32)atoi(argv[++i]);
            if (params->ch0_format > 3) {
                printf("ERROR: Invalid ofmt value, must be 0-3\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-dnr3") == 0 && i + 1 < argc) {
            params->dnr3_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-again") == 0 && i + 1 < argc) {
            params->again_set = K_TRUE;
            params->again_value = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-scene_name") == 0 && i + 1 < argc) {
            strncpy(params->scene_name, argv[++i], sizeof(params->scene_name) - 1);
            params->scene_set = K_TRUE;
        } else if (strcmp(argv[i], "-scene_path") == 0 && i + 1 < argc) {
            strncpy(params->scene_path, argv[++i], sizeof(params->scene_path) - 1);
            params->scene_set = K_TRUE;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
            return 1;  // Help requested
        } else {
            printf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    
    if (!params->c_set) {
        printf("ERROR: Missing required arguments (-c)\n");
        return -1;
    }
    
    // If exposure is set manually, AE must be disabled
    if (params->exp_set && params->ae_enable == K_TRUE) {
        printf("ERROR: When -exp is set, -ae must be set to 0\n");
        return -1;
    }
    
    // If again is set manually, AE must be disabled
    if (params->again_set && params->ae_enable == K_TRUE) {
        printf("ERROR: When -again is set, -ae must be set to 0\n");
        return -1;
    }
    
    return 0;
}

// ---------------------------------------------------------------------------
// VB init
// ---------------------------------------------------------------------------

// Helper function to get sensor resolution
static k_s32 get_sensor_resolution(k_vicap_dev dev_chn, k_u32 *width, k_u32 *height, k_u32 *fps,
                                   k_u32 req_width, k_u32 req_height, k_u32 req_fps)
{
    k_vicap_sensor_info sensor_info;
    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_type sensor_type;
    
    memset(&sensor_info, 0, sizeof(sensor_info));
    memset(&probe_cfg, 0, sizeof(probe_cfg));
    
    probe_cfg.csi_num = dev_chn;
    probe_cfg.width   = req_width;
    probe_cfg.height  = req_height;
    probe_cfg.fps     = req_fps;
    
    if (0x00 != kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info)) {
        printf("ERROR: kd_mpi_sensor_adapt_get failed on CSI %d\n", probe_cfg.csi_num);
        return -1;
    }
    
    sensor_type = sensor_info.sensor_type;
    
    k_s32 ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_get_sensor_info failed, ret=%d\n", ret);
        return ret;
    }
    printf("DEBUG: sensor_name=%s, sensor_type=%d, csi_num=%d\n", 
           sensor_info.sensor_name ? sensor_info.sensor_name : "NULL", 
           sensor_info.sensor_type, sensor_info.csi_num);
    
    if (width) *width = sensor_info.width;
    if (height) *height = sensor_info.height;
    if (fps) *fps = sensor_info.fps;
    
    // Save full sensor info for later use
    memcpy(&g_sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));
    printf("INFO: Saved sensor: %s, type=%d, csi=%d\n",
           g_sensor_info.sensor_name ? g_sensor_info.sensor_name : "NULL",
           g_sensor_info.sensor_type, g_sensor_info.csi_num);
    
    return K_SUCCESS;
}

static k_s32 sample_vb_init(void)
{
    k_vb_config config;
    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;

    k_s32 ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("ERROR: kd_mpi_vb_set_config failed, ret=%d\n", ret);
        return ret;
    }

    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;

    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("ERROR: kd_mpi_vb_set_supplement_config failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vb_init();
    if (ret) {
        printf("ERROR: kd_mpi_vb_init failed, ret=%d\n", ret);
        return ret;
    }

    return K_SUCCESS;
}

// ---------------------------------------------------------------------------
// VICAP init (sensor → VICAP dev0/ch0, 输出 NV12，指定宽高)
// ---------------------------------------------------------------------------

static k_s32 sample_vicap_init(k_vicap_dev dev_chn, k_u32 out_width, k_u32 out_height,
                                      k_bool ae_en, k_bool awb_en, k_bool hdr_en, k_bool dw_en,
                                      k_bool dnr3_en, k_u32 ch0_format)
{
    k_vicap_dev_attr     dev_attr;
    k_vicap_chn_attr     chn_attr;
    k_vicap_sensor_info  sensor_info = {0};
    k_s32 ret;

    // Width must be multiple of 8 (hardware requirement)
    if ((out_width & 7) != 0) {
        out_width = (out_width + 7) & ~7;  // Align to 8
        printf("INFO: VICAP output width aligned to %u\n", out_width);
    }

    // Use saved sensor info (from get_sensor_resolution in main)
    memcpy(&sensor_info, &g_sensor_info, sizeof(k_vicap_sensor_info));
    printf("INFO: Using sensor: %s, %ux%u, type=%d, csi=%d\n",
           sensor_info.sensor_name ? sensor_info.sensor_name : "NULL",
           sensor_info.width, sensor_info.height,
           sensor_info.sensor_type, sensor_info.csi_num);

    if(sensor_info.sensor_name == NULL)
    {
        printf("no sensor find in csi %d,please check\n", sensor_info.csi_num);
        return -1;
    }

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width  = sensor_info.width;   // Use actual sensor width
    dev_attr.acq_win.height = sensor_info.height;  // Use actual sensor height
    dev_attr.mode           = VICAP_WORK_ONLINE_MODE;
    dev_attr.buffer_num     = 6;
    dev_attr.buffer_size    = VB_ALIGN_UP(sensor_info.width * sensor_info.height * 2, 1024);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(sensor_info));  // Copy sensor info
    
    // Configure pipe control with AE/AWB/HDR/DNR3 settings
    dev_attr.pipe_ctrl.data = 0xFFFFFFFF;
    dev_attr.pipe_ctrl.bits.ae_enable = ae_en;
    dev_attr.pipe_ctrl.bits.awb_enable = awb_en;
    dev_attr.pipe_ctrl.bits.ahdr_enable = hdr_en;
    dev_attr.pipe_ctrl.bits.dnr3_enable = dnr3_en;
    dev_attr.dw_enable = dw_en;

    /* 简单起见：固定使用 VICAP_DEV_ID_0 做采集，CSI 从 g_vicap_dev_id 控制 */
    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_dev_attr failed, ret=%d\n", ret);
        return ret;
    }

    // Configure CHN0 for dump
    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = dev_attr.acq_win.width;   // Sensor native width
    chn_attr.out_win.height = dev_attr.acq_win.height;  // Sensor native height
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_FALSE;
    chn_attr.scale_enable   = K_FALSE;  // Disable scaling for CHN0
    chn_attr.chn_enable     = K_TRUE;
    
    // Set CHN0 format based on ofmt parameter
    switch (ch0_format) {
        case 0:
            chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            chn_attr.buffer_size = VB_ALIGN_UP(dev_attr.acq_win.width * dev_attr.acq_win.height * 3 / 2, 4096);
            break;
        case 1:
            chn_attr.pix_format = PIXEL_FORMAT_RGB_888;
            chn_attr.buffer_size = VB_ALIGN_UP(dev_attr.acq_win.width * dev_attr.acq_win.height * 3, 4096);
            break;
        case 2:
            chn_attr.pix_format = PIXEL_FORMAT_RGB_888_PLANAR;
            chn_attr.buffer_size = VB_ALIGN_UP(dev_attr.acq_win.width * dev_attr.acq_win.height * 3, 4096);
            break;
        case 3:
            chn_attr.pix_format = PIXEL_FORMAT_RGB_BAYER_10BPP;
            chn_attr.buffer_size = VB_ALIGN_UP(dev_attr.acq_win.width * dev_attr.acq_win.height * 2, 4096);
            break;
        default:
            chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            chn_attr.buffer_size = VB_ALIGN_UP(dev_attr.acq_win.width * dev_attr.acq_win.height * 3 / 2, 4096);
            break;
    }
    chn_attr.buffer_num     = 6;
    chn_attr.alignment      = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr CHN0 failed, ret=%d\n", ret);
        return ret;
    }
    
    // Configure CHN1 for preview (YUV420SP)
    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = out_width;
    chn_attr.out_win.height = out_height;
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_TRUE;
    chn_attr.scale_enable   = K_FALSE;
    chn_attr.chn_enable     = K_TRUE;
    chn_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num     = 6;
    chn_attr.buffer_size    = VB_ALIGN_UP(out_width * out_height * 3 / 2, 4096);
    chn_attr.alignment      = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_1, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_init failed, ret=%d\n", ret);
        return ret;
    }

    // Get sensor fd for exposure control
    k_vicap_sensor_attr sensor_attr;
    sensor_attr.dev_num = VICAP_DEV_ID_0;
    ret = kd_mpi_vicap_get_sensor_fd(&sensor_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_get_sensor_fd failed, ret=%d\n", ret);
        return ret;
    }
    g_sensor_fd = sensor_attr.sensor_fd;
    printf("INFO: Sensor fd = %d\n", g_sensor_fd);

    return K_SUCCESS;
}

// ---------------------------------------------------------------------------
// VICAP ↔ VO bind / unbind
// ---------------------------------------------------------------------------

static void sample_vicap_bind_vo(k_vo_layer_id layer_id)
{
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn vo_mpp_chn;

    memset(&vi_mpp_chn, 0, sizeof(vi_mpp_chn));
    memset(&vo_mpp_chn, 0, sizeof(vo_mpp_chn));

    /* 源：VI / VICAP DEV0 CH1 (preview) */
    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = VICAP_CHN_ID_1;

    /* 目的：VO 显示设备上的指定 video layer */
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = layer_id;

    k_s32 ret = kd_mpi_sys_bind(&vi_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("ERROR: kd_mpi_sys_bind VICAP->VO failed, ret=0x%x\n", ret);
    } else {
        printf("Bind VICAP(dev=%d, ch=1) -> VO(layer=%d) OK\n",
               VICAP_DEV_ID_0, layer_id);
    }
}

static void sample_vicap_unbind_vo(k_vo_layer_id layer_id)
{
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn vo_mpp_chn;

    memset(&vi_mpp_chn, 0, sizeof(vi_mpp_chn));
    memset(&vo_mpp_chn, 0, sizeof(vo_mpp_chn));

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = VICAP_CHN_ID_1;

    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = layer_id;

    k_s32 ret = kd_mpi_sys_unbind(&vi_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("WARN: kd_mpi_sys_unbind VICAP->VO failed, ret=0x%x\n", ret);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    k_s32 ret;
    k_connector_type connector_type = 0;
    k_u32 width = 0, height = 0;  // Auto-detected from connector
    k_vo_layer_id layer_id = 1;
    int rot_val = 0;
    // offset_x and offset_y always 0 for fullscreen
    int csi_idx = 2; // Default CSI 2
    
    // VICAP features
    k_bool ae_enable = K_TRUE;    // Default enable AE
    k_bool awb_enable = K_TRUE;   // Default enable AWB
    k_bool hdr_enable = K_FALSE;   // Default disable HDR
    k_bool dw_enable = K_FALSE;    // Default disable dewarp
    k_bool dnr3_enable = K_TRUE;   // Default enable DNR3
    
    // Manual exposure (in microseconds)
    k_bool exp_set = K_FALSE;
    k_u32 exp_value_us = 0;
    
    // Manual again
    k_bool again_set = K_FALSE;
    float again_value = 1.0f;
    
    // Sensor configuration
    k_u32 sensor_width = 1920;   // Default sensor width
    k_u32 sensor_height = 1080;  // Default sensor height
    k_u32 sensor_fps = 30;       // Default sensor FPS

    sample_params_t params;
    k_s32 parse_ret = parse_parameters(argc, argv, &params);
    if (parse_ret == 1) {
        print_usage(argv[0]);
        return 0;
    } else if (parse_ret < 0) {
        print_usage(argv[0]);
        return -1;
    }
    
    // Copy parameters to local variables
    connector_type = params.connector_type;
    rot_val = params.rot_val;
    csi_idx = params.csi_idx;
    ae_enable = params.ae_enable;
    awb_enable = params.awb_enable;
    hdr_enable = params.hdr_enable;
    dw_enable = params.dw_enable;
    dnr3_enable = params.dnr3_enable;
    exp_set = params.exp_set;
    exp_value_us = params.exp_value_us;
    sensor_width = params.sensor_width;
    sensor_height = params.sensor_height;
    sensor_fps = params.sensor_fps;
    g_ch0_format = params.ch0_format;
    again_set = params.again_set;
    again_value = params.again_value;

    g_vicap_dev_id = (k_vicap_dev)csi_idx;

    // Get actual sensor resolution before calculating aspect ratio
    ret = get_sensor_resolution(g_vicap_dev_id, &g_sensor_width, &g_sensor_height, NULL, sensor_width, sensor_height, sensor_fps);
    if (ret != K_SUCCESS) {
        printf("ERROR: Failed to get sensor resolution\n");
        return -1;
    }
    printf("INFO: Sensor resolution: %ux%u\n", g_sensor_width, g_sensor_height);

       /* 1. 初始化 VB */
    ret = sample_vb_init();
    if (ret != K_SUCCESS) {
        printf("ERROR: sample_vb_init failed, ret=%d\n", ret);
        return -1;
    }

    // Initialize display first to get correct resolution
    k_gdma_rotation_e rotate = GDMA_ROTATE_DEGREE_0;
    if (rot_val == 90) rotate = GDMA_ROTATE_DEGREE_90;
    else if (rot_val == 180) rotate = GDMA_ROTATE_DEGREE_180;
    else if (rot_val == 270) rotate = GDMA_ROTATE_DEGREE_270;

    if(0x00 != kd_display_init(connector_type, 0, 0, rotate)) {
        printf("ERROR: connector init failed\n");
        return -1;
    }

    // Get display resolution from connector (after init)
    k_vo_size display_resolution;
    ret = kd_mpi_vo_get_resolution(&display_resolution);
    if (ret != 0) {
        printf("WARNING: Failed to get display resolution, using default\n");
        display_resolution.width = 1920;
        display_resolution.height = 1080;
    }

    // Calculate output size maintaining sensor aspect ratio to avoid distortion
    // Consider three cases: sensor > screen, sensor < screen, sensor == screen
    float sensor_aspect = (float)g_sensor_width / (float)g_sensor_height;
    float screen_aspect = (float)display_resolution.width / (float)display_resolution.height;

    // Case 1: Sensor resolution larger than screen (need downscale)
    // Case 2: Sensor resolution smaller than screen (can upscale or fit)
    // Case 3: Sensor resolution equals screen (direct match)
    
    // Always maintain aspect ratio, fit within screen bounds
    if (sensor_aspect > screen_aspect) {
        // Sensor is wider: fit to screen width, letterbox top/bottom
        width = display_resolution.width;
        height = (k_u32)((float)width / sensor_aspect);
    } else if (sensor_aspect < screen_aspect) {
        // Sensor is taller: fit to screen height, letterbox left/right
        height = display_resolution.height;
        width = (k_u32)((float)height * sensor_aspect);
    } else {
        // Same aspect ratio: direct match
        width = display_resolution.width;
        height = display_resolution.height;
    }
    
    // Ensure output dimensions don't exceed screen bounds
    if (width > display_resolution.width) {
        width = display_resolution.width;
        height = (k_u32)((float)width / sensor_aspect);
    }
    if (height > display_resolution.height) {
        height = display_resolution.height;
        width = (k_u32)((float)height * sensor_aspect);
    }

    // Align width to 8 pixels (hardware requirement)
    if ((width & 7) != 0) {
        width = (width + 7) & ~7;
    }
    // Align height to 8 pixels
    if ((height & 7) != 0) {
        height = (height + 7) & ~7;
    }

    printf("sample_vicap_sensor: connector=%d, screen_size=%ux%u, output_size=%ux%u, layer=%d, rotate=%d, csi=%d\n",
           connector_type, display_resolution.width, display_resolution.height, width, height, layer_id, rot_val, g_vicap_dev_id);
    printf("Aspect ratio mode: sensor=%.2f, screen=%.2f\n", sensor_aspect, screen_aspect);
    printf("CHN0 format: %d (0=yuv, 1=rgb888, 2=rgb888p, 3=raw)\n", g_ch0_format);

    printf("Fullscreen mode (offset_x=0, offset_y=0)\n");
    printf("VICAP features: AE=%d, AWB=%d, HDR=%d, Dewarp=%d, DNR3=%d",
           ae_enable, awb_enable, hdr_enable, dw_enable, dnr3_enable);
    if (exp_set) {
        printf(", Manual Exp=%u us", exp_value_us);
    }
    printf("\n");

    signal(SIGINT, handle_signal);

    // 如果设置了场景参数，注册并加载场景
    if (params.scene_set) {
        printf("\nRegistering scene '%s' with path '%s'...\n", params.scene_name, params.scene_path);
        ret = kd_mpi_vicap_register_scene(params.scene_name, params.scene_path);
        if (ret < 0) {
            printf("ERROR: Failed to register scene\n");
            goto cleanup_display;
        }
        
        // 加载场景（在启动流之前）
        printf("Loading scene '%s'...\n", params.scene_name);
        ret = kd_mpi_vicap_load_scene(params.scene_name);
        if (ret < 0) {
            printf("ERROR: Failed to load scene\n");
            goto cleanup_display;
        }
        
        const char *current = kd_mpi_vicap_get_scene();
        printf("Current scene: %s\n", current ? current : "none");
    }

    /* 2. 初始化 VICAP（sensor + dev0/ch0 输出到指定分辨率） */
    ret = sample_vicap_init(g_vicap_dev_id, width, height, ae_enable, awb_enable, hdr_enable, dw_enable, dnr3_enable, g_ch0_format);
    if (ret != K_SUCCESS) {
        printf("ERROR: sample_vicap_init failed, ret=%d\n", ret);
        goto cleanup_display;
    }


    // Configure layer for fullscreen (offset_x=0, offset_y=0)
    ret = kd_display_layer_configure(layer_id, PIXEL_FORMAT_YUV_SEMIPLANAR_420, width, height, 0, 0);
    if (0x00 != ret) {
        printf("ERROR: failed to configure layer, ret=%d\n", ret);
        goto cleanup_vicap;
    }
    kd_display_layer_enable(layer_id);

    /* 4. 绑定 VICAP → VO */
    sample_vicap_bind_vo(layer_id);

    /* 5. 启动 VICAP 流，开始预览 */
    printf("Starting VICAP stream on dev0 ch0 ...\n");
    ret = kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vicap_start_stream failed, ret=%d\n", ret);
        goto cleanup_bind;
    }

    /* 6. 如果设置了手动曝光，在 sensor start 之后设置曝光 */
    if (exp_set) {
        
        k_sensor_exposure_time_range range;
        ret = kd_mpi_sensor_get_exposure_time_range(g_sensor_fd, &range);
        if (ret != 0) {
            printf("ERROR: kd_mpi_sensor_get_exposure_time_range failed, ret=%d\n", ret);
            goto cleanup_stream;
        }
        
        if (exp_value_us < (k_u32)range.min_intg_time_us || exp_value_us > (k_u32)range.max_intg_time_us) {
            printf("ERROR: Exposure value %u us is out of range [%.0f us, %.0f us]\n", 
                   exp_value_us, range.min_intg_time_us, range.max_intg_time_us);
            printf("Please set -exp to a value between %.0f and %.0f (microseconds)\n", range.min_intg_time_us, range.max_intg_time_us);
            goto cleanup_stream;
        }
        
        // Convert microseconds to seconds for the API call
        float exp_value_sec = (float)exp_value_us / 1000000.0f;
        k_sensor_intg_time intg_time;
        intg_time.intg_time[0] = exp_value_sec;  // SENSOR_LINEAR_PARAS = 0
        ret = kd_mpi_sensor_intg_time_set(g_sensor_fd, intg_time);
        if (ret != 0) {
            printf("ERROR: kd_mpi_sensor_intg_time_set failed, ret=%d\n", ret);
            goto cleanup_stream;
        }
        printf("INFO: Manual exposure set to %u us (%.6f sec), range: %.6f-%.6f sec\n", 
               exp_value_us, exp_value_sec, (float)range.min_intg_time_us / 1000000.0f, (float)range.max_intg_time_us / 1000000.0f);
    }

    // Set manual again if specified
    if (again_set) {
        k_sensor_gain_info gain_range;
        ret = kd_mpi_sensor_get_gain_range(g_sensor_fd, &gain_range);
        if (ret != 0) {
            printf("ERROR: kd_mpi_sensor_get_gain_range failed, ret=%d\n", ret);
            goto cleanup_stream;
        }
        
        if (again_value < gain_range.min || again_value > gain_range.max) {
            printf("ERROR: Again value %.2f is out of range [%.2f, %.2f]\n", 
                   again_value, gain_range.min, gain_range.max);
            printf("Please set -again to a value between %.2f and %.2f\n", gain_range.min, gain_range.max);
            goto cleanup_stream;
        }
        
        k_sensor_gain gain;
        gain.gain[0] = again_value;
        ret = kd_mpi_sensor_again_set(g_sensor_fd, gain);
        if (ret != 0) {
            printf("ERROR: kd_mpi_sensor_again_set failed, ret=%d\n", ret);
            goto cleanup_stream;
        }
        printf("INFO: Manual again set to %.2f (range: %.2f-%.2f, step: %.6f)\n", 
               again_value, gain_range.min, gain_range.max, gain_range.step);
    }

    printf("Preview running.\n");
    printf("CHN0: dump (format=%d), CHN1: preview\n", g_ch0_format);
    printf("Commands: d=dump 1 frame, d <n>=dump n frames, q=quit\n");
    
    k_u32 dump_count = 0;
    char cmd_buf[64];
    
    while (g_app_run) {
        printf("\n---------------------------------------\n");
        printf(" Input command:\n");
        printf("   d      - Dump one frame\n");
        printf("   d <n>  - Dump n frames\n");
        printf("   q      - Quit\n");
        printf("---------------------------------------\n");
        printf("Command: ");
        fflush(stdout);
        
        if (fgets(cmd_buf, sizeof(cmd_buf), stdin) == NULL) {
            continue;
        }
        
        // Remove newline
        cmd_buf[strcspn(cmd_buf, "\n")] = 0;
        
        char cmd = cmd_buf[0];
        int count = 1;
        
        // Parse "d <n>" format
        if (cmd == 'd' || cmd == 'D') {
            char *space = strchr(cmd_buf, ' ');
            if (space) {
                count = atoi(space + 1);
                if (count <= 0) count = 1;
            }
            printf("Dumping %d frame(s)...\n", count);
            for (int i = 0; i < count; i++) {
                sample_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &dump_count);
            }
        } else if (cmd == 'q' || cmd == 'Q') {
            printf("Exiting...\n");
            break;
        } else if (cmd != '\0') {
            printf("Unknown command: %c\n", cmd);
        }
    }

cleanup_stream:
    printf("Stopping VICAP stream ...\n");
    kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);

cleanup_bind:
    sample_vicap_unbind_vo(layer_id);

cleanup_display:
    kd_display_deinit();

cleanup_vicap:
    printf("Deinitializing VICAP dev0 ...\n");
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);

    printf("Deinitializing VB ...\n");
    kd_mpi_vb_exit();

    printf("vo_test_video exit.\n");
    return 0;
}
