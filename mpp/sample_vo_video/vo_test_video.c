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
 * Very simple “sensor -> VICAP -> VO” preview sample.
 *
 *参考 sample_uvc_dev_vicap/main.c 和 sample_wbc_dump/sample_wbc.c，
 *做了一条最精简的链路：
 *
 *   Sensor → VICAP(CH0, YUV420SP, 可缩放到指定宽高)
 *        └─ kd_mpi_sys_bind ─→ VO layer(NV12/YUV420SP) → Connector
 *
 * 命令行参数：
 *   vo_test_video -c <connector_type> -l <vo_layer_id> [-w <width>] [-h <height>] [options]
 *
 * 例如先用 list_connector 查看 connector type，VO layer = 2，CSI 用默认 2：
 *   vo_test_video -c <connector_type> -l 2
 *
 * 注意：
 *   - VICAP 输出格式固定为 PIXEL_FORMAT_YUV_SEMIPLANAR_420（NV12 / YUV420SP）
 *   - VO layer 也配置成 NV12，大小与 VICAP 输出一致
 *   - width 需要是 8 像素对齐（驱动硬件要求），800 满足要求
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <signal.h>
#include <unistd.h>

#include "k_module.h"
#include "k_vb_comm.h"
#include "k_vicap_comm.h"
#include "k_vo_comm.h"

#include "mpi_sensor_api.h"
#include "mpi_connector_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"

#include "kd_display.h"

// ---------------------------------------------------------------------------
// Global Defines and State
// ---------------------------------------------------------------------------

/* ISP 采集窗口固定为 1920x1080，VICAP 通过 scale 输出到用户指定宽高 */
#define ISP_WIDTH   (1920)
#define ISP_HEIGHT  (1080)

/* 简单 demo：只用 VICAP DEV0 / CHN0 做采集，CSI 默认用 2 号（可通过参数覆盖） */
static k_vicap_dev g_vicap_dev_id = VICAP_DEV_ID_2;

/* Ctrl+C 退出标志 */
static volatile bool g_app_run = true;

static int resolve_connector_size(k_connector_type *connector_type, k_u32 *width, k_u32 *height)
{
    k_connector_info info;

    memset(&info, 0, sizeof(info));
    if (kd_mpi_get_connector_info(*connector_type, &info) == K_SUCCESS) {
        if (*width == 0)
            *width = info.resolution.hactive;
        if (*height == 0)
            *height = info.resolution.vactive;
    }

    if (*width == 0)
        *width = K_CONN_WIDTH(*connector_type);
    if (*height == 0)
        *height = K_CONN_HEIGHT(*connector_type);

    if (*width == 0 || *height == 0) {
        printf("ERROR: unable to resolve display size for connector type %u, please pass -w/-h\n", *connector_type);
        return -1;
    }

    return 0;
}

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

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -c <type>   Connector type. Use list_connector to view supported types\n");
    printf("  -w <width>  Width (multiple of 8) [default: panel width]\n");
    printf("  -h <height> Height [default: panel height]\n");
    printf("  -l <layer>  VO Layer ID (0-3 for Video)\n");
    printf("  -r <degree> Rotation (0, 90, 180, 270)\n");
    printf("  -x <offset> X offset\n");
    printf("  -y <offset> Y offset\n");
    printf("  -s <csi>    CSI index (0-2) [default: 2]\n");
    printf("\nExample:\n");
    printf("  list_connector\n");
    printf("  %s -c <connector_type> -l 2 -x 10 -y 10\n", prog);
}

// ---------------------------------------------------------------------------
// VB init
// ---------------------------------------------------------------------------

/* 与 sample_uvc_dev_vicap/main.c / sample_wbc_dump/sample_wbc.c 相同风格 */
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

static k_s32 sample_vicap_init(k_vicap_dev dev_chn, k_u32 out_width, k_u32 out_height)
{
    k_vicap_dev_attr     dev_attr;
    k_vicap_chn_attr     chn_attr;
    k_vicap_sensor_info  sensor_info;
    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_type  sensor_type;

    if (out_width == 0 || out_height == 0) {
        printf("ERROR: VICAP output size invalid: %ux%u\n", out_width, out_height);
        return -1;
    }

    if ((out_width & 7) != 0) {
        printf("ERROR: VICAP output width (%u) must be multiple of 8\n", out_width);
        return -1;
    }

    memset(&sensor_info, 0, sizeof(sensor_info));
    memset(&probe_cfg, 0, sizeof(probe_cfg));

    /* 这里沿用 SDK 示例的写法：用 dev_chn 作为 CSI 编号做 sensor 自适应 */
    probe_cfg.csi_num = dev_chn;
    probe_cfg.width   = ISP_WIDTH;
    probe_cfg.height  = ISP_HEIGHT;
    probe_cfg.fps     = 30;

    if (0x00 != kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info)) {
        printf("ERROR: kd_mpi_sensor_adapt_get failed on CSI %d, want %dx%d@%d\n",
               probe_cfg.csi_num, probe_cfg.width, probe_cfg.height, probe_cfg.fps);
        return -1;
    }

    sensor_type = sensor_info.sensor_type;

    k_s32 ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_get_sensor_info failed, ret=%d\n", ret);
        return ret;
    }

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width  = ISP_WIDTH;
    dev_attr.acq_win.height = ISP_HEIGHT;
    dev_attr.mode           = VICAP_WORK_ONLINE_MODE;
    dev_attr.buffer_num     = 6;
    dev_attr.buffer_size    = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 1024);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(sensor_info));

    /* 简单起见：固定使用 VICAP_DEV_ID_0 做采集，CSI 从 g_vicap_dev_id 控制 */
    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_dev_attr failed, ret=%d\n", ret);
        return ret;
    }

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = out_width;
    chn_attr.out_win.height = out_height;
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_FALSE;
    chn_attr.scale_enable   = K_TRUE;
    chn_attr.chn_enable     = K_TRUE;
    chn_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420; /* NV12 / YUV420SP */
    chn_attr.buffer_num     = 6;
    chn_attr.buffer_size    = VB_ALIGN_UP(out_width * out_height * 3 / 2, 4096);
    chn_attr.alignment      = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_init failed, ret=%d\n", ret);
        return ret;
    }

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

    /* 源：VI / VICAP DEV0 CH0 */
    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = VICAP_CHN_ID_0;

    /* 目的：VO 显示设备上的指定 video layer */
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = layer_id;

    k_s32 ret = kd_mpi_sys_bind(&vi_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("ERROR: kd_mpi_sys_bind VICAP->VO failed, ret=0x%x\n", ret);
    } else {
        printf("Bind VICAP(dev=%d, ch=0) -> VO(layer=%d) OK\n",
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
    vi_mpp_chn.chn_id = VICAP_CHN_ID_0;

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
    k_u32 width = 0, height = 0;
    k_vo_layer_id layer_id = 2;
    int rot_val = 0;
    k_s32 offset_x = 0, offset_y = 0;
    int csi_idx = 2; // Default CSI 2

    int opt;
    bool c_set = false;

    // Parsing args: c=connector, w=width, h=height, l=layer, r=rotate, x=offset_x, y=offset_y, s=csi
    while ((opt = getopt(argc, argv, "c:w:h:l:r:x:y:s:")) != -1) {
        switch (opt) {
            case 'c': connector_type = (k_connector_type)atoi(optarg); c_set = true; break;
            case 'w': width = (k_u32)atoi(optarg); break;
            case 'h': height = (k_u32)atoi(optarg); break;
            case 'l': layer_id = (k_vo_layer_id)atoi(optarg); break;
            case 'r': rot_val = atoi(optarg); break;
            case 'x': offset_x = atoi(optarg); break;
            case 'y': offset_y = atoi(optarg); break;
            case 's': csi_idx = atoi(optarg); break;
            default: print_usage(argv[0]); return -1;
        }
    }

    if (!c_set) {
        printf("ERROR: Missing required arguments (-c)\n");
        print_usage(argv[0]);
        return -1;
    }

    if (resolve_connector_size(&connector_type, &width, &height) != 0) {
        print_usage(argv[0]);
        return -1;
    }

    g_vicap_dev_id = (k_vicap_dev)csi_idx;

    k_gdma_rotation_e rotate = GDMA_ROTATE_DEGREE_0;
    if (rot_val == 90) rotate = GDMA_ROTATE_DEGREE_90;
    else if (rot_val == 180) rotate = GDMA_ROTATE_DEGREE_180;
    else if (rot_val == 270) rotate = GDMA_ROTATE_DEGREE_270;

    printf("vo_test_video: connector=%u, size=%ux%u, layer=%d, rotate=%d, offset=(%d,%d), csi=%d\n",
           connector_type, width, height, layer_id, rot_val, offset_x, offset_y, g_vicap_dev_id);

    signal(SIGINT, handle_signal);

    /* 1. 初始化 VB */
    ret = sample_vb_init();
    if (ret != K_SUCCESS) {
        printf("ERROR: sample_vb_init failed, ret=%d\n", ret);
        return -1;
    }

    /* 2. 初始化 VICAP（sensor + dev0/ch0 输出到指定分辨率） */
    ret = sample_vicap_init(g_vicap_dev_id, width, height);
    if (ret != K_SUCCESS) {
        printf("ERROR: sample_vicap_init failed, ret=%d\n", ret);
        goto cleanup_vb;
    }

    /* 3. 初始化 VO + connector，配置指定 layer 为 NV12 + 指定分辨率 */
    if(0x00 != kd_display_init(connector_type, 0, 0, rotate)) {
        printf("ERROR: connector init failed\n");
        goto cleanup_vb;
    }

    ret = kd_display_layer_configure(layer_id, PIXEL_FORMAT_YUV_SEMIPLANAR_420, width, height, offset_x, offset_y);
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

    printf("Preview running, press Ctrl+C to stop.\n");
    while (g_app_run) {
        usleep(100 * 1000); /* 100ms 轮询退出标志，不做任何 CPU 拷贝 */
    }

    printf("Stopping VICAP stream ...\n");
    kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);

cleanup_bind:
    sample_vicap_unbind_vo(layer_id);

    kd_display_deinit();

cleanup_vicap:
    printf("Deinitializing VICAP dev0 ...\n");
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);

cleanup_vb:
    printf("Deinitializing VB ...\n");
    kd_mpi_vb_exit();

    printf("vo_test_video exit.\n");
    return 0;
}
