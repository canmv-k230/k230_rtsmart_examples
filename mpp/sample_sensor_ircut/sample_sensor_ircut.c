/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
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
 * IMX415 IR-Cut preview.
 *
 *   Sensor → VICAP DEV0 → CHN1 → VO
 *
 * -ir_cut 0  color, IR-Cut filter on
 * -ir_cut 1  infrared, IR-Cut filter off, ISP database gets the -ir suffix
 * -auto      poll exposure every 500 ms; switch after 3 confirms in one direction
 *
 * The starting mode is given to VICAP before kd_mpi_vicap_init. q quits.
 */

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>

#include "k_module.h"
#include "k_sys_comm.h"
#include "k_vb_comm.h"
#include "k_vicap_comm.h"
#include "k_vo_comm.h"

#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_isp_api.h"

#include "kd_display.h"

#define SAMPLE_ALIGN_UP(x, a) (((x) + ((a) - 1u)) & ~((a) - 1u))
#define IR_DAY2NIGHT       0.75f
#define IR_NIGHT2DAY       0.20f
/* Sophgo gray-world start: 190/256 .. 280/256. Not an IMX415 lamp calibration. */
#define IR_RG_MIN          0.74f
#define IR_RG_MAX          1.09f
#define IR_BG_MIN          0.74f
#define IR_BG_MAX          1.09f
#define IR_CONFIRM_COUNT   3
#define IR_POLL_MS         500
#define IR_SETTLE_MS       2000

typedef struct {
    k_connector_type connector_type;
    int csi_idx;
    k_s32 sensor_type;
    k_ircut_mode ircut;
    float rg_min;
    float rg_max;
    float bg_min;
    float bg_max;
    bool c_set;
    bool stype_set;
    bool ircut_set;
    bool ir_auto;
} sample_params_t;

static volatile bool g_app_run = true;
static bool g_vicap_alive = false;
static bool g_vicap_inited = false;
static k_vicap_sensor_info g_sensor_info;
static k_ircut_mode g_mode = K_IRCUT_MODE_COLOR;
static float g_rg_min = IR_RG_MIN;
static float g_rg_max = IR_RG_MAX;
static float g_bg_min = IR_BG_MIN;
static float g_bg_max = IR_BG_MAX;
static k_isp_ir_switch g_last_suggest = K_ISP_IR_SWITCH_NONE;
static int g_confirm = 0;
static bool g_hold_logged = false;
static k_u32 g_out_w = 0;
static k_u32 g_out_h = 0;

static void handle_signal(int sig)
{
    if (sig == SIGINT) {
        printf("Caught SIGINT, exiting...\n");
        g_app_run = false;
    }
}

static const char *mipi_lanes_name(k_vicap_mipi_lanes lanes)
{
    switch (lanes) {
    case VICAP_MIPI_1LANE:
        return "1lane";
    case VICAP_MIPI_2LANE:
        return "2lane";
    case VICAP_MIPI_4LANE:
        return "4lane";
    default:
        return "unknown";
    }
}

static const char *ircut_name(k_ircut_mode mode)
{
    return mode == K_IRCUT_MODE_IR ? "ir (filter off)" : "color (filter on)";
}

static void align_preview_size(k_u32 *w, k_u32 *h)
{
    if ((*w & 15) != 0)
        *w = SAMPLE_ALIGN_UP(*w, 16);
    if ((*h & 1) != 0)
        *h = *h & ~1u;
}

static k_vicap_work_mode select_work_mode(k_u32 acq_w, k_u32 acq_h)
{
    if (acq_w > VICAP_SENSOR_MAX_WIDTH || acq_h > VICAP_SENSOR_MAX_HEIGH) {
        printf("INFO: acq %ux%u > %ux%u → VICAP_WORK_SW_TILE_MODE\n",
               acq_w, acq_h, VICAP_SENSOR_MAX_WIDTH, VICAP_SENSOR_MAX_HEIGH);
        return VICAP_WORK_SW_TILE_MODE;
    }
    return VICAP_WORK_ONLINE_MODE;
}

static void calc_preview_size(k_u32 sensor_w, k_u32 sensor_h,
                              k_u32 screen_w, k_u32 screen_h,
                              k_u32 *out_w, k_u32 *out_h)
{
    float sensor_aspect = (float)sensor_w / (float)sensor_h;
    float screen_aspect = (float)screen_w / (float)screen_h;
    k_u32 width, height;

    if (sensor_aspect > screen_aspect) {
        width = screen_w;
        height = (k_u32)((float)width / sensor_aspect);
    } else if (sensor_aspect < screen_aspect) {
        height = screen_h;
        width = (k_u32)((float)height * sensor_aspect);
    } else {
        width = screen_w;
        height = screen_h;
    }

    align_preview_size(&width, &height);
    *out_w = width;
    *out_h = height;
}

static void print_usage(const char *prog)
{
    printf("Usage: %s -c <connector> -s <csi> -stype <id> -ir_cut <0|1> [-auto]\n", prog);
    printf("  -c <type>     Connector type\n");
    printf("  -s <0|1|2>    CSI index [default: 0]\n");
    printf("  -stype <id>   Sensor type (see board list_sensor)\n");
    printf("  -ir_cut <0|1> Starting mode. 0 color (IR-Cut on), 1 infrared (IR-Cut off)\n");
    printf("  -auto         Switch from exposure. day2night=%.2f night2day=%.2f, %d confirms\n",
           IR_DAY2NIGHT, IR_NIGHT2DAY, IR_CONFIRM_COUNT);
    printf("  -rg <min> <max>  IR-lamp R/G window, default %.2f %.2f\n", IR_RG_MIN, IR_RG_MAX);
    printf("  -bg <min> <max>  IR-lamp B/G window, default %.2f %.2f\n", IR_BG_MIN, IR_BG_MAX);
    printf("\nExample:\n");
    printf("  %s -c 605274432 -s 0 -stype <id> -ir_cut 0 -auto\n", prog);
}

static k_s32 parse_parameters(int argc, char **argv, sample_params_t *params)
{
    memset(params, 0, sizeof(*params));
    params->csi_idx = 0;
    params->rg_min = IR_RG_MIN;
    params->rg_max = IR_RG_MAX;
    params->bg_min = IR_BG_MIN;
    params->bg_max = IR_BG_MAX;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            params->connector_type = (k_connector_type)atoi(argv[++i]);
            params->c_set = true;
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            params->csi_idx = atoi(argv[++i]);
            if (params->csi_idx < 0 || params->csi_idx > 2) {
                printf("ERROR: Invalid -s %d, must be 0, 1 or 2\n", params->csi_idx);
                return -1;
            }
        } else if (strcmp(argv[i], "-stype") == 0 && i + 1 < argc) {
            params->sensor_type = atoi(argv[++i]);
            if (params->sensor_type < 0) {
                printf("ERROR: Invalid -stype %d\n", params->sensor_type);
                return -1;
            }
            params->stype_set = true;
        } else if (strcmp(argv[i], "-ir_cut") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v != 0 && v != 1) {
                printf("ERROR: Invalid -ir_cut %d, must be 0 or 1\n", v);
                return -1;
            }
            params->ircut = (v == 1) ? K_IRCUT_MODE_IR : K_IRCUT_MODE_COLOR;
            params->ircut_set = true;
        } else if (strcmp(argv[i], "-auto") == 0) {
            params->ir_auto = true;
        } else if (strcmp(argv[i], "-rg") == 0 && i + 2 < argc) {
            params->rg_min = strtof(argv[++i], NULL);
            params->rg_max = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "-bg") == 0 && i + 2 < argc) {
            params->bg_min = strtof(argv[++i], NULL);
            params->bg_max = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
            return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (!params->c_set || !params->stype_set || !params->ircut_set) {
        printf("ERROR: -c, -stype and -ir_cut are required\n");
        return -1;
    }
    return 0;
}

static k_s32 get_sensor_resolution(const sample_params_t *p)
{
    k_vicap_sensor_info sensor_info;
    k_s32 ret;

    memset(&sensor_info, 0, sizeof(sensor_info));
    ret = kd_mpi_vicap_get_sensor_info((k_vicap_sensor_type)p->sensor_type, &sensor_info);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_get_sensor_info(type=%d) failed, ret=%d\n",
               p->sensor_type, ret);
        return ret;
    }
    if (sensor_info.csi_num != (k_vicap_csi_num)(p->csi_idx + VICAP_CSI0)) {
        printf("ERROR: sensor type %d is on CSI%d, but -s %d was requested\n",
               p->sensor_type, (int)sensor_info.csi_num - 1, p->csi_idx);
        return -1;
    }

    printf("Using sensor type %d: %s %ux%u@%u %s, IR-Cut %s\n",
           p->sensor_type,
           sensor_info.sensor_name ? sensor_info.sensor_name : "unknown",
           sensor_info.width, sensor_info.height, sensor_info.fps,
           mipi_lanes_name(sensor_info.mipi_lanes),
           ircut_name(p->ircut));
    memcpy(&g_sensor_info, &sensor_info, sizeof(g_sensor_info));
    return K_SUCCESS;
}

static k_s32 sample_vb_init(void)
{
    k_vb_config config;
    k_s32 ret;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("ERROR: kd_mpi_vb_set_config failed, ret=%d\n", ret);
        return ret;
    }
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("ERROR: kd_mpi_vb_init failed, ret=%d\n", ret);
        return ret;
    }
    return K_SUCCESS;
}

static void fill_chn0_attr(k_vicap_chn_attr *chn_attr, const k_vicap_dev_attr *dev_attr,
                           k_u32 buf_num)
{
    k_u32 src_w = dev_attr->acq_win.width;
    k_u32 src_h = dev_attr->acq_win.height;
    k_u32 ch0_w = src_w;
    k_u32 ch0_h = src_h;

    memset(chn_attr, 0, sizeof(*chn_attr));
    if (((src_w & 15u) != 0) || ((src_h & 1u) != 0) || src_w > 1920 || src_h > 1080) {
        if (src_w >= 1920 && src_h >= 1080) {
            ch0_w = 1920;
            ch0_h = 1080;
        } else {
            ch0_w = src_w & ~15u;
            ch0_h = src_h & ~1u;
        }
        chn_attr->crop_win.h_start = (src_w > ch0_w) ? ((src_w - ch0_w) / 2) & ~1u : 0;
        chn_attr->crop_win.v_start = (src_h > ch0_h) ? ((src_h - ch0_h) / 2) & ~1u : 0;
        chn_attr->crop_win.width = ch0_w;
        chn_attr->crop_win.height = ch0_h;
        chn_attr->crop_enable = K_TRUE;
    } else {
        chn_attr->crop_win = dev_attr->acq_win;
        chn_attr->crop_enable = K_FALSE;
    }

    chn_attr->out_win.width = ch0_w;
    chn_attr->out_win.height = ch0_h;
    chn_attr->scale_win = chn_attr->out_win;
    chn_attr->scale_enable = K_FALSE;
    chn_attr->chn_enable = K_TRUE;
    chn_attr->pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr->buffer_size = VB_ALIGN_UP(ch0_w * ch0_h * 3 / 2, 4096);
    chn_attr->buffer_num = buf_num;
    chn_attr->alignment = 12;
    chn_attr->buffer_pool_id = VB_INVALID_POOLID;
}

static void fill_chn1_preview_attr(k_vicap_chn_attr *chn_attr,
                                   const k_vicap_dev_attr *dev_attr,
                                   k_u32 out_w, k_u32 out_h, k_u32 buf_num)
{
    k_u32 src_w = dev_attr->acq_win.width;
    k_u32 src_h = dev_attr->acq_win.height;
    k_u32 crop_w = (src_w >= 1920) ? 1920 : SAMPLE_ALIGN_UP(src_w, 16);
    k_u32 crop_h = (src_h >= 1080) ? 1080 : (src_h & ~1u);

    memset(chn_attr, 0, sizeof(*chn_attr));
    chn_attr->out_win.width = out_w;
    chn_attr->out_win.height = out_h;
    chn_attr->crop_win.h_start = (src_w > crop_w) ? ((src_w - crop_w) / 2) & ~1u : 0;
    chn_attr->crop_win.v_start = (src_h > crop_h) ? ((src_h - crop_h) / 2) & ~1u : 0;
    chn_attr->crop_win.width = crop_w;
    chn_attr->crop_win.height = crop_h;
    chn_attr->scale_win = chn_attr->out_win;
    chn_attr->crop_enable = K_TRUE;
    chn_attr->scale_enable = (out_w != crop_w || out_h != crop_h) ? K_TRUE : K_FALSE;
    chn_attr->chn_enable = K_TRUE;
    chn_attr->pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr->buffer_num = buf_num;
    chn_attr->buffer_size = VB_ALIGN_UP(out_w * out_h * 3 / 2, 4096);
    chn_attr->alignment = 12;
    chn_attr->buffer_pool_id = VB_INVALID_POOLID;
}

static k_s32 sample_vicap_init(k_ircut_mode ircut, k_u32 out_w, k_u32 out_h)
{
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_s32 ret;
    const k_u32 buf_num = 6;

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width = g_sensor_info.width;
    dev_attr.acq_win.height = g_sensor_info.height;
    dev_attr.mode = select_work_mode(dev_attr.acq_win.width, dev_attr.acq_win.height);
    dev_attr.buffer_num = buf_num;
    dev_attr.buffer_size = VB_ALIGN_UP(g_sensor_info.width * g_sensor_info.height * 2, 4096);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    memcpy(&dev_attr.sensor_info, &g_sensor_info, sizeof(g_sensor_info));
    dev_attr.pipe_ctrl.data = 0xFFFFFFFF;
    dev_attr.pipe_ctrl.bits.ae_enable = K_TRUE;
    dev_attr.pipe_ctrl.bits.awb_enable = K_TRUE;
    dev_attr.pipe_ctrl.bits.ahdr_enable = K_FALSE;
    dev_attr.pipe_ctrl.bits.dnr3_enable = K_TRUE;
    dev_attr.dw_enable = K_FALSE;
    dev_attr.mirror = VICAP_MIRROR_NONE;

    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_dev_attr failed, ret=%d\n", ret);
        return ret;
    }

    fill_chn0_attr(&chn_attr, &dev_attr, buf_num);
    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr CHN0 failed, ret=%d\n", ret);
        return ret;
    }

    fill_chn1_preview_attr(&chn_attr, &dev_attr, out_w, out_h, buf_num);
    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_1, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr CHN1 failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vicap_set_ircut(VICAP_DEV_ID_0, ircut);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_ircut(%s) failed, ret=%d\n", ircut_name(ircut), ret);
        return ret;
    }

    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_init failed, ret=%d\n", ret);
        return ret;
    }
    return K_SUCCESS;
}

static k_s32 bind_vo(k_vo_layer_id layer_id)
{
    k_mpp_chn vi_mpp_chn = {0};
    k_mpp_chn vo_mpp_chn = {0};

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = VICAP_CHN_ID_1;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = layer_id;

    k_s32 ret = kd_mpi_sys_bind(&vi_mpp_chn, &vo_mpp_chn);
    if (ret)
        printf("ERROR: kd_mpi_sys_bind VICAP->VO failed, ret=0x%x\n", ret);
    return ret;
}

static void unbind_vo(k_vo_layer_id layer_id)
{
    k_mpp_chn vi_mpp_chn = {0};
    k_mpp_chn vo_mpp_chn = {0};

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = VICAP_CHN_ID_1;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = layer_id;
    kd_mpi_sys_unbind(&vi_mpp_chn, &vo_mpp_chn);
}

static const char *suggest_name(k_isp_ir_switch suggest)
{
    switch (suggest) {
    case K_ISP_IR_SWITCH_TO_IR:
        return "to_ir";
    case K_ISP_IR_SWITCH_TO_COLOR:
        return "to_color";
    default:
        return "none";
    }
}

/* deinit clears dev/chn attr, so a mode change rebuilds them before init. */
static k_s32 ir_restart(k_ircut_mode mode)
{
    k_s32 ret;

    ret = sample_vicap_init(mode, g_out_w, g_out_h);
    if (ret)
        return ret;
    g_vicap_inited = true;
    ret = kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);
    if (ret) {
        printf("[ir] start_stream(%s) failed, ret=%d\n", ircut_name(mode), ret);
        kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
        g_vicap_inited = false;
    }
    return ret;
}

/* 1: switched and streaming. 0: stayed on the old mode. -1: pipeline is down. */
static int ir_switch_mode(k_ircut_mode next)
{
    k_s32 ret;

    printf("[ir] switch %s -> %s\n", ircut_name(g_mode), ircut_name(next));
    kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
    g_vicap_alive = false;
    g_vicap_inited = false;

    ret = ir_restart(next);
    if (ret == K_SUCCESS) {
        g_mode = next;
        g_vicap_alive = true;
        g_hold_logged = false;
        printf("[ir] switch ok, mode=%s, settle %d ms\n", ircut_name(g_mode), IR_SETTLE_MS);
        return 1;
    }

    printf("[ir] switch to %s failed, ret=%d, restore %s\n",
           ircut_name(next), ret, ircut_name(g_mode));
    ret = ir_restart(g_mode);
    if (ret) {
        printf("[ir] restore %s failed, ret=%d\n", ircut_name(g_mode), ret);
        return -1;
    }
    g_vicap_alive = true;
    printf("[ir] restored %s\n", ircut_name(g_mode));
    return 0;
}

static k_s32 ir_read_meter(k_isp_ir_meter *meter)
{
    return kd_mpi_isp_ir_query(ISP_DEV_ID_0, meter);
}

static void ir_log_confirm(int count, k_isp_ir_switch suggest)
{
    k_isp_ir_meter meter;

    if (ir_read_meter(&meter) != K_SUCCESS) {
        printf("[ir] confirm %d/%d %s, query failed\n",
               count, IR_CONFIRM_COUNT, suggest_name(suggest));
        return;
    }
    printf("[ir] confirm %d/%d %s index=%.3f rg=%.3f bg=%.3f\n",
           count, IR_CONFIRM_COUNT, suggest_name(suggest),
           meter.exposure_index, meter.rg, meter.bg);
}

/* One line when the IR lamp keeps the ratios inside the window. */
static void ir_log_hold_once(void)
{
    k_isp_ir_meter meter;
    bool hold;

    if (g_mode != K_IRCUT_MODE_IR) {
        g_hold_logged = false;
        return;
    }
    if (ir_read_meter(&meter) != K_SUCCESS)
        return;
    hold = meter.exposure_index < IR_NIGHT2DAY &&
           meter.rg >= g_rg_min && meter.rg <= g_rg_max &&
           meter.bg >= g_bg_min && meter.bg <= g_bg_max;
    if (!hold) {
        g_hold_logged = false;
        return;
    }
    if (g_hold_logged)
        return;
    g_hold_logged = true;
    printf("[ir] hold ir index=%.3f rg=%.3f bg=%.3f\n",
           meter.exposure_index, meter.rg, meter.bg);
}

/* Returns settle time in ms after a successful switch, 0 to keep polling, -1 if down. */
static int ir_auto_once(void)
{
    k_isp_ir_auto_attr attr;
    k_isp_ir_switch suggest;
    k_s32 ret;

    memset(&attr, 0, sizeof(attr));
    attr.enable = K_TRUE;
    attr.day2night = IR_DAY2NIGHT;
    attr.night2day = IR_NIGHT2DAY;
    attr.rg_min = g_rg_min;
    attr.rg_max = g_rg_max;
    attr.bg_min = g_bg_min;
    attr.bg_max = g_bg_max;
    attr.mode = g_mode;

    ret = kd_mpi_isp_ir_auto_run_once(ISP_DEV_ID_0, &attr);
    if (ret != K_SUCCESS) {
        printf("[ir] run_once failed, ret=%d\n", ret);
        return 0;
    }

    suggest = attr.suggest;
    if (suggest == K_ISP_IR_SWITCH_NONE || suggest != g_last_suggest) {
        g_last_suggest = suggest;
        g_confirm = (suggest == K_ISP_IR_SWITCH_NONE) ? 0 : 1;
        if (suggest == K_ISP_IR_SWITCH_NONE)
            ir_log_hold_once();
        else
            ir_log_confirm(g_confirm, suggest);
        return 0;
    }

    g_confirm++;
    ir_log_confirm(g_confirm, suggest);
    if (g_confirm < IR_CONFIRM_COUNT)
        return 0;

    g_confirm = 0;
    g_last_suggest = K_ISP_IR_SWITCH_NONE;
    if (ir_switch_mode(suggest == K_ISP_IR_SWITCH_TO_IR ? K_IRCUT_MODE_IR
                                                       : K_IRCUT_MODE_COLOR) == 1)
        return IR_SETTLE_MS;
    return g_vicap_alive ? 0 : -1;
}

static bool wait_key_or_timeout(int timeout_ms)
{
    fd_set rfds;
    struct timeval tv;
    int sel;

    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    sel = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (sel > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
        int key = getchar();
        if (key == 'q' || key == 'Q')
            return true;
        if (key == EOF)
            usleep(IR_POLL_MS * 1000);
    } else if (sel < 0 && errno != EINTR) {
        perror("select");
        return true;
    }
    return !g_app_run;
}

static void wait_quit(bool ir_auto)
{
    int settle_ms = 0;

    if (ir_auto) {
        k_isp_ir_meter meter;

        printf("[ir] auto on, start=%s day2night=%.2f night2day=%.2f "
               "rg=%.2f..%.2f bg=%.2f..%.2f\n",
               ircut_name(g_mode), IR_DAY2NIGHT, IR_NIGHT2DAY,
               g_rg_min, g_rg_max, g_bg_min, g_bg_max);
        k_s32 qret = ir_read_meter(&meter);

        if (qret != K_SUCCESS)
            printf("[ir] query failed, ret=%d\n", qret);
        else
            printf("[ir] index=%.3f rg=%.3f bg=%.3f int=%.6fs again=%.3f\n",
                   meter.exposure_index, meter.rg, meter.bg,
                   meter.integration_time_s, meter.again);
        printf("Preview running. q to quit.\n");
    } else {
        printf("Preview running. q to quit. Pass -auto to switch from exposure.\n");
    }

    while (g_app_run) {
        if (wait_key_or_timeout(IR_POLL_MS))
            break;
        if (!ir_auto || !g_vicap_alive)
            continue;
        if (settle_ms > 0) {
            settle_ms -= IR_POLL_MS;
            if (settle_ms < 0)
                settle_ms = 0;
            continue;
        }
        settle_ms = ir_auto_once();
        if (settle_ms < 0) {
            printf("[ir] pipeline is down, exit\n");
            break;
        }
    }
}

int main(int argc, char **argv)
{
    sample_params_t params;
    k_s32 ret;
    k_u32 width = 0, height = 0;
    k_vo_layer_id layer_id = 1;
    k_vo_size display_resolution;
    ret = parse_parameters(argc, argv, &params);
    if (ret == 1) {
        print_usage(argv[0]);
        return 0;
    }
    if (ret < 0) {
        print_usage(argv[0]);
        return -1;
    }

    ret = get_sensor_resolution(&params);
    if (ret != K_SUCCESS)
        return -1;

    ret = sample_vb_init();
    if (ret != K_SUCCESS)
        return -1;

    if (kd_display_init(params.connector_type, 0, 0, GDMA_ROTATE_DEGREE_0) != 0) {
        printf("ERROR: connector init failed\n");
        goto cleanup_vb;
    }

    ret = kd_mpi_vo_get_resolution(&display_resolution);
    if (ret != 0) {
        display_resolution.width = 1920;
        display_resolution.height = 1080;
    }
    calc_preview_size(g_sensor_info.width, g_sensor_info.height,
                      display_resolution.width, display_resolution.height,
                      &width, &height);
    g_out_w = width;
    g_out_h = height;
    g_mode = params.ircut;
    g_rg_min = params.rg_min;
    g_rg_max = params.rg_max;
    g_bg_min = params.bg_min;
    g_bg_max = params.bg_max;
    printf("sample_sensor_ircut: connector=%d screen=%ux%u preview=%ux%u csi=%d ir_cut=%s auto=%d\n",
           params.connector_type, display_resolution.width, display_resolution.height,
           width, height, params.csi_idx, ircut_name(params.ircut), params.ir_auto);

    signal(SIGINT, handle_signal);

    ret = sample_vicap_init(params.ircut, width, height);
    if (ret != K_SUCCESS)
        goto cleanup_display;
    g_vicap_inited = true;

    ret = kd_display_layer_configure(layer_id, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                     width, height, 0, 0);
    if (ret != 0) {
        printf("ERROR: configure layer failed, ret=%d\n", ret);
        goto cleanup_vicap;
    }
    kd_display_layer_enable(layer_id);
    if (bind_vo(layer_id) != 0)
        goto cleanup_vicap;

    ret = kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);
    if (ret != K_SUCCESS) {
        printf("ERROR: start_stream failed, ret=%d\n", ret);
        goto cleanup_bind;
    }
    g_vicap_alive = true;

    wait_quit(params.ir_auto);

    if (g_vicap_alive) {
        printf("Stopping VICAP stream...\n");
        kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
    }

cleanup_bind:
    unbind_vo(layer_id);
cleanup_vicap:
    if (g_vicap_inited) {
        printf("Deinit VICAP...\n");
        kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
        g_vicap_inited = false;
    }
cleanup_display:
    kd_display_deinit();
cleanup_vb:
    printf("Deinit VB...\n");
    kd_mpi_vb_exit();
    return ret == K_SUCCESS ? 0 : -1;
}
