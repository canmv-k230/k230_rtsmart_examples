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
 * VICAP sensor preview + dump demo.
 *
 *   Sensor → VICAP DEV0
 *        ├─ CHN0 → Dump (YUV / RGB / RAW)
 *        └─ CHN1 → VO layer → Connector
 *
 * Examples:
 *   sample_vicap_sensor -c 20
 *   sample_vicap_sensor -c 20 -ofmt 3 -dump 1          # RAW dump (ONLINE)
 *   sample_vicap_sensor -c 20 -width 2592 -height 1944 -lane 4
 *
 * Interactive: d / d <n> dump frames; q quit.
 */

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define SAMPLE_ALIGN_UP(x, a) (((x) + ((a) - 1u)) & ~((a) - 1u))

enum {
    CH0_FMT_YUV420SP = 0,
    CH0_FMT_RGB888   = 1,
    CH0_FMT_RGB888P  = 2,
    CH0_FMT_RAW      = 3,
};

typedef struct {
    k_connector_type connector_type;
    int rot_val;
    int csi_idx;
    k_bool ae_enable;
    k_bool awb_enable;
    k_bool dw_enable;
    k_bool dnr3_enable;
    k_bool force_tile; /* force SW_TILE even for RAW / small res */
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
    k_bool mirror_en;
    k_bool flip_en;
    k_bool sensor_type_set;
    k_s32 sensor_type;
    k_u32 auto_dump_count; /* 0=interactive; >0 dump N then quit */
    k_vicap_mipi_lane_pref lane_pref;
} sample_params_t;

static k_vicap_dev g_vicap_csi = VICAP_DEV_ID_2;
static volatile bool g_app_run = true;
static k_u32 g_sensor_width = 1920;
static k_u32 g_sensor_height = 1080;
static k_s32 g_sensor_fd = -1;
static k_vicap_sensor_info g_sensor_info;
static k_u32 g_ch0_format = CH0_FMT_YUV420SP;

/* -------------------------------------------------------------------------- */
/* Utils                                                                      */
/* -------------------------------------------------------------------------- */

static void handle_signal(int sig)
{
    if (sig == SIGINT) {
        printf("Caught SIGINT, exiting...\n");
        g_app_run = false;
    }
}

static k_vicap_mirror mirror_flip_to_mode(k_bool mirror_en, k_bool flip_en)
{
    if (mirror_en && flip_en)
        return VICAP_MIRROR_BOTH;
    if (mirror_en)
        return VICAP_MIRROR_HOR;
    if (flip_en)
        return VICAP_MIRROR_VER;
    return VICAP_MIRROR_NONE;
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

static const char *work_mode_name(k_vicap_work_mode mode)
{
    switch (mode) {
    case VICAP_WORK_ONLINE_MODE:
        return "ONLINE";
    case VICAP_WORK_OFFLINE_MODE:
        return "OFFLINE";
    case VICAP_WORK_SW_TILE_MODE:
        return "SW_TILE";
    case VICAP_WORK_MCM_MIPI_MODE:
        return "MCM";
    default:
        return "UNKNOWN";
    }
}

static k_s32 parse_bool01(const char *name, int val)
{
    if (val != 0 && val != 1) {
        printf("ERROR: Invalid %s value %d, must be 0 or 1\n", name, val);
        return -1;
    }
    return 0;
}

static void align_preview_size(k_u32 *w, k_u32 *h)
{
    if ((*w & 15) != 0)
        *w = SAMPLE_ALIGN_UP(*w, 16);
    if ((*h & 1) != 0)
        *h = *h & ~1u;
}

/**
 * Select VICAP work mode.
 * - ofmt=3 (RAW dump): ONLINE by default (RAW path does not need SW_TILE).
 * - -tile 1: force SW_TILE for A/B.
 * - else: SW_TILE when acq exceeds VICAP max; otherwise ONLINE.
 */
static k_vicap_work_mode select_work_mode(k_u32 acq_w, k_u32 acq_h,
                                          k_u32 ch0_format, k_bool force_tile)
{
    if (force_tile) {
        printf("INFO: force VICAP_WORK_SW_TILE_MODE (-tile 1), acq=%ux%u\n",
               acq_w, acq_h);
        return VICAP_WORK_SW_TILE_MODE;
    }

    if (ch0_format == CH0_FMT_RAW) {
        printf("INFO: RAW dump (-ofmt 3): use VICAP_WORK_ONLINE_MODE, acq=%ux%u\n",
               acq_w, acq_h);
        return VICAP_WORK_ONLINE_MODE;
    }

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

    if (width > screen_w) {
        width = screen_w;
        height = (k_u32)((float)width / sensor_aspect);
    }
    if (height > screen_h) {
        height = screen_h;
        width = (k_u32)((float)height * sensor_aspect);
    }

    align_preview_size(&width, &height);
    *out_w = width;
    *out_h = height;
}

/* -------------------------------------------------------------------------- */
/* Dump                                                                       */
/* -------------------------------------------------------------------------- */

static void sample_vicap_dump_frame(k_vicap_dev dev, k_vicap_chn chn, k_u32 *dump_count)
{
    k_video_frame_info dump_info;
    k_s32 ret;
    k_char *suffix = "bin";
    k_char filename[256];
    k_u32 w, h, stride0, stride1;
    k_u32 map_size = 0;
    k_u32 packed_size = 0;
    k_u8 *virt0 = NULL;
    k_u8 *virt1 = NULL;
    k_u32 map1_size = 0;
    FILE *file = NULL;
    k_u32 y;

    memset(&dump_info, 0, sizeof(dump_info));

    k_vicap_dump_format dump_fmt =
        (g_ch0_format == CH0_FMT_RAW) ? VICAP_DUMP_RAW : VICAP_DUMP_YUV;
    ret = kd_mpi_vicap_dump_frame(dev, chn, dump_fmt, &dump_info, 1000);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_dump_frame failed, ret=%d\n", ret);
        return;
    }

    w = dump_info.v_frame.width;
    h = dump_info.v_frame.height;
    stride0 = dump_info.v_frame.stride[0];
    stride1 = dump_info.v_frame.stride[1];

    printf("dump meta: %ux%u fmt=%d stride=[%u,%u,%u] phys=[0x%llx,0x%llx]\n",
           w, h, (int)dump_info.v_frame.pixel_format,
           dump_info.v_frame.stride[0], dump_info.v_frame.stride[1],
           dump_info.v_frame.stride[2],
           (unsigned long long)dump_info.v_frame.phys_addr[0],
           (unsigned long long)dump_info.v_frame.phys_addr[1]);

    if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_420) {
        suffix = "yuv420sp";
        stride0 = SAMPLE_ALIGN_UP((stride0 > w) ? stride0 : w, 16);
        stride1 = SAMPLE_ALIGN_UP((stride1 > w) ? stride1 : w, 16);
        map_size = SAMPLE_ALIGN_UP(stride0 * h, 4096) +
                   SAMPLE_ALIGN_UP(stride1 * (h / 2), 4096);
        packed_size = w * h + w * (h / 2);
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_888) {
        suffix = "rgb888";
        {
            k_u32 align_w = SAMPLE_ALIGN_UP(w, 16);
            if (stride0 < align_w * 3)
                stride0 = align_w * 3;
        }
        map_size = stride0 * h;
        packed_size = w * h * 3;
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_888_PLANAR) {
        suffix = "rgb888p";
        stride0 = SAMPLE_ALIGN_UP((stride0 > w) ? stride0 : w, 16);
        map_size = SAMPLE_ALIGN_UP(stride0 * h, 4096) * 3;
        packed_size = w * h * 3;
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_BAYER_10BPP ||
               dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_BAYER_12BPP) {
        suffix = (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_BAYER_10BPP)
                     ? "raw10"
                     : "raw12";
        {
            k_u32 stride_bytes = ((w + 7u) / 8u) * 16u;
            if (stride0 > w)
                stride_bytes = stride0 * 2u;
            stride0 = stride_bytes;
        }
        map_size = stride0 * h;
        packed_size = w * h * 2;
    } else {
        suffix = "yuv420sp";
        stride0 = SAMPLE_ALIGN_UP((stride0 > w) ? stride0 : w, 16);
        stride1 = SAMPLE_ALIGN_UP((stride1 > w) ? stride1 : w, 16);
        map_size = SAMPLE_ALIGN_UP(stride0 * h, 4096) +
                   SAMPLE_ALIGN_UP(stride1 * (h / 2), 4096);
        packed_size = w * h + w * (h / 2);
    }

    virt0 = kd_mpi_sys_mmap(dump_info.v_frame.phys_addr[0], map_size);
    if (!virt0) {
        printf("ERROR: Failed to mmap dump plane0 (size=%u)\n", map_size);
        goto out_release;
    }

    if (dump_info.v_frame.phys_addr[1] &&
        dump_info.v_frame.phys_addr[1] != dump_info.v_frame.phys_addr[0] &&
        dump_info.v_frame.pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_420) {
        map1_size = SAMPLE_ALIGN_UP(stride1 * (h / 2), 4096);
        virt1 = kd_mpi_sys_mmap(dump_info.v_frame.phys_addr[1], map1_size);
        if (!virt1) {
            printf("WARN: mmap plane1 failed, fallback to contiguous after Y\n");
            map1_size = 0;
        }
    }

    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "vicap_dev%d_chn%d_%dx%d_%04d.%s",
             dev, chn, w, h, *dump_count, suffix);

    printf("Saving packed dump to %s (stride0=%u stride1=%u map=%u packed=%u)...\n",
           filename, stride0, stride1, map_size, packed_size);
    file = fopen(filename, "wb");
    if (!file) {
        printf("ERROR: Failed to open dump file\n");
        goto out_unmap;
    }

    if (strcmp(suffix, "yuv420sp") == 0) {
        for (y = 0; y < h; y++)
            fwrite(virt0 + (size_t)y * stride0, 1, w, file);
        {
            k_u8 *uv = virt1;
            if (!uv)
                uv = virt0 + SAMPLE_ALIGN_UP(stride0 * h, 4096);
            for (y = 0; y < h / 2; y++)
                fwrite(uv + (size_t)y * stride1, 1, w, file);
        }
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_888) {
        for (y = 0; y < h; y++)
            fwrite(virt0 + (size_t)y * stride0, 1, w * 3, file);
    } else if (dump_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_888_PLANAR) {
        k_u32 plane;
        for (plane = 0; plane < 3; plane++) {
            k_u8 *p = virt0 + (size_t)plane * SAMPLE_ALIGN_UP(stride0 * h, 4096);
            for (y = 0; y < h; y++)
                fwrite(p + (size_t)y * stride0, 1, w, file);
        }
    } else {
        for (y = 0; y < h; y++)
            fwrite(virt0 + (size_t)y * stride0, 1, w * 2, file);
    }

    fclose(file);
    file = NULL;
    printf("Dump saved: %s (%u bytes packed, map=%u)\n", filename, packed_size, map_size);

out_unmap:
    if (virt1)
        kd_mpi_sys_munmap(virt1, map1_size);
    if (virt0)
        kd_mpi_sys_munmap(virt0, map_size);
out_release:
    ret = kd_mpi_vicap_dump_release(dev, chn, &dump_info);
    if (ret)
        printf("ERROR: kd_mpi_vicap_dump_release failed, ret=%d\n", ret);

    (*dump_count)++;
}

static void run_dump_loop(k_u32 auto_dump_count)
{
    k_u32 dump_count = 0;

    printf("Preview running. CHN0 dump (ofmt=%u), CHN1 preview\n", g_ch0_format);

    if (auto_dump_count > 0) {
        sleep(1);
        printf("Auto-dumping %u frame(s)...\n", auto_dump_count);
        for (k_u32 i = 0; i < auto_dump_count; i++)
            sample_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &dump_count);
        printf("Auto-dump done, exiting.\n");
        return;
    }

    printf("Commands: d / d <n> dump, q quit\n");
    while (g_app_run) {
        char cmd_buf[64];

        printf("Command: ");
        fflush(stdout);
        if (fgets(cmd_buf, sizeof(cmd_buf), stdin) == NULL)
            continue;

        cmd_buf[strcspn(cmd_buf, "\n")] = 0;
        char cmd = cmd_buf[0];
        int count = 1;

        if (cmd == 'd' || cmd == 'D') {
            char *space = strchr(cmd_buf, ' ');
            if (space) {
                count = atoi(space + 1);
                if (count <= 0)
                    count = 1;
            }
            printf("Dumping %d frame(s)...\n", count);
            for (int i = 0; i < count; i++)
                sample_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &dump_count);
        } else if (cmd == 'q' || cmd == 'Q') {
            printf("Exiting...\n");
            break;
        } else if (cmd != '\0') {
            printf("Unknown command: %c\n", cmd);
        }
    }
}

/* -------------------------------------------------------------------------- */
/* CLI                                                                        */
/* -------------------------------------------------------------------------- */

static void print_usage(const char *prog)
{
    printf("Usage: %s -c <connector> [options]\n", prog);
    printf("Options:\n");
    printf("  -c <type>        Connector type [REQUIRED]\n");
    printf("  -r <0|90|180|270> Rotation [default: 0]\n");
    printf("  -s <0|1|2>       CSI index [default: 2]\n");
    printf("  -L, -lane, --lane <2|4>  Probe lane preference [default: ANY / unset]\n");
    printf("  -stype <id>      Force sensor type (see board list_sensor)\n");
    printf("  -width/-height/-fps  Probe target [default: 1920x1080@30]\n");
    printf("  -ofmt <0|1|2|3>  CHN0 format: yuv / rgb888 / rgb888p / raw [default: 0]\n");
    printf("                   ofmt=3 (RAW) uses ONLINE work mode (unless -tile 1)\n");
    printf("  -dump <n>        Dump n frames then quit\n");
    printf("  -ae/-awb/-dw/-dnr3 <0|1>\n");
    printf("  -tile <0|1>      Force SW_TILE [default: 0]\n");
    printf("  -exp <us> / -again <x>  Manual AE (requires -ae 0)\n");
    printf("  -mirror/-flip <0|1>\n");
    printf("  -scene_name / -scene_path\n");
    printf("\nExamples:\n");
    printf("  %s -c 20 -s 0 -width 1920 -height 1080\n", prog);
    printf("  %s -c 20 -s 0 -ofmt 3 -dump 1              # RAW ONLINE dump\n", prog);
    printf("  %s -c 20 -s 0 -width 2592 -height 1944 -lane 4\n", prog);
    printf("  %s -c 20 -s 0 -stype <id> -r 90 -dump 1\n", prog);
}

static k_s32 parse_parameters(int argc, char **argv, sample_params_t *params)
{
    memset(params, 0, sizeof(*params));

    params->csi_idx = 2;
    params->ae_enable = K_TRUE;
    params->awb_enable = K_TRUE;
    params->dw_enable = K_FALSE;
    params->dnr3_enable = K_TRUE;
    params->force_tile = K_FALSE;
    params->sensor_width = 1920;
    params->sensor_height = 1080;
    params->sensor_fps = 30;
    params->ch0_format = CH0_FMT_YUV420SP;
    params->again_value = 1.0f;
    params->lane_pref = VICAP_MIPI_LANE_PREF_ANY;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            params->connector_type = (k_connector_type)atoi(argv[++i]);
            params->c_set = true;
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            params->rot_val = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            params->csi_idx = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-L") == 0 || strcmp(argv[i], "-lane") == 0 ||
                    strcmp(argv[i], "--lane") == 0) && i + 1 < argc) {
            int lane = atoi(argv[++i]);
            if (lane == 4)
                params->lane_pref = VICAP_MIPI_LANE_PREF_4LANE;
            else if (lane == 2)
                params->lane_pref = VICAP_MIPI_LANE_PREF_2LANE;
            else {
                printf("ERROR: Invalid -lane %d, must be 2 or 4\n", lane);
                return -1;
            }
        } else if (strcmp(argv[i], "-ae") == 0 && i + 1 < argc) {
            params->ae_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-awb") == 0 && i + 1 < argc) {
            params->awb_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
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
        } else if (strcmp(argv[i], "-stype") == 0 && i + 1 < argc) {
            params->sensor_type_set = K_TRUE;
            params->sensor_type = atoi(argv[++i]);
            if (params->sensor_type < 0) {
                printf("ERROR: Invalid -stype %d\n", params->sensor_type);
                return -1;
            }
        } else if (strcmp(argv[i], "-dump") == 0 && i + 1 < argc) {
            int n = atoi(argv[++i]);
            params->auto_dump_count = (n > 0) ? (k_u32)n : 1;
        } else if (strcmp(argv[i], "-ofmt") == 0 && i + 1 < argc) {
            params->ch0_format = (k_u32)atoi(argv[++i]);
            if (params->ch0_format > CH0_FMT_RAW) {
                printf("ERROR: Invalid ofmt, must be 0-3\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-dnr3") == 0 && i + 1 < argc) {
            params->dnr3_enable = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-tile") == 0 && i + 1 < argc) {
            params->force_tile = (atoi(argv[++i]) == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-again") == 0 && i + 1 < argc) {
            params->again_set = K_TRUE;
            params->again_value = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-scene_name") == 0 && i + 1 < argc) {
            strncpy(params->scene_name, argv[++i], sizeof(params->scene_name) - 1);
            params->scene_set = K_TRUE;
        } else if (strcmp(argv[i], "-scene_path") == 0 && i + 1 < argc) {
            strncpy(params->scene_path, argv[++i], sizeof(params->scene_path) - 1);
            params->scene_set = K_TRUE;
        } else if (strcmp(argv[i], "-mirror") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (parse_bool01("-mirror", v) < 0)
                return -1;
            params->mirror_en = (v == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-flip") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (parse_bool01("-flip", v) < 0)
                return -1;
            params->flip_en = (v == 1) ? K_TRUE : K_FALSE;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
            return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (!params->c_set) {
        printf("ERROR: Missing required -c\n");
        return -1;
    }
    if (params->exp_set && params->ae_enable == K_TRUE) {
        printf("ERROR: When -exp is set, -ae must be 0\n");
        return -1;
    }
    if (params->again_set && params->ae_enable == K_TRUE) {
        printf("ERROR: When -again is set, -ae must be 0\n");
        return -1;
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/* Sensor / VB / VICAP                                                        */
/* -------------------------------------------------------------------------- */

static k_s32 get_sensor_resolution(const sample_params_t *p)
{
    k_vicap_sensor_info sensor_info;
    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_type sensor_type;
    k_s32 ret;
    memset(&sensor_info, 0, sizeof(sensor_info));
    memset(&probe_cfg, 0, sizeof(probe_cfg));
    if (p->sensor_type_set) {
        sensor_type = (k_vicap_sensor_type)p->sensor_type;
        ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
        if (ret) {
            printf("ERROR: kd_mpi_vicap_get_sensor_info(type=%d) failed, ret=%d\n",
                   p->sensor_type, ret);
            return ret;
        }
        if (sensor_info.csi_num != (k_vicap_csi_num)(g_vicap_csi + VICAP_CSI0)) {
            printf("ERROR: sensor type %d is on CSI%d, but -s %d was requested\n",
                   p->sensor_type, (int)sensor_info.csi_num - 1, g_vicap_csi);
            return -1;
        }
        printf("Using forced sensor type %d: %s %ux%u@%u %s\n",
               p->sensor_type,
               sensor_info.sensor_name ? sensor_info.sensor_name : "unknown",
               sensor_info.width, sensor_info.height, sensor_info.fps,
               mipi_lanes_name(sensor_info.mipi_lanes));
    } else {
        probe_cfg.csi_num = g_vicap_csi;
        probe_cfg.width = p->sensor_width;
        probe_cfg.height = p->sensor_height;
        probe_cfg.fps = p->sensor_fps;
        if (kd_mpi_sensor_adapt_get_ex(&probe_cfg, &sensor_info, p->lane_pref) != 0) {
            printf("ERROR: kd_mpi_sensor_adapt_get_ex failed on CSI %d (lane_pref=%d)\n",
                   probe_cfg.csi_num, (int)p->lane_pref);
            return -1;
        }

        sensor_type = sensor_info.sensor_type;
        ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
        if (ret) {
            printf("ERROR: kd_mpi_vicap_get_sensor_info failed, ret=%d\n", ret);
            return ret;
        }
        printf("Probed sensor type %d: %s %ux%u@%u %s\n",
               sensor_type,
               sensor_info.sensor_name ? sensor_info.sensor_name : "unknown",
               sensor_info.width, sensor_info.height, sensor_info.fps,
               mipi_lanes_name(sensor_info.mipi_lanes));

        if ((p->sensor_width && p->sensor_width != sensor_info.width) ||
            (p->sensor_height && p->sensor_height != sensor_info.height) ||
            (p->sensor_fps && p->sensor_fps != sensor_info.fps)) {
            printf("WARN: requested %ux%u@%u != selected %ux%u@%u\n",
                   p->sensor_width, p->sensor_height, p->sensor_fps,
                   sensor_info.width, sensor_info.height, sensor_info.fps);
        }
    }

    g_sensor_width = sensor_info.width;
    g_sensor_height = sensor_info.height;
    memcpy(&g_sensor_info, &sensor_info, sizeof(g_sensor_info));
    return K_SUCCESS;
}

static k_s32 sample_vb_init(void)
{
    k_vb_config config;
    k_vb_supplement_config supplement_config;
    k_s32 ret;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("ERROR: kd_mpi_vb_set_config failed, ret=%d\n", ret);
        return ret;
    }

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

static void fill_chn0_attr(k_vicap_chn_attr *chn_attr, const k_vicap_dev_attr *dev_attr,
                           k_u32 ch0_format, k_u32 buf_num)
{
    memset(chn_attr, 0, sizeof(*chn_attr));

    if (ch0_format == CH0_FMT_RAW) {
        chn_attr->out_win.width = dev_attr->acq_win.width;
        chn_attr->out_win.height = dev_attr->acq_win.height;
        chn_attr->crop_win = dev_attr->acq_win;
        chn_attr->scale_win = chn_attr->out_win;
        chn_attr->crop_enable = K_FALSE;
        chn_attr->scale_enable = K_FALSE;
    } else {
        k_u32 src_w = dev_attr->acq_win.width;
        k_u32 src_h = dev_attr->acq_win.height;
        k_u32 ch0_w = src_w;
        k_u32 ch0_h = src_h;
        k_bool need_crop = K_FALSE;
        const k_bool unaligned = ((src_w & 15u) != 0) || ((src_h & 1u) != 0);
        const k_bool too_large = (src_w > 1920) || (src_h > 1080);

        if (unaligned || too_large) {
            if (src_w >= 1920 && src_h >= 1080) {
                ch0_w = 1920;
                ch0_h = 1080;
            } else {
                ch0_w = src_w & ~15u;
                ch0_h = src_h & ~1u;
            }
            need_crop = K_TRUE;
        }

        chn_attr->out_win.width = ch0_w;
        chn_attr->out_win.height = ch0_h;
        chn_attr->scale_win = chn_attr->out_win;
        chn_attr->scale_enable = K_FALSE;
        if (need_crop) {
            chn_attr->crop_win.h_start =
                (src_w > ch0_w) ? ((src_w - ch0_w) / 2) & ~1u : 0;
            chn_attr->crop_win.v_start =
                (src_h > ch0_h) ? ((src_h - ch0_h) / 2) & ~1u : 0;
            chn_attr->crop_win.width = ch0_w;
            chn_attr->crop_win.height = ch0_h;
            chn_attr->crop_enable = K_TRUE;
            printf("INFO: CHN0 YUV/RGB center-crop %ux%u → %ux%u\n",
                   src_w, src_h, ch0_w, ch0_h);
        } else {
            chn_attr->crop_win = dev_attr->acq_win;
            chn_attr->crop_enable = K_FALSE;
            printf("INFO: CHN0 YUV/RGB native %ux%u\n", ch0_w, ch0_h);
        }
    }

    chn_attr->chn_enable = K_TRUE;
    switch (ch0_format) {
    case CH0_FMT_RGB888:
        chn_attr->pix_format = PIXEL_FORMAT_RGB_888;
        chn_attr->buffer_size =
            VB_ALIGN_UP(chn_attr->out_win.width * chn_attr->out_win.height * 3, 4096);
        break;
    case CH0_FMT_RGB888P:
        chn_attr->pix_format = PIXEL_FORMAT_RGB_888_PLANAR;
        chn_attr->buffer_size =
            VB_ALIGN_UP(chn_attr->out_win.width * chn_attr->out_win.height * 3, 4096);
        break;
    case CH0_FMT_RAW:
        chn_attr->pix_format = PIXEL_FORMAT_RGB_BAYER_10BPP;
        chn_attr->buffer_size =
            VB_ALIGN_UP(chn_attr->out_win.width * chn_attr->out_win.height * 2, 4096);
        break;
    case CH0_FMT_YUV420SP:
    default:
        chn_attr->pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        chn_attr->buffer_size =
            VB_ALIGN_UP(chn_attr->out_win.width * chn_attr->out_win.height * 3 / 2, 4096);
        break;
    }
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
    chn_attr->scale_enable =
        (out_w != crop_w || out_h != crop_h) ? K_TRUE : K_FALSE;
    chn_attr->chn_enable = K_TRUE;
    chn_attr->pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr->buffer_num = buf_num;
    chn_attr->buffer_size = VB_ALIGN_UP(out_w * out_h * 3 / 2, 4096);
    chn_attr->alignment = 12;
    chn_attr->buffer_pool_id = VB_INVALID_POOLID;

    printf("INFO: CHN1 preview crop %ux%u@(%u,%u) → out %ux%u\n",
           crop_w, crop_h, chn_attr->crop_win.h_start, chn_attr->crop_win.v_start,
           out_w, out_h);
}

static k_s32 sample_vicap_init(const sample_params_t *p, k_u32 out_width, k_u32 out_height)
{
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_vicap_sensor_info sensor_info = {0};
    k_s32 ret;
    k_u32 buf_num;
    k_bool dnr3_use;

    align_preview_size(&out_width, &out_height);
    memcpy(&sensor_info, &g_sensor_info, sizeof(sensor_info));
    if (sensor_info.sensor_name == NULL) {
        printf("no sensor find in csi %d, please check\n", sensor_info.csi_num);
        return -1;
    }

    buf_num = (p->ch0_format == CH0_FMT_RAW) ? 4 : 6;
    dnr3_use = p->dnr3_enable;
    if (p->ch0_format == CH0_FMT_RAW && dnr3_use) {
        dnr3_use = K_FALSE;
        printf("INFO: RAW dump (-ofmt 3): auto disable DNR3\n");
    }

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width = sensor_info.width;
    dev_attr.acq_win.height = sensor_info.height;
    dev_attr.mode = select_work_mode(dev_attr.acq_win.width, dev_attr.acq_win.height,
                                     p->ch0_format, p->force_tile);
    printf("INFO: VICAP work mode = %s\n", work_mode_name(dev_attr.mode));

    dev_attr.buffer_num = buf_num;
    dev_attr.buffer_size =
        VB_ALIGN_UP(sensor_info.width * sensor_info.height * 2, 4096);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(sensor_info));

    dev_attr.pipe_ctrl.data = 0xFFFFFFFF;
    dev_attr.pipe_ctrl.bits.ae_enable = p->ae_enable;
    dev_attr.pipe_ctrl.bits.awb_enable = p->awb_enable;
    dev_attr.pipe_ctrl.bits.ahdr_enable = K_FALSE;
    dev_attr.pipe_ctrl.bits.dnr3_enable = dnr3_use;
    dev_attr.dw_enable = p->dw_enable;
    /* mirror must be set before set_dev_attr / init */
    dev_attr.mirror = mirror_flip_to_mode(p->mirror_en, p->flip_en);

    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_dev_attr failed, ret=%d\n", ret);
        return ret;
    }

    fill_chn0_attr(&chn_attr, &dev_attr, p->ch0_format, buf_num);
    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr CHN0 failed, ret=%d\n", ret);
        return ret;
    }

    fill_chn1_preview_attr(&chn_attr, &dev_attr, out_width, out_height, buf_num);
    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_1, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr CHN1 failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_init failed, ret=%d\n", ret);
        return ret;
    }

    {
        k_vicap_sensor_attr sensor_attr;
        sensor_attr.dev_num = VICAP_DEV_ID_0;
        ret = kd_mpi_vicap_get_sensor_fd(&sensor_attr);
        if (ret) {
            printf("ERROR: kd_mpi_vicap_get_sensor_fd failed, ret=%d\n", ret);
            return ret;
        }
        g_sensor_fd = sensor_attr.sensor_fd;
    }
    return K_SUCCESS;
}

static void sample_vicap_bind_vo(k_vo_layer_id layer_id)
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
    else
        printf("Bind VICAP(dev0,ch1) -> VO(layer=%d) OK\n", layer_id);
}

static void sample_vicap_unbind_vo(k_vo_layer_id layer_id)
{
    k_mpp_chn vi_mpp_chn = {0};
    k_mpp_chn vo_mpp_chn = {0};

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = VICAP_CHN_ID_1;

    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = layer_id;

    k_s32 ret = kd_mpi_sys_unbind(&vi_mpp_chn, &vo_mpp_chn);
    if (ret)
        printf("WARN: kd_mpi_sys_unbind VICAP->VO failed, ret=0x%x\n", ret);
}

static k_s32 apply_manual_exp_again(const sample_params_t *p)
{
    k_s32 ret;

    if (p->exp_set) {
        k_sensor_exposure_time_range range;
        ret = kd_mpi_sensor_get_exposure_time_range(g_sensor_fd, &range);
        if (ret) {
            printf("ERROR: get_exposure_time_range failed, ret=%d\n", ret);
            return ret;
        }
        if (p->exp_value_us < (k_u32)range.min_intg_time_us ||
            p->exp_value_us > (k_u32)range.max_intg_time_us) {
            printf("ERROR: Exposure %u us out of range [%.0f, %.0f]\n",
                   p->exp_value_us, range.min_intg_time_us, range.max_intg_time_us);
            return -1;
        }
        {
            float exp_sec = (float)p->exp_value_us / 1000000.0f;
            k_sensor_intg_time intg_time;
            intg_time.intg_time[0] = exp_sec;
            ret = kd_mpi_sensor_intg_time_set(g_sensor_fd, intg_time);
            if (ret) {
                printf("ERROR: intg_time_set failed, ret=%d\n", ret);
                return ret;
            }
            printf("INFO: Manual exposure %u us\n", p->exp_value_us);
        }
    }

    if (p->again_set) {
        k_sensor_gain_info gain_range;
        ret = kd_mpi_sensor_get_gain_range(g_sensor_fd, &gain_range);
        if (ret) {
            printf("ERROR: get_gain_range failed, ret=%d\n", ret);
            return ret;
        }
        if (p->again_value < gain_range.min || p->again_value > gain_range.max) {
            printf("ERROR: Again %.2f out of range [%.2f, %.2f]\n",
                   p->again_value, gain_range.min, gain_range.max);
            return -1;
        }
        {
            k_sensor_gain gain;
            gain.gain[0] = p->again_value;
            ret = kd_mpi_sensor_again_set(g_sensor_fd, gain);
            if (ret) {
                printf("ERROR: again_set failed, ret=%d\n", ret);
                return ret;
            }
            printf("INFO: Manual again %.2f\n", p->again_value);
        }
    }
    return K_SUCCESS;
}

/* -------------------------------------------------------------------------- */
/* main                                                                       */
/* -------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    sample_params_t params;
    k_s32 ret;
    k_u32 width = 0, height = 0;
    k_vo_layer_id layer_id = 1;
    k_vo_size display_resolution;
    k_gdma_rotation_e rotate = GDMA_ROTATE_DEGREE_0;

    ret = parse_parameters(argc, argv, &params);
    if (ret == 1) {
        print_usage(argv[0]);
        return 0;
    }
    if (ret < 0) {
        print_usage(argv[0]);
        return -1;
    }

    g_ch0_format = params.ch0_format;
    g_vicap_csi = (k_vicap_dev)params.csi_idx;

    ret = get_sensor_resolution(&params);
    if (ret != K_SUCCESS) {
        printf("ERROR: Failed to get sensor resolution\n");
        return -1;
    }

    printf("sensor_info: type=%d name=%s phy_freq=%d hdr_mode=%d lanes=%s\n",
           g_sensor_info.sensor_type, g_sensor_info.sensor_name,
           g_sensor_info.phy_freq, g_sensor_info.hdr_mode,
           mipi_lanes_name(g_sensor_info.mipi_lanes));

    ret = sample_vb_init();
    if (ret != K_SUCCESS)
        return -1;

    if (params.rot_val == 90)
        rotate = GDMA_ROTATE_DEGREE_90;
    else if (params.rot_val == 180)
        rotate = GDMA_ROTATE_DEGREE_180;
    else if (params.rot_val == 270)
        rotate = GDMA_ROTATE_DEGREE_270;

    if (kd_display_init(params.connector_type, 0, 0, rotate) != 0) {
        printf("ERROR: connector init failed\n");
        goto cleanup_vb;
    }

    ret = kd_mpi_vo_get_resolution(&display_resolution);
    if (ret != 0) {
        printf("WARNING: get display resolution failed, use 1920x1080\n");
        display_resolution.width = 1920;
        display_resolution.height = 1080;
    }

    calc_preview_size(g_sensor_width, g_sensor_height,
                      display_resolution.width, display_resolution.height,
                      &width, &height);

    printf("sample_vicap_sensor: connector=%d screen=%ux%u preview=%ux%u "
           "rotate=%d csi=%d ofmt=%u\n",
           params.connector_type, display_resolution.width, display_resolution.height,
           width, height, params.rot_val, g_vicap_csi, g_ch0_format);
    printf("VICAP features: AE=%d AWB=%d DW=%d DNR3=%d tile=%d\n",
           params.ae_enable, params.awb_enable,
           params.dw_enable, params.dnr3_enable, params.force_tile);

    signal(SIGINT, handle_signal);

    if (params.scene_set) {
        printf("Registering scene '%s' path '%s'\n",
               params.scene_name, params.scene_path);
        ret = kd_mpi_vicap_register_scene(params.scene_name, params.scene_path);
        if (ret < 0) {
            printf("ERROR: register scene failed\n");
            goto cleanup_display;
        }
        ret = kd_mpi_vicap_load_scene(params.scene_name);
        if (ret < 0) {
            printf("ERROR: load scene failed\n");
            goto cleanup_display;
        }
    }

    ret = sample_vicap_init(&params, width, height);
    if (ret != K_SUCCESS) {
        printf("ERROR: sample_vicap_init failed, ret=%d\n", ret);
        goto cleanup_display;
    }

    ret = kd_display_layer_configure(layer_id, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                     width, height, 0, 0);
    if (ret != 0) {
        printf("ERROR: configure layer failed, ret=%d\n", ret);
        goto cleanup_vicap;
    }
    kd_display_layer_enable(layer_id);
    sample_vicap_bind_vo(layer_id);

    ret = kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);
    if (ret != K_SUCCESS) {
        printf("ERROR: start_stream failed, ret=%d\n", ret);
        goto cleanup_bind;
    }
    printf("VICAP stream started\n");

    if (apply_manual_exp_again(&params) != K_SUCCESS)
        goto cleanup_stream;

    run_dump_loop(params.auto_dump_count);

cleanup_stream:
    printf("Stopping VICAP stream...\n");
    kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);

cleanup_bind:
    sample_vicap_unbind_vo(layer_id);

cleanup_display:
    kd_display_deinit();

cleanup_vicap:
    printf("Deinit VICAP...\n");
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);

cleanup_vb:
    printf("Deinit VB...\n");
    kd_mpi_vb_exit();
    return 0;
}
