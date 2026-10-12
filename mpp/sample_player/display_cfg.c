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
#include "display_cfg.h"
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "mpi_vo_api.h"
#include "mpi_sys_api.h"
#include "k_video_comm.h"
#include "k_vo_comm.h"
#include "kd_display.h"

#define ENABLE_VO_LAYER   1

static k_connector_type g_connector_type = HX8377_V2_MIPI_4LAN_1080X1920_30FPS;


void display_set_connector_type(k_connector_type type)
{
    g_connector_type = type;
}

typedef struct {
    k_connector_type type;
    k_vo_layer_id layer_id;
    int width;
    int height;
    int ratation_90;
} sample_vo_info;

static k_s32 sample_vo_init(sample_vo_info* vo_info)
{
    k_s32 ret;
    k_gdma_rotation_e rotate = vo_info->ratation_90 ? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;

    ret = kd_display_init(vo_info->type, 0, 0, rotate);
    if (ret != K_SUCCESS) {
        printf("kd_display_init failed, ret=%d\n", ret);
        return ret;
    }

    kd_display_layer_disable(vo_info->layer_id);

    printf("%s>vo init width %d height %d\n", __func__, vo_info->width, vo_info->height);

    ret = kd_display_layer_configure_ex(vo_info->layer_id, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                        vo_info->width, vo_info->height, 0, 0, 0xff, rotate,
                                        vo_info->ratation_90 ? 3 : 0, 0);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_display_layer_configure failed, ret=%d\n", ret);
        kd_display_deinit();
        return ret;
    }

    ret = kd_display_layer_enable(vo_info->layer_id);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_display_layer_enable failed, ret=%d\n", ret);
        kd_display_deinit();
        return ret;
    }

    printf("enable vo layer:%d\n", vo_info->layer_id);
    return 0;
}

k_s32 display_layer_init(k_u32 width,k_u32 height)
{
    sample_vo_info vo_info;
    vo_info.type = g_connector_type;
    vo_info.layer_id = ENABLE_VO_LAYER;
    vo_info.width = width;
    vo_info.height = height;
    vo_info.ratation_90 = 0;

    if (g_connector_type == ST7701_V1_MIPI_2LAN_480X800_30FPS)
    {
        vo_info.ratation_90 = 1;
    }
    else if (g_connector_type == LT9611_MIPI_4LAN_1920X1080_30FPS)
    {
        vo_info.ratation_90 = 0;
    }

    return sample_vo_init(&vo_info);

}

void display_layer_deinit()
{
    kd_display_layer_disable(ENABLE_VO_LAYER);
}

k_s32 vo_init()
{
    return K_SUCCESS;
}

k_s32 vo_deinit()
{
    return kd_display_deinit();
}
