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

#include "k_connector_comm.h"
#include "mpi_connector_api.h"

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
    k_u32 ret = 0;
    k_s32 connector_fd;
    k_connector_type connector_type = vo_info->type;
    k_connector_info connector_info;

    memset(&connector_info, 0, sizeof(k_connector_info));

    //connector get sensor info
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("sample_vicap, the sensor type not supported!\n");
        return ret;
    }

    connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return K_ERR_VO_NOTREADY;
    }

    // connector init
    kd_mpi_connector_init(connector_fd, connector_info);
    // set connect power
    kd_mpi_connector_power_set(connector_fd, K_TRUE);

    kd_mpi_vo_disable_layer(vo_info->layer_id);

    printf("%s>vo init width %d height %d\n", __func__, vo_info->width, vo_info->height);

    k_vo_layer_attr vo_layer_attr;
    memset(&vo_layer_attr, 0, sizeof(vo_layer_attr));
    vo_layer_attr.layer_id           = vo_info->layer_id;
    vo_layer_attr.position.x         = 0;
    vo_layer_attr.position.y         = 0;
    vo_layer_attr.img_size.width     = vo_info->width;
    vo_layer_attr.img_size.height    = vo_info->height;
    vo_layer_attr.pixel_format       = PIXEL_FORMAT_YUV_SEMIPLANAR_420; /* NV12 / YUV420SP */
    vo_layer_attr.func               = (vo_info->ratation_90 == 1) ? GDMA_ROTATE_DEGREE_90 :GDMA_ROTATE_DEGREE_0;
    vo_layer_attr.rot_buf_nr         = (vo_info->ratation_90 == 1) ? 3:0; /* 旋转时使用少量 GSDMA buffer */
    vo_layer_attr.global_alpha       = 0xff;
    vo_layer_attr.rot_buf_bpp        = 0;

    printf("sample_vo_init: layer_id %d, width %d, height %d, pixel_format %d, func %d,buf_nr:%d\n",
           vo_layer_attr.layer_id, vo_layer_attr.img_size.width, vo_layer_attr.img_size.height,
           vo_layer_attr.pixel_format, vo_layer_attr.func,vo_layer_attr.rot_buf_nr);

    ret = kd_mpi_vo_set_layer_attr(vo_info->layer_id,&vo_layer_attr);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_set_layer_attr failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vo_enable_layer(vo_info->layer_id);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_enable_layer failed, ret=%d\n", ret);
        return ret;
    }
    //exit ;
    return 0;
}

void display_layer_init(k_u32 width,k_u32 height)
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

    sample_vo_init(&vo_info);

}

void display_layer_deinit()
{
    kd_mpi_vo_disable_layer(ENABLE_VO_LAYER);
}

k_s32 vo_init()
{
    return K_SUCCESS;
}

k_s32 vo_deinit()
{
    return K_SUCCESS;
}