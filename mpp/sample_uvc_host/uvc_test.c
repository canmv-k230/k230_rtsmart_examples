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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <sys/mman.h>
#include <signal.h>

#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_sys_comm.h"
#include "mpi_vb_api.h"
#include "mpi_sys_api.h"
#include "mpi_vo_api.h"
#include "k_vo_comm.h"

#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "mpi_vdec_api.h"

#include "mpi_uvc_api.h"

#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align)-1))
#define VO_POOL_BLOCK_COUNT 4
#define OUTPUT_BUF_CNT 6

typedef struct
{
    int ch;
    k_vo_layer_id chn_id;
    bool is_mjpeg;

    bool vb_inited;
    bool uvc_started;
    bool vo_layer_enabled;
    bool vo_pool_created;
    bool vo_block_created;
    bool vo_mapped;
    bool vdec_pool_created;
    bool vdec_attached;
    bool vdec_created;
    bool vdec_started;
    bool vdec_bound;

    k_s32 vo_poolid;
    k_s32 vdec_poolid;
    k_vb_blk_handle vo_block;
    void *vo_vaddr;
    k_u32 vo_map_size;
} sample_uvc_runtime;

int vb_init(void)
{
    k_s32 ret = 0;
    k_vb_config config;

    memset(&config, 0, sizeof(config));

    config.max_pool_cnt = 10;

    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("kd_mpi_vb_set_config fail, ret = %d\n", ret);
        goto out;
    }

    ret = kd_mpi_vb_init();
    if (ret) {
        printf("kd_mpi_vb_init fail, ret = %d\n", ret);
    }

out:
    return ret;
}

int vb_create_vo_pool(int width, int height)
{
    k_s32 pool_id;
    k_vb_pool_config pool_config;
    k_u32 frame_size = ALIGN_UP(width * height * 3 / 2, 0x1000);

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = VO_POOL_BLOCK_COUNT;
    pool_config.blk_size = frame_size;
    pool_config.mode = VB_REMAP_MODE_NONE;
    pool_id = kd_mpi_vb_create_pool(&pool_config);

    if (VB_INVALID_POOLID == pool_id) {
        printf("create vo pool fail\n");
        return VB_INVALID_POOLID;
    }

    return pool_id;
}

k_s32 sample_connector_init(k_connector_type type)
{
    k_u32 ret = 0;
    k_s32 connector_fd = -1;
    k_u32 chip_id = 0x00;
    k_connector_type connector_type = type;
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

    // set connect power
    ret = kd_mpi_connector_power_set(connector_fd, 1);
    if (ret) {
        goto out;
    }
    // set connect get id
    ret = kd_mpi_connector_id_get(connector_fd, &chip_id);
    if (ret) {
        goto out;
    }
    // connector init
    ret = kd_mpi_connector_init(connector_fd, connector_info);
    if (ret) {
        goto out;
    }

out:
    if (connector_fd >= 0) {
        kd_mpi_connector_close(connector_fd);
    }
    return ret;
}

static int sample_vo_layer_start(k_vo_layer_id layer_id, int width, int height, bool rotation)
{
    k_vo_layer_attr attr;

    if (layer_id < K_VO_LAYER_VIDEO1 || layer_id > K_VO_LAYER_VIDEO3) {
        printf("input layer id %d not supported\n", layer_id);
        return -1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.layer_id = layer_id;
    attr.position.x = 0;
    attr.position.y = 0;
    attr.img_size.width = width;
    attr.img_size.height = height;
    attr.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    attr.func = rotation ? GDMA_ROTATE_DEGREE_270 : GDMA_ROTATE_DEGREE_0;
    attr.rot_buf_nr = rotation ? 2 : 0;
    attr.global_alpha = 0xff;

    if (kd_mpi_vo_set_layer_attr(layer_id, &attr) != K_SUCCESS) {
        printf("kd_mpi_vo_set_layer_attr failed\n");
        return -1;
    }

    if (kd_mpi_vo_enable_layer(layer_id) != K_SUCCESS) {
        printf("kd_mpi_vo_enable_layer failed\n");
        return -1;
    }

    return 0;
}

static int sample_vo_prepare_frame(sample_uvc_runtime *rt, k_video_frame_info *vf_info,
                                   int width, int height)
{
    k_u64 phys_addr = 0;
    k_u32 y_size;
    k_u32 frame_size;

    if (!rt || !vf_info) {
        return -1;
    }

    y_size = width * height;
    frame_size = ALIGN_UP(y_size * 3 / 2, 0x1000);

    memset(vf_info, 0, sizeof(*vf_info));
    vf_info->pool_id = rt->vo_poolid;
    vf_info->mod_id = K_ID_VO;
    vf_info->v_frame.width = width;
    vf_info->v_frame.height = height;
    vf_info->v_frame.stride[0] = width;
    vf_info->v_frame.stride[1] = width;
    vf_info->v_frame.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;

    rt->vo_block = kd_mpi_vb_get_block(rt->vo_poolid, frame_size, NULL);
    if (rt->vo_block == VB_INVALID_HANDLE) {
        printf("%s get vb block error\n", __func__);
        return -1;
    }

    phys_addr = kd_mpi_vb_handle_to_phyaddr(rt->vo_block);
    if (phys_addr == 0) {
        printf("%s get phys addr error\n", __func__);
        kd_mpi_vb_release_block(rt->vo_block);
        rt->vo_block = VB_INVALID_HANDLE;
        return -1;
    }

    rt->vo_vaddr = kd_mpi_sys_mmap(phys_addr, frame_size);
    if (rt->vo_vaddr == NULL) {
        printf("%s mmap error\n", __func__);
        kd_mpi_vb_release_block(rt->vo_block);
        rt->vo_block = VB_INVALID_HANDLE;
        return -1;
    }

    vf_info->v_frame.phys_addr[0] = phys_addr;
    vf_info->v_frame.phys_addr[1] = phys_addr + y_size;
    rt->vo_map_size = frame_size;
    rt->vo_block_created = true;
    rt->vo_mapped = true;
    return 0;
}

static k_s32 vb_create_vdec_pool(int width, int height)
{
    k_vb_pool_config pool_config;

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = OUTPUT_BUF_CNT;
    pool_config.blk_size = ALIGN_UP(width * height, 0x1000) * 2;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

static k_s32 vb_destroy_vdec_pool(k_s32 vdec_poolid)
{
    return kd_mpi_vb_destory_pool(vdec_poolid);
}

void yuyv_to_nv12(const char *yuyv, char *nv12, int width, int height) {
    char *y_plane = nv12;               // Y 平面
    char *uv_plane = nv12 + width * height; // UV 交错平面

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i += 2) {
            int index = j * width * 2 + i * 2; // YUYV 格式的索引

            // Y 分量
            y_plane[j * width + i] = yuyv[index];
            y_plane[j * width + i + 1] = yuyv[index + 2];

            // UV 分量（仅偶数行）
            if (j % 2 == 0) {
                uv_plane[(j / 2) * width + i] = yuyv[index + 1];     // U
                uv_plane[(j / 2) * width + i + 1] = yuyv[index + 3]; // V
            }
        }
    }
}

void uyvy_to_nv12(const char *uyvy, char *nv12, int width, int height) {
    char *y_plane = nv12;
    char *uv_plane = nv12 + width * height;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i += 2) {
            int index = j * width * 2 + i * 2;

            y_plane[j * width + i] = uyvy[index + 1];
            y_plane[j * width + i + 1] = uyvy[index + 3];

            if (j % 2 == 0) {
                uv_plane[(j / 2) * width + i] = uyvy[index + 0];
                uv_plane[(j / 2) * width + i + 1] = uyvy[index + 2];
            }
        }
    }
}

void i420_to_nv12(const char *i420, char *nv12, int width, int height) {
    int y_size = width * height;
    int uv_plane_size = y_size / 4;
    const unsigned char *y = (const unsigned char *)i420;
    const unsigned char *u = y + y_size;
    const unsigned char *v = u + uv_plane_size;
    unsigned char *dst_y = (unsigned char *)nv12;
    unsigned char *dst_uv = dst_y + y_size;

    memcpy(dst_y, y, y_size);
    for (int i = 0; i < uv_plane_size; i++) {
        dst_uv[i * 2] = u[i];
        dst_uv[i * 2 + 1] = v[i];
    }
}

static k_s32 sample_vdec_bind_vo(k_u32 chn_id)
{
    k_mpp_chn vdec_mpp_chn;
    k_mpp_chn vvo_mpp_chn;

    vdec_mpp_chn.mod_id = K_ID_VDEC;
    vdec_mpp_chn.dev_id = 0;
    vdec_mpp_chn.chn_id = 0;
    vvo_mpp_chn.mod_id = K_ID_VO;
    vvo_mpp_chn.dev_id = 0;//VVO_DISPLAY_DEV_ID;
    vvo_mpp_chn.chn_id = chn_id;//VVO_DISPLAY_CHN_ID;

    return kd_mpi_sys_bind(&vdec_mpp_chn, &vvo_mpp_chn);
}

static k_s32 sample_vdec_unbind_vo(k_u32 chn_id)
{
    k_mpp_chn vdec_mpp_chn;
    k_mpp_chn vvo_mpp_chn;

    vdec_mpp_chn.mod_id = K_ID_VDEC;
    vdec_mpp_chn.dev_id = 0;
    vdec_mpp_chn.chn_id = 0;
    vvo_mpp_chn.mod_id = K_ID_VO;
    vvo_mpp_chn.dev_id = 0;//VVO_DISPLAY_DEV_ID;
    vvo_mpp_chn.chn_id = chn_id;//VVO_DISPLAY_CHN_ID;

    return kd_mpi_sys_unbind(&vdec_mpp_chn, &vvo_mpp_chn);
}

static volatile sig_atomic_t exit_flag;

static void sig_handler(int sig_no) {
    (void)sig_no;
    exit_flag = 1;
}

static int parse_fourcc_arg(const char *arg, unsigned int *fourcc)
{
    char *endptr = NULL;
    unsigned long value;

    if (!arg || !fourcc) {
        return -1;
    }

    if (!strcmp(arg, "YUY2")) {
        *fourcc = USBH_VIDEO_FOURCC_YUY2;
        return 0;
    }
    if (!strcmp(arg, "UYVY")) {
        *fourcc = USBH_VIDEO_FOURCC_UYVY;
        return 0;
    }
    if (!strcmp(arg, "NV12")) {
        *fourcc = USBH_VIDEO_FOURCC_NV12;
        return 0;
    }
    if (!strcmp(arg, "I420")) {
        *fourcc = USBH_VIDEO_FOURCC_I420;
        return 0;
    }
    if (!strcmp(arg, "MJPEG") || !strcmp(arg, "MJPG")) {
        *fourcc = USBH_VIDEO_FOURCC_MJPEG;
        return 0;
    }

    value = strtoul(arg, &endptr, 0);
    if ((endptr != arg) && (*endptr == '\0') && (value <= 0xffffffffUL)) {
        *fourcc = (unsigned int)value;
        return 0;
    }

    return -1;
}

static int parse_int_arg(const char *arg, int *value)
{
    char *endptr = NULL;
    long tmp;

    if (!arg || !value) {
        return -1;
    }

    errno = 0;
    tmp = strtol(arg, &endptr, 0);
    if (errno || endptr == arg || *endptr != '\0' || tmp < INT_MIN || tmp > INT_MAX) {
        return -1;
    }

    *value = (int)tmp;
    return 0;
}

static const char *fourcc_to_str(unsigned int fourcc, char text[5])
{
    text[0] = (char)(fourcc & 0xff);
    text[1] = (char)((fourcc >> 8) & 0xff);
    text[2] = (char)((fourcc >> 16) & 0xff);
    text[3] = (char)((fourcc >> 24) & 0xff);
    text[4] = '\0';

    for (int i = 0; i < 4; i++) {
        if (!isprint((unsigned char)text[i])) {
            text[i] = '.';
        }
    }

    return text;
}

static void print_usage(void)
{
    printf("Usage: ./sample_uvc [connector_type] [rotation] [fourcc] [width] [height] [total_frame]\n");
    printf("  [fourcc] supports: YUY2 UYVY NV12 I420 MJPEG (or numeric, e.g. 0x47504a4d)\n");
}

static int sample_copy_raw_to_nv12(unsigned int fourcc, const struct uvc_frame *frame,
                                   void *dst, int width, int height)
{
    switch (fourcc) {
    case USBH_VIDEO_FOURCC_YUY2:
        yuyv_to_nv12(frame->userptr, (char *)dst, width, height);
        return 0;
    case USBH_VIDEO_FOURCC_UYVY:
        uyvy_to_nv12(frame->userptr, (char *)dst, width, height);
        return 0;
    case USBH_VIDEO_FOURCC_NV12:
        memcpy(dst, frame->userptr, width * height * 3 / 2);
        return 0;
    case USBH_VIDEO_FOURCC_I420:
        i420_to_nv12(frame->userptr, (char *)dst, width, height);
        return 0;
    default:
        printf("unsupported uncompressed fourcc: 0x%08x\n", fourcc);
        return -1;
    }
}

static int sample_vdec_start(sample_uvc_runtime *rt, int width, int height)
{
    k_vdec_chn_attr attr;
    int ret;

    rt->vdec_poolid = vb_create_vdec_pool(width, height);
    if (rt->vdec_poolid == VB_INVALID_POOLID) {
        printf("fail to create vdec pool\n");
        return -1;
    }
    rt->vdec_pool_created = true;

    ret = kd_mpi_vdec_attach_vb_pool(rt->ch, rt->vdec_poolid);
    if (ret) {
        printf("kd_mpi_vdec_attach_vb_pool fail, ret = %d\n", ret);
        return ret;
    }
    rt->vdec_attached = true;

    memset(&attr, 0, sizeof(attr));
    attr.pic_width = width;
    attr.pic_height = height;
    attr.stream_buf_size = ALIGN_UP(width * height, 0x1000);
    attr.type = K_PT_JPEG;

    ret = kd_mpi_vdec_create_chn(rt->ch, &attr);
    if (ret) {
        printf("kd_mpi_vdec_create_chn fail, ret = %d\n", ret);
        return ret;
    }
    rt->vdec_created = true;

    ret = kd_mpi_vdec_start_chn(rt->ch);
    if (ret) {
        printf("kd_mpi_vdec_start_chn fail, ret = %d\n", ret);
        return ret;
    }
    rt->vdec_started = true;

    ret = sample_vdec_bind_vo(rt->chn_id);
    if (ret) {
        printf("sample_vdec_bind_vo fail, ret = %d\n", ret);
        return ret;
    }
    rt->vdec_bound = true;

    return 0;
}

static void sample_cleanup(sample_uvc_runtime *rt)
{
    if (rt->vdec_bound) {
        sample_vdec_unbind_vo(rt->chn_id);
        rt->vdec_bound = false;
    }
    if (rt->vdec_started) {
        kd_mpi_vdec_stop_chn(rt->ch);
        rt->vdec_started = false;
    }
    if (rt->vdec_created) {
        kd_mpi_vdec_destroy_chn(rt->ch);
        rt->vdec_created = false;
    }
    if (rt->vdec_attached) {
        kd_mpi_vdec_detach_vb_pool(rt->ch);
        rt->vdec_attached = false;
    }
    if (rt->vdec_pool_created) {
        vb_destroy_vdec_pool(rt->vdec_poolid);
        rt->vdec_pool_created = false;
        rt->vdec_poolid = VB_INVALID_POOLID;
    }

    if (rt->vo_layer_enabled) {
        kd_mpi_vo_disable_layer(rt->chn_id);
        rt->vo_layer_enabled = false;
    }

    if (rt->uvc_started) {
        uvc_host_exit();
        rt->uvc_started = false;
    }

    if (rt->vo_mapped) {
        kd_mpi_sys_munmap(rt->vo_vaddr, rt->vo_map_size);
        rt->vo_mapped = false;
        rt->vo_vaddr = NULL;
        rt->vo_map_size = 0;
    }
    if (rt->vo_block_created) {
        kd_mpi_vb_release_block(rt->vo_block);
        rt->vo_block_created = false;
        rt->vo_block = VB_INVALID_HANDLE;
    }
    if (rt->vo_pool_created) {
        kd_mpi_vb_destory_pool(rt->vo_poolid);
        rt->vo_pool_created = false;
        rt->vo_poolid = VB_INVALID_POOLID;
    }

    if (rt->vb_inited) {
        kd_mpi_vb_exit();
        rt->vb_inited = false;
    }
}

int main(int argc, char **argv)
{
    int ret = 0;
    int width, height;
    int total_frame;
    int frame_num = 0;
    char input_fourcc_text[5];
    char negotiated_fourcc_text[5];
    k_connector_type type;
    bool rotation;
    struct uvc_format format = { 0 };
    k_video_frame_info vf_info;
    sample_uvc_runtime rt = {
        .ch = 0,
        .chn_id = K_VO_LAYER_VIDEO1,
        .vo_poolid = VB_INVALID_POOLID,
        .vdec_poolid = VB_INVALID_POOLID,
        .vo_block = VB_INVALID_HANDLE,
    };

    if (argc != 7) {
        print_usage();
        return -1;
    }

    {
        int tmp_type;
        int tmp_rotation;

        if (parse_int_arg(argv[1], &tmp_type) || parse_int_arg(argv[2], &tmp_rotation)) {
            print_usage();
            return -1;
        }
        type = (k_connector_type)tmp_type;
        rotation = (tmp_rotation != 0);
    }

    if (parse_fourcc_arg(argv[3], &format.fourcc)) {
        printf("invalid fourcc: %s\n", argv[3]);
        print_usage();
        return -1;
    }
    if (parse_int_arg(argv[4], &width) || parse_int_arg(argv[5], &height) || parse_int_arg(argv[6], &total_frame) ||
        width <= 0 || height <= 0 || total_frame <= 0) {
        print_usage();
        return -1;
    }

    printf("type = %d, rotation = %d, fourcc = %s (0x%08x), (%d X %d) = %d frame\n",
           type, rotation, fourcc_to_str(format.fourcc, input_fourcc_text),
           format.fourcc, width, height, total_frame);
    rt.is_mjpeg = (format.fourcc == USBH_VIDEO_FOURCC_MJPEG);

    exit_flag = 0;
    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    ret = vb_init();
    if (ret) {
        goto cleanup;
    }
    rt.vb_inited = true;

    format.width = width;
    format.height = height;
    ret = uvc_host_init(&format);
    if (ret) {
        printf("uvc_host_init fail\n");
        goto cleanup;
    }

    ret = uvc_host_start_stream();
    if (ret) {
        printf("uvc start stream fail\n");
        goto cleanup;
    }
    rt.uvc_started = true;

    width = format.width;
    height = format.height;
    printf("uvc resolution is (%d X %d), negotiated fourcc=%s (0x%08x)\n",
           width, height, fourcc_to_str(format.fourcc, negotiated_fourcc_text),
           format.fourcc);

    ret = sample_connector_init(type);
    if (ret) {
        goto cleanup;
    }

    if (sample_vo_layer_start(rt.chn_id, width, height, rotation) != 0) {
        ret = -1;
        goto cleanup;
    }
    rt.vo_layer_enabled = true;

    if (rt.is_mjpeg) {
        ret = sample_vdec_start(&rt, width, height);
        if (ret) {
            goto cleanup;
        }
    } else {
        rt.vo_poolid = vb_create_vo_pool(width, height);
        if (rt.vo_poolid == VB_INVALID_POOLID) {
            ret = -1;
            goto cleanup;
        }
        rt.vo_pool_created = true;

        ret = sample_vo_prepare_frame(&rt, &vf_info, width, height);
        if (ret) {
            ret = -1;
            goto cleanup;
        }
    }

    while (!exit_flag) {
        struct uvc_frame frame;

        memset(&frame, 0, sizeof(frame));
        ret = uvc_host_get_frame(&frame, 5000);
        if (ret) {
            if (!exit_flag) {
                printf("uvc_host_get_frame fail\n");
            } else {
                ret = 0;
            }
            break;
        }

        if (rt.is_mjpeg) {
            ret = kd_mpi_vdec_send_stream(rt.ch, &frame.v_stream, -1);
            if (ret) {
                printf("kd_mpi_vdec_send_stream fail\n");
            }
        } else {
            ret = sample_copy_raw_to_nv12(format.fourcc, &frame, rt.vo_vaddr, width, height);
            if (!ret) {
                ret = kd_mpi_vo_insert_frame(rt.chn_id, &vf_info);
                if (ret) {
                    printf("kd_mpi_vo_insert_frame fail\n");
                }
            }
        }

        {
            int put_ret = uvc_host_put_frame(&frame);
            if (put_ret) {
                printf("uvc_host_put_frame fail\n");
                if (!ret) {
                    ret = put_ret;
                }
            }
        }

        if (ret) {
            break;
        }

        if (++frame_num >= total_frame) {
            break;
        }
    }

cleanup:
    sample_cleanup(&rt);

    if (ret) {
        return ret;
    }
    return 0;
}
