/* Copyright (c) 2025, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 */

/*
 * AE ROI traversal sample.
 *
 * A single AE ROI walks over the complete sensor image in raster order. The
 * same rectangle is drawn over the preview so the active metering region is
 * visible. No face detector or kmodel is required.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <signal.h>
#include <unistd.h>

#include "k_isp_comm.h"
#include "k_module.h"
#include "k_sys_comm.h"
#include "k_vb_comm.h"
#include "k_vicap_comm.h"
#include "k_vo_comm.h"
#include "kd_display.h"
#include "mpi_isp_api.h"
#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_vo_api.h"

#define CAPTURE_DEV       VICAP_DEV_ID_0
#define PREVIEW_CHN       VICAP_CHN_ID_0
#define PREVIEW_LAYER     ((k_vo_layer_id)1)
#define OSD_LAYER         K_VO_LAYER_OSD0
#define OSD_BUFFER_COUNT  2
#define RECT_COLOR        0xFFFF0000U
#define RECT_THICKNESS    3U
#define PREVIEW_ALIGN_WIDTH  16U
#define PREVIEW_ALIGN_HEIGHT 8U

typedef struct {
    k_connector_type connector;
    k_u32 request_width;
    k_u32 request_height;
    k_u32 request_fps;
    k_u32 roi_width;
    k_u32 roi_height;
    k_u32 step_x;
    k_u32 step_y;
    k_u32 interval_ms;
    int csi;
    int rotation;
    bool connector_set;
} sample_options;

typedef struct {
    k_u32 x;
    k_u32 y;
    k_u32 width;
    k_u32 height;
} sample_rect;

typedef struct {
    k_u32 width;
    k_u32 height;
    k_u32 size;
    k_s32 pool_id;
    k_vb_blk_handle blocks[OSD_BUFFER_COUNT];
    k_u64 phys[OSD_BUFFER_COUNT];
    k_u32 *pixels[OSD_BUFFER_COUNT];
} overlay_buffers;

static volatile bool g_running = true;

static void handle_signal(int signal_number)
{
    if (signal_number == SIGINT || signal_number == SIGTERM)
        g_running = false;
}

static void print_usage(const char *program)
{
    printf("Usage: %s -c <connector_type> [options]\n", program);
    printf("Options:\n");
    printf("  -c <type>       Connector type reported by list_connector (required)\n");
    printf("  -s <csi>        Sensor CSI index [default: 2]\n");
    printf("  -r <degree>     Display rotation: 0, 90, 180, or 270 [default: auto]\n");
    printf("  -width <px>     Requested sensor width [default: 1920]\n");
    printf("  -height <px>    Requested sensor height [default: 1080]\n");
    printf("  -fps <value>    Requested sensor frame rate [default: 30]\n");
    printf("  -roi-width <px> AE ROI width in sensor pixels [default: 320]\n");
    printf("  -roi-height <px> AE ROI height in sensor pixels [default: 320]\n");
    printf("  -step-x <px>    Horizontal distance per update [default: ROI width]\n");
    printf("  -step-y <px>    Vertical distance per row [default: ROI height]\n");
    printf("  -interval <ms>  Time between ROI updates [default: 1000]\n");
    printf("  -h, -help       Show this help\n");
    printf("Run list_connector first, then use a reported connector type.\n");
    printf("Example: %s -c <connector_type> -s 2 -roi-width 320 -roi-height 240 -interval 1000\n",
           program);
}

static bool parse_u32(const char *option, const char *text, k_u32 *value)
{
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);

    if (!text[0] || (end && *end) || parsed == 0 || parsed > UINT32_MAX) {
        printf("ERROR: invalid value for %s: %s\n", option, text);
        return false;
    }
    *value = (k_u32)parsed;
    return true;
}

static int parse_options(int argc, char **argv, sample_options *options)
{
    memset(options, 0, sizeof(*options));
    options->csi = 2;
    options->request_width = 1920;
    options->request_height = 1080;
    options->request_fps = 30;
    options->roi_width = 320;
    options->roi_height = 320;
    options->interval_ms = 1000;
    options->rotation = -1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help"))
            return 1;
        if (i + 1 >= argc) {
            printf("ERROR: missing value for %s\n", argv[i]);
            return -1;
        }

        const char *value = argv[++i];
        if (!strcmp(argv[i - 1], "-c")) {
            options->connector = (k_connector_type)atoi(value);
            options->connector_set = true;
        } else if (!strcmp(argv[i - 1], "-s")) {
            options->csi = atoi(value);
        } else if (!strcmp(argv[i - 1], "-r")) {
            options->rotation = atoi(value);
        } else if (!strcmp(argv[i - 1], "-width")) {
            if (!parse_u32("-width", value, &options->request_width)) return -1;
        } else if (!strcmp(argv[i - 1], "-height")) {
            if (!parse_u32("-height", value, &options->request_height)) return -1;
        } else if (!strcmp(argv[i - 1], "-fps")) {
            if (!parse_u32("-fps", value, &options->request_fps)) return -1;
        } else if (!strcmp(argv[i - 1], "-roi-width")) {
            if (!parse_u32("-roi-width", value, &options->roi_width)) return -1;
        } else if (!strcmp(argv[i - 1], "-roi-height")) {
            if (!parse_u32("-roi-height", value, &options->roi_height)) return -1;
        } else if (!strcmp(argv[i - 1], "-step-x")) {
            if (!parse_u32("-step-x", value, &options->step_x)) return -1;
        } else if (!strcmp(argv[i - 1], "-step-y")) {
            if (!parse_u32("-step-y", value, &options->step_y)) return -1;
        } else if (!strcmp(argv[i - 1], "-interval")) {
            if (!parse_u32("-interval", value, &options->interval_ms)) return -1;
        } else {
            printf("ERROR: unknown option: %s\n", argv[i - 1]);
            return -1;
        }
    }

    if (!options->connector_set) {
        printf("ERROR: -c <connector_type> is required\n");
        return -1;
    }
    if (options->csi < 0 || options->csi > 2) {
        printf("ERROR: CSI index must be 0, 1, or 2\n");
        return -1;
    }
    if (options->rotation != -1 && options->rotation != 0 && options->rotation != 90 &&
        options->rotation != 180 && options->rotation != 270) {
        printf("ERROR: rotation must be 0, 90, 180, or 270\n");
        return -1;
    }
    if (!options->step_x) options->step_x = options->roi_width;
    if (!options->step_y) options->step_y = options->roi_height;
    return 0;
}

static k_gdma_rotation_e display_rotation(int rotation)
{
    switch (rotation) {
    case 90: return GDMA_ROTATE_DEGREE_90;
    case 180: return GDMA_ROTATE_DEGREE_180;
    case 270: return GDMA_ROTATE_DEGREE_270;
    default: return GDMA_ROTATE_DEGREE_0;
    }
}

static bool rotation_swaps_axes(int rotation)
{
    return rotation == 90 || rotation == 270;
}

static int resolve_display_rotation(const sample_options *options)
{
    k_u32 connector_width;
    k_u32 connector_height;

    if (options->rotation >= 0)
        return options->rotation;

    connector_width = K_CONN_WIDTH(options->connector);
    connector_height = K_CONN_HEIGHT(options->connector);
    return connector_width != 0 && connector_height != 0 &&
           connector_width < connector_height ? 90 : 0;
}

static k_s32 probe_sensor(const sample_options *options, k_vicap_sensor_info *sensor_info)
{
    k_vicap_probe_config probe;
    memset(&probe, 0, sizeof(probe));
    memset(sensor_info, 0, sizeof(*sensor_info));

    probe.csi_num = options->csi;
    probe.width = options->request_width;
    probe.height = options->request_height;
    probe.fps = options->request_fps;

    k_s32 ret = kd_mpi_sensor_adapt_get(&probe, sensor_info);
    if (ret) {
        printf("ERROR: no sensor found on CSI%d for %ux%u@%u\n", options->csi,
               options->request_width, options->request_height, options->request_fps);
        return ret;
    }

    ret = kd_mpi_vicap_get_sensor_info(sensor_info->sensor_type, sensor_info);
    if (ret)
        printf("ERROR: failed to get sensor information, ret=%d\n", ret);
    return ret;
}

static k_s32 vb_init(void)
{
    k_vb_config config;
    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;

    k_s32 ret = kd_mpi_vb_set_config(&config);
    if (ret) return ret;
    return kd_mpi_vb_init();
}

static k_s32 choose_preview_size(k_u32 sensor_width, k_u32 sensor_height,
                                 k_u32 screen_width, k_u32 screen_height,
                                 k_u32 *width, k_u32 *height, k_u32 *x, k_u32 *y)
{
    if (sensor_width == 0 || sensor_height == 0 ||
        screen_width == 0 || screen_height == 0)
        return K_FAILED;

    if ((uint64_t)sensor_width * screen_height > (uint64_t)screen_width * sensor_height) {
        *width = screen_width;
        *height = (k_u32)((uint64_t)screen_width * sensor_height / sensor_width);
    } else {
        *height = screen_height;
        *width = (k_u32)((uint64_t)screen_height * sensor_width / sensor_height);
    }

    *width &= ~(PREVIEW_ALIGN_WIDTH - 1U);
    *height &= ~(PREVIEW_ALIGN_HEIGHT - 1U);
    if (*width == 0 || *height == 0)
        return K_FAILED;

    *x = (screen_width - *width) / 2;
    *y = (screen_height - *height) / 2;
    return K_SUCCESS;
}

static k_s32 get_logical_display_size(int rotation,
                                      k_u32 reported_width, k_u32 reported_height,
                                      k_u32 *width, k_u32 *height)
{
    k_connector_info connector_info;
    k_u32 display_width = reported_width;
    k_u32 display_height = reported_height;

    memset(&connector_info, 0, sizeof(connector_info));
    if (kd_display_get_connector_info(&connector_info) == K_SUCCESS &&
        connector_info.resolution.hactive != 0 &&
        connector_info.resolution.vactive != 0) {
        display_width = connector_info.resolution.hactive;
        display_height = connector_info.resolution.vactive;
        if (rotation_swaps_axes(rotation)) {
            k_u32 value = display_width;
            display_width = display_height;
            display_height = value;
        }
    }

    if (display_width == 0 || display_height == 0)
        return K_FAILED;

    *width = display_width;
    *height = display_height;
    return K_SUCCESS;
}

static k_s32 vicap_init(const k_vicap_sensor_info *sensor_info,
                        k_u32 preview_width, k_u32 preview_height)
{
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    memset(&dev_attr, 0, sizeof(dev_attr));
    memset(&chn_attr, 0, sizeof(chn_attr));

    dev_attr.acq_win.width = sensor_info->width;
    dev_attr.acq_win.height = sensor_info->height;
    dev_attr.input_type = VICAP_INPUT_TYPE_SENSOR;
    dev_attr.mode = (sensor_info->width == 3840 && sensor_info->height == 2160) ?
                    VICAP_WORK_SW_TILE_MODE : VICAP_WORK_ONLINE_MODE;
    dev_attr.buffer_num = 6;
    dev_attr.buffer_size = VB_ALIGN_UP(sensor_info->width * sensor_info->height * 2, 4096);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    dev_attr.pipe_ctrl.data = 0xFFFFFFFF;
    dev_attr.pipe_ctrl.bits.af_enable = 0;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.pipe_ctrl.bits.dnr3_enable = 0;
    dev_attr.pipe_ctrl.bits.ae_enable = 1;
    dev_attr.pipe_ctrl.bits.awb_enable = 1;
    memcpy(&dev_attr.sensor_info, sensor_info, sizeof(*sensor_info));

    k_s32 ret = kd_mpi_vicap_set_dev_attr(CAPTURE_DEV, dev_attr);
    if (ret) return ret;

    chn_attr.out_win.width = preview_width;
    chn_attr.out_win.height = preview_height;
    chn_attr.crop_win = dev_attr.acq_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_TRUE;
    chn_attr.scale_enable = (preview_width != sensor_info->width ||
                             preview_height != sensor_info->height) ? K_TRUE : K_FALSE;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num = 6;
    chn_attr.buffer_size = VB_ALIGN_UP(preview_width * preview_height * 3 / 2, 4096);
    chn_attr.alignment = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(CAPTURE_DEV, PREVIEW_CHN, chn_attr);
    if (ret) return ret;
    return kd_mpi_vicap_init(CAPTURE_DEV);
}

static k_s32 bind_preview(void)
{
    k_mpp_chn source = { K_ID_VI, CAPTURE_DEV, PREVIEW_CHN };
    k_mpp_chn destination = { K_ID_VO, K_VO_DISPLAY_DEV_ID, PREVIEW_LAYER };
    return kd_mpi_sys_bind(&source, &destination);
}

static k_s32 unbind_preview(void)
{
    k_mpp_chn source = { K_ID_VI, CAPTURE_DEV, PREVIEW_CHN };
    k_mpp_chn destination = { K_ID_VO, K_VO_DISPLAY_DEV_ID, PREVIEW_LAYER };
    return kd_mpi_sys_unbind(&source, &destination);
}

static k_s32 overlay_init(overlay_buffers *overlay, k_u32 width, k_u32 height)
{
    k_vb_pool_config config;
    memset(overlay, 0, sizeof(*overlay));
    memset(&config, 0, sizeof(config));
    overlay->pool_id = VB_INVALID_POOLID;
    for (k_u32 i = 0; i < OSD_BUFFER_COUNT; ++i)
        overlay->blocks[i] = VB_INVALID_HANDLE;

    overlay->width = width;
    overlay->height = height;
    overlay->size = VB_ALIGN_UP(width * height * sizeof(k_u32), 4096);
    config.blk_cnt = OSD_BUFFER_COUNT;
    config.blk_size = overlay->size;
    config.mode = VB_REMAP_MODE_NOCACHE;

    overlay->pool_id = kd_mpi_vb_create_pool(&config);
    if (overlay->pool_id < 0)
        return overlay->pool_id;

    for (k_u32 i = 0; i < OSD_BUFFER_COUNT; ++i) {
        overlay->blocks[i] = kd_mpi_vb_get_block(overlay->pool_id, overlay->size, NULL);
        if (overlay->blocks[i] == VB_INVALID_HANDLE)
            return K_FAILED;
        overlay->phys[i] = kd_mpi_vb_handle_to_phyaddr(overlay->blocks[i]);
        overlay->pixels[i] = kd_mpi_sys_mmap(overlay->phys[i], overlay->size);
        if (!overlay->pixels[i])
            return K_FAILED;
        memset(overlay->pixels[i], 0, overlay->size);
    }
    return K_SUCCESS;
}

static void overlay_deinit(overlay_buffers *overlay)
{
    for (k_u32 i = 0; i < OSD_BUFFER_COUNT; ++i) {
        if (overlay->pixels[i])
            kd_mpi_sys_munmap(overlay->pixels[i], overlay->size);
        if (overlay->blocks[i] != VB_INVALID_HANDLE)
            kd_mpi_vb_release_block(overlay->blocks[i]);
    }
    if (overlay->pool_id >= 0)
        kd_mpi_vb_destory_pool(overlay->pool_id);
}

static void set_ae_roi(const sample_rect *rect)
{
    k_isp_ae_roi roi;
    memset(&roi, 0, sizeof(roi));
    roi.roiNum = 1;
    roi.roiWeight = 1.0f;
    roi.roiWindow[0].window.hOffset = (k_u16)rect->x;
    roi.roiWindow[0].window.vOffset = (k_u16)rect->y;
    roi.roiWindow[0].window.width = (k_u16)rect->width;
    roi.roiWindow[0].window.height = (k_u16)rect->height;
    roi.roiWindow[0].weight = 1.0f;

    k_s32 ret = kd_mpi_isp_ae_set_roi((k_isp_dev)CAPTURE_DEV, roi);
    if (ret)
        printf("WARNING: failed to set AE ROI, ret=%d\n", ret);
}

static void draw_roi(overlay_buffers *overlay, k_u32 buffer_index,
                     const sample_rect *rect, k_u32 sensor_width, k_u32 sensor_height,
                     k_u32 preview_width, k_u32 preview_height,
                     k_u32 preview_x, k_u32 preview_y)
{
    k_u32 *pixels = overlay->pixels[buffer_index];
    k_u32 x0 = preview_x + (k_u32)((uint64_t)rect->x * preview_width / sensor_width);
    k_u32 y0 = preview_y + (k_u32)((uint64_t)rect->y * preview_height / sensor_height);
    k_u32 x1 = preview_x + (k_u32)((uint64_t)(rect->x + rect->width) * preview_width / sensor_width);
    k_u32 y1 = preview_y + (k_u32)((uint64_t)(rect->y + rect->height) * preview_height / sensor_height);
    k_video_frame_info frame;

    if (x1 >= overlay->width) x1 = overlay->width - 1;
    if (y1 >= overlay->height) y1 = overlay->height - 1;
    memset(pixels, 0, overlay->size);

    for (k_u32 thickness = 0; thickness < RECT_THICKNESS; ++thickness) {
        if (x0 + thickness > x1 || y0 + thickness > y1) break;
        for (k_u32 x = x0 + thickness; x <= x1 - thickness; ++x) {
            pixels[(y0 + thickness) * overlay->width + x] = RECT_COLOR;
            pixels[(y1 - thickness) * overlay->width + x] = RECT_COLOR;
        }
        for (k_u32 y = y0 + thickness; y <= y1 - thickness; ++y) {
            pixels[y * overlay->width + x0 + thickness] = RECT_COLOR;
            pixels[y * overlay->width + x1 - thickness] = RECT_COLOR;
        }
    }

    memset(&frame, 0, sizeof(frame));
    frame.mod_id = K_ID_VO;
    frame.pool_id = overlay->pool_id;
    frame.v_frame.width = overlay->width;
    frame.v_frame.height = overlay->height;
    frame.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    frame.v_frame.stride[0] = overlay->width * sizeof(k_u32);
    frame.v_frame.phys_addr[0] = overlay->phys[buffer_index];
    kd_display_layer_push_frame(OSD_LAYER, &frame);
}

static void advance_roi(sample_rect *rect, k_u32 image_width, k_u32 image_height,
                        k_u32 step_x, k_u32 step_y)
{
    if (rect->x + rect->width < image_width) {
        k_u32 next_x = rect->x + step_x;
        rect->x = next_x + rect->width > image_width ? image_width - rect->width : next_x;
        return;
    }

    rect->x = 0;
    if (rect->y + rect->height < image_height) {
        k_u32 next_y = rect->y + step_y;
        rect->y = next_y + rect->height > image_height ? image_height - rect->height : next_y;
    } else {
        rect->y = 0;
    }
}

int main(int argc, char **argv)
{
    sample_options options;
    k_vicap_sensor_info sensor_info;
    k_u32 reported_width = 0, reported_height = 0;
    k_u32 screen_width = 0, screen_height = 0;
    k_u32 preview_width, preview_height, preview_x, preview_y;
    overlay_buffers overlay;
    bool vb_ready = false, display_ready = false, vicap_ready = false;
    bool layer_ready = false, osd_ready = false, overlay_ready = false;
    bool bound = false, streaming = false, roi_enabled = false;
    int rotation;

    memset(&overlay, 0, sizeof(overlay));

    int parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) {
        print_usage(argv[0]);
        return parse_result > 0 ? 0 : -1;
    }

    k_s32 ret = probe_sensor(&options, &sensor_info);
    if (ret) return ret;

    if (options.roi_width > sensor_info.width) options.roi_width = sensor_info.width;
    if (options.roi_height > sensor_info.height) options.roi_height = sensor_info.height;
    rotation = resolve_display_rotation(&options);

    ret = vb_init();
    if (ret) {
        printf("ERROR: VB initialization failed, ret=%d\n", ret);
        goto cleanup;
    }
    vb_ready = true;

    ret = kd_display_init(options.connector, 0, 0, display_rotation(rotation));
    if (ret) {
        printf("ERROR: display initialization failed, ret=%d\n", ret);
        goto cleanup;
    }
    display_ready = true;

    ret = kd_display_get_resolution(&reported_width, &reported_height);
    if (ret) {
        printf("ERROR: could not get display resolution, ret=%d\n", ret);
        goto cleanup;
    }

    ret = get_logical_display_size(rotation, reported_width, reported_height,
                                   &screen_width, &screen_height);
    if (ret) {
        printf("ERROR: invalid display size %ux%u\n", reported_width, reported_height);
        goto cleanup;
    }

    ret = choose_preview_size(sensor_info.width, sensor_info.height,
                              screen_width, screen_height,
                              &preview_width, &preview_height, &preview_x, &preview_y);
    if (ret) {
        printf("ERROR: cannot choose preview size for sensor %ux%u and display %ux%u\n",
               sensor_info.width, sensor_info.height, screen_width, screen_height);
        goto cleanup;
    }

    ret = vicap_init(&sensor_info, preview_width, preview_height);
    if (ret) {
        printf("ERROR: VICAP initialization failed, ret=%d\n", ret);
        goto cleanup;
    }
    vicap_ready = true;

    ret = kd_display_layer_configure(PREVIEW_LAYER, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                     preview_width, preview_height, preview_x, preview_y);
    if (ret) {
        printf("ERROR: preview layer configuration failed, ret=%d\n", ret);
        goto cleanup;
    }
    ret = kd_display_layer_enable(PREVIEW_LAYER);
    if (ret) goto cleanup;
    layer_ready = true;

    overlay_ready = true;
    ret = overlay_init(&overlay, screen_width, screen_height);
    if (ret) {
        printf("ERROR: OSD buffer allocation failed, ret=%d\n", ret);
        goto cleanup;
    }
    ret = kd_display_layer_configure(OSD_LAYER, PIXEL_FORMAT_ARGB_8888,
                                     screen_width, screen_height, 0, 0);
    if (ret) {
        printf("ERROR: OSD layer configuration failed, ret=%d\n", ret);
        goto cleanup;
    }
    ret = kd_display_layer_enable(OSD_LAYER);
    if (ret) goto cleanup;
    osd_ready = true;

    ret = bind_preview();
    if (ret) {
        printf("ERROR: VICAP to VO bind failed, ret=%d\n", ret);
        goto cleanup;
    }
    bound = true;

    ret = kd_mpi_vicap_start_stream(CAPTURE_DEV);
    if (ret) {
        printf("ERROR: VICAP stream start failed, ret=%d\n", ret);
        goto cleanup;
    }
    streaming = true;

    ret = kd_mpi_isp_ae_roi_set_enable((k_isp_dev)CAPTURE_DEV, K_TRUE);
    if (ret) {
        printf("ERROR: AE ROI enable failed, ret=%d\n", ret);
        goto cleanup;
    }
    roi_enabled = true;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    printf("AE ROI traversal running: CSI%d, sensor %ux%u, logical display %ux%u, "
           "preview %ux%u at (%u,%u), rotation %d, ROI %ux%u\n",
           options.csi, sensor_info.width, sensor_info.height, screen_width, screen_height,
           preview_width, preview_height, preview_x, preview_y, rotation,
           options.roi_width, options.roi_height);
    printf("Press Ctrl+C to stop.\n");

    sample_rect rect = { 0, 0, options.roi_width, options.roi_height };
    k_u32 buffer_index = 0;
    while (g_running) {
        set_ae_roi(&rect);
        draw_roi(&overlay, buffer_index, &rect, sensor_info.width, sensor_info.height,
                 preview_width, preview_height, preview_x, preview_y);
        buffer_index = (buffer_index + 1) % OSD_BUFFER_COUNT;
        usleep(options.interval_ms * 1000U);
        advance_roi(&rect, sensor_info.width, sensor_info.height,
                    options.step_x, options.step_y);
    }

    ret = 0;

cleanup:
    if (roi_enabled)
        kd_mpi_isp_ae_roi_set_enable((k_isp_dev)CAPTURE_DEV, K_FALSE);
    if (streaming)
        kd_mpi_vicap_stop_stream(CAPTURE_DEV);
    if (bound)
        unbind_preview();
    if (osd_ready)
        kd_display_layer_disable(OSD_LAYER);
    if (layer_ready)
        kd_display_layer_disable(PREVIEW_LAYER);
    if (overlay_ready)
        overlay_deinit(&overlay);
    if (vicap_ready)
        kd_mpi_vicap_deinit(CAPTURE_DEV);
    if (display_ready)
        kd_display_deinit();
    if (vb_ready)
        kd_mpi_vb_exit();
    return ret;
}
