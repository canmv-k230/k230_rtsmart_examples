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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
 */

#include <getopt.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k_gsdma_comm.h"
#include "demos/benchmark/lv_demo_benchmark.h"
#include "lv_k230_display.h"
#include "lv_k230_vglite.h"
#include "lvgl.h"
#include "mpi_vb_api.h"

#define MAX_DISPLAY_WIDTH  1920U
#define MAX_DISPLAY_HEIGHT 1920U

typedef struct {
    const char *name;
    k_connector_type type;
} connector_entry_t;

static const connector_entry_t g_connectors[] = {
    { "hx8399-1080x1920", HX8399_1080_1920_DSI_V1 },
    { "ili9806-480x800", ILI9806_480_800_DSI_V1 },
    { "ili9881-800x1280", ILI9881_800_1280_DSI_V1 },
    { "nt35516-536x960", NT35516_536_960_DSI_V1 },
    { "nt35532-1080x1920", NT35532_1080_1920_DSI_V1 },
    { "gc9503-480x800", GC9503_480_800_DSI_V1 },
    { "st7102-480x640", ST7102_480_640_DSI_V1 },
    { "aml020t-480x360", AML020T_480_360_DSI_V1 },
    { "st7701-480x800", ST7701_480_800_DSI_V1 },
    { "st7701-480x854", ST7701_480_854_DSI_V1 },
    { "st7701-480x640", ST7701_480_640_DSI_V1 },
    { "st7701-368x544", ST7701_368_544_DSI_V1 },
    { "jd9852-240x320", JD9852_240_320_DSI_V1 },
    { "lt9611-1920x1080-30", LT9611_1920_1080_HDMI_V1 },
    { "lt9611-1920x1080-60", LT9611_1920_1080_HDMI_V2 },
    { "lt9611-1280x720-60", LT9611_1280_720_HDMI_V1 },
    { "lt9611-1280x720-50", LT9611_1280_720_HDMI_V2 },
    { "lt9611-1280x720-30", LT9611_1280_720_HDMI_V3 },
    { "lt9611-640x480-60", LT9611_640_480_HDMI_V1 },
    { "lt9611-vesa-1920x1080-30", LT9611_1920_1080_HDMI_V3 },
    { "lt9611-vesa-1920x1080-60", LT9611_1920_1080_HDMI_V4 },
    { "lt9611-vesa-1280x720-60", LT9611_1280_720_HDMI_V4 },
    { "lt9611-vesa-1280x720-50", LT9611_1280_720_HDMI_V5 },
    { "lt9611-vesa-1280x720-30", LT9611_1280_720_HDMI_V6 },
    { "lt9611-vesa-640x480-60", LT9611_640_480_HDMI_V2 },
    { "st7789-320x240", ST7789_320_240_SPI_V1 },
    { "st7789-240x280", ST7789_240_280_I8080_V1 },
    { "nv3030b-240x240", NV3030B_240_240_QSPI_V1 },
};

static volatile sig_atomic_t g_interrupted;
static volatile bool g_benchmark_finished;

static const char *g_connector_name = "st7701-480x800";
static k_connector_type g_connector_type = ST7701_480_800_DSI_V1;
static lv_color_format_t g_color_format = LV_COLOR_FORMAT_XRGB8888;
static const char *g_color_format_name = "XRGB8888";
static k_gdma_rotation_e g_display_rotation = GDMA_ROTATE_DEGREE_0;
static int g_rotation_degrees;
static uint32_t g_width;
static uint32_t g_height;
static bool g_self_test;

static void signal_handler(int signum)
{
    (void)signum;
    g_interrupted = 1;
}

static const connector_entry_t *find_connector_by_name(const char *name)
{
    size_t i;

    for (i = 0; i < sizeof(g_connectors) / sizeof(g_connectors[0]); i++) {
        if (strcmp(name, g_connectors[i].name) == 0) {
            return &g_connectors[i];
        }
    }

    return NULL;
}

static const connector_entry_t *find_connector_by_type(k_connector_type type)
{
    size_t i;

    for (i = 0; i < sizeof(g_connectors) / sizeof(g_connectors[0]); i++) {
        if (type == g_connectors[i].type) {
            return &g_connectors[i];
        }
    }

    return NULL;
}

static void list_connectors(void)
{
    size_t i;

    printf("Supported connector modes:\n");
    for (i = 0; i < sizeof(g_connectors) / sizeof(g_connectors[0]); i++) {
        printf("  %-31s 0x%08" PRIx32 " (%" PRIu32 ")\n",
               g_connectors[i].name, (uint32_t)g_connectors[i].type,
               (uint32_t)g_connectors[i].type);
    }
}

static bool parse_connector(const char *value)
{
    const connector_entry_t *entry = find_connector_by_name(value);
    if (entry) {
        g_connector_name = entry->name;
        g_connector_type = entry->type;
        return true;
    }

    char *end = NULL;
    unsigned long raw = strtoul(value, &end, 0);
    if (!value[0] || !end || *end != '\0' || raw > UINT32_MAX) {
        return false;
    }

    g_connector_type = (k_connector_type)raw;
    entry = find_connector_by_type(g_connector_type);
    g_connector_name = entry ? entry->name : "numeric";
    return true;
}

static bool parse_color_format(const char *value)
{
    if (strcmp(value, "rgb565") == 0) {
        g_color_format = LV_COLOR_FORMAT_RGB565;
        g_color_format_name = "RGB565";
    } else if (strcmp(value, "rgb888") == 0) {
        g_color_format = LV_COLOR_FORMAT_RGB888;
        g_color_format_name = "RGB888";
    } else if (strcmp(value, "xrgb8888") == 0) {
        g_color_format = LV_COLOR_FORMAT_XRGB8888;
        g_color_format_name = "XRGB8888";
    } else if (strcmp(value, "argb8888") == 0) {
        g_color_format = LV_COLOR_FORMAT_ARGB8888;
        g_color_format_name = "ARGB8888";
    } else {
        return false;
    }

    return true;
}

static bool parse_rotation(const char *value)
{
    char *end = NULL;
    long degrees = strtol(value, &end, 10);

    if (!value[0] || !end || *end != '\0') {
        return false;
    }

    switch (degrees) {
    case 0:
        g_display_rotation = GDMA_ROTATE_DEGREE_0;
        break;
    case 90:
        g_display_rotation = GDMA_ROTATE_DEGREE_90;
        break;
    case 180:
        g_display_rotation = GDMA_ROTATE_DEGREE_180;
        break;
    case 270:
        g_display_rotation = GDMA_ROTATE_DEGREE_270;
        break;
    default:
        return false;
    }

    g_rotation_degrees = (int)degrees;
    return true;
}

static bool parse_renderer(const char *value)
{
    lv_k230_renderer_t renderer;

    if (strcmp(value, "sw") == 0) {
        renderer = LV_K230_RENDERER_SW;
    } else if (strcmp(value, "rvv") == 0) {
        renderer = LV_K230_RENDERER_RVV;
    } else if (strcmp(value, "vglite") == 0) {
        renderer = LV_K230_RENDERER_VGLITE;
    } else {
        return false;
    }

    return lv_k230_renderer_set(renderer);
}

static void print_usage(const char *program)
{
    printf("Usage: %s [options]\n", program);
    printf("  -c, --connector <name|value>  Connector mode (default: st7701-480x800)\n");
    printf("  -f, --format <format>         rgb565, rgb888, xrgb8888, argb8888 (default: xrgb8888)\n");
    printf("  -r, --rotate <degrees>        0, 90, 180, 270 (default: 0)\n");
    printf("  -R, --renderer <renderer>     sw, rvv, vglite (default: vglite)\n");
    printf("  -T, --self-test               Run RVV pixel-equivalence test and exit\n");
    printf("  -L, --list-connectors         List supported connector modes\n");
    printf("  -h, --help                    Show this help\n");
}

static int vb_init(void)
{
    k_vb_config config;
    k_vb_supplement_config supplement_config;
    k_s32 ret;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = VB_MAX_POOLS;
    config.comm_pool[0].blk_cnt = 1;
    config.comm_pool[0].mode = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[0].blk_size =
        VB_ALIGN_UP(MAX_DISPLAY_WIDTH * MAX_DISPLAY_HEIGHT * 4U, 4096);

    ret = kd_mpi_vb_set_config(&config);
    if (ret != K_SUCCESS) {
        printf("LVGL_BENCHMARK_ERROR vb_set_config=%d\n", ret);
        return ret;
    }

    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config = VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret != K_SUCCESS) {
        printf("LVGL_BENCHMARK_ERROR vb_set_supplement_config=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vb_init();
    if (ret != K_SUCCESS) {
        printf("LVGL_BENCHMARK_ERROR vb_init=%d\n", ret);
    }
    return ret;
}

static void print_scene_result(const lv_demo_benchmark_scene_dsc_t *scene)
{
    uint32_t count = scene->measurement_cnt;

    if (count == 0) {
        printf("LVGL_BENCHMARK_RESULT {\"kind\":\"scene\",\"name\":\"%s\","
               "\"samples\":0,\"cpu_pct\":null,\"fps\":null,"
               "\"render_ms\":null,\"flush_ms\":null}\n", scene->name);
        return;
    }

    printf("LVGL_BENCHMARK_RESULT {\"kind\":\"scene\",\"name\":\"%s\","
           "\"samples\":%" PRIu32 ",\"cpu_pct\":%" PRIu32 ","
           "\"fps\":%" PRIu32 ",\"render_ms\":%" PRIu32 ","
           "\"flush_ms\":%" PRIu32 "}\n",
           scene->name, count, scene->cpu_avg_usage / count,
           scene->fps_avg / count, scene->render_avg_time / count,
           scene->flush_avg_time / count);
}

static void benchmark_finished_cb(const lv_demo_benchmark_summary_t *summary)
{
    const lv_demo_benchmark_scene_dsc_t *scene;
    lv_k230_renderer_stats_t renderer_stats;
    int32_t count = summary->valid_scene_cnt;

    lv_k230_renderer_get_stats(&renderer_stats);

    for (scene = summary->scenes; scene->create_cb; scene++) {
        print_scene_result(scene);
    }

    if (count > 0) {
        printf("LVGL_BENCHMARK_RESULT {\"kind\":\"summary\","
               "\"connector\":\"%s\",\"connector_type\":%" PRIu32 ","
               "\"renderer\":\"%s\",\"color_format\":\"%s\","
               "\"rotation\":%d,"
               "\"width\":%" PRIu32 ",\"height\":%" PRIu32 ","
               "\"scene_count\":%d,\"cpu_pct\":%d,\"fps\":%d,"
               "\"render_ms\":%d,\"flush_ms\":%d,"
               "\"rvv_tasks\":%" PRIu64 ",\"rvv_pixels\":%" PRIu64 ","
               "\"vglite_tasks\":%" PRIu64 ",\"vglite_pixels\":%" PRIu64 "}\n",
               g_connector_name, (uint32_t)g_connector_type,
               lv_k230_vglite_renderer_name(), g_color_format_name,
               g_rotation_degrees, g_width, g_height, count,
               summary->total_avg_cpu / count,
               summary->total_avg_fps / count,
               summary->total_avg_render_time / count,
               summary->total_avg_flush_time / count,
               renderer_stats.rvv_tasks, renderer_stats.rvv_pixels,
               renderer_stats.vglite_tasks, renderer_stats.vglite_pixels);
    }

    printf("LVGL_BENCHMARK_END status=PASS\n");
    fflush(stdout);
    g_benchmark_finished = true;
}

int main(int argc, char **argv)
{
    static const struct option options[] = {
        { "connector", required_argument, NULL, 'c' },
        { "format", required_argument, NULL, 'f' },
        { "rotate", required_argument, NULL, 'r' },
        { "renderer", required_argument, NULL, 'R' },
        { "self-test", no_argument, NULL, 'T' },
        { "list-connectors", no_argument, NULL, 'L' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 },
    };
    lv_display_t *display = NULL;
    bool vb_initialized = false;
    bool display_initialized = false;
    int opt;
    int result = EXIT_FAILURE;

    while ((opt = getopt_long(argc, argv, "c:f:r:R:TLh", options, NULL)) != -1) {
        switch (opt) {
        case 'c':
            if (!parse_connector(optarg)) {
                fprintf(stderr, "Invalid connector: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'f':
            if (!parse_color_format(optarg)) {
                fprintf(stderr, "Invalid color format: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'r':
            if (!parse_rotation(optarg)) {
                fprintf(stderr, "Invalid rotation: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'R':
            if (!parse_renderer(optarg)) {
                fprintf(stderr, "Invalid or unavailable renderer: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'T':
            g_self_test = true;
            break;
        case 'L':
            list_connectors();
            return EXIT_SUCCESS;
        case 'h':
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        default:
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (g_self_test) {
        lv_k230_renderer_set(LV_K230_RENDERER_RVV);
        bool passed = lv_k230_renderer_rvv_self_test();
        printf("LVGL_RENDERER_SELF_TEST status=%s\n", passed ? "PASS" : "FAIL");
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("LVGL_BENCHMARK_BEGIN connector=%s connector_type=%" PRIu32
           " renderer=%s color_format=%s rotation=%d lvgl=%d.%d.%d\n",
           g_connector_name, (uint32_t)g_connector_type,
           lv_k230_vglite_renderer_name(), g_color_format_name,
           g_rotation_degrees, LVGL_VERSION_MAJOR, LVGL_VERSION_MINOR,
           LVGL_VERSION_PATCH);
    fflush(stdout);

    if (vb_init() != K_SUCCESS) {
        goto cleanup;
    }
    vb_initialized = true;

    if (kd_display_init(g_connector_type, 0, 0, g_display_rotation) != K_SUCCESS) {
        printf("LVGL_BENCHMARK_END status=FAIL stage=connector_init\n");
        goto cleanup;
    }
    display_initialized = true;

    if (kd_display_get_resolution(&g_width, &g_height) != K_SUCCESS) {
        printf("LVGL_BENCHMARK_END status=FAIL stage=display_resolution\n");
        goto cleanup;
    }

    lv_init();
    display = lv_k230_display_create(K_VO_LAYER_OSD0, 255);
    if (!display) {
        printf("LVGL_BENCHMARK_END status=FAIL stage=display_create\n");
        goto cleanup;
    }

    lv_display_set_color_format(display, g_color_format);
    lv_color_format_t actual_format = lv_display_get_color_format(display);
    if (actual_format != g_color_format) {
        printf("LVGL_BENCHMARK_END status=FAIL stage=color_format requested=%s actual=%d\n",
               g_color_format_name, actual_format);
        goto cleanup;
    }

    printf("LVGL_BENCHMARK_CONFIG width=%" PRIu32 " height=%" PRIu32 "\n",
           g_width, g_height);
    lv_k230_renderer_reset_stats();
    lv_demo_benchmark_set_end_cb(benchmark_finished_cb);
    lv_demo_benchmark();

    while (!g_interrupted && !g_benchmark_finished) {
        uint32_t delay_ms = lv_timer_handler();
        if (delay_ms > 20U) {
            delay_ms = 20U;
        }
        if (delay_ms == 0U) {
            delay_ms = 1U;
        }
        usleep(delay_ms * 1000U);
    }

    if (g_benchmark_finished) {
        result = EXIT_SUCCESS;
    } else {
        printf("LVGL_BENCHMARK_END status=INTERRUPTED\n");
    }

cleanup:
    if (display) {
        lv_display_delete(display);
    }
    if (display_initialized) {
        kd_display_deinit();
    }
    if (vb_initialized) {
        kd_mpi_vb_exit();
    }
    fflush(stdout);
    return result;
}
