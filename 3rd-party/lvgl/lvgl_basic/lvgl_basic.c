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
#include <string.h>
#include <unistd.h>

#include <getopt.h>
#include <signal.h>

#include "lvgl.h"

#include "lv_k230_display.h"
#include "lv_k230_input_touch.h"

#include "k_gsdma_comm.h"
#include "mpi_vb_api.h"

#define MAX_DISPLATY_WIDTH  (1920)
#define MAX_DISPLATY_HEIGHT (1080)

// Global variables for signal handling
static volatile int  g_signal_received = 0;
static lv_display_t* g_display         = NULL;

// Signal handler for graceful shutdown
static void signal_handler(int signum)
{
    printf("\nReceived signal %d, shutting down...\n", signum);
    g_signal_received = 1;
}

/* Screen objects */
static lv_obj_t *main_screen;
static lv_obj_t *counter_label;
static lv_obj_t *color_preview;
static int touch_counter = 0;

/* Function prototypes */
static void create_simple_ui(lv_obj_t *parent);
static void btn_click_cb(lv_event_t *e);
static void slider_cb(lv_event_t *e);
static void switch_cb(lv_event_t *e);
static void touch_cb(lv_event_t *e);

void lv_demo_touch_simple(void)
{
    /* Create main screen */
    main_screen = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(main_screen, lv_color_hex(0x333333), 0);
    
    /* Add touch callback to whole screen */
    lv_obj_add_event_cb(main_screen, touch_cb, LV_EVENT_CLICKED, NULL);
    
    /* Create UI */
    create_simple_ui(main_screen);
    
    /* Load screen */
    lv_screen_load(main_screen);
}

static void create_simple_ui(lv_obj_t *parent)
{
    /* Use simple padding that works with rotation */
    lv_obj_set_flex_flow(parent, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(parent, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_all(parent, 10, 0);
    lv_obj_set_style_pad_row(parent, 10, 0);
    
    /* Title */
    lv_obj_t *title = lv_label_create(parent);
    lv_label_set_text(title, "LVGL Touch Demo");
    lv_obj_set_style_text_color(title, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(title, &lv_font_montserrat_14, 0);
    
    /* Touch counter */
    counter_label = lv_label_create(parent);
    lv_label_set_text_fmt(counter_label, "Touches: %d", touch_counter);
    lv_obj_set_style_text_color(counter_label, lv_color_hex(0x00FF00), 0);
    
    /* Color preview box (shows touch feedback) */
    color_preview = lv_obj_create(parent);
    lv_obj_set_size(color_preview, 100, 100);
    lv_obj_set_style_bg_color(color_preview, lv_color_hex(0x555555), 0);
    lv_obj_set_style_radius(color_preview, 10, 0);
    lv_obj_set_style_border_width(color_preview, 2, 0);
    lv_obj_set_style_border_color(color_preview, lv_color_hex(0xFFFFFF), 0);
    
    /* Touch button */
    lv_obj_t *btn = lv_btn_create(parent);
    lv_obj_set_size(btn, 150, 50);
    lv_obj_set_style_bg_color(btn, lv_color_hex(0x2196F3), 0);
    lv_obj_set_style_radius(btn, 25, 0);
    lv_obj_add_event_cb(btn, btn_click_cb, LV_EVENT_CLICKED, NULL);
    
    lv_obj_t *btn_label = lv_label_create(btn);
    lv_label_set_text(btn_label, "Touch Me");
    lv_obj_center(btn_label);
    
    /* Slider */
    lv_obj_t *slider = lv_slider_create(parent);
    lv_obj_set_width(slider, 200);
    lv_obj_add_event_cb(slider, slider_cb, LV_EVENT_VALUE_CHANGED, NULL);
    
    /* Switch with label */
    lv_obj_t *switch_row = lv_obj_create(parent);
    lv_obj_set_size(switch_row, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(switch_row, 0, 0);
    lv_obj_set_style_border_width(switch_row, 0, 0);
    lv_obj_set_flex_flow(switch_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_style_pad_column(switch_row, 10, 0);
    
    lv_obj_t *switch_label = lv_label_create(switch_row);
    lv_label_set_text(switch_label, "Toggle:");
    lv_obj_set_style_text_color(switch_label, lv_color_hex(0xFFFFFF), 0);
    
    lv_obj_t *sw = lv_switch_create(switch_row);
    lv_obj_add_event_cb(sw, switch_cb, LV_EVENT_VALUE_CHANGED, NULL);
}

/* Button click callback */
static void btn_click_cb(lv_event_t *e)
{
    touch_counter++;
    lv_label_set_text_fmt(counter_label, "Touches: %d", touch_counter);
    
    /* Flash the color preview */
    lv_obj_set_style_bg_color(color_preview, lv_color_hex(0xFF0000), 0);
    lv_obj_set_style_bg_color(color_preview, lv_color_hex(0x555555), 300);
}

/* Slider callback */
static void slider_cb(lv_event_t *e)
{
    lv_obj_t *slider = lv_event_get_target(e);
    int32_t val = lv_slider_get_value(slider);
    
    /* Change preview color based on slider */
    lv_color_t color = lv_color_hsv_to_rgb(val * 3.6, 100, 100);
    lv_obj_set_style_bg_color(color_preview, color, 0);
}

/* Switch callback */
static void switch_cb(lv_event_t *e)
{
    lv_obj_t *sw = lv_event_get_target(e);
    
    if (lv_obj_has_state(sw, LV_STATE_CHECKED)) {
        lv_obj_set_style_border_color(color_preview, lv_color_hex(0x00FF00), 0);
    } else {
        lv_obj_set_style_border_color(color_preview, lv_color_hex(0xFFFFFF), 0);
    }
}

/* Screen touch callback */
static void touch_cb(lv_event_t *e)
{
    touch_counter++;
    lv_label_set_text_fmt(counter_label, "Touches: %d", touch_counter);
    
    /* Get touch coordinates */
    lv_point_t point;
    lv_indev_get_point(lv_indev_active(), &point);
    
    /* Move preview to touch position (bounded) */
    lv_obj_t *screen = lv_event_get_target(e);
    lv_coord_t w = lv_obj_get_width(screen);
    lv_coord_t h = lv_obj_get_height(screen);
    
    lv_coord_t x = point.x - 50;
    lv_coord_t y = point.y - 50;
    
    /* Keep preview on screen */
    x = LV_CLAMP(0, x, w - 100);
    y = LV_CLAMP(0, y, h - 100);
    
    lv_obj_set_pos(color_preview, x, y);
}

int vb_init(void)
{
    k_s32                  ret;
    k_vb_config            config;
    k_vb_supplement_config supplement_config;

    memset(&config, 0x00, sizeof(config));
    config.max_pool_cnt = VB_MAX_POOLS;

    // for gdma rotate
    config.comm_pool[0].blk_cnt  = 1;
    config.comm_pool[0].mode     = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[0].blk_size = VB_ALIGN_UP(MAX_DISPLATY_WIDTH * MAX_DISPLATY_HEIGHT * 4, 4096);

    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        return ret;
    }

    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("vb_set_supplement_config failed ret:%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        return ret;
    }

    return 0;
}

int vb_deinit(void)
{
    kd_mpi_vb_exit();

    return 0;
}

// Print usage information
void print_usage(const char* progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Options:\n");
    printf("  -c, --connector <type>  Connector type (default: 0)\n");
    printf("  -l, --layer <id>        OSD layer ID (default: 1)\n");
    printf("  -f, --format <format>   Color format (l8, rgb565, rgb888, argb8888; default: rgb888)\n");
    printf("  -r, --rotate <deg>      Rotation angle (0, 90, 180, 270; default: 0)\n");
    printf("  -H, --help              Show this help message\n");
}

// Parse color format from argument
static lv_color_format_t parse_color_format(const char* format)
{
    if (strcmp(format, "l8") == 0) {
        return LV_COLOR_FORMAT_L8;
    } else if (strcmp(format, "rgb565") == 0) {
        return LV_COLOR_FORMAT_RGB565;
    } else if (strcmp(format, "rgb888") == 0) {
        return LV_COLOR_FORMAT_RGB888;
    } else if (strcmp(format, "argb8888") == 0) {
        return LV_COLOR_FORMAT_ARGB8888;
    } else {
        printf("Unknown color format: %s, using default: rgb888\n", format);
        return LV_COLOR_FORMAT_RGB888;
    }
}

// Get color format as string
static const char* color_format_to_string(lv_color_format_t format)
{
    switch (format) {
    case LV_COLOR_FORMAT_L8:
        return "L8";
    case LV_COLOR_FORMAT_RGB565:
        return "RGB565";
    case LV_COLOR_FORMAT_RGB888:
        return "RGB888";
    case LV_COLOR_FORMAT_ARGB8888:
        return "ARGB8888";
    default:
        return "Unknown";
    }
}

// Convert degrees to LVGL display rotation type
static lv_display_rotation_t parse_rotation(int degrees)
{
    /* gdma rotate 90 with lvgl rotate is reversed */
    switch (degrees) {
    case 90:
        return LV_DISPLAY_ROTATION_270;
    case 180:
        return LV_DISPLAY_ROTATION_180;
    case 270:
        return LV_DISPLAY_ROTATION_90;
    default:
        return LV_DISPLAY_ROTATION_0;
    }
}

int main(int argc, char* argv[])
{
    int opt;
    int rotation_angle = 0;

    lv_display_t*     disp;
    k_connector_type  connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
    k_vo_layer_id     osd_layer      = K_VO_LAYER_OSD0;
    lv_color_format_t color_format   = LV_COLOR_FORMAT_RGB888;

    // Parse command line arguments
    const char*          optstring      = "c:l:f:r:?H";
    static struct option long_options[] = {
        { "connector", required_argument, NULL, 'c' },
        { "layer", required_argument, NULL, 'l' },
        { "format", required_argument, NULL, 'f' },
        { "rotate", required_argument, NULL, 'r' },
        { "help", no_argument, NULL, 'H' },
        { NULL, 0, NULL, 0 },
    };

    while ((opt = getopt_long(argc, argv, optstring, long_options, NULL)) != -1) {
        switch (opt) {
        case 'c':
            connector_type = atoi(optarg);
            break;
        case 'l':
            osd_layer = atoi(optarg);
            break;
        case 'f':
            color_format = parse_color_format(optarg);
            break;
        case 'r':
            rotation_angle = atoi(optarg);
            break;
        case 'H':
        case '?':
            print_usage(argv[0]);
            return 0;
        }
    }

    // Print configuration
    printf("LVGL K230 OSD Test Starting...\n");
    printf("Configuration:\n");
    printf("  Connector: (%d)\n", connector_type);
    printf("  OSD Layer: %d\n", osd_layer);
    printf("  Rotation:  %d deg\n", rotation_angle);
    printf("  Color Format: %s\n", color_format_to_string(color_format));
    printf("\n");

    // Setup signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);

    vb_init();

    if (0x00 != kd_display_init(connector_type)) {
        printf("failed int init connector\n");

        goto _failed_init_connector;
    }

    // Initialize LVGL
    lv_init();

    // Create LVGL display on OSD layer (auto detect size)
    disp = lv_k230_display_create(osd_layer, 255);
    if (!disp) {
        printf("Failed to create LVGL display\n");
        return -1;
    }
    g_display = disp; // Store for signal handler

    printf("LVGL display created successfully\n");

    // Set rotation
    lv_display_set_rotation(disp, parse_rotation(rotation_angle));

    // Set color format
    printf("Setting color format to %s...\n", color_format_to_string(color_format));
    lv_display_set_color_format(disp, color_format);
    printf("Color format set successfully\n");

    lv_k230_touch_init(0);

    // Create demo widgets
    lv_demo_touch_simple();
    printf("Demo widgets created\n");
    printf("Press Ctrl+C to exit...\n");

    // Main loop
    while (!g_signal_received) {
        // Handle LVGL tasks
        uint32_t delay_ms = lv_task_handler();

        if (100 < delay_ms) {
            delay_ms = 100;
        }

        // Use a shorter sleep to be more responsive to signals
        usleep(delay_ms * 1000);
    }

    // Cleanup
    printf("Cleaning up...\n");
    if (disp) {
        lv_display_delete(disp);
        g_display = NULL;
    }

    printf("LVGL application exited gracefully\n");

_failed_init_connector:
    vb_deinit();

    return 0;
}