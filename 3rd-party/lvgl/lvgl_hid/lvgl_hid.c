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

#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lvgl.h"

#include "lv_k230_display.h"
#include "lv_k230_input_hid.h"

#include "k_gsdma_comm.h"
#include "mpi_vb_api.h"

#define MAX_DISPLATY_WIDTH   1920
#define MAX_DISPLATY_HEIGHT  1080
#define CURSOR_SIZE          18

static volatile int  g_signal_received = 0;
static lv_display_t* g_display = NULL;
static lv_group_t*   g_keypad_group = NULL;
static lv_indev_t*   g_pointer_indev = NULL;
static lv_indev_t*   g_keypad_indev = NULL;

static lv_obj_t* g_cursor;
static lv_obj_t* g_status_label;
static lv_obj_t* g_key_label;
static lv_obj_t* g_info_label;
static lv_obj_t* g_button;
static lv_obj_t* g_slider;
static lv_obj_t* g_switch;
static lv_obj_t* g_dropdown;

static void signal_handler(int signum)
{
    printf("\nReceived signal %d, shutting down...\n", signum);
    g_signal_received = 1;
}

static void update_status(const char* text)
{
    if (g_status_label != NULL) {
        lv_label_set_text(g_status_label, text);
    }
}

static void btn_click_cb(lv_event_t* e)
{
    (void)e;
    update_status("Button clicked by mouse or keyboard");
}

static void slider_cb(lv_event_t* e)
{
    char text[64];
    lv_obj_t* slider = lv_event_get_target(e);
    int32_t value = lv_slider_get_value(slider);

    snprintf(text, sizeof(text), "Slider value: %ld", (long)value);
    update_status(text);
}

static void switch_cb(lv_event_t* e)
{
    lv_obj_t* sw = lv_event_get_target(e);
    update_status(lv_obj_has_state(sw, LV_STATE_CHECKED) ? "Switch enabled" : "Switch disabled");
}

static void dropdown_cb(lv_event_t* e)
{
    char text[64];
    char option[32];
    lv_obj_t* dropdown = lv_event_get_target(e);

    lv_dropdown_get_selected_str(dropdown, option, sizeof(option));
    snprintf(text, sizeof(text), "Selected: %s", option);
    update_status(text);
}

static void key_event_cb(lv_event_t* e)
{
    uint32_t key;
    char text[64];

    if (lv_event_get_code(e) != LV_EVENT_KEY) {
        return;
    }

    key = lv_event_get_key(e);
    if (key >= 32 && key < 127) {
        snprintf(text, sizeof(text), "Last key: '%c'", (char)key);
    } else {
        snprintf(text, sizeof(text), "Last key code: %lu", (unsigned long)key);
    }

    if (g_key_label != NULL) {
        lv_label_set_text(g_key_label, text);
    }
}

static void create_cursor_overlay(lv_obj_t* parent)
{
    g_cursor = lv_obj_create(parent);
    lv_obj_remove_style_all(g_cursor);
    lv_obj_set_size(g_cursor, CURSOR_SIZE, CURSOR_SIZE);
    lv_obj_add_flag(g_cursor, LV_OBJ_FLAG_IGNORE_LAYOUT);
    lv_obj_set_style_border_width(g_cursor, 2, 0);
    lv_obj_set_style_border_color(g_cursor, lv_color_hex(0xff5a36), 0);
    lv_obj_set_style_radius(g_cursor, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_bg_opa(g_cursor, LV_OPA_TRANSP, 0);
    lv_obj_move_foreground(g_cursor);
}

static void create_ui(void)
{
    lv_obj_t* screen = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(screen, lv_color_hex(0x101820), 0);
    lv_obj_set_style_bg_grad_color(screen, lv_color_hex(0x1d3557), 0);
    lv_obj_set_style_bg_grad_dir(screen, LV_GRAD_DIR_VER, 0);
    lv_obj_add_event_cb(screen, key_event_cb, LV_EVENT_KEY, NULL);
    lv_obj_remove_flag(screen, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* panel = lv_obj_create(screen);
    lv_obj_set_size(panel, lv_pct(82), lv_pct(78));
    lv_obj_center(panel);
    lv_obj_set_style_radius(panel, 18, 0);
    lv_obj_set_style_bg_color(panel, lv_color_hex(0xf4f1ea), 0);
    lv_obj_set_style_border_width(panel, 0, 0);
    lv_obj_set_style_pad_all(panel, 18, 0);
    lv_obj_set_flex_flow(panel, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(panel, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_pad_row(panel, 14, 0);
    lv_obj_remove_flag(panel, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* title = lv_label_create(panel);
    lv_label_set_text(title, "USB HID LVGL Demo");
    lv_obj_set_style_text_font(title, LV_FONT_DEFAULT, 0);
    lv_obj_set_style_text_color(title, lv_color_hex(0x0f172a), 0);

    g_info_label = lv_label_create(panel);
    lv_label_set_text(g_info_label,
                      "Move the USB mouse to drive the cursor.\n"
                      "Use keyboard arrows, tab, enter, and text keys to interact.");
    lv_obj_set_style_text_color(g_info_label, lv_color_hex(0x334155), 0);
    lv_obj_set_style_text_align(g_info_label, LV_TEXT_ALIGN_CENTER, 0);

    g_status_label = lv_label_create(panel);
    lv_label_set_text(g_status_label, "Waiting for HID input...");
    lv_obj_set_style_text_color(g_status_label, lv_color_hex(0xb42318), 0);

    g_key_label = lv_label_create(panel);
    lv_label_set_text(g_key_label, "Last key: none");
    lv_obj_set_style_text_color(g_key_label, lv_color_hex(0x0369a1), 0);

    g_button = lv_btn_create(panel);
    lv_obj_set_size(g_button, 220, 52);
    lv_obj_set_style_radius(g_button, 26, 0);
    lv_obj_set_style_bg_color(g_button, lv_color_hex(0xe76f51), 0);
    lv_obj_add_event_cb(g_button, btn_click_cb, LV_EVENT_CLICKED, NULL);
    lv_obj_add_event_cb(g_button, key_event_cb, LV_EVENT_KEY, NULL);

    lv_obj_t* btn_label = lv_label_create(g_button);
    lv_label_set_text(btn_label, "Activate Action");
    lv_obj_center(btn_label);

    g_slider = lv_slider_create(panel);
    lv_obj_set_width(g_slider, 260);
    lv_slider_set_range(g_slider, 0, 100);
    lv_slider_set_value(g_slider, 35, LV_ANIM_OFF);
    lv_obj_add_event_cb(g_slider, slider_cb, LV_EVENT_VALUE_CHANGED, NULL);
    lv_obj_add_event_cb(g_slider, key_event_cb, LV_EVENT_KEY, NULL);

    lv_obj_t* row = lv_obj_create(panel);
    lv_obj_set_size(row, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(row, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(row, 0, 0);
    lv_obj_set_style_pad_all(row, 0, 0);
    lv_obj_set_style_pad_column(row, 12, 0);
    lv_obj_set_flex_flow(row, LV_FLEX_FLOW_ROW);
    lv_obj_remove_flag(row, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t* switch_label = lv_label_create(row);
    lv_label_set_text(switch_label, "Option");
    lv_obj_set_style_text_color(switch_label, lv_color_hex(0x334155), 0);

    g_switch = lv_switch_create(row);
    lv_obj_add_event_cb(g_switch, switch_cb, LV_EVENT_VALUE_CHANGED, NULL);
    lv_obj_add_event_cb(g_switch, key_event_cb, LV_EVENT_KEY, NULL);

    g_dropdown = lv_dropdown_create(panel);
    lv_dropdown_set_options(g_dropdown, "Alpha\nBeta\nGamma\nDelta");
    lv_obj_set_width(g_dropdown, 220);
    lv_obj_add_event_cb(g_dropdown, dropdown_cb, LV_EVENT_VALUE_CHANGED, NULL);
    lv_obj_add_event_cb(g_dropdown, key_event_cb, LV_EVENT_KEY, NULL);

    create_cursor_overlay(screen);
    lv_screen_load(screen);
}

static void setup_keypad_group(void)
{
    if (g_keypad_indev == NULL) {
        return;
    }

    g_keypad_group = lv_group_create();
    if (g_keypad_group == NULL) {
        printf("failed to create keypad group\n");
        return;
    }

    lv_group_add_obj(g_keypad_group, g_button);
    lv_group_add_obj(g_keypad_group, g_slider);
    lv_group_add_obj(g_keypad_group, g_switch);
    lv_group_add_obj(g_keypad_group, g_dropdown);
    lv_group_focus_obj(g_button);
    lv_indev_set_group(g_keypad_indev, g_keypad_group);
}

static int vb_init(void)
{
    k_s32                  ret;
    k_vb_config            config;
    k_vb_supplement_config supplement_config;

    memset(&config, 0x00, sizeof(config));
    config.max_pool_cnt = VB_MAX_POOLS;
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

static int vb_deinit(void)
{
    kd_mpi_vb_exit();
    return 0;
}

static void print_usage(const char* progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Options:\n");
    printf("  -c, --connector <type>  Connector type (default: 0)\n");
    printf("  -l, --layer <id>        OSD layer ID (default: 1)\n");
    printf("  -r, --rotate <deg>      Rotation angle (0, 90, 180, 270; default: 0)\n");
    printf("  -H, --help              Show this help message\n");
}

static lv_display_rotation_t parse_rotation(int degrees)
{
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
    lv_display_t* disp;
    k_connector_type connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
    k_vo_layer_id osd_layer = K_VO_LAYER_OSD0;
    const char* optstring = "c:l:r:?H";
    static struct option long_options[] = {
        { "connector", required_argument, NULL, 'c' },
        { "layer", required_argument, NULL, 'l' },
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
        case 'r':
            rotation_angle = atoi(optarg);
            break;
        case 'H':
        case '?':
            print_usage(argv[0]);
            return 0;
        }
    }

    signal(SIGINT, signal_handler);

    if (vb_init() != 0) {
        return -1;
    }

    if (kd_display_init(connector_type) != 0) {
        printf("failed to init connector\n");
        vb_deinit();
        return -1;
    }

    lv_init();

    disp = lv_k230_display_create(osd_layer, 255);
    if (disp == NULL) {
        printf("failed to create display\n");
        kd_display_deinit();
        vb_deinit();
        return -1;
    }

    g_display = disp;
    lv_display_set_rotation(disp, parse_rotation(rotation_angle));
    lv_display_set_color_format(disp, LV_COLOR_FORMAT_RGB888);

    g_pointer_indev = lv_k230_hid_pointer_init_auto();
    g_keypad_indev = lv_k230_hid_keypad_init_auto();

    create_ui();
    setup_keypad_group();

    if (g_pointer_indev != NULL) {
        lv_indev_set_cursor(g_pointer_indev, g_cursor);
    } else {
        update_status("No USB HID pointer found");
    }

    if (g_keypad_indev == NULL && g_key_label != NULL) {
        lv_label_set_text(g_key_label, "No USB HID keyboard found");
    }

    while (!g_signal_received) {
        uint32_t delay_ms = lv_task_handler();

        if (delay_ms > 50) {
            delay_ms = 50;
        }

        usleep(delay_ms * 1000);
    }

    if (g_keypad_group != NULL) {
        lv_group_delete(g_keypad_group);
        g_keypad_group = NULL;
    }

    if (disp != NULL) {
        lv_display_delete(disp);
        g_display = NULL;
    }

    kd_display_deinit();
    vb_deinit();
    return 0;
}