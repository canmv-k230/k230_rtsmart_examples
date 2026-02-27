#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "lv_k230_display.h"
#include "lvgl.h"

#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"

// ============================================================================
// Global Defines and Variables
// ============================================================================
#define ISP_WIDTH      1920
#define ISP_HEIGHT     1080
#define DISPLAY_WIDTH  640
#define DISPLAY_HEIGHT 480
#define RGB888_SIZE    (DISPLAY_WIDTH * DISPLAY_HEIGHT * 3)

// HUD Theme Colors
#define HUD_GREEN      lv_color_hex(0x00FF41) // Matrix Green
#define HUD_DARK_GREEN lv_color_hex(0x003B00) // Deep Background
#define HUD_BLACK      lv_color_hex(0x000000)
#define HUD_WHITE      lv_color_hex(0xFFFFFF)

static volatile int  g_app_run      = 1;
static lv_display_t* g_display      = NULL;
static uint8_t*      g_frame_buffer = NULL;
static int           g_frame_count  = 0;
static float         avg_fps        = 0;

// UI Components
static lv_obj_t*      img_widget;
static lv_obj_t*      label_fps;
static lv_obj_t*      label_uptime;
static lv_obj_t*      fps_bar;
static lv_image_dsc_t sensor_dsc;

static k_connector_type g_connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
static k_vicap_dev      g_csi_idx        = VICAP_DEV_ID_2;

static void handle_sig(int sig)
{
    (void)sig;
    g_app_run = 0;
}

// ============================================================================
// VICAP Logic (Provided by User - Untouched)
// ============================================================================
static k_s32 sample_vicap_init(k_vicap_dev csi_idx)
{
    k_vicap_dev_attr     dev_attr;
    k_vicap_chn_attr     chn_attr;
    k_vicap_sensor_info  sensor_info;
    k_vicap_probe_config probe_cfg;
    k_s32                ret;

    printf("Initializing VICAP for CSI %d...\n", csi_idx);

    memset(&probe_cfg, 0, sizeof(probe_cfg));
    probe_cfg.csi_num = csi_idx;
    probe_cfg.width   = ISP_WIDTH;
    probe_cfg.height  = ISP_HEIGHT;
    probe_cfg.fps     = 30;

    memset(&sensor_info, 0, sizeof(sensor_info));
    ret = kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info);
    if (ret != 0)
        return -1;

    ret = kd_mpi_vicap_get_sensor_info(sensor_info.sensor_type, &sensor_info);
    if (ret)
        return ret;

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width              = ISP_WIDTH;
    dev_attr.acq_win.height             = ISP_HEIGHT;
    dev_attr.mode                       = VICAP_WORK_OFFLINE_MODE;
    dev_attr.buffer_num                 = 6;
    dev_attr.buffer_size                = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 1024);
    dev_attr.buffer_pool_id             = VB_INVALID_POOLID;
    dev_attr.pipe_ctrl.data             = 0xffffffff;
    // dev_attr.pipe_ctrl.bits.af_enable   = 0;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.pipe_ctrl.bits.dnr3_enable = 0;

    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret)
        return ret;

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = DISPLAY_WIDTH;
    chn_attr.out_win.height = DISPLAY_HEIGHT;
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_FALSE;
    chn_attr.scale_enable   = K_TRUE;
    chn_attr.chn_enable     = K_TRUE;
    chn_attr.pix_format     = PIXEL_FORMAT_RGB_888;
    chn_attr.buffer_num     = 6;
    chn_attr.buffer_size    = VB_ALIGN_UP(RGB888_SIZE, 4096);
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret)
        return ret;

    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    return ret;
}

static void sample_vicap_deinit(void)
{
    kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
    usleep(100000);
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
}

// ============================================================================
// VB & Display Helpers
// ============================================================================
static k_s32 sample_vb_init(void)
{
    k_vb_config config;
    memset(&config, 0, sizeof(config));
    config.max_pool_cnt          = 64;
    config.comm_pool[0].blk_cnt  = 10;
    config.comm_pool[0].blk_size = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 4096);
    config.comm_pool[1].blk_cnt  = 10;
    config.comm_pool[1].blk_size = VB_ALIGN_UP(RGB888_SIZE, 4096);
    return kd_mpi_vb_set_config(&config) || kd_mpi_vb_init();
}

// ============================================================================
// Modern HUD UI Construction (14pt Font Only)
// ============================================================================
static void create_hud_ui(void)
{
    sensor_dsc.header.cf     = LV_COLOR_FORMAT_RGB888;
    sensor_dsc.header.w      = DISPLAY_WIDTH;
    sensor_dsc.header.h      = DISPLAY_HEIGHT;
    sensor_dsc.header.stride = DISPLAY_WIDTH * 3;
    sensor_dsc.data_size     = RGB888_SIZE;
    sensor_dsc.data          = g_frame_buffer;

    img_widget = lv_image_create(lv_scr_act());
    lv_image_set_src(img_widget, &sensor_dsc);
    lv_obj_center(img_widget);

    // Sidebar Container
    lv_obj_t* panel = lv_obj_create(lv_scr_act());
    lv_obj_set_size(panel, 180, 180);
    lv_obj_align(panel, LV_ALIGN_TOP_LEFT, 20, 20);
    lv_obj_set_style_bg_color(panel, HUD_BLACK, 0);
    lv_obj_set_style_bg_opa(panel, LV_OPA_60, 0);
    lv_obj_set_style_border_width(panel, 1, 0);
    lv_obj_set_style_border_color(panel, HUD_GREEN, 0);
    lv_obj_set_style_radius(panel, 0, 0);
    lv_obj_set_style_pad_all(panel, 10, 0);

    lv_obj_t* title = lv_label_create(panel);
    lv_label_set_text(title, "[ SENSOR HUD ]");
    lv_obj_set_style_text_color(title, HUD_GREEN, 0);
    lv_obj_set_style_text_font(title, &lv_font_montserrat_14, 0);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 0);

    label_fps = lv_label_create(panel);
    lv_label_set_text(label_fps, "FPS: 0.0");
    lv_obj_set_style_text_color(label_fps, HUD_WHITE, 0);
    lv_obj_set_style_text_font(label_fps, &lv_font_montserrat_14, 0);
    lv_obj_align(label_fps, LV_ALIGN_TOP_LEFT, 0, 40);

    fps_bar = lv_bar_create(panel);
    lv_obj_set_size(fps_bar, 140, 8);
    lv_obj_align(fps_bar, LV_ALIGN_TOP_LEFT, 0, 65);
    lv_bar_set_range(fps_bar, 0, 60);
    lv_obj_set_style_bg_color(fps_bar, HUD_DARK_GREEN, LV_PART_MAIN);
    lv_obj_set_style_bg_color(fps_bar, HUD_GREEN, LV_PART_INDICATOR);

    label_uptime = lv_label_create(panel);
    lv_label_set_text(label_uptime, "UP: 00:00:00");
    lv_obj_set_style_text_color(label_uptime, HUD_WHITE, 0);
    lv_obj_set_style_text_font(label_uptime, &lv_font_montserrat_14, 0);
    lv_obj_align(label_uptime, LV_ALIGN_BOTTOM_LEFT, 0, 0);
}

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char* argv[])
{
    signal(SIGINT, handle_sig);

    g_frame_buffer = malloc(RGB888_SIZE);
    memset(g_frame_buffer, 0, RGB888_SIZE);

    sample_vb_init();
    kd_display_init(g_connector_type);
    sample_vicap_init(g_csi_idx);
    kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);

    lv_init();
    g_display = lv_k230_display_create(K_VO_LAYER_OSD0, 255);
    lv_display_set_rotation(g_display, LV_DISPLAY_ROTATION_270);
    lv_display_set_color_format(g_display, LV_COLOR_FORMAT_RGB888);

    create_hud_ui();

    struct timespec start_t, last_t, curr_t;
    clock_gettime(CLOCK_MONOTONIC, &start_t);
    last_t = start_t;

    while (g_app_run) {
        k_video_frame_info vf_info;
        if (kd_mpi_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, VICAP_DUMP_YUV, &vf_info, 100) == K_SUCCESS) {
            void* vaddr = kd_mpi_sys_mmap(vf_info.v_frame.phys_addr[0], RGB888_SIZE);
            if (vaddr != (void*)-1) {
                memcpy(g_frame_buffer, vaddr, RGB888_SIZE);
                g_frame_count++;

                clock_gettime(CLOCK_MONOTONIC, &curr_t);
                double elapsed = (curr_t.tv_sec - last_t.tv_sec) + (curr_t.tv_nsec - last_t.tv_nsec) / 1000000000.0;
                if (elapsed >= 1.0) {
                    avg_fps = (g_frame_count % 100) / elapsed;
                    char buf[32];
                    snprintf(buf, sizeof(buf), "FPS: %.1f", avg_fps);
                    lv_label_set_text(label_fps, buf);
                    lv_bar_set_value(fps_bar, (int)avg_fps, LV_ANIM_ON);

                    long up = curr_t.tv_sec - start_t.tv_sec;
                    snprintf(buf, sizeof(buf), "UP: %02ld:%02ld:%02ld", up / 3600, (up % 3600) / 60, up % 60);
                    lv_label_set_text(label_uptime, buf);
                    last_t = curr_t;
                }

                lv_obj_invalidate(img_widget);
                kd_mpi_sys_munmap(vaddr, RGB888_SIZE);
            }
            kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &vf_info);
        }

        lv_timer_handler();
        usleep(5000);
    }

    sample_vicap_deinit();
    kd_display_deinit();
    free(g_frame_buffer);
    kd_mpi_vb_exit();

    return 0;
}
