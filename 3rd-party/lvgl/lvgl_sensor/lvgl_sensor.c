#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lv_k230_display.h"
#include "lv_k230_image_convert.h"
#include "lv_k230_vglite.h"
#include "lvgl.h"

#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"

#include "hal_utils.h"
#include "canmv_misc.h"

#define ISP_WIDTH      1920
#define ISP_HEIGHT     1080
#define DISPLAY_WIDTH  640
#define DISPLAY_HEIGHT 480
#define RGB888_SIZE    (DISPLAY_WIDTH * DISPLAY_HEIGHT * 3)
#define FRAME_MAP_CACHE_SIZE 8

#define HUD_GREEN lv_color_hex(0x00FF41)
#define HUD_BLACK lv_color_hex(0x000000)
#define HUD_WHITE lv_color_hex(0xFFFFFF)

static volatile sig_atomic_t g_app_run = 1;
static lv_display_t* g_display = NULL;
static lv_obj_t* img_widget = NULL;
static lv_obj_t* label_fps = NULL;
static lv_image_dsc_t sensor_dsc;

static k_connector_type g_connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
static k_vicap_dev g_csi_idx = VICAP_DEV_ID_2;

typedef struct _frame_map {
    k_u64 pa;
    void* va;
    k_u32 size;
} frame_map_t;

static frame_map_t g_frame_maps[FRAME_MAP_CACHE_SIZE];

static void handle_sig(int sig)
{
    printf("\nReceived signal %d, shutting down...\n", sig);
    g_app_run = 0;
}

static void* frame_map_get(k_u64 pa, k_u32 size)
{
    int free_idx = -1;

    for (int i = 0; i < FRAME_MAP_CACHE_SIZE; i++) {
        if (g_frame_maps[i].va && g_frame_maps[i].pa == pa && g_frame_maps[i].size >= size) {
            return g_frame_maps[i].va;
        }
        if (!g_frame_maps[i].va && free_idx < 0) {
            free_idx = i;
        }
    }

    if (free_idx < 0) {
        return NULL;
    }

    void* va = kd_mpi_sys_mmap(pa, size);
    if (!va) {
        return NULL;
    }

    lv_k230_vglite_register_mpp_buffer(va, pa, size, false, PIXEL_FORMAT_RGB_888);
    g_frame_maps[free_idx].pa = pa;
    g_frame_maps[free_idx].va = va;
    g_frame_maps[free_idx].size = size;
    return va;
}

static void frame_map_cleanup(void)
{
    for (int i = 0; i < FRAME_MAP_CACHE_SIZE; i++) {
        if (g_frame_maps[i].va) {
            lv_k230_vglite_unregister_buffer(g_frame_maps[i].va);
            kd_mpi_sys_munmap(g_frame_maps[i].va, g_frame_maps[i].size);
            g_frame_maps[i].pa = 0;
            g_frame_maps[i].va = NULL;
            g_frame_maps[i].size = 0;
        }
    }
}

#if !LV_USE_DRAW_VG_LITE
static void rgb888_to_lv_rgb888_inplace(uint8_t * data, uint32_t stride,
                                        uint32_t width, uint32_t height)
{
    for(uint32_t y = 0; y < height; y++) {
        uint8_t * pixel = data + (size_t)y * stride;
        size_t remaining = width;

        while(remaining > 0) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"(remaining));
            asm volatile(
                "vlseg3e8.v v0, (%0)\n\t"
                "vmv.v.v v3, v0\n\t"
                "vmv.v.v v0, v2\n\t"
                "vmv.v.v v2, v3\n\t"
                "vsseg3e8.v v0, (%0)\n\t"
                :
                : "r"(pixel)
                : "v0", "v1", "v2", "v3", "memory");

            pixel += vl * 3u;
            remaining -= vl;
        }
    }
}
#endif

static k_s32 sample_vicap_init(k_vicap_dev csi_idx)
{
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_vicap_sensor_info sensor_info;
    k_vicap_probe_config probe_cfg;
    k_s32 ret;

    printf("Initializing VICAP for CSI %d...\n", csi_idx);

    memset(&probe_cfg, 0, sizeof(probe_cfg));
    probe_cfg.csi_num = csi_idx;
    probe_cfg.width = ISP_WIDTH;
    probe_cfg.height = ISP_HEIGHT;
    probe_cfg.fps = 60;

    memset(&sensor_info, 0, sizeof(sensor_info));
    ret = kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info);
    if (ret != 0) {
        return ret;
    }

    ret = kd_mpi_vicap_get_sensor_info(sensor_info.sensor_type, &sensor_info);
    if (ret) {
        return ret;
    }

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width = sensor_info.width;
    dev_attr.acq_win.height = sensor_info.height;
    dev_attr.mode = VICAP_WORK_ONLINE_MODE;
    dev_attr.buffer_num = 6;
    dev_attr.buffer_size = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 1024);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    dev_attr.pipe_ctrl.data = 0xffffffff;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.pipe_ctrl.bits.dnr3_enable = 0;

    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret) {
        return ret;
    }

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width = DISPLAY_WIDTH;
    chn_attr.out_win.height = DISPLAY_HEIGHT;
    chn_attr.crop_win = dev_attr.acq_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_FALSE;
    chn_attr.scale_enable = K_TRUE;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_RGB_888;
    chn_attr.buffer_num = 6;
    chn_attr.buffer_size = VB_ALIGN_UP(RGB888_SIZE, 4096);
    chn_attr.alignment = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        return ret;
    }

    return kd_mpi_vicap_init(VICAP_DEV_ID_0);
}

static void sample_vicap_deinit(void)
{
    usleep(100000);
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
}

static k_s32 sample_vb_init(void)
{
    k_vb_config config;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;
    config.comm_pool[0].blk_cnt = 10;
    config.comm_pool[0].blk_size = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 4096);
    config.comm_pool[1].blk_cnt = 10;
    config.comm_pool[1].blk_size = VB_ALIGN_UP(RGB888_SIZE, 4096);

    return kd_mpi_vb_set_config(&config) || kd_mpi_vb_init();
}

static void create_hud_ui(void)
{
    img_widget = lv_image_create(lv_scr_act());
    lv_obj_center(img_widget);

    lv_obj_set_style_bg_color(lv_scr_act(), HUD_BLACK, 0);
    lv_obj_set_style_bg_opa(lv_scr_act(), LV_OPA_COVER, 0);

    lv_obj_t* header = lv_obj_create(lv_scr_act());
    lv_obj_set_size(header, lv_pct(100), 60);
    lv_obj_align(header, LV_ALIGN_TOP_MID, 0, 0);
    lv_obj_set_style_bg_color(header, lv_color_hex(0x222222), 0);
    lv_obj_set_style_bg_opa(header, LV_OPA_60, 0);
    lv_obj_set_style_border_width(header, 0, 0);
    lv_obj_set_style_radius(header, 0, 0);

    lv_obj_t* title = lv_label_create(header);
    lv_label_set_text(title, LV_SYMBOL_VIDEO " LVGL + VG-Lite");
    lv_obj_set_style_text_color(title, HUD_WHITE, 0);
    lv_obj_center(title);

    lv_obj_t* status_box = lv_obj_create(lv_scr_act());
    lv_obj_set_size(status_box, lv_pct(88), 60);
    lv_obj_align(status_box, LV_ALIGN_BOTTOM_MID, 0, -20);
    lv_obj_set_style_bg_color(status_box, HUD_BLACK, 0);
    lv_obj_set_style_bg_opa(status_box, LV_OPA_50, 0);
    lv_obj_set_style_border_color(status_box, lv_palette_main(LV_PALETTE_BLUE), 0);
    lv_obj_set_style_border_width(status_box, 1, 0);

    label_fps = lv_label_create(status_box);
    lv_label_set_text(label_fps, "640x480 | RGB888 auto | FPS: 0.0 | CPU: 0");
    lv_label_set_long_mode(label_fps, LV_LABEL_LONG_CLIP);
    lv_obj_set_width(label_fps, lv_pct(96));
    lv_obj_set_style_text_color(label_fps, HUD_GREEN, 0);
    lv_obj_center(label_fps);
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    k_video_frame_info held_frame;
    bool frame_held = false;
    uint64_t last_ticks_ms = utils_cpu_ticks_ms();
    int frame_count = 0;
    int stream_started = 0;

    signal(SIGINT, handle_sig);
    signal(SIGTERM, handle_sig);
    signal(SIGPIPE, SIG_IGN);
    memset(&held_frame, 0, sizeof(held_frame));

    if (K_SUCCESS != sample_vb_init()) {
        printf("sample_vb_init failed\n");
        return -1;
    }

    if (K_SUCCESS != kd_display_init(g_connector_type)) {
        printf("kd_display_init failed\n");
        goto cleanup_vb;
    }

    if (K_SUCCESS != sample_vicap_init(g_csi_idx)) {
        printf("sample_vicap_init failed\n");
        goto cleanup_display;
    }

    if (K_SUCCESS != kd_mpi_vicap_start_stream(VICAP_DEV_ID_0)) {
        printf("kd_mpi_vicap_start_stream failed\n");
        goto cleanup_vicap;
    }
    stream_started = 1;

    lv_init();
    g_display = lv_k230_display_create(K_VO_LAYER_OSD0, 255);
    lv_display_set_rotation(g_display, LV_DISPLAY_ROTATION_270);
    lv_display_set_color_format(g_display, LV_COLOR_FORMAT_ARGB8888);

    create_hud_ui();

    memset(&sensor_dsc, 0, sizeof(sensor_dsc));
    sensor_dsc.header.magic = LV_IMAGE_HEADER_MAGIC;
    sensor_dsc.header.cf = LV_COLOR_FORMAT_RGB888;
    sensor_dsc.header.flags = LV_IMAGE_FLAGS_MODIFIABLE;
    sensor_dsc.header.w = DISPLAY_WIDTH;
    sensor_dsc.header.h = DISPLAY_HEIGHT;
    sensor_dsc.header.stride = DISPLAY_WIDTH * 3;

    while (g_app_run) {
        k_video_frame_info vf_info;

        if (kd_mpi_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, VICAP_DUMP_YUV, &vf_info, 20) == K_SUCCESS) {
            bool frame_ready = false;

            if (vf_info.v_frame.pixel_format == PIXEL_FORMAT_RGB_888 &&
                vf_info.v_frame.width == DISPLAY_WIDTH && vf_info.v_frame.height == DISPLAY_HEIGHT) {
                uint32_t source_stride = vf_info.v_frame.stride[0];
                if (source_stride < DISPLAY_WIDTH * 3) {
                    source_stride = DISPLAY_WIDTH * 3;
                }

                uint64_t frame_size = (uint64_t)source_stride * DISPLAY_HEIGHT;
                if (frame_size <= UINT32_MAX) {
                    void * vaddr = frame_map_get(vf_info.v_frame.phys_addr[0], (uint32_t)frame_size);
                    if (vaddr) {
#if !LV_USE_DRAW_VG_LITE
                        rgb888_to_lv_rgb888_inplace(vaddr, source_stride, DISPLAY_WIDTH, DISPLAY_HEIGHT);
#endif
                        sensor_dsc.header.stride = source_stride;
                        sensor_dsc.data_size = (uint32_t)frame_size;
                        sensor_dsc.data = vaddr;
                        lv_image_set_src(img_widget, &sensor_dsc);
                        frame_count++;

                        uint64_t current_ticks_ms = utils_cpu_ticks_ms();
                        uint64_t delta = current_ticks_ms - last_ticks_ms;
                        if (delta >= 1000) {
                            int usage = 0;
                            float avg_fps = (float)frame_count / (delta / 1000.0f);
                            char buf[80];

                            canmv_misc_get_cpu_usage(&usage);
                            snprintf(buf, sizeof(buf), "%dx%d | RGB888 auto %s | FPS: %.1f | CPU: %d",
                                     DISPLAY_WIDTH, DISPLAY_HEIGHT, lv_k230_image_convert_backend(), avg_fps, usage);
                            lv_label_set_text(label_fps, buf);

                            frame_count = 0;
                            last_ticks_ms = current_ticks_ms;
                        }

                        frame_ready = true;
                    }
                }
            }

            if (frame_ready) {
                if (frame_held) {
                    kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &held_frame);
                }
                held_frame = vf_info;
                frame_held = true;
            }
            else {
                kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &vf_info);
            }
        }

        lv_timer_handler();
        usleep(1000);
    }

    if (frame_held) {
        lv_k230_vglite_wait_idle();
        kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &held_frame);
        frame_held = false;
    }

    if (g_display) {
        lv_display_delete(g_display);
        g_display = NULL;
    }

cleanup_vicap:
    if (stream_started) {
        kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
    }
    sample_vicap_deinit();
cleanup_display:
    kd_display_deinit();
cleanup_vb:
    frame_map_cleanup();
    kd_mpi_vb_exit();

    return 0;
}
