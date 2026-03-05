#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "lv_k230_display.h"
#include "lvgl.h"

#include "mpi_gsdma_api.h"
#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"

#include "hal_utils.h"

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

static volatile int  g_app_run = 1;
static lv_display_t* g_display = NULL;

// UI Components
static lv_obj_t* img_widget;
static lv_obj_t* label_fps;

static k_connector_type g_connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
static k_vicap_dev      g_csi_idx        = VICAP_DEV_ID_2;

static void handle_sig(int sig)
{
    (void)sig;
    g_app_run = 0;
}

// ============================================================================
// MMZ Buffer
// ============================================================================
typedef struct _mmz_buffer {
    k_u64 pa;
    void* va;
    k_u32 size;
} mmz_buffer_t;

k_s32 mmz_buffer_alloc(mmz_buffer_t* buffer, k_u32 size)
{
    if (!buffer || !size) {
        return K_FAILED;
    }

    buffer->size = size;

    if (K_SUCCESS != kd_mpi_sys_mmz_alloc_cached(&buffer->pa, &buffer->va, "mmz_buf", "anonymous", size)) {
        return K_FAILED;
    }

    return K_SUCCESS;
}

k_s32 mmz_buffer_free(mmz_buffer_t* buffer)
{
    if (!buffer) {
        return K_FAILED;
    }

    return kd_mpi_sys_mmz_free(buffer->pa, buffer->va);
}

k_s32 mmz_buffer_flush_cache(mmz_buffer_t* buffer)
{
    if (!buffer) {
        return K_FAILED;
    }

    return kd_mpi_sys_mmz_flush_cache(buffer->pa, buffer->va, buffer->size);
}

k_s32 mmz_buffer_invalidate_cache(mmz_buffer_t* buffer)
{
    if (!buffer) {
        return K_FAILED;
    }

    return kd_mpi_sys_mmz_invalidate_cache(buffer->pa, buffer->va, buffer->size);
}

// ============================================================================
// VICAP Logic
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
    probe_cfg.fps     = 60;

    memset(&sensor_info, 0, sizeof(sensor_info));
    ret = kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info);
    if (ret != 0)
        return -1;

    ret = kd_mpi_vicap_get_sensor_info(sensor_info.sensor_type, &sensor_info);
    if (ret)
        return ret;

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width  = ISP_WIDTH;
    dev_attr.acq_win.height = ISP_HEIGHT;
    dev_attr.mode           = VICAP_WORK_OFFLINE_MODE;
    dev_attr.buffer_num     = 6;
    dev_attr.buffer_size    = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 1024);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    dev_attr.pipe_ctrl.data = 0xffffffff;
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
    // 1. Create image widget
    img_widget = lv_image_create(lv_scr_act());
    lv_obj_center(img_widget);

    // 2. Make screen background transparent to see the sensor behind it
    lv_obj_set_style_bg_opa(lv_scr_act(), LV_OPA_TRANSP, 0);

    // 3. Header Bar (Semi-transparent dark)
    lv_obj_t* header = lv_obj_create(lv_scr_act());
    lv_obj_set_size(header, DISPLAY_WIDTH, 60);
    lv_obj_align(header, LV_ALIGN_TOP_MID, 0, 0);
    lv_obj_set_style_bg_color(header, lv_color_hex(0x222222), 0);
    lv_obj_set_style_bg_opa(header, LV_OPA_60, 0);
    lv_obj_set_style_border_width(header, 0, 0);
    lv_obj_set_style_radius(header, 0, 0);

    lv_obj_t* title = lv_label_create(header);
    lv_label_set_text(title, LV_SYMBOL_VIDEO " LIVE STREAMING");
    lv_obj_set_style_text_color(title, HUD_WHITE, 0);
    lv_obj_center(title);

    // 4. Status Indicator (Bottom Box)
    lv_obj_t* status_box = lv_obj_create(lv_scr_act());
    lv_obj_set_size(status_box, DISPLAY_WIDTH / 2, 60);
    lv_obj_align(status_box, LV_ALIGN_BOTTOM_MID, 0, -20);
    lv_obj_set_style_bg_color(status_box, HUD_BLACK, 0);
    lv_obj_set_style_bg_opa(status_box, LV_OPA_40, 0);
    lv_obj_set_style_border_color(status_box, lv_palette_main(LV_PALETTE_BLUE), 0);
    lv_obj_set_style_border_width(status_box, 1, 0);

    label_fps = lv_label_create(status_box);
    lv_label_set_text(label_fps, "0x0 | RGB888 | FPS: 0.0");
    lv_obj_set_style_text_color(label_fps, HUD_GREEN, 0);
    lv_obj_center(label_fps);
}

// for 640x480, vector took about 500us, c took about 1000us, 2x faster.
static void rgb888_to_bgr888_inplace(uint8_t* data, size_t num_pixels)
{
#if 0
    uint8_t *start = data;
    const uint8_t *end = data + num_pixels * 3;
    while (start < end) {
        uint8_t r = *start;
        uint8_t g = *(start + 1);
        uint8_t b = *(start + 2);

        // Swap R and B
        *start = b;
        *(start + 1) = g; // G stays the same
        *(start + 2) = r;

        start += 3; // Move to the next pixel
    }
#else
    size_t   vl;
    uint8_t* ptr = data;

    while (num_pixels > 0) {
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"(num_pixels));

        asm volatile(
            // Load RGB
            "vlseg3e8.v v0, (%0)\n\t" // v0=R, v1=G, v2=B

            // Swap R and B
            "vmv.v.v v3, v0\n\t" // tmp = R
            "vmv.v.v v0, v2\n\t" // v0 = B
            "vmv.v.v v2, v3\n\t" // v2 = R

            // Store BGR
            "vsseg3e8.v v0, (%0)\n\t"
            :
            : "r"(ptr)
            : "v0", "v1", "v2", "v3", "memory");

        ptr += vl * 3;
        num_pixels -= vl;
    }
#endif
}

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    mmz_buffer_t image_buffer;
    // Use uint64_t for ms to prevent overflow issues
    uint64_t current_ticks_ms = 0;
    uint64_t last_ticks_ms    = utils_cpu_ticks_ms();

    int frame_count = 0;

    signal(SIGINT, handle_sig);

    if (K_SUCCESS != mmz_buffer_alloc(&image_buffer, RGB888_SIZE)) {
        printf("mmz_buffer_alloc failed\n");
        return -1;
    }

    sample_vb_init();
    kd_display_init(g_connector_type);
    sample_vicap_init(g_csi_idx);
    kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);

    lv_init();
    g_display = lv_k230_display_create(K_VO_LAYER_OSD0, 255);
    lv_display_set_rotation(g_display, LV_DISPLAY_ROTATION_270);
    lv_display_set_color_format(g_display, LV_COLOR_FORMAT_ARGB8888);

    create_hud_ui();

    lv_image_dsc_t sensor_dsc;
    sensor_dsc.header.cf     = LV_COLOR_FORMAT_RGB888;
    sensor_dsc.header.w      = DISPLAY_WIDTH;
    sensor_dsc.header.h      = DISPLAY_HEIGHT;
    sensor_dsc.header.stride = DISPLAY_WIDTH * 3;
    sensor_dsc.data_size     = RGB888_SIZE;
    sensor_dsc.data          = (uint8_t*)image_buffer.va;
    lv_image_set_src(img_widget, &sensor_dsc);

    while (g_app_run) {
        k_video_frame_info vf_info;

        // 1. Capture Frame
        if (kd_mpi_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, VICAP_DUMP_YUV, &vf_info, 100) == K_SUCCESS) {

            if (PIXEL_FORMAT_RGB_888 == vf_info.v_frame.pixel_format) {
                // 2. Process In-place using RVV
                void* vaddr = kd_mpi_sys_mmap(vf_info.v_frame.phys_addr[0], RGB888_SIZE);
                if (vaddr != NULL) {
                    rgb888_to_bgr888_inplace(vaddr, vf_info.v_frame.width * vf_info.v_frame.height);

                    kd_mpi_sys_mmz_flush_cache(vf_info.v_frame.phys_addr[0], vaddr, RGB888_SIZE);
                }
                kd_mpi_sys_munmap(vaddr, RGB888_SIZE);
            }

            // 3. Move processed data to stable MMZ buffer for LVGL
            // This ensures LVGL doesn't crash if it tries to redraw while vicap is busy
            kd_mpi_dma_memcpy((void*)image_buffer.pa, (void*)vf_info.v_frame.phys_addr[0], RGB888_SIZE);

            // Tell LVGL the image data has changed
            lv_obj_invalidate(img_widget);

            // 4. Improved FPS Calculation
            current_ticks_ms = utils_cpu_ticks_ms();
            uint64_t delta   = current_ticks_ms - last_ticks_ms;

            frame_count++;

            if (delta >= 1000) { // Update every 1000ms (1 second)
                float elapsed_sec = delta / 1000.0f;
                float avg_fps     = (float)frame_count / elapsed_sec;

                char buf[64];
                // Displaying as BGR888 because we performed the swap
                snprintf(buf, sizeof(buf), "%dx%d | BGR888 | FPS: %.1f", DISPLAY_WIDTH, DISPLAY_HEIGHT, avg_fps);

                lv_label_set_text(label_fps, buf);

                // Reset for next window
                frame_count   = 0;
                last_ticks_ms = current_ticks_ms;
            }

            kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &vf_info);
        }

        // 5. Handle LVGL tasks
        lv_timer_handler();

        // Reduced sleep to maximize frame capture rate
        usleep(1000);
    }

    // ... Cleanup ...
    if (g_display) {
        lv_display_delete(g_display);
        g_display = NULL;
    }
    mmz_buffer_free(&image_buffer);

    sample_vicap_deinit();

    kd_display_deinit();
    kd_mpi_vb_exit();

    return 0;
}
