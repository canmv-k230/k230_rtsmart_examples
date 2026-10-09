/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lvgl.h"
#include "lv_k230_display.h"
#include "lv_k230_input_touch.h"
#include "mpi_vb_api.h"
#include "mpi_vo_api.h"

#define TEST_LAYER K_VO_LAYER_OSD0

static bool inject;
static int pool_limit = -1, pool_calls, live_pools, cleanup_errors;
static int enable_failures, enable_calls, wait_error, wait_calls;

k_s32 __real_kd_mpi_vb_create_pool(k_vb_pool_config *config);
k_s32 __real_kd_mpi_vb_destory_pool(k_u32 pool);
k_s32 __real_kd_mpi_vo_enable_layer(k_vo_layer_id layer);
k_s32 __real_kd_mpi_vo_wait_frame_release(k_vo_layer_id layer,
                                          const k_video_frame_info *frame, k_s32 timeout_ms);

k_s32 __wrap_kd_mpi_vb_create_pool(k_vb_pool_config *config)
{
    if (inject && ++pool_calls > pool_limit && pool_limit >= 0) return VB_INVALID_POOLID;
    k_s32 pool = __real_kd_mpi_vb_create_pool(config);
    if (inject && (k_u32)pool != VB_INVALID_POOLID) live_pools++;
    return pool;
}

k_s32 __wrap_kd_mpi_vb_destory_pool(k_u32 pool)
{
    k_s32 result = __real_kd_mpi_vb_destory_pool(pool);
    if (inject) {
        if (result == 0) live_pools--;
        else cleanup_errors++;
    }
    return result;
}

k_s32 __wrap_kd_mpi_vo_enable_layer(k_vo_layer_id layer)
{
    if (inject) {
        enable_calls++;
        if (enable_failures > 0) {
            enable_failures--;
            return K_FAILED;
        }
    }
    return __real_kd_mpi_vo_enable_layer(layer);
}

k_s32 __wrap_kd_mpi_vo_wait_frame_release(k_vo_layer_id layer,
                                          const k_video_frame_info *frame, k_s32 timeout_ms)
{
    if (inject) {
        wait_calls++;
        if (wait_error) {
            errno = wait_error;
            wait_error = 0;
            return -1;
        }
    }
    return __real_kd_mpi_vo_wait_frame_release(layer, frame, timeout_ms);
}

static bool old_buffer_preserved(lv_display_t *display, const lv_draw_buf_t *before,
                                 const void *pixels)
{
    const lv_draw_buf_t *after = lv_display_get_buf_active(display);
    return lv_display_get_color_format(display) == before->header.cf &&
           after->data == before->data && after->data_size == before->data_size &&
           after->header.cf == before->header.cf && after->header.stride == before->header.stride &&
           memcmp(after->data, pixels, before->data_size) == 0;
}

static bool render_color(lv_display_t *display, uint32_t rgb)
{
    lv_obj_t *screen = lv_display_get_screen_active(display);
    lv_obj_remove_style_all(screen);
    lv_obj_set_style_bg_color(screen, lv_color_hex(rgb), 0);
    lv_obj_set_style_bg_opa(screen, LV_OPA_COVER, 0);
    lv_obj_invalidate(screen);
    lv_refr_now(display);
    const lv_draw_buf_t *buffer = lv_display_get_buf_active(display);
    uint32_t bpp = lv_color_format_get_size(buffer->header.cf);
    const uint8_t expected[] = {rgb & 255, (rgb >> 8) & 255, (rgb >> 16) & 255};
    const uint8_t *last = buffer->data + (buffer->header.h - 1) * buffer->header.stride +
                          (buffer->header.w - 1) * bpp;
    bool ok = memcmp(buffer->data, expected, 3) == 0 && memcmp(last, expected, 3) == 0;
    if (!ok) {
        printf("Recovery pixels: cf=%d first=%02x%02x%02x last=%02x%02x%02x expected=%02x%02x%02x\n",
               buffer->header.cf, buffer->data[0], buffer->data[1], buffer->data[2],
               last[0], last[1], last[2], expected[0], expected[1], expected[2]);
    }
    return ok;
}

bool lvgl_run_display_recovery_tests(bool rotate)
{
    bool ok = true;
    const int errors[] = {EBUSY, EINTR, ENOTTY};
    inject = true;
    live_pools = cleanup_errors = 0;
    for (unsigned i = 0; i < sizeof(errors) / sizeof(errors[0]); i++) {
        wait_error = errors[i];
        wait_calls = 0;
        lv_display_t *probe = lv_k230_display_create(TEST_LAYER, 255);
        bool passed = wait_error == 0 && (errors[i] == ENOTTY ? probe == NULL : probe != NULL);
        if (errors[i] == EINTR) passed &= wait_calls == 2;
        if (probe) lv_display_delete(probe);
        passed &= live_pools == 0 && cleanup_errors == 0;
        printf("Display startup errno=%d: %s\n", errors[i], passed ? "PASS" : "FAIL");
        ok &= passed;
    }

    lv_display_t *display = lv_k230_display_create(TEST_LAYER, 255);
    void *pixels = NULL;
    if (!display) { ok = false; goto cleanup; }
#if LV_USE_PERF_MONITOR
    /* The benchmark overlay changes the pixels at the bottom-right corner. */
    lv_sysmon_hide_performance(display);
#endif
    lv_display_set_color_format(display, LV_COLOR_FORMAT_XRGB8888);
    ok &= render_color(display, 0x00ff00);
    lv_draw_buf_t before = *lv_display_get_buf_active(display);
    pixels = malloc(before.data_size);
    if (!pixels) { ok = false; goto cleanup; }
    memcpy(pixels, before.data, before.data_size);

    pool_calls = 0;
    pool_limit = 0;
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB565);
    bool passed = pool_calls == 1 && live_pools == 1 && old_buffer_preserved(display, &before, pixels);
    printf("Display allocation rollback: %s\n", passed ? "PASS" : "FAIL");
    ok &= passed;

    /* Only the candidate allocation is allowed. Rollback must reuse the old
     * buffers even if every subsequent allocation would fail. */
    pool_calls = enable_calls = 0;
    pool_limit = 1;
    enable_failures = 1;
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB565);
    /* Idle displays may pause normally; a successful rollback must allow
     * the next refresh request to resume them. */
    lv_display_send_event(display, LV_EVENT_REFR_REQUEST, NULL);
    k_vo_layer_attr attr;
    passed = pool_calls == 1 && enable_calls == 2 && live_pools == 1 &&
             old_buffer_preserved(display, &before, pixels) &&
             kd_mpi_vo_get_layer_attr(TEST_LAYER, &attr) == 0 &&
             attr.pixel_format == PIXEL_FORMAT_ARGB_8888 &&
             !lv_timer_get_paused(lv_display_get_refr_timer(display));
    printf("Display OSD rollback without allocation: %s\n", passed ? "PASS" : "FAIL");
    ok &= passed;

    pool_calls = enable_calls = 0;
    enable_failures = 2;
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB565);
    lv_display_send_event(display, LV_EVENT_REFR_REQUEST, NULL);
    passed = pool_calls == 1 && enable_calls == 2 && live_pools == 1 &&
             old_buffer_preserved(display, &before, pixels) &&
             lv_timer_get_paused(lv_display_get_refr_timer(display));
    printf("Display failed OSD restore: %s\n", passed ? "PASS" : "FAIL");
    ok &= passed;

    pool_limit = -1;
    enable_failures = 0;
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB888);
    passed = lv_display_get_color_format(display) == LV_COLOR_FORMAT_RGB888 &&
             !lv_timer_get_paused(lv_display_get_refr_timer(display)) &&
             live_pools == 1 && render_color(display, 0xff00ff);
    printf("Display recovery after OSD failure: %s cf=%d pools=%d paused=%d\n",
           passed ? "PASS" : "FAIL", lv_display_get_color_format(display), live_pools,
           lv_timer_get_paused(lv_display_get_refr_timer(display)));
    ok &= passed;

    passed = true;
    for (unsigned i = 0; i < 128; i++) {
        k_vo_dev_attr dev_attr;
        passed &= kd_mpi_vo_get_dev_attr(&dev_attr) == 0 &&
                  dev_attr.dev_rot_flg == (rotate ? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0);
    }
    lv_indev_t *touch = lv_k230_touch_init(0);
    passed &= touch != NULL;
    if (touch) lv_indev_delete(touch);
    printf("VO device getter / touch initialization: %s\n", passed ? "PASS" : "FAIL");
    ok &= passed;

cleanup:
    free(pixels);
    pool_limit = -1;
    enable_failures = wait_error = 0;
    if (display) lv_display_delete(display);
    ok &= live_pools == 0 && cleanup_errors == 0;
    printf("Display recovery cleanup: %s pools=%d errors=%d\n",
           live_pools == 0 && cleanup_errors == 0 ? "PASS" : "FAIL", live_pools, cleanup_errors);
    inject = false;
    return ok;
}
