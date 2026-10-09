/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 * SPDX-License-Identifier: BSD-2-Clause
 */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "lvgl.h"
#include "lv_k230_display.h"
#include "lv_k230_vglite.h"
#include "lvgl_test_thread.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vo_api.h"

#define TEST_LAYER K_VO_LAYER_OSD0
#define RECT_COUNT 8

bool lvgl_run_display_recovery_tests(bool rotate);

typedef struct {
    k_video_frame_info frame;
    int timeout_ms;
    int result;
    int error;
    uint64_t elapsed_us;
} release_wait_t;

static void *release_wait(void *arg)
{
    release_wait_t *wait = arg;
    uint64_t start = time_us();
    wait->result = kd_mpi_vo_wait_frame_release(TEST_LAYER, &wait->frame, wait->timeout_ms);
    wait->error = errno;
    wait->elapsed_us = time_us() - start;
    return NULL;
}

static bool start_release_wait(test_thread_t *thread, const pthread_attr_t *attr,
                               release_wait_t *wait)
{
    if (attr) return test_thread_start(thread, attr, release_wait, wait);

    /* On K230's single RT-Smart CPU, a higher-priority waiter enters the
     * driver before the application thread can start a competing operation. */
    pthread_attr_t waiter_attr;
    struct sched_param priority = {.sched_priority = 10};
    if (pthread_attr_init(&waiter_attr)) return false;
    bool ok = pthread_attr_setschedpolicy(&waiter_attr, SCHED_OTHER) == 0 &&
              pthread_attr_setschedparam(&waiter_attr, &priority) == 0 &&
              pthread_attr_setinheritsched(&waiter_attr, PTHREAD_EXPLICIT_SCHED) == 0;
    if (ok) ok = test_thread_start(thread, &waiter_attr, release_wait, wait);
    pthread_attr_destroy(&waiter_attr);
    return ok;
}

static void interrupt_wait(int signal_number)
{
    (void)signal_number;
}

static bool test_release_timeout_limits(const k_video_frame_info *held,
                                       const k_video_frame_info *free_frame)
{
    /* The K230 test kernel runs at 1000 ticks/s with 32-bit ticks. */
    const int maximum_ms = INT32_MAX - 1;
    const k_video_frame_info *frames[] = {free_frame, held};
    bool ok = true;
    for (unsigned i = 0; i < sizeof(frames) / sizeof(frames[0]); i++) {
        uint64_t start = time_us();
        int ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, frames[i], INT32_MAX);
        int error = errno;
        uint64_t elapsed = time_us() - start;
        bool rejected = ret == -1 && error == EINVAL && elapsed < 100000;
        ok &= rejected;
        printf("VO timeout overflow: %s frame=%s result=%d errno=%d elapsed_us=%llu\n",
               rejected ? "PASS" : "FAIL", i ? "held" : "free", ret, error,
               (unsigned long long)elapsed);
    }
    ok &= kd_mpi_vo_wait_frame_release(TEST_LAYER, free_frame, maximum_ms) == 0;
    int ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, free_frame, -2);
    ok &= ret == -1 && errno == EINVAL;

    struct sigaction action = {0}, previous;
    action.sa_handler = interrupt_wait;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1, &action, &previous)) return false;
    release_wait_t wait = {.frame = *held, .timeout_ms = maximum_ms};
    test_thread_t thread;
    if (start_release_wait(&thread, NULL, &wait)) {
        test_thread_join(&thread, true);
        ok &= wait.result == -1 && wait.error == EINTR;
    } else ok = false;
    sigaction(SIGUSR1, &previous, NULL);
    printf("VO timeout boundary: %s maximum_ms=%d interrupted=%d/%d elapsed_us=%llu\n",
           ok ? "PASS" : "FAIL", maximum_ms, wait.result, wait.error,
           (unsigned long long)wait.elapsed_us);
    return ok;
}

typedef struct {
    int result;
    uint64_t elapsed_us;
} wbc_wait_t;

static void *block_vo_control(void *arg)
{
    wbc_wait_t *wait = arg;
    k_video_frame_info frame;
    uint64_t start = time_us();
    wait->result = kd_mpi_wbc_dump_frame(&frame, 500);
    wait->elapsed_us = time_us() - start;
    if (wait->result == 0) kd_mpi_wbc_dump_release(&frame);
    return NULL;
}

static bool wait_for_vo_contention(const test_thread_t *blocker,
                                   const k_video_frame_info *free_frame)
{
    uint64_t deadline = time_us() + LVGL_TEST_THREAD_TIMEOUT_MS * 1000u;
    while (!test_thread_done(blocker) && time_us() < deadline) {
        int ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, free_frame, 0);
        if (ret == -1 && errno == EBUSY) return !test_thread_done(blocker);
        if (ret != 0) return false;
        usleep(1000);
    }
    return false;
}

static bool test_release_contention(const k_video_frame_info *held, const k_video_frame_info *free_frame)
{
    k_vo_wbc_attr attr = {.blk_cnt = 2};
    k_video_frame_info captured;
    bool captured_valid = false, ok = false;
    if (kd_mpi_vo_set_wbc_attr(&attr) || kd_mpi_vo_enable_wbc()) return false;
    if (kd_mpi_wbc_dump_frame(&captured, 1000)) goto cleanup;
    captured_valid = true;

    /* One captured block and one scanout block leave no free WBC buffer.
     * The next dump holds the VO control mutex until its 500 ms timeout. */
    wbc_wait_t blocker = {0};
    test_thread_t blocker_thread, waiter_threads[2];
    struct sigaction action = {0}, previous;
    action.sa_handler = interrupt_wait;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1, &action, &previous)) goto cleanup;
    if (!test_thread_start(&blocker_thread, NULL, block_vo_control, &blocker)) goto restore_signal;
    if (!wait_for_vo_contention(&blocker_thread, free_frame)) {
        test_thread_join(&blocker_thread, false);
        goto restore_signal;
    }

    uint64_t start = time_us();
    int ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, free_frame, 0);
    int error = errno;
    int poll_result = ret, poll_error = error;
    uint64_t poll_us = time_us() - start;
    ok = ret == -1 && error == EBUSY && poll_us < 100000;
    start = time_us();
    ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, free_frame, 40);
    error = errno;
    int deadline_result = ret, deadline_error = error;
    uint64_t deadline_us = time_us() - start;
    ok &= ret == -1 && error == ETIMEDOUT && deadline_us >= 35000 &&
          !test_thread_done(&blocker_thread);

    release_wait_t waits[2] = {{.frame = *free_frame, .timeout_ms = -1}};
    if (start_release_wait(&waiter_threads[0], NULL, &waits[0])) {
        test_thread_join(&waiter_threads[0], true);
        ok &= waits[0].result == -1 && waits[0].error == EINTR &&
              !test_thread_done(&blocker_thread);
    } else ok = false;
    ok &= wait_for_vo_contention(&blocker_thread, free_frame);
    start = time_us();
    ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, free_frame, 700);
    uint64_t acquire_us = time_us() - start;
    ok &= ret == 0;
    test_thread_join(&blocker_thread, false);
    ok &= blocker.result != 0 && blocker.elapsed_us >= 450000;
    printf("VO contention entry: %s poll_us=%llu deadline_us=%llu signal_us=%llu acquire_us=%llu\n",
           ok ? "PASS" : "FAIL", (unsigned long long)poll_us,
           (unsigned long long)deadline_us, (unsigned long long)waits[0].elapsed_us,
           (unsigned long long)acquire_us);
    if (!ok) printf("  poll=%d/%d deadline=%d/%d signal=%d/%d blocker=%d/%llu\n",
                    poll_result, poll_error, deadline_result, deadline_error,
                    waits[0].result, waits[0].error, blocker.result,
                    (unsigned long long)blocker.elapsed_us);

    /* Register both frame waiters before the WBC thread takes the mutex.
     * Timeout/signal cleanup must not reacquire that busy mutex. */
    waits[0] = (release_wait_t){.frame = *held, .timeout_ms = 100};
    waits[1] = (release_wait_t){.frame = *held, .timeout_ms = 1000};
    int created = 0;
    for (; created < 2; created++) {
        if (!start_release_wait(&waiter_threads[created], NULL, &waits[created])) break;
    }
    bool blocked = test_thread_start(&blocker_thread, NULL, block_vo_control, &blocker);
    bool contended = blocked && wait_for_vo_contention(&blocker_thread, free_frame);
    if (created == 2) test_thread_join(&waiter_threads[1], true);
    if (created > 0) test_thread_join(&waiter_threads[0], false);
    /* Cleanup must complete while the WBC operation still owns the mutex. */
    bool cleaned_while_blocked = contended && !test_thread_done(&blocker_thread) &&
                                 wait_for_vo_contention(&blocker_thread, free_frame);
    if (blocked) test_thread_join(&blocker_thread, false);
    bool cleanup_ok = cleaned_while_blocked && created == 2 && blocker.result != 0 && blocker.elapsed_us >= 450000 &&
                      waits[0].result == -1 && waits[0].error == ETIMEDOUT &&
                      waits[0].elapsed_us >= 90000 &&
                      waits[1].result == -1 && waits[1].error == EINTR;
    ok &= cleanup_ok;
    printf("VO contention cleanup: %s timeout_us=%llu signal_us=%llu\n",
           cleanup_ok ? "PASS" : "FAIL", (unsigned long long)waits[0].elapsed_us,
           (unsigned long long)waits[1].elapsed_us);

restore_signal:
    sigaction(SIGUSR1, &previous, NULL);
cleanup:
    if (captured_valid) kd_mpi_wbc_dump_release(&captured);
    ok &= kd_mpi_vo_disable_wbc() == 0;
    return ok;
}

static bool test_release_preemption(const k_video_frame_info *frame)
{
    enum { ROUNDS = 64, WAITERS = 6 };
    pthread_attr_t attr;
    struct sched_param priority = {.sched_priority = 10};
    struct sigaction action = {0}, previous;
    unsigned released = 0, expired = 0, interrupted = 0, rounds = 0;
    bool ok = false;

    action.sa_handler = interrupt_wait;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1, &action, &previous)) return false;
    if (pthread_attr_init(&attr)) goto restore_signal;
    /* RT-Smart uses lower numbers for higher priorities. The application
     * thread runs at 25; these waiters preempt thread-context notifications. */
    if (pthread_attr_setschedpolicy(&attr, SCHED_OTHER) ||
        pthread_attr_setschedparam(&attr, &priority) ||
        pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED)) goto destroy_attr;

    ok = true;
    for (; rounds < ROUNDS && ok; rounds++) {
        release_wait_t waits[WAITERS];
        test_thread_t threads[WAITERS];
        int created = 0;
        if (kd_mpi_vo_enable_layer(TEST_LAYER) || kd_mpi_vo_insert_frame(TEST_LAYER, frame)) {
            ok = false;
            break;
        }
        for (; created < WAITERS; created++) {
            waits[created] = (release_wait_t){.frame = *frame,
                .timeout_ms = created < 2 ? 1 + (int)(rounds % 4) : 1000};
            if (!start_release_wait(&threads[created], &attr, &waits[created])) break;
        }

        /* Alternate notifications near short deadlines and after expiry.
         * Signals also remove registered waiters before the next traversal. */
        usleep((rounds % 4) * 1000);
        for (int i = 2; i < 4 && i < created; i++) {
            test_thread_join(&threads[i], true);
        }
        ok &= kd_mpi_vo_disable_layer(TEST_LAYER) == 0;
        for (int i = 0; i < created; i++) {
            if (i < 2 || i >= 4) test_thread_join(&threads[i], false);
            if (waits[i].result == 0) released++;
            else if (waits[i].result == -1 && waits[i].error == ETIMEDOUT && i < 2) expired++;
            else if (waits[i].result == -1 && waits[i].error == EINTR && i >= 2 && i < 4) interrupted++;
            else {
                printf("VO preemption waiter: round=%u index=%d result=%d errno=%d\n",
                       rounds, i, waits[i].result, waits[i].error);
                ok = false;
            }
        }
        ok &= created == WAITERS;
        ok &= kd_mpi_vo_wait_frame_release(TEST_LAYER, frame, 0) == 0;
    }
    ok &= rounds == ROUNDS && released > 0 && expired > 0 && interrupted > 0;
    printf("VO preemption: %s rounds=%u released=%u expired=%u interrupted=%u\n",
           ok ? "PASS" : "FAIL", rounds, released, expired, interrupted);

destroy_attr:
    pthread_attr_destroy(&attr);
restore_signal:
    sigaction(SIGUSR1, &previous, NULL);
    return ok;
}

static bool test_release_notifications(void)
{
    const uint32_t size = 64 * 64 * 4;
    k_u32 pool = kd_mpi_vb_create_pool_ex(size, 3, VB_REMAP_MODE_NOCACHE);
    k_vb_blk_handle blocks[3] = {VB_INVALID_HANDLE, VB_INVALID_HANDLE, VB_INVALID_HANDLE};
    k_video_frame_info frames[3] = {0};
    bool ok = pool != VB_INVALID_POOLID;
    if (!ok) return false;

    for (int i = 0; i < 3; i++) {
        blocks[i] = kd_mpi_vb_get_block(pool, size, NULL);
        if (blocks[i] == VB_INVALID_HANDLE) { ok = false; goto cleanup; }
        frames[i].pool_id = pool;
        frames[i].v_frame.phys_addr[0] = kd_mpi_vb_handle_to_phyaddr(blocks[i]);
        frames[i].v_frame.width = frames[i].v_frame.height = 64;
        frames[i].v_frame.stride[0] = 256;
        frames[i].v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
        void *pixels = kd_mpi_sys_mmap(frames[i].v_frame.phys_addr[0], size);
        if (!pixels) { ok = false; goto cleanup; }
        memset(pixels, 0xff, size);
        kd_mpi_sys_munmap(pixels, size);
    }

    k_vo_layer_attr attr = {0};
    attr.img_size.width = attr.img_size.height = 64;
    attr.pixel_format = PIXEL_FORMAT_ARGB_8888;
    attr.global_alpha = 255;
    attr.func = GDMA_ROTATE_DEGREE_0;
    if (kd_mpi_vo_set_layer_attr(TEST_LAYER, &attr) || kd_mpi_vo_enable_layer(TEST_LAYER)) {
        ok = false;
        goto cleanup;
    }
    ok &= kd_mpi_vo_wait_frame_release(TEST_LAYER, &frames[0], 0) == 0;
    ok &= kd_mpi_vo_insert_frame(TEST_LAYER, &frames[0]) == 0;
    usleep(60000);
    int ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, &frames[0], 0);
    ok &= ret == -1 && errno == EBUSY;
    uint64_t start = time_us();
    ret = kd_mpi_vo_wait_frame_release(TEST_LAYER, &frames[0], 10);
    ok &= ret == -1 && errno == ETIMEDOUT && time_us() - start >= 9000;
    ok &= test_release_timeout_limits(&frames[0], &frames[1]);

    /* Both waiters must wake, and insertion must not block on their wait. */
    release_wait_t waits[2] = {{.frame = frames[0], .timeout_ms = 1000},
                              {.frame = frames[0], .timeout_ms = 1000}};
    test_thread_t threads[2];
    int created = 0;
    for (; created < 2; created++) {
        if (!start_release_wait(&threads[created], NULL, &waits[created])) break;
    }
    ok &= kd_mpi_vo_insert_frame(TEST_LAYER, &frames[1]) == 0;
    for (int i = 0; i < created; i++) {
        test_thread_join(&threads[i], false);
        ok &= waits[i].result == 0;
    }
    ok &= created == 2;
    ok &= kd_mpi_vb_inquire_user_cnt(blocks[0]) == 1;

    /* Rapid replacement exercises pending-slot releases as well as scanout. */
    for (int i = 0; i < 100 && ok; i++) {
        ok &= kd_mpi_vo_insert_frame(TEST_LAYER, &frames[2]) == 0;
        ok &= kd_mpi_vo_insert_frame(TEST_LAYER, &frames[1]) == 0;
        ok &= kd_mpi_vo_wait_frame_release(TEST_LAYER, &frames[2], 100) == 0;
    }

    struct sigaction action = {0}, previous;
    action.sa_handler = interrupt_wait;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1, &action, &previous)) { ok = false; goto cleanup; }
    waits[0].frame = frames[1];
    if (start_release_wait(&threads[0], NULL, &waits[0])) {
        test_thread_join(&threads[0], true);
        ok &= waits[0].result == -1 && waits[0].error == EINTR;
    } else ok = false;
    sigaction(SIGUSR1, &previous, NULL);

    ok &= test_release_contention(&frames[1], &frames[0]);

    if (start_release_wait(&threads[0], NULL, &waits[0])) {
        ok &= kd_mpi_vo_disable_layer(TEST_LAYER) == 0;
        test_thread_join(&threads[0], false);
        ok &= waits[0].result == 0;
    } else ok = false;
    ok &= test_release_preemption(&frames[0]);
    for (int i = 0; i < 3; i++) ok &= kd_mpi_vb_inquire_user_cnt(blocks[i]) == 1;
    ret = kd_mpi_vo_wait_frame_release(K_MAX_VO_LAYER_NR, &frames[0], 0);
    ok &= ret == -1 && errno == EINVAL;

cleanup:
    kd_mpi_vo_disable_layer(TEST_LAYER);
    for (int i = 0; i < 3; i++) {
        if (blocks[i] != VB_INVALID_HANDLE) kd_mpi_vb_release_block(blocks[i]);
    }
    ok &= kd_mpi_vb_destory_pool(pool) == 0;
    printf("VO release: %s (free/busy/timeout/multiple waiters/replacement/signal/disable)\n", ok ? "PASS" : "FAIL");
    return ok;
}

typedef struct {
    uint64_t flush_start;
    uint64_t flush_us;
    unsigned flushes;
    unsigned ownership_errors;
    void *rendered;
    void *buffers[3];
    unsigned buffer_count;
} display_stats_t;

static void monitor_display(lv_event_t *event)
{
    lv_display_t *display = lv_event_get_target(event);
    display_stats_t *stats = lv_event_get_user_data(event);
    lv_event_code_t code = lv_event_get_code(event);
    if (code == LV_EVENT_FLUSH_START) stats->flush_start = time_us();
    if (code == LV_EVENT_FLUSH_FINISH) {
        stats->flush_us += time_us() - stats->flush_start;
        stats->flushes++;
        stats->rendered = lv_display_get_buf_active(display)->data;
    }
    if (code != LV_EVENT_RENDER_START) return;

    lv_draw_buf_t *buffer = lv_display_get_buf_active(display);
    unsigned i;
    for (i = 0; i < stats->buffer_count; i++) if (stats->buffers[i] == buffer->data) break;
    if (i == stats->buffer_count && i < 3) stats->buffers[stats->buffer_count++] = buffer->data;
    k_video_frame_info frame = {0};
    k_sys_virmem_info memory;
    if (kd_mpi_sys_get_virmem_info(buffer->data, &memory) != 0) {
        stats->ownership_errors++;
        return;
    }
    frame.v_frame.phys_addr[0] = memory.phy_addr;
    k_vb_blk_handle block = kd_mpi_vb_phyaddr_to_handle(frame.v_frame.phys_addr[0]);
    frame.pool_id = kd_mpi_vb_handle_to_pool_id(block);
    if (kd_mpi_vo_wait_frame_release(TEST_LAYER, &frame, 0) != 0) stats->ownership_errors++;
}

static void update_reference(uint8_t *reference, uint32_t stride, lv_color_format_t format,
                             unsigned rect, uint32_t color)
{
    unsigned bpp = lv_color_format_get_size(format);
    unsigned x = 20 + (rect % 4) * 48;
    unsigned y = 20 + (rect / 4) * 80;
    uint8_t pixel[4] = {color & 255, (color >> 8) & 255, (color >> 16) & 255, 255};
    if (format == LV_COLOR_FORMAT_RGB565) {
        uint16_t rgb = ((color >> 8) & 0xf800) | ((color >> 5) & 0x07e0) | ((color >> 3) & 0x001f);
        memcpy(pixel, &rgb, sizeof(rgb));
    }
    for (unsigned row = y; row < y + 20; row++) {
        for (unsigned col = x; col < x + 20; col++) memcpy(reference + row * stride + col * bpp, pixel, bpp);
    }
}

static bool test_display_format(lv_display_t *display, lv_color_format_t format)
{
    static const uint32_t colors[] = {0xff0000, 0x00ff00, 0x0000ff, 0xffffff};
    display_stats_t stats = {0};
    lv_display_set_color_format(display, format);
    lv_obj_t *screen = lv_display_get_screen_active(display);
    lv_obj_clean(screen);
    lv_obj_remove_style_all(screen);
    lv_obj_set_style_bg_color(screen, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(screen, LV_OPA_COVER, 0);
    lv_obj_t *rects[RECT_COUNT];
    for (unsigned i = 0; i < RECT_COUNT; i++) {
        rects[i] = lv_obj_create(screen);
        lv_obj_remove_style_all(rects[i]);
        lv_obj_set_pos(rects[i], 20 + (i % 4) * 48, 20 + (i / 4) * 80);
        lv_obj_set_size(rects[i], 20, 20);
        lv_obj_set_style_bg_opa(rects[i], LV_OPA_COVER, 0);
        lv_obj_set_style_bg_color(rects[i], lv_color_black(), 0);
    }
    lv_display_add_event_cb(display, monitor_display, LV_EVENT_ALL, &stats);
    lv_refr_now(display);
    /* FLUSH_FINISH runs before LVGL rotates its own double-buffer pointer. */
    lv_draw_buf_t *active = lv_display_get_buf_active(display);
    size_t size = active->header.stride * active->header.h;
    uint8_t *reference = malloc(size);
    if (!reference) { lv_display_remove_event_cb_with_user_data(display, monitor_display, &stats); return false; }
    memcpy(reference, stats.rendered, size);

    bool ok = true;
    unsigned mismatches = 0;
    /* Vary separate rectangles so a buffer can miss multiple disjoint updates. */
    for (unsigned i = 0; i < 40; i++) {
        unsigned rect = i % RECT_COUNT;
        uint32_t color = colors[(i / RECT_COUNT) % 4];
        lv_obj_set_style_bg_color(rects[rect], lv_color_hex(color), 0);
        update_reference(reference, active->header.stride, format, rect, color);
        lv_refr_now(display);
        if (!stats.rendered || memcmp(reference, stats.rendered, size)) mismatches++;
        if (i % 7 == 0) usleep(40000);
    }
    ok &= mismatches == 0 && stats.ownership_errors == 0;
    free(reference);

    stats.flush_us = stats.flushes = 0;
    uint64_t start = time_us();
    for (unsigned i = 0; i < 180; i++) {
        lv_obj_set_style_bg_color(rects[0], lv_color_hex(colors[i % 4]), 0);
        lv_refr_now(display);
    }
    uint64_t elapsed = time_us() - start;
    printf("Display cf=%d: %s, buffers=%u, mismatches=%u, ownership_errors=%u, "
           "submit_fps=%.1f, flush_avg_us=%llu\n", format, ok ? "PASS" : "FAIL",
           stats.buffer_count, mismatches, stats.ownership_errors,
           stats.flushes * 1000000.0 / elapsed,
           (unsigned long long)(stats.flushes ? stats.flush_us / stats.flushes : 0));
    ok &= stats.ownership_errors == 0;
    lv_display_remove_event_cb_with_user_data(display, monitor_display, &stats);
    return ok;
}

bool lvgl_run_display_regressions(bool rotate)
{
    if (kd_display_init(ST7701_V1_MIPI_2LAN_480X800_30FPS, 0, 0,
                        rotate ? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0)) return false;
    bool ok = rotate || test_release_notifications();
    ok &= lvgl_run_display_recovery_tests(rotate);
    lv_display_t *display = lv_k230_display_create(TEST_LAYER, 255);
    if (!display) { kd_display_deinit(); return false; }
    ok &= test_display_format(display, LV_COLOR_FORMAT_RGB565);
    ok &= test_display_format(display, LV_COLOR_FORMAT_RGB888);
    ok &= test_display_format(display, LV_COLOR_FORMAT_XRGB8888);
    lv_display_delete(display);
    kd_display_deinit();
    return ok;
}
