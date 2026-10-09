/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef LVGL_TEST_THREAD_H
#define LVGL_TEST_THREAD_H

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifndef LVGL_TEST_THREAD_TIMEOUT_MS
#define LVGL_TEST_THREAD_TIMEOUT_MS 2000
#endif

typedef struct {
    pthread_t id;
    void *(*run)(void *);
    void *arg;
    bool started;
    bool done;
} test_thread_t;

static inline uint64_t time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static inline void test_thread_failed(const char *operation, int error)
{
    fprintf(stderr, "VO test thread %s failed: %d\n", operation, error);
    fflush(NULL);
    /* A worker may still own driver resources and refer to the caller's stack.
     * Do not unwind or free its buffers when the driver cannot be interrupted. */
    _Exit(EXIT_FAILURE);
}

static inline void *test_thread_entry(void *arg)
{
    test_thread_t *thread = arg;
    __atomic_store_n(&thread->started, true, __ATOMIC_RELEASE);
    thread->run(thread->arg);
    __atomic_store_n(&thread->done, true, __ATOMIC_RELEASE);
    return NULL;
}

static inline bool test_thread_start(test_thread_t *thread, const pthread_attr_t *attr,
                                     void *(*run)(void *), void *arg)
{
    *thread = (test_thread_t){.run = run, .arg = arg};
    int error = pthread_create(&thread->id, attr, test_thread_entry, thread);
    if (error) return false;

    uint64_t deadline = time_us() + LVGL_TEST_THREAD_TIMEOUT_MS * 1000u;
    while (!__atomic_load_n(&thread->started, __ATOMIC_ACQUIRE)) {
        if (time_us() >= deadline) test_thread_failed("start", ETIMEDOUT);
        usleep(1000);
    }
    return true;
}

static inline bool test_thread_done(const test_thread_t *thread)
{
    return __atomic_load_n(&thread->done, __ATOMIC_ACQUIRE);
}

static inline void test_thread_join(test_thread_t *thread, bool interrupt)
{
    uint64_t deadline = time_us() + LVGL_TEST_THREAD_TIMEOUT_MS * 1000u;
    for (;;) {
        int error = pthread_tryjoin_np(thread->id, NULL);
        if (error == 0) return;
        if (error != EBUSY) test_thread_failed("join", error);
        if (time_us() >= deadline) {
            fprintf(stderr, "VO test worker: done=%d interrupt=%d\n",
                    test_thread_done(thread), interrupt);
            test_thread_failed("join", ETIMEDOUT);
        }

        /* started does not prove entry into the syscall. Retry until completion
         * so a signal handled just before entry cannot leave the worker stuck. */
        if (interrupt && !test_thread_done(thread)) {
            error = pthread_kill(thread->id, SIGUSR1);
            if (error && error != ESRCH) test_thread_failed("interrupt", error);
        }
        usleep(1000);
    }
}

#endif
