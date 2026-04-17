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

#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "drv_pmu.h"

#define TEST_PMU_DEFAULT_CLEANUP_MS            1000

struct test_pmu_context {
    drv_pmu_inst_t *pmu;
    int cleanup_ms;
    bool cleanup_done;
};

static volatile sig_atomic_t g_test_pmu_stop;

static int test_pmu_open_instance(drv_pmu_inst_t **pmu);

static void test_pmu_signal_handler(int signo)
{
    if (signo == SIGINT || signo == SIGTERM)
        g_test_pmu_stop = 1;
}

static void test_pmu_print_usage(const char *prog)
{
    printf("Usage:\n");
    printf("  %s [listen [cleanup_ms]]\n", prog);
    printf("  %s powercycle <shutdown_after_s> <poweron_after_s>\n", prog);
    printf("  %s cancel\n", prog);
    printf("  poweron_after_s is counted after shutdown\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s listen 1000\n", prog);
    printf("  %s powercycle 10 30\n", prog);
    printf("  %s cancel\n", prog);
}

static int test_pmu_install_signal_handlers(void)
{
    if (signal(SIGINT, test_pmu_signal_handler) == SIG_ERR) {
        perror("signal(SIGINT)");
        return -1;
    }

    if (signal(SIGTERM, test_pmu_signal_handler) == SIG_ERR) {
        perror("signal(SIGTERM)");
        return -1;
    }

    return 0;
}

static int test_pmu_prepare(struct test_pmu_context *ctx)
{
    if (test_pmu_open_instance(&ctx->pmu) < 0)
        return -1;

    if (drv_pmu_register_notify(ctx->pmu, 0) < 0) {
        fprintf(stderr, "register PMU notify failed\n");
        return -1;
    }

    printf("PMU shutdown listener started\n");
    printf("  pid        : %d\n", getpid());
    printf("  signal     : SIGUSR1 (HAL managed)\n");
    printf("  cleanup_ms : %d\n", ctx->cleanup_ms);
    printf("Waiting for PMU long-press/release shutdown events...\n");

    return 0;
}

static void test_pmu_cleanup(struct test_pmu_context *ctx)
{
    if (ctx != NULL)
        drv_pmu_inst_destroy(&ctx->pmu);
}

static int test_pmu_open_instance(drv_pmu_inst_t **pmu)
{
    if (drv_pmu_inst_create(pmu) < 0) {
        fprintf(stderr, "create PMU instance failed\n");
        return -1;
    }

    return 0;
}

static int test_pmu_parse_u32(const char *text, uint32_t *value)
{
    char *end = NULL;
    unsigned long parsed;

    parsed = strtoul(text, &end, 10);
    if ((text == end) || (end == NULL) || (*end != '\0') ||
        (parsed > UINT32_MAX))
        return -1;

    *value = (uint32_t)parsed;
    return 0;
}

static void test_pmu_do_cleanup(struct test_pmu_context *ctx)
{
    if (ctx->cleanup_done)
        return;

    printf("[test_pmu] long press detected, do cleanup for %d ms...\n",
           ctx->cleanup_ms);
    if (ctx->cleanup_ms > 0) {
        struct timespec delay = {
            .tv_sec = ctx->cleanup_ms / 1000,
            .tv_nsec = (ctx->cleanup_ms % 1000) * 1000000L,
        };

        nanosleep(&delay, NULL);
    }

    ctx->cleanup_done = true;
    printf("[test_pmu] cleanup done, wait key release before ACK\n");
}

static int test_pmu_handle_event(struct test_pmu_context *ctx)
{
    drv_pmu_event_t event;
    int ret;

    ret = drv_pmu_wait_event(ctx->pmu, &event, -1);
    if (ret < 0) {
        fprintf(stderr, "wait PMU event failed\n");
        return -1;
    }
    if (ret > 0)
        return 0;

    event &= DRV_PMU_EVENT_LONG_PRESS | DRV_PMU_EVENT_KEY_RELEASE;
    if (event == 0U)
        return 0;

    printf("[test_pmu] got event=0x%x\n", event);

    if (drv_pmu_event_has_long_press(event))
        test_pmu_do_cleanup(ctx);

    if (!drv_pmu_event_has_key_release(event))
        return 0;

    if (!ctx->cleanup_done) {
        printf("[test_pmu] release arrived before cleanup, do cleanup now\n");
        test_pmu_do_cleanup(ctx);
    }

    printf("[test_pmu] key released, send shutdown ACK\n");
    if (drv_pmu_ack_shutdown(ctx->pmu) < 0) {
        fprintf(stderr, "PMU shutdown ACK failed\n");
        return -1;
    }

    printf("[test_pmu] shutdown ACK sent\n");
    ctx->cleanup_done = false;
    return 0;
}

static int test_pmu_run_listener(int cleanup_ms)
{
    struct test_pmu_context ctx = {
        .cleanup_ms = cleanup_ms,
    };
    int ret = 1;

    if (test_pmu_install_signal_handlers() < 0)
        goto out;

    if (test_pmu_prepare(&ctx) < 0)
        goto out;

    while (!g_test_pmu_stop) {
        if (test_pmu_handle_event(&ctx) < 0)
            break;
    }

    printf("Exiting, unregister PMU listener\n");
    ret = 0;

out:
    test_pmu_cleanup(&ctx);
    return ret;
}

static int test_pmu_run_power_cycle(uint32_t shutdown_after_s,
                                    uint32_t poweron_after_s)
{
    drv_pmu_inst_t *pmu = NULL;
    int ret = 1;

    if (test_pmu_open_instance(&pmu) < 0)
        goto out;

    if (drv_pmu_schedule_power_cycle(pmu, shutdown_after_s,
                                     poweron_after_s) < 0) {
        fprintf(stderr, "schedule PMU power cycle failed\n");
        goto out;
    }

    printf("[test_pmu] scheduled: shutdown after %u s, power on after %u s\n",
           shutdown_after_s, poweron_after_s);
    ret = 0;

out:
    drv_pmu_inst_destroy(&pmu);
    return ret;
}

static int test_pmu_run_cancel(void)
{
    drv_pmu_inst_t *pmu = NULL;
    int ret = 1;

    if (test_pmu_open_instance(&pmu) < 0)
        goto out;

    if (drv_pmu_cancel_power_cycle(pmu) < 0) {
        fprintf(stderr, "cancel PMU power cycle failed\n");
        goto out;
    }

    printf("[test_pmu] canceled PMU power cycle\n");
    ret = 0;

out:
    drv_pmu_inst_destroy(&pmu);
    return ret;
}

static int test_pmu_run_default_listener(int argc, char **argv)
{
    uint32_t cleanup_ms = TEST_PMU_DEFAULT_CLEANUP_MS;

    if (argc > 2) {
        test_pmu_print_usage(argv[0]);
        return 1;
    }

    if ((argc == 2) && (test_pmu_parse_u32(argv[1], &cleanup_ms) < 0)) {
        fprintf(stderr, "invalid cleanup_ms: %s\n", argv[1]);
        return 1;
    }

    return test_pmu_run_listener((int)cleanup_ms);
}

static int test_pmu_run_listen_cmd(int argc, char **argv)
{
    uint32_t cleanup_ms = TEST_PMU_DEFAULT_CLEANUP_MS;

    if (argc > 3) {
        test_pmu_print_usage(argv[0]);
        return 1;
    }

    if ((argc == 3) && (test_pmu_parse_u32(argv[2], &cleanup_ms) < 0)) {
        fprintf(stderr, "invalid cleanup_ms: %s\n", argv[2]);
        return 1;
    }

    return test_pmu_run_listener((int)cleanup_ms);
}

static int test_pmu_run_powercycle_cmd(int argc, char **argv)
{
    uint32_t shutdown_after_s;
    uint32_t poweron_after_s;

    if (argc != 4) {
        test_pmu_print_usage(argv[0]);
        return 1;
    }

    if ((test_pmu_parse_u32(argv[2], &shutdown_after_s) < 0) ||
        (test_pmu_parse_u32(argv[3], &poweron_after_s) < 0)) {
        fprintf(stderr, "invalid powercycle arguments\n");
        return 1;
    }

    return test_pmu_run_power_cycle(shutdown_after_s, poweron_after_s);
}

int main(int argc, char **argv)
{
    const char *cmd;
    uint32_t cleanup_ms;

    if (argc <= 1)
        return test_pmu_run_default_listener(argc, argv);

    cmd = argv[1];
    if (strcmp(cmd, "listen") == 0)
        return test_pmu_run_listen_cmd(argc, argv);

    if (strcmp(cmd, "powercycle") == 0)
        return test_pmu_run_powercycle_cmd(argc, argv);

    if ((strcmp(cmd, "cancel") == 0) && (argc == 2))
        return test_pmu_run_cancel();

    if ((argc == 2) && (test_pmu_parse_u32(cmd, &cleanup_ms) == 0))
        return test_pmu_run_listener((int)cleanup_ms);

    test_pmu_print_usage(argv[0]);
    return 1;
}
