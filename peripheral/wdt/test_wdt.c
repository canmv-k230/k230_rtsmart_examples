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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <signal.h>
#include <stdbool.h>
#include <unistd.h>

#include "drv_wdt.h"

volatile sig_atomic_t stop_flag        = false;
bool                  stop_wdt_on_exit = true;

// Signal handler for Ctrl+C
void handle_sigint(int sig)
{
    (void)sig;
    stop_flag = true;
    printf("\nReceived Ctrl+C - ");

    if (stop_wdt_on_exit) {
        printf("Stopping watchdog before exit...\n");
    } else {
        printf("Feeding watchdog before exit...\n");
        wdt_feed();
    }
}

void print_usage(const char* prog_name)
{
    printf("Usage: %s <timeout_sec> [--no-stop]\n", prog_name);
    printf("  <timeout_sec>  : Watchdog timeout in seconds (1-60)\n");
    printf("  --no-stop      : Keep WDT running on exit (feed instead of stop)\n");
}

int main(int argc, char** argv)
{
    uint32_t timeout_sec;

    // Parse command line arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    timeout_sec = (uint32_t)atoi(argv[1]);
    if (timeout_sec < 1 || timeout_sec > 60) {
        fprintf(stderr, "Error: Timeout must be between 1-60 seconds\n");
        return 1;
    }

    if (argc > 2) {
        if (strcmp(argv[2], "--no-stop") == 0) {
            stop_wdt_on_exit = false;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    // Register signal handler
    signal(SIGINT, handle_sigint);

    printf("Watchdog Timer Test\n");
    printf("===================\n");
    printf("Setting timeout: %u seconds\n", timeout_sec);
    printf("Behavior on exit: %s\n", stop_wdt_on_exit ? "STOP WDT" : "KEEP WDT RUNNING");
    printf("Press Ctrl+C to quit\n\n");

    // Initialize and start watchdog
    if (wdt_set_timeout(timeout_sec) != 0) {
        fprintf(stderr, "Error: Failed to set WDT timeout\n");
        return 1;
    }

    uint32_t current_timeout = wdt_get_timeout();
    printf("Current WDT timeout: %u seconds\n", current_timeout);

    if (wdt_start() != 0) {
        fprintf(stderr, "Error: Failed to start WDT\n");
        return 1;
    }

    printf("Watchdog started successfully\n");

    // Main loop - feed the watchdog periodically
    while (!stop_flag) {
        printf("Feeding watchdog...\n");
        if (wdt_feed() != 0) {
            fprintf(stderr, "Error: Failed to feed WDT\n");
            break;
        }

        // Sleep for half the timeout period
        sleep(timeout_sec / 2);
    }

    // Cleanup
    if (stop_wdt_on_exit) {
        printf("Stopping watchdog...\n");
        if (wdt_stop() != 0) {
            fprintf(stderr, "Error: Failed to stop WDT\n");
            return 1;
        }
    }

    printf("Test completed\n");
    return 0;
}
