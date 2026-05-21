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

volatile sig_atomic_t stop_flag = false;
bool                  feed_wdt  = true;

// Signal handler for Ctrl+C
void handle_sigint(int sig)
{
    (void)sig;
    stop_flag = true;
    printf("\nReceived Ctrl+C - closing watchdog handle and handing off to kernel...\n");
}

void print_usage(const char* prog_name)
{
    printf("Usage: %s <timeout_sec> [--no-feed]\n", prog_name);
    printf("  <timeout_sec>  : Watchdog timeout in seconds (1-60)\n");
    printf("  --no-feed      : Do not feed WDT after start; use this to test timeout reboot\n");
}

int main(int argc, char** argv)
{
    uint32_t timeout_sec;
    uint32_t current_timeout;

    // Parse command line arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    timeout_sec = (uint32_t)atoi(argv[1]);

    if (argc > 2) {
        int argi;

        for (argi = 2; argi < argc; ++argi) {
            if (strcmp(argv[argi], "--no-feed") == 0) {
                feed_wdt = false;
            } else {
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    if (!feed_wdt) {
        printf("No-feed mode enabled; system should reboot when watchdog expires unless interrupted first.\n");
    }

    // Register signal handler
    signal(SIGINT, handle_sigint);

    printf("Watchdog Timer Test\n");
    printf("===================\n");
    printf("Setting timeout: %u seconds\n", timeout_sec);
    printf("Behavior on exit: CLOSE HANDLE AND HAND OFF TO KERNEL\n");
    printf("Feeding mode: %s\n", feed_wdt ? "FEED PERIODICALLY" : "DO NOT FEED; EXPECT TIMEOUT REBOOT");
    printf("Press Ctrl+C to quit\n\n");

    // Initialize and start watchdog
    if (wdt_set_timeout(timeout_sec) != 0) {
        fprintf(stderr, "Error: Failed to set WDT timeout\n");
        return 1;
    }

    current_timeout = wdt_get_timeout();
    printf("Current WDT timeout: %u seconds\n", current_timeout);
    if (current_timeout != timeout_sec) {
        printf("Requested timeout was rounded to the nearest hardware-supported value.\n");
    }

    if (wdt_start() != 0) {
        fprintf(stderr, "Error: Failed to start WDT\n");
        return 1;
    }

    printf("Watchdog started successfully\n");

    // Main loop - feed the watchdog periodically
    while (!stop_flag) {
        if (feed_wdt) {
            printf("Feeding watchdog...\n");
            if (wdt_feed() != 0) {
                fprintf(stderr, "Error: Failed to feed WDT\n");
                break;
            }

            sleep(current_timeout / 2);
        } else {
            printf("Watchdog running without feed; waiting for timeout reboot...\n");
            sleep(current_timeout + 1);
        }
    }

    // Cleanup
    if (wdt_close() != 0) {
        fprintf(stderr, "Error: Failed to close WDT\n");
        return 1;
    }

    printf("Test completed\n");
    return 0;
}
