/**
 * Copyright (c) 2023, Canaan Bright Sight Co., Ltd
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "drv_fft.h"

#define PI       3.14159265358979323846
#define MAX_DIFF 5

static int g_verbose = 1;

static void generate_test_signal(int point, short* real, short* imag)
{
    for (int i = 0; i < point; i++) {
        float phase = 2.0f * (float)PI * i / point;
        real[i]     = (short)(10 * cosf(phase) + 20 * cosf(2 * phase) + 30 * cosf(3 * phase) + 0.2f * cosf(4 * phase)
                          + 1000 * cosf(5 * phase));
        imag[i]     = 0;
    }
}

static long timespec_diff_us(const struct timespec* a, const struct timespec* b)
{
    return (b->tv_sec - a->tv_sec) * 1000000L + (b->tv_nsec - a->tv_nsec) / 1000L;
}

static int verify_roundtrip(int point, const short* orig_real, const short* orig_imag, const short* ifft_real,
                            const short* ifft_imag, long fft_us, long ifft_us)
{
    int max_dr = 0, max_di = 0;
    int idx_dr = 0, idx_di = 0;

    for (int i = 0; i < point; i++) {
        int dr = abs(ifft_real[i] - orig_real[i]);
        int di = abs(ifft_imag[i] - orig_imag[i]);

        if (g_verbose >= 2)
            printf("  [%4d] real: ifft=%6d orig=%6d diff=%d  "
                   "imag: ifft=%6d orig=%6d diff=%d\n",
                   i, ifft_real[i], orig_real[i], dr, ifft_imag[i], orig_imag[i], di);

        if (dr > max_dr) {
            max_dr = dr;
            idx_dr = i;
        }
        if (di > max_di) {
            max_di = di;
            idx_di = i;
        }
    }

    int pass = (max_dr <= MAX_DIFF) && (max_di <= MAX_DIFF);

    printf("  point %4d  fft %ld us  ifft %ld us  "
           "max_diff(real=%d@%d, imag=%d@%d)  %s\n",
           point, fft_us, ifft_us, max_dr, idx_dr, max_di, idx_di, pass ? "PASS" : "FAIL");

    return pass ? 0 : -1;
}

static int fft_roundtrip_test(drv_fft_inst_t* inst, int point)
{
    short* in_real  = malloc(point * sizeof(short));
    short* in_imag  = malloc(point * sizeof(short));
    short* fft_real = malloc(point * sizeof(short));
    short* fft_imag = malloc(point * sizeof(short));
    short* out_real = malloc(point * sizeof(short));
    short* out_imag = malloc(point * sizeof(short));

    if (!in_real || !in_imag || !fft_real || !fft_imag || !out_real || !out_imag) {
        printf("  point %d: alloc failed\n", point);
        free(in_real);
        free(in_imag);
        free(fft_real);
        free(fft_imag);
        free(out_real);
        free(out_imag);
        return -1;
    }

    generate_test_signal(point, in_real, in_imag);

    drv_fft_cfg_t cfg = {
        .point       = point,
        .input_mode  = RIRI,
        .output_mode = RR_II_OUT,
        .shift       = 0x555,
        .timeout_ms  = 0,
    };

    struct timespec t0, t1, t2;
    int             ret;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    ret = drv_fft_fft(inst, &cfg, in_real, in_imag, fft_real, fft_imag);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (ret != 0) {
        printf("  point %d: fft failed (%d)\n", point, ret);
        goto out;
    }

    cfg.shift = 0xaaa;
    ret       = drv_fft_ifft(inst, &cfg, fft_real, fft_imag, out_real, out_imag);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    if (ret != 0) {
        printf("  point %d: ifft failed (%d)\n", point, ret);
        goto out;
    }

    ret = verify_roundtrip(point, in_real, in_imag, out_real, out_imag, timespec_diff_us(&t0, &t1), timespec_diff_us(&t1, &t2));
out:
    free(in_real);
    free(in_imag);
    free(fft_real);
    free(fft_imag);
    free(out_real);
    free(out_imag);
    return ret;
}

static const int test_points[] = { 64, 128, 256, 512, 1024, 2048, 4096 };

int main(int argc, char* argv[])
{
    if (argc >= 2)
        g_verbose = atoi(argv[1]);

    printf("main start verbose=%d\n", g_verbose);

    drv_fft_inst_t* inst;
    printf("before drv_fft_open\n");
    int ret = drv_fft_open(&inst);
    if (ret != 0) {
        printf("drv_fft_open failed ret=%d\n", ret);
        return 1;
    }
    printf("after drv_fft_open inst=%p\n", inst);

    int failures = 0;
    for (int i = 0; i < (int)(sizeof(test_points) / sizeof(test_points[0])); i++) {
        printf("start point=%d\n", test_points[i]);
        if (fft_roundtrip_test(inst, test_points[i]) != 0)
            failures++;
    }

    printf("before drv_fft_close\n");
    drv_fft_close(&inst);
    printf("done failures=%d\n", failures);

    printf("\n%d/%d tests passed\n", (int)(sizeof(test_points) / sizeof(test_points[0])) - failures,
           (int)(sizeof(test_points) / sizeof(test_points[0])));
    return failures ? 1 : 0;
}
