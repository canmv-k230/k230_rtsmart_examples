#include "drv_adc.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile sig_atomic_t stop_flag;
static int total_checks;
static int total_failures;

#define CHECK(condition, format, ...)                             \
    do {                                                          \
        int passed = !!(condition);                               \
        total_checks++;                                           \
        if (!passed)                                              \
            total_failures++;                                     \
        printf("%s: " format "\n", passed ? "PASS" : "FAIL",    \
               ##__VA_ARGS__);                                    \
    } while (0)

static void handle_sigint(int sig)
{
    (void)sig;
    stop_flag = 1;
}

static int parse_u32(const char* text, uint32_t min, uint32_t max,
                     uint32_t* value)
{
    char* end;
    unsigned long parsed;

    if (text == NULL || *text == '\0')
        return -1;
    parsed = strtoul(text, &end, 10);
    if (*end != '\0' || parsed < min || parsed > max)
        return -1;
    *value = (uint32_t)parsed;
    return 0;
}

static int valid_reference(uint32_t ref_uv)
{
    return ref_uv == 1800000 || ref_uv == 3600000;
}

static int run_self_test(uint32_t samples, uint32_t ref_uv)
{
    uint32_t uv;

    printf("ADC Self-Test: %u samples/channel, reference %u uV\n",
           samples, ref_uv);

    CHECK(drv_adc_read(0) == DRV_ADC_INVALID_VALUE,
          "reject read before initialization");
    CHECK(drv_adc_init() == 0, "initialize ADC owner 1");
    CHECK(drv_adc_init() == 0, "initialize ADC owner 2");

    CHECK(drv_adc_read(-1) == DRV_ADC_INVALID_VALUE,
          "reject channel -1");
    CHECK(drv_adc_read(DRV_ADC_MAX_CHANNEL) == DRV_ADC_INVALID_VALUE,
          "reject channel %d", DRV_ADC_MAX_CHANNEL);
    CHECK(drv_adc_read_uv(0, 0) == DRV_ADC_INVALID_VALUE,
          "reject zero reference voltage");

    CHECK(drv_adc_raw_to_uv(0, ref_uv, &uv) == 0 && uv == 0,
          "convert zero-scale raw sample");
    CHECK(drv_adc_raw_to_uv(DRV_ADC_RESOLUTION, ref_uv, &uv) == 0
              && uv == ref_uv,
          "convert full-scale raw sample");
    uint32_t expected_mid = (uint32_t)(((uint64_t)2048 * ref_uv
                                        + DRV_ADC_RESOLUTION / 2)
                                       / DRV_ADC_RESOLUTION);
    CHECK(drv_adc_raw_to_uv(2048, ref_uv, &uv) == 0
              && uv == expected_mid,
          "convert midpoint raw sample to %u uV", expected_mid);
    CHECK(drv_adc_raw_to_uv(DRV_ADC_RESOLUTION + 1, ref_uv, &uv) != 0,
          "reject raw value above 12-bit range");
    CHECK(drv_adc_raw_to_uv(1, 0, &uv) != 0,
          "reject zero reference in conversion helper");
    CHECK(drv_adc_raw_to_uv(1, ref_uv, NULL) != 0,
          "reject NULL conversion output");

    for (int channel = 0; channel < DRV_ADC_MAX_CHANNEL; channel++) {
        uint32_t minimum = DRV_ADC_INVALID_VALUE;
        uint32_t maximum = 0;
        uint64_t sum = 0;
        int samples_ok = 1;

        for (uint32_t i = 0; i < samples; i++) {
            uint32_t raw = drv_adc_read(channel);
            uint32_t measured_uv = drv_adc_read_uv(channel, ref_uv);

            if (raw == DRV_ADC_INVALID_VALUE
                || raw > DRV_ADC_RESOLUTION
                || measured_uv == DRV_ADC_INVALID_VALUE
                || measured_uv > ref_uv) {
                samples_ok = 0;
                break;
            }
            if (raw < minimum)
                minimum = raw;
            if (raw > maximum)
                maximum = raw;
            sum += raw;
        }

        CHECK(samples_ok,
              "channel %d range: min=%u max=%u mean=%u",
              channel, minimum, maximum,
              samples_ok ? (uint32_t)(sum / samples) : 0);
    }

    CHECK(drv_adc_deinit() == 0, "release ADC owner 1");
    CHECK(drv_adc_read(0) != DRV_ADC_INVALID_VALUE,
          "ADC remains open for owner 2");
    CHECK(drv_adc_deinit() == 0, "release ADC owner 2");
    CHECK(drv_adc_read(0) == DRV_ADC_INVALID_VALUE,
          "final deinit closes ADC");

    int lifecycle_ok = 1;
    for (int i = 0; i < 100; i++) {
        if (drv_adc_init() != 0) {
            lifecycle_ok = 0;
            break;
        }
        if (drv_adc_read(i % DRV_ADC_MAX_CHANNEL) == DRV_ADC_INVALID_VALUE) {
            lifecycle_ok = 0;
            (void)drv_adc_deinit();
            break;
        }
        if (drv_adc_deinit() != 0) {
            lifecycle_ok = 0;
            break;
        }
    }
    CHECK(lifecycle_ok, "100 init/read/deinit cycles");
    CHECK(drv_adc_read(0) == DRV_ADC_INVALID_VALUE,
          "ADC remains closed after lifecycle test");

    printf("Checks: %d passed, %d failed, %d total\n",
           total_checks - total_failures, total_failures, total_checks);
    return total_failures == 0 ? 0 : 1;
}

static int run_monitor(int channel, uint32_t ref_uv)
{
    signal(SIGINT, handle_sigint);

    printf("ADC Monitor\n");
    printf("Reading channel %d with reference %u uV\n", channel, ref_uv);
    printf("Press Ctrl+C to stop.\n\n");

    if (drv_adc_init() != 0) {
        fprintf(stderr, "ADC initialization failed\n");
        return 1;
    }

    while (!stop_flag) {
        uint32_t raw = drv_adc_read(channel);
        uint32_t uv;

        if (raw == DRV_ADC_INVALID_VALUE
            || drv_adc_raw_to_uv(raw, ref_uv, &uv) != 0) {
            fprintf(stderr, "ADC read failed\n");
            drv_adc_deinit();
            return 1;
        }
        printf("Channel %d: Raw=0x%03X (%4u), Voltage=%u uV\n",
               channel, raw, raw, uv);
        sleep(1);
    }

    drv_adc_deinit();
    printf("ADC monitor completed\n");
    return 0;
}

static void print_usage(const char* program)
{
    printf("Usage:\n");
    printf("  %s [CHANNEL [REF_UV]]\n", program);
    printf("  %s --self-test [SAMPLES [REF_UV]]\n", program);
}

int main(int argc, char** argv)
{
    uint32_t channel = 0;
    uint32_t ref_uv = DRV_ADC_DEFAULT_REF_UV;

    if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
        uint32_t samples = 10;

        if (argc > 4
            || (argc > 2 && parse_u32(argv[2], 1, 10000, &samples) != 0)
            || (argc > 3 && parse_u32(argv[3], 1, UINT32_MAX, &ref_uv) != 0)
            || !valid_reference(ref_uv)) {
            print_usage(argv[0]);
            return 2;
        }
        return run_self_test(samples, ref_uv);
    }

    if (argc > 3
        || (argc > 1
            && parse_u32(argv[1], 0, DRV_ADC_MAX_CHANNEL - 1, &channel) != 0)
        || (argc > 2 && parse_u32(argv[2], 1, UINT32_MAX, &ref_uv) != 0)
        || !valid_reference(ref_uv)) {
        print_usage(argv[0]);
        return 2;
    }

    return run_monitor((int)channel, ref_uv);
}
