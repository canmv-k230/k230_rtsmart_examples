#include "drv_fpioa.h"
#include "drv_pwm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PWM0_PIN     42
#define PWM1_PIN     43
#define PWM3_PIN     47
#define PWM0_CHANNEL 0
#define PWM1_CHANNEL 1
#define PWM3_CHANNEL 3

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

static int value_within_percent(uint32_t actual, uint32_t expected,
                                uint32_t percent)
{
    uint64_t difference = actual > expected ? actual - expected
                                            : expected - actual;

    return difference * 100 <= (uint64_t)expected * percent;
}

static int setup_pwm_pins(void)
{
    int ret0 = drv_fpioa_set_pin_func(PWM0_PIN, PWM0);
    int ret1 = drv_fpioa_set_pin_func(PWM1_PIN, PWM1);
    int ret3 = drv_fpioa_set_pin_func(PWM3_PIN, PWM3);

    CHECK(ret0 == 0, "map PWM0 to pin %d", PWM0_PIN);
    CHECK(ret1 == 0, "map PWM1 to pin %d", PWM1_PIN);
    CHECK(ret3 == 0, "map PWM3 to pin %d", PWM3_PIN);
    return ret0 == 0 && ret1 == 0 && ret3 == 0 ? 0 : -1;
}

static int begin_test(const char* name)
{
    int ret;

    printf("\n=== %s ===\n", name);
    ret = drv_pwm_init();
    CHECK(ret == 0, "initialize PWM");
    return ret;
}

static void end_test(void)
{
    CHECK(drv_pwm_deinit() == 0, "deinitialize PWM");
}

static void test_invalid_parameters(void)
{
    uint32_t period_ns;

    if (begin_test("Invalid Parameter Tests") != 0)
        return;

    CHECK(drv_pwm_set_freq(-1, 1000) != 0, "reject channel -1");
    CHECK(drv_pwm_set_freq(6, 1000) != 0, "reject channel 6");
    CHECK(drv_pwm_set_freq(255, 1000) != 0, "reject channel 255");
    CHECK(drv_pwm_set_freq(PWM0_CHANNEL, 0) != 0,
          "reject zero frequency");
    CHECK(drv_pwm_set_freq(PWM0_CHANNEL, 2000000000U) != 0,
          "reject frequency above nanosecond resolution");
    CHECK(drv_pwm_enable(PWM0_CHANNEL) != 0,
          "reject enable before period configuration");
    CHECK(drv_pwm_set_duty(PWM0_CHANNEL, 50) != 0,
          "reject duty before period configuration");

    CHECK(drv_pwm_set_freq(PWM0_CHANNEL, 1000) == 0,
          "configure valid 1000 Hz period");
    CHECK(drv_pwm_set_duty(PWM0_CHANNEL, 101) != 0,
          "reject 101 percent duty");
    CHECK(drv_pwm_set_duty(PWM0_CHANNEL, 255) != 0,
          "reject 255 percent duty");

    period_ns = 1000000000U / 1000U;
    CHECK(drv_pwm_set_duty_ns(PWM0_CHANNEL, period_ns + 1) != 0,
          "reject pulse wider than period");
    end_test();
}

static void test_readback_matrix(void)
{
    static const uint32_t frequencies[] = {
        1, 100, 500, 1000, 2000, 5000, 10000, 1000000,
    };
    static const uint32_t duties[] = { 0, 10, 25, 50, 75, 90, 100 };
    uint32_t actual;

    if (begin_test("Frequency and Duty Readback") != 0)
        return;

    for (size_t i = 0; i < sizeof(frequencies) / sizeof(frequencies[0]); i++) {
        uint32_t requested = frequencies[i];
        int set_ret = drv_pwm_set_freq(PWM0_CHANNEL, requested);
        int get_ret = set_ret == 0
                          ? drv_pwm_get_freq(PWM0_CHANNEL, &actual)
                          : -1;

        CHECK(set_ret == 0 && get_ret == 0
                  && value_within_percent(actual, requested, 1),
              "frequency request %u Hz, readback %u Hz", requested,
              get_ret == 0 ? actual : 0);
    }

    CHECK(drv_pwm_set_freq(PWM0_CHANNEL, 1000) == 0,
          "restore 1000 Hz before duty matrix");
    for (size_t i = 0; i < sizeof(duties) / sizeof(duties[0]); i++) {
        uint32_t requested = duties[i];
        int set_ret = drv_pwm_set_duty(PWM0_CHANNEL, requested);
        int get_ret = set_ret == 0
                          ? drv_pwm_get_duty(PWM0_CHANNEL, &actual)
                          : -1;
        uint32_t difference = get_ret == 0 && actual > requested
                                  ? actual - requested
                                  : requested - (get_ret == 0 ? actual : 0);

        CHECK(set_ret == 0 && get_ret == 0 && difference <= 1,
              "duty request %u percent, readback %u percent", requested,
              get_ret == 0 ? actual : 0);
    }
    end_test();
}

static void test_shared_instance(void)
{
    uint32_t freq0 = 0, freq1 = 0, duty0 = 0, duty1 = 0;

    if (begin_test("Shared-Instance Period and Independent Duty") != 0)
        return;

    CHECK(drv_pwm_set_freq(PWM0_CHANNEL, 1000) == 0,
          "set PWM0/PWM1 shared period to 1000 Hz");
    CHECK(drv_pwm_set_duty(PWM0_CHANNEL, 25) == 0,
          "set PWM0 duty to 25 percent");
    CHECK(drv_pwm_set_duty(PWM1_CHANNEL, 75) == 0,
          "set PWM1 duty to 75 percent");
    CHECK(drv_pwm_enable(PWM0_CHANNEL) == 0, "enable PWM0");
    CHECK(drv_pwm_enable(PWM1_CHANNEL) == 0, "enable PWM1");

    CHECK(drv_pwm_set_freq(PWM1_CHANNEL, 2000) == 0,
          "change shared PWM0/PWM1 period to 2000 Hz");
    int readback_ok = drv_pwm_get_freq(PWM0_CHANNEL, &freq0) == 0
                      && drv_pwm_get_freq(PWM1_CHANNEL, &freq1) == 0
                      && drv_pwm_get_duty(PWM0_CHANNEL, &duty0) == 0
                      && drv_pwm_get_duty(PWM1_CHANNEL, &duty1) == 0;
    CHECK(readback_ok && value_within_percent(freq0, 2000, 1)
              && value_within_percent(freq1, 2000, 1),
          "shared frequency readback PWM0=%u PWM1=%u", freq0, freq1);
    CHECK(readback_ok && duty0 >= 24 && duty0 <= 25
              && duty1 >= 74 && duty1 <= 75,
          "preserve sibling duties PWM0=%u PWM1=%u", duty0, duty1);

    CHECK(drv_pwm_disable(PWM0_CHANNEL) == 0, "disable PWM0");
    CHECK(drv_pwm_disable(PWM1_CHANNEL) == 0, "disable PWM1");
    end_test();
}

static void test_separate_instances(void)
{
    uint32_t freq0 = 0, freq3 = 0, duty0 = 0, duty3 = 0;

    if (begin_test("Separate-Instance Independent Configuration") != 0)
        return;

    CHECK(drv_pwm_set_freq(PWM0_CHANNEL, 1000) == 0,
          "set PWM0 to 1000 Hz");
    CHECK(drv_pwm_set_duty(PWM0_CHANNEL, 30) == 0,
          "set PWM0 to 30 percent");
    CHECK(drv_pwm_set_freq(PWM3_CHANNEL, 2500) == 0,
          "set PWM3 to 2500 Hz");
    CHECK(drv_pwm_set_duty(PWM3_CHANNEL, 70) == 0,
          "set PWM3 to 70 percent");
    CHECK(drv_pwm_enable(PWM0_CHANNEL) == 0, "enable PWM0");
    CHECK(drv_pwm_enable(PWM3_CHANNEL) == 0, "enable PWM3");

    int readback_ok = drv_pwm_get_freq(PWM0_CHANNEL, &freq0) == 0
                      && drv_pwm_get_freq(PWM3_CHANNEL, &freq3) == 0
                      && drv_pwm_get_duty(PWM0_CHANNEL, &duty0) == 0
                      && drv_pwm_get_duty(PWM3_CHANNEL, &duty3) == 0;
    CHECK(readback_ok && value_within_percent(freq0, 1000, 1)
              && value_within_percent(freq3, 2500, 1),
          "independent frequency readback PWM0=%u PWM3=%u", freq0, freq3);
    CHECK(readback_ok && duty0 >= 29 && duty0 <= 30
              && duty3 >= 69 && duty3 <= 70,
          "independent duty readback PWM0=%u PWM3=%u", duty0, duty3);

    CHECK(drv_pwm_disable(PWM0_CHANNEL) == 0, "disable PWM0");
    CHECK(drv_pwm_disable(PWM3_CHANNEL) == 0, "disable PWM3");
    end_test();
}

static void test_repeated_lifecycle(void)
{
    uint32_t frequency = 0, duty = 0;

    if (begin_test("Repeated Enable/Disable") != 0)
        return;

    CHECK(drv_pwm_set_freq(PWM0_CHANNEL, 2000) == 0,
          "configure lifecycle frequency");
    CHECK(drv_pwm_set_duty(PWM0_CHANNEL, 40) == 0,
          "configure lifecycle duty");

    int cycles_ok = 1;
    for (int i = 0; i < 100; i++) {
        if (drv_pwm_enable(PWM0_CHANNEL) != 0
            || drv_pwm_enable(PWM0_CHANNEL) != 0
            || drv_pwm_disable(PWM0_CHANNEL) != 0
            || drv_pwm_disable(PWM0_CHANNEL) != 0) {
            cycles_ok = 0;
            break;
        }
    }
    CHECK(cycles_ok, "100 idempotent double-enable/double-disable cycles");

    int readback_ok = drv_pwm_get_freq(PWM0_CHANNEL, &frequency) == 0
                      && drv_pwm_get_duty(PWM0_CHANNEL, &duty) == 0;
    CHECK(readback_ok && value_within_percent(frequency, 2000, 1)
              && duty >= 39 && duty <= 40,
          "configuration persists after cycles: %u Hz, %u percent",
          frequency, duty);
    end_test();
}

static void run_measurement_hold(unsigned int seconds)
{
    if (seconds == 0)
        return;

    if (begin_test("External Measurement Hold") != 0)
        return;

    int configured = drv_pwm_set_freq(PWM0_CHANNEL, 1000) == 0
                     && drv_pwm_set_duty(PWM0_CHANNEL, 25) == 0
                     && drv_pwm_set_freq(PWM3_CHANNEL, 2500) == 0
                     && drv_pwm_set_duty(PWM3_CHANNEL, 70) == 0
                     && drv_pwm_enable(PWM0_CHANNEL) == 0
                     && drv_pwm_enable(PWM3_CHANNEL) == 0;
    CHECK(configured, "configure external measurement outputs");
    if (configured) {
        printf("MEASURE: pin %d = 1000 Hz at 25 percent, pin %d = 2500 Hz at 70 percent for %u seconds\n",
               PWM0_PIN, PWM3_PIN, seconds);
        sleep(seconds);
    }
    CHECK(drv_pwm_disable(PWM0_CHANNEL) == 0, "disable measured PWM0");
    CHECK(drv_pwm_disable(PWM3_CHANNEL) == 0, "disable measured PWM3");
    end_test();
}

static int parse_hold_seconds(int argc, char** argv, unsigned int* seconds)
{
    char* end;
    unsigned long value;

    *seconds = 0;
    if (argc == 1)
        return 0;
    if (argc != 3 || strcmp(argv[1], "--hold") != 0)
        return -1;

    value = strtoul(argv[2], &end, 10);
    if (*argv[2] == '\0' || *end != '\0' || value == 0 || value > 3600)
        return -1;
    *seconds = (unsigned int)value;
    return 0;
}

int main(int argc, char** argv)
{
    unsigned int hold_seconds;

    if (parse_hold_seconds(argc, argv, &hold_seconds) != 0) {
        printf("Usage: %s [--hold SECONDS]\n", argv[0]);
        return 2;
    }

    printf("Starting PWM Test Suite\n");
    if (setup_pwm_pins() != 0)
        goto done;

    test_invalid_parameters();
    test_readback_matrix();
    test_shared_instance();
    test_separate_instances();
    test_repeated_lifecycle();
    run_measurement_hold(hold_seconds);

done:
    printf("\nChecks: %d passed, %d failed, %d total\n",
           total_checks - total_failures, total_failures, total_checks);
    return total_failures == 0 ? 0 : 1;
}
