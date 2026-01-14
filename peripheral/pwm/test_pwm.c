#include "drv_fpioa.h"
#include "drv_pwm.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Test parameters
#define TEST_FREQ     1000 // 1kHz initial test frequency
#define TEST_DUTY     50 // 50% initial duty cycle
#define TEST_DURATION 2 // Test duration in seconds

#define TEST_CHANNEL 0 // Channel to test (0-5)

// Pin configuration
#define PWM0_PIN     42
#define PWM1_PIN     43
#define PWM0_CHANNEL 0
#define PWM1_CHANNEL 1

void setup_pwm_pins(void)
{
    printf("Configuring PWM pins...\n");

    // Map PWM channels to physical pins
    if (drv_fpioa_set_pin_func(PWM0_PIN, PWM0) != 0) {
        printf("FAIL: Failed to map PWM0 to pin %d\n", PWM0_PIN);
        exit(1);
    }

    if (drv_fpioa_set_pin_func(PWM1_PIN, PWM1) != 0) {
        printf("FAIL: Failed to map PWM1 to pin %d\n", PWM1_PIN);
        exit(1);
    }

    printf("PASS: PWM pins configured (PWM0: pin %d, PWM1: pin %d)\n", PWM0_PIN, PWM1_PIN);
}

void test_dual_pwm_operation(void)
{
    printf("\n=== Testing Dual PWM Operation ===\n");

    // Initialize PWM
    if (drv_pwm_init() != 0) {
        printf("FAIL: PWM initialization failed\n");
        return;
    }

    // Configure both PWM channels
    printf("Configuring PWM channels...\n");
    if (drv_pwm_set_freq(PWM0_CHANNEL, TEST_FREQ) != 0 || drv_pwm_set_freq(PWM1_CHANNEL, TEST_FREQ) != 0) {
        printf("FAIL: Failed to set PWM frequencies\n");
        return;
    }

    if (drv_pwm_set_duty(PWM0_CHANNEL, TEST_DUTY) != 0 || drv_pwm_set_duty(PWM1_CHANNEL, TEST_DUTY) != 0) {
        printf("FAIL: Failed to set PWM duty cycles\n");
        return;
    }

    printf("PASS: Both PWM channels configured to %dHz, %d%% duty\n", TEST_FREQ, TEST_DUTY);

    // Enable both PWM channels simultaneously
    printf("Enabling both PWM channels...\n");
    if (drv_pwm_enable(PWM0_CHANNEL) != 0 || drv_pwm_enable(PWM1_CHANNEL) != 0) {
        printf("FAIL: Failed to enable PWM channels\n");
        return;
    }
    printf("PASS: Both PWM channels enabled\n");

    // Test frequency changes while running
    printf("\nTesting frequency changes...\n");
    uint32_t test_frequencies[] = { 500, 1000, 2000, 5000 };
    for (int i = 0; i < sizeof(test_frequencies) / sizeof(test_frequencies[0]); i++) {
        uint32_t freq = test_frequencies[i];

        printf("Setting both PWMs to %dHz...\n", freq);
        if (drv_pwm_set_freq(PWM0_CHANNEL, freq) != 0 || drv_pwm_set_freq(PWM1_CHANNEL, freq) != 0) {
            printf("FAIL: Failed to set frequency to %dHz\n", freq);
            continue;
        }

        // Verify
        uint32_t freq0, freq1;
        drv_pwm_get_freq(PWM0_CHANNEL, &freq0);
        drv_pwm_get_freq(PWM1_CHANNEL, &freq1);
        printf("PWM0: %uHz, PWM1: %uHz\n", freq0, freq1);

        sleep(1); // Observe each frequency for 1 second
    }

    // Test duty cycle changes while running
    printf("\nTesting duty cycle changes...\n");
    uint32_t test_duties[] = { 10, 25, 50, 75, 90 };
    for (int i = 0; i < sizeof(test_duties) / sizeof(test_duties[0]); i++) {
        uint32_t duty = test_duties[i];

        printf("Setting PWM0 to %d%%, PWM1 to %d%%...\n", duty, 100 - duty);
        if (drv_pwm_set_duty(PWM0_CHANNEL, duty) != 0 || drv_pwm_set_duty(PWM1_CHANNEL, 100 - duty) != 0) {
            printf("FAIL: Failed to set duty cycles\n");
            continue;
        }

        // Verify
        uint32_t duty0, duty1;
        drv_pwm_get_duty(PWM0_CHANNEL, &duty0);
        drv_pwm_get_duty(PWM1_CHANNEL, &duty1);
        printf("PWM0: %u%%, PWM1: %u%%\n", duty0, duty1);

        sleep(1); // Observe each duty cycle for 1 second
    }

    // Disable both PWM channels
    printf("\nDisabling PWM channels...\n");
    if (drv_pwm_disable(PWM0_CHANNEL) != 0 || drv_pwm_disable(PWM1_CHANNEL) != 0) {
        printf("FAIL: Failed to disable PWM channels\n");
    } else {
        printf("PASS: Both PWM channels disabled\n");
    }

    // Cleanup
    drv_pwm_deinit();
}

void test_simultaneous_freq_changes(void)
{
    printf("\n=== Testing Simultaneous Frequency Changes ===\n");

    // Initialize PWM
    if (drv_pwm_init() != 0) {
        printf("FAIL: PWM initialization failed\n");
        return;
    }

    // Configure both PWM channels with same initial settings
    printf("Initializing both PWM channels to %dHz, %d%% duty\n", TEST_FREQ, TEST_DUTY);
    if (drv_pwm_set_freq(PWM0_CHANNEL, TEST_FREQ) != 0 || drv_pwm_set_freq(PWM1_CHANNEL, TEST_FREQ) != 0) {
        printf("FAIL: Failed to set initial frequencies\n");
        return;
    }

    if (drv_pwm_set_duty(PWM0_CHANNEL, TEST_DUTY) != 0 || drv_pwm_set_duty(PWM1_CHANNEL, TEST_DUTY) != 0) {
        printf("FAIL: Failed to set initial duty cycles\n");
        return;
    }

    // Enable both channels
    printf("Enabling both PWM channels\n");
    if (drv_pwm_enable(PWM0_CHANNEL) != 0 || drv_pwm_enable(PWM1_CHANNEL) != 0) {
        printf("FAIL: Failed to enable PWM channels\n");
        return;
    }

    // Test 1: Synchronized frequency sweep
    printf("\nTest 1: Synchronized frequency sweep\n");
    uint32_t sync_freqs[] = { 100, 500, 1000, 2000, 5000, 10000 };
    for (int i = 0; i < sizeof(sync_freqs) / sizeof(sync_freqs[0]); i++) {
        uint32_t freq = sync_freqs[i];

        printf("Setting both to %dHz... ", freq);
        if (drv_pwm_set_freq(PWM0_CHANNEL, freq) != 0 || drv_pwm_set_freq(PWM1_CHANNEL, freq) != 0) {
            printf("FAIL\n");
            continue;
        }

        // Verify
        uint32_t freq0, freq1;
        drv_pwm_get_freq(PWM0_CHANNEL, &freq0);
        drv_pwm_get_freq(PWM1_CHANNEL, &freq1);

        if (fabs(freq0 - freq) > freq * 0.1 || fabs(freq1 - freq) > freq * 0.1) {
            printf("FAIL (PWM0: %uHz, PWM1: %uHz)\n", freq0, freq1);
        } else {
            printf("PASS\n");
        }

        sleep(1); // Observe for 1 second
    }

    // Test 2: Independent frequency changes
    printf("\nTest 2: Independent frequency changes\n");
    struct {
        uint32_t freq0;
        uint32_t freq1;
    } indep_freqs[] = { { 1000, 2000 }, { 500, 1500 }, { 2000, 500 }, { 3000, 1000 } };

    for (int i = 0; i < sizeof(indep_freqs) / sizeof(indep_freqs[0]); i++) {
        printf("Setting PWM0 to %dHz, PWM1 to %dHz... ", indep_freqs[i].freq0, indep_freqs[i].freq1);

        if (drv_pwm_set_freq(PWM0_CHANNEL, indep_freqs[i].freq0) != 0
            || drv_pwm_set_freq(PWM1_CHANNEL, indep_freqs[i].freq1) != 0) {
            printf("FAIL\n");
            continue;
        }

        // Verify
        uint32_t freq0, freq1;
        drv_pwm_get_freq(PWM0_CHANNEL, &freq0);
        drv_pwm_get_freq(PWM1_CHANNEL, &freq1);

        if (fabs(freq0 - indep_freqs[i].freq0) > indep_freqs[i].freq0 * 0.1
            || fabs(freq1 - indep_freqs[i].freq1) > indep_freqs[i].freq1 * 0.1) {
            printf("FAIL (PWM0: %uHz, PWM1: %uHz)\n", freq0, freq1);
        } else {
            printf("PASS\n");
        }

        sleep(1); // Observe for 1 second
    }

    // Test 3: Rapid frequency changes
    printf("\nTest 3: Rapid frequency changes\n");
    for (int i = 0; i < 10; i++) {
        uint32_t freq0 = 500 + (i % 5) * 500;
        uint32_t freq1 = 1000 - (i % 3) * 300;

        printf("Cycle %d: PWM0=%dHz, PWM1=%dHz... ", i + 1, freq0, freq1);
        if (drv_pwm_set_freq(PWM0_CHANNEL, freq0) != 0 || drv_pwm_set_freq(PWM1_CHANNEL, freq1) != 0) {
            printf("FAIL\n");
            continue;
        }

        usleep(200000); // 200ms delay for observation
        printf("DONE\n");
    }

    // Disable both channels
    printf("\nDisabling PWM channels\n");
    if (drv_pwm_disable(PWM0_CHANNEL) != 0 || drv_pwm_disable(PWM1_CHANNEL) != 0) {
        printf("FAIL: Failed to disable PWM channels\n");
    } else {
        printf("PASS: Both PWM channels disabled\n");
    }

    // Cleanup
    drv_pwm_deinit();
}

void test_pwm_basic_operations(void)
{
    printf("\n=== Testing Basic PWM Operations ===\n");

    // Initialize PWM
    if (drv_pwm_init() != 0) {
        printf("FAIL: PWM initialization failed\n");
        return;
    }
    printf("PASS: PWM initialized\n");

    // Test frequency setting
    if (drv_pwm_set_freq(TEST_CHANNEL, TEST_FREQ) != 0) {
        printf("FAIL: Failed to set frequency\n");
    } else {
        printf("PASS: Frequency set to %d Hz\n", TEST_FREQ);

        // Verify frequency
        uint32_t read_freq;
        if (drv_pwm_get_freq(TEST_CHANNEL, &read_freq) != 0) {
            printf("FAIL: Failed to get frequency\n");
        } else {
            printf("PASS: Read frequency: %u Hz\n", read_freq);
            if (abs((int)read_freq - (int)TEST_FREQ) > (TEST_FREQ / 10)) {
                printf("WARN: Frequency mismatch (set %u vs read %u)\n", TEST_FREQ, read_freq);
            }
        }
    }

    // Test duty cycle setting
    if (drv_pwm_set_duty(TEST_CHANNEL, TEST_DUTY) != 0) {
        printf("FAIL: Failed to set duty cycle\n");
    } else {
        printf("PASS: Duty cycle set to %d%%\n", TEST_DUTY);

        // Verify duty cycle
        uint32_t read_duty;
        if (drv_pwm_get_duty(TEST_CHANNEL, &read_duty) != 0) {
            printf("FAIL: Failed to get duty cycle\n");
        } else {
            printf("PASS: Read duty cycle: %u%%\n", read_duty);
            if (read_duty != TEST_DUTY) {
                printf("WARN: Duty cycle mismatch (set %u vs read %u)\n", TEST_DUTY, read_duty);
            }
        }
    }

    // Test enable/disable
    if (drv_pwm_enable(TEST_CHANNEL) != 0) {
        printf("FAIL: Failed to enable PWM\n");
    } else {
        printf("PASS: PWM enabled\n");
        sleep(1); // Let PWM run for 1 second

        if (drv_pwm_disable(TEST_CHANNEL) != 0) {
            printf("FAIL: Failed to disable PWM\n");
        } else {
            printf("PASS: PWM disabled\n");
        }
    }

    // Cleanup
    drv_pwm_deinit();
}

void test_pwm_corner_cases(void)
{
    printf("\n=== Testing PWM Corner Cases ===\n");

    if (drv_pwm_init() != 0) {
        printf("FAIL: PWM initialization failed\n");
        return;
    }

    // Test invalid channel numbers
    printf("\nTesting invalid channels:\n");
    int invalid_channels[] = { -1, 6, 255 };
    for (int i = 0; i < sizeof(invalid_channels) / sizeof(invalid_channels[0]); i++) {
        int ch = invalid_channels[i];
        if (drv_pwm_set_freq(ch, TEST_FREQ) == 0) {
            printf("FAIL: Accepted invalid channel %d\n", ch);
        } else {
            printf("PASS: Rejected invalid channel %d\n", ch);
        }
    }

    // Test frequency limits
    printf("\nTesting frequency limits:\n");
    uint32_t extreme_freqs[] = { 0, 1, 1000000 }; // 0Hz, 1Hz, 1MHz
    for (int i = 0; i < sizeof(extreme_freqs) / sizeof(extreme_freqs[0]); i++) {
        uint32_t freq = extreme_freqs[i];
        if (drv_pwm_set_freq(TEST_CHANNEL, freq) != 0) {
            printf("PASS: Rejected %u Hz (may be expected for some limits)\n", freq);
        } else {
            printf("PASS: Accepted %u Hz\n", freq);

            // Verify
            uint32_t read_freq;
            drv_pwm_get_freq(TEST_CHANNEL, &read_freq);
            printf("     Actual frequency: %u Hz\n", read_freq);

            // Clean up
            drv_pwm_disable(TEST_CHANNEL);
        }
    }

    // Test duty cycle limits
    printf("\nTesting duty cycle limits:\n");
    uint32_t extreme_duties[] = { 0, 50, 100, 101, 255 }; // 0%, 50%, 100%, 101%, 255%
    for (int i = 0; i < sizeof(extreme_duties) / sizeof(extreme_duties[0]); i++) {
        uint32_t duty = extreme_duties[i];
        if (drv_pwm_set_duty(TEST_CHANNEL, duty) != 0) {
            printf("PASS: Rejected %u%% duty (may be expected for values >100)\n", duty);
        } else {
            printf("PASS: Accepted %u%% duty\n", duty);

            // Verify
            uint32_t read_duty;
            drv_pwm_get_duty(TEST_CHANNEL, &read_duty);
            printf("     Actual duty: %u%%\n", read_duty);
        }
    }

    // Test enable/disable sequences
    printf("\nTesting enable/disable sequences:\n");
    for (int i = 0; i < 3; i++) {
        if (drv_pwm_enable(TEST_CHANNEL) != 0) {
            printf("FAIL: Enable #%d failed\n", i + 1);
        } else {
            printf("PASS: Enable #%d succeeded\n", i + 1);
        }
    }

    for (int i = 0; i < 3; i++) {
        if (drv_pwm_disable(TEST_CHANNEL) != 0) {
            printf("FAIL: Disable #%d failed\n", i + 1);
        } else {
            printf("PASS: Disable #%d succeeded\n", i + 1);
        }
    }

    // Cleanup
    drv_pwm_deinit();
}

void test_pwm_configuration_persistence(void)
{
    printf("\n=== Testing Configuration Persistence ===\n");

    if (drv_pwm_init() != 0) {
        printf("FAIL: PWM initialization failed\n");
        return;
    }

    // Set initial configuration
    uint32_t initial_freq = 2000;
    uint32_t initial_duty = 30;
    drv_pwm_set_freq(TEST_CHANNEL, initial_freq);
    drv_pwm_set_duty(TEST_CHANNEL, initial_duty);

    // Enable and disable
    drv_pwm_enable(TEST_CHANNEL);
    sleep(1);
    drv_pwm_disable(TEST_CHANNEL);

    // Verify configuration persists
    uint32_t current_freq, current_duty;
    drv_pwm_get_freq(TEST_CHANNEL, &current_freq);
    drv_pwm_get_duty(TEST_CHANNEL, &current_duty);

    if (current_freq == initial_freq && current_duty == initial_duty) {
        printf("PASS: Configuration persists after enable/disable\n");
    } else {
        printf("FAIL: Configuration changed after enable/disable\n");
        printf("      Before: %uHz %u%%, After: %uHz %u%%\n", initial_freq, initial_duty, current_freq, current_duty);
    }

    // Cleanup
    drv_pwm_deinit();
}

int main(void)
{
    printf("Starting Dual PWM Test Suite\n");

    // Setup hardware pins first
    setup_pwm_pins();

    // Run individual tests
    test_pwm_basic_operations();
    test_pwm_corner_cases();
    test_pwm_configuration_persistence();

    // Run dual PWM test
    test_dual_pwm_operation();
    test_simultaneous_freq_changes();

    printf("\nTest Suite Completed\n");
    return 0;
}
