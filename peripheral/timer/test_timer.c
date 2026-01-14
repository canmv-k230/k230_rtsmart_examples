#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdatomic.h>
#include "drv_timer.h"

// 测试结果统计
static int test_passed = 0;
static int test_failed = 0;

// 中断测试变量
static atomic_int hard_timer_irq_count = 0;
static atomic_int soft_timer_irq_count = 0;
static atomic_int irq_test_running = 0;

// 测试宏
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s:%d - %s\n", __func__, __LINE__, msg); \
        test_failed++; \
        return -1; \
    } else { \
        printf("[PASS] %s\n", msg); \
        test_passed++; \
    } \
} while(0)

#define WAIT_FOR_IRQ(count_var, expected, timeout_ms) do { \
    int waited = 0; \
    while (atomic_load(&count_var) < expected && waited < timeout_ms) { \
        usleep(1000); /* 1ms */ \
        waited++; \
    } \
    TEST_ASSERT(atomic_load(&count_var) >= expected, "IRQ occurred expected times"); \
} while(0)

// Timer callback functions
void hard_timer_callback(void* args) {
    atomic_fetch_add(&hard_timer_irq_count, 1);
}

void soft_timer_callback(void* args) {
    atomic_fetch_add(&soft_timer_irq_count, 1);
}

// Helper function to print test summary
void print_test_summary() {
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", test_passed);
    printf("Failed: %d\n", test_failed);
    printf("===================\n");
}

// Test cases for hard timer
int test_hard_timer() {
    printf("\n=== Testing Hard Timer ===\n");

    drv_hard_timer_inst_t* timer = NULL;
    rt_hwtimer_info_t info;
    uint32_t freq;
    int id;
    int ret;

    // Test 1: Create with invalid ID
    ret = drv_hard_timer_inst_create(KD_TIMER_MAX_NUM, &timer);
    TEST_ASSERT(ret == -1, "Create with invalid ID should fail");

    // Test 2: Create with valid ID
    ret = drv_hard_timer_inst_create(0, &timer);
    TEST_ASSERT(ret == 0 && timer != NULL, "Create with valid ID");

    // Test 3: Get timer info
    ret = drv_hard_timer_get_info(timer, &info);
    TEST_ASSERT(ret == 0, "Get timer info");

    // Test 4: Set invalid mode while running
    ret = drv_hard_timer_start(timer);
    TEST_ASSERT(ret == 0, "Start timer");
    ret = drv_hard_timer_set_mode(timer, HWTIMER_MODE_PERIOD);
    TEST_ASSERT(ret == -1, "Set mode while running should fail");
    ret = drv_hard_timer_stop(timer);
    TEST_ASSERT(ret == 0, "Stop timer");

    // Test 5: Set valid mode
    ret = drv_hard_timer_set_mode(timer, HWTIMER_MODE_PERIOD);
    TEST_ASSERT(ret == 0, "Set valid mode");

    // Test 6: Set invalid frequency (out of range)
    uint32_t invalid_freq = info.minfreq - 1;
    ret = drv_hard_timer_set_freq(timer, invalid_freq);
    TEST_ASSERT(ret == -1, "Set invalid frequency should fail");

    // Test 7: Set valid frequency
    uint32_t valid_freq = (info.minfreq + info.maxfreq) / 2;
    ret = drv_hard_timer_set_freq(timer, valid_freq);
    TEST_ASSERT(ret == 0, "Set valid frequency");

    // Test 8: Get frequency
    ret = drv_hard_timer_get_freq(timer, &freq);
    TEST_ASSERT(ret == 0 && freq == valid_freq, "Get frequency");

    // Test 9: Set invalid period (out of range)
    float freq_kHz = (float)freq / 1000.0f;
    float minPeriod_ms = 1.0f / freq_kHz;
    float maxPeriod_ms = (float)info.maxcnt / freq_kHz;
    uint32_t too_small = (uint32_t)minPeriod_ms - 1;
    uint32_t too_large = (uint32_t)maxPeriod_ms + 1;

    ret = drv_hard_timer_set_period(timer, too_small);
    TEST_ASSERT(ret == -1, "Set too small period should fail");
    ret = drv_hard_timer_set_period(timer, too_large);
    TEST_ASSERT(ret == -1, "Set too large period should fail");

    // Test 10: Set valid period
    uint32_t valid_period = 100; // 100ms
    ret = drv_hard_timer_set_period(timer, valid_period);
    TEST_ASSERT(ret == 0, "Set valid period");

    // Test 11: Register IRQ
    atomic_store(&hard_timer_irq_count, 0);
    ret = drv_hard_timer_register_irq(timer, hard_timer_callback, NULL);
    TEST_ASSERT(ret == 0, "Register IRQ");

    // Test 12: Start timer and verify callbacks
    ret = drv_hard_timer_start(timer);
    TEST_ASSERT(ret == 0, "Start timer");

    // Wait for at least 3 callbacks (300ms period)
    WAIT_FOR_IRQ(hard_timer_irq_count, 3, 500);

    // Test 13: Check if timer is started
    ret = drv_hard_timer_is_started(timer);
    TEST_ASSERT(ret == 1, "Timer should be started");

    // Test 14: Get timer ID
    id = drv_hard_timer_get_id(timer);
    TEST_ASSERT(id == 0, "Get timer ID");

    // Test 15: Stop timer
    ret = drv_hard_timer_stop(timer);
    TEST_ASSERT(ret == 0, "Stop timer");

    // Test 16: Unregister IRQ
    ret = drv_hard_timer_unregister_irq(timer);
    TEST_ASSERT(ret == 0, "Unregister IRQ");

    // Test 17: Destroy timer
    drv_hard_timer_inst_destroy(&timer);
    TEST_ASSERT(timer == NULL, "Destroy timer");

    return 0;
}

// Test cases for soft timer
int test_soft_timer() {
    printf("\n=== Testing Soft Timer ===\n");

    drv_soft_timer_inst_t* timer = NULL;
    int ret;

    // Test 1: Create timer
    ret = drv_soft_timer_create(&timer);
    TEST_ASSERT(ret == 0 && timer != NULL, "Create soft timer");

    // Test 2: Set mode (oneshot)
    ret = drv_soft_timer_set_mode(timer, HWTIMER_MODE_ONESHOT);
    TEST_ASSERT(ret == 0, "Set oneshot mode");

    // Test 3: Set period
    ret = drv_soft_timer_set_period(timer, 100); // 100ms
    TEST_ASSERT(ret == 0, "Set period");

    // Test 4: Register IRQ
    atomic_store(&soft_timer_irq_count, 0);
    ret = drv_soft_timer_register_irq(timer, soft_timer_callback, NULL);
    TEST_ASSERT(ret == 0, "Register IRQ");

    // Test 5: Start timer (oneshot)
    ret = drv_soft_timer_start(timer);
    TEST_ASSERT(ret == 0, "Start oneshot timer");

    // Wait for the callback (100ms period + margin)
    WAIT_FOR_IRQ(soft_timer_irq_count, 1, 150);

    // Test 6: Check timer state after oneshot (should still be running)
    ret = drv_soft_timer_is_started(timer);
    TEST_ASSERT(ret == 1, "Timer should still be running after oneshot");

    // Test 7: Stop timer
    ret = drv_soft_timer_stop(timer);
    TEST_ASSERT(ret == 0, "Stop timer");

    // Test 8: Set mode (periodic)
    ret = drv_soft_timer_set_mode(timer, HWTIMER_MODE_PERIOD);
    TEST_ASSERT(ret == 0, "Set periodic mode");

    // Test 9: Start timer (periodic)
    atomic_store(&soft_timer_irq_count, 0);
    ret = drv_soft_timer_start(timer);
    TEST_ASSERT(ret == 0, "Start periodic timer");

    // Wait for at least 3 callbacks (300ms period)
    WAIT_FOR_IRQ(soft_timer_irq_count, 3, 350);

    // Test 10: Stop timer
    ret = drv_soft_timer_stop(timer);
    TEST_ASSERT(ret == 0, "Stop periodic timer");

    // Test 11: Unregister IRQ
    ret = drv_soft_timer_unregister_irq(timer);
    TEST_ASSERT(ret == 0, "Unregister IRQ");

    // Test 12: Destroy timer
    drv_soft_timer_destroy(&timer);
    TEST_ASSERT(timer == NULL, "Destroy timer");

    return 0;
}

int main() {
    printf("Starting Timer Driver Tests\n");

    test_hard_timer();
    test_soft_timer();

    print_test_summary();

    return test_failed ? -1 : 0;
}
