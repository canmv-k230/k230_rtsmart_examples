#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>

#include "drv_gpio.h"
#include "drv_fpioa.h"

#define TEST_GPIO_PIN1      11  // 用于基本功能测试
#define TEST_GPIO_PIN2      12  // 用于输入输出测试
#define TEST_GPIO_PIN_IRQ   11  // 用于中断测试（必须小于64）
#define TEST_GPIO_PIN_MAX   63  // 最大引脚号; 71或63，某些板子无法测试到PMU GPIO
#define TEST_GPIO_PIN_INVALID 72  // 无效引脚号
#define TEST_GPIO_CONCURRENCY_THREADS 6
#define TEST_GPIO_CONCURRENCY_MIN_THREADS 4
#define TEST_GPIO_CONCURRENCY_LOOPS 2000

// 测试结果统计
static int test_passed = 0;
static int test_failed = 0;

// 中断测试变量
static volatile int irq_count = 0;
static volatile int irq_test_running = 0;

static int g_run_manual_tests = 1;

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

#define TEST_START(name) do { \
    printf("\n========== Test: %s ==========\n", name); \
} while(0)

// GPIO 中断回调函数
static void gpio_irq_handler(void* args)
{
    int *count = (int*)args;
    if (count) {
        (*count)++;
    }
    printf("[IRQ] GPIO interrupt triggered, count=%d\n", irq_count);
}

struct gpio_concurrency_ctx {
    drv_gpio_inst_t* gpio;
    int              pin;
    int              is_output;
    int              completed_loops;
    int              failures;
    volatile int*    start_flag;
};

static void* gpio_concurrency_worker(void* args)
{
    struct gpio_concurrency_ctx* ctx = (struct gpio_concurrency_ctx*)args;

    while (!*ctx->start_flag) {
        usleep(100);
    }

    for (int i = 0; i < TEST_GPIO_CONCURRENCY_LOOPS; i++) {
        if (ctx->is_output) {
            gpio_pin_value_t expected = (i & 0x01) ? GPIO_PV_HIGH : GPIO_PV_LOW;

            if ((i % 32) == 0 && drv_gpio_mode_set(ctx->gpio, GPIO_DM_OUTPUT) != 0) {
                ctx->failures++;
                break;
            }

            if (drv_gpio_value_set(ctx->gpio, expected) != 0) {
                ctx->failures++;
                break;
            }

            if (drv_gpio_value_get(ctx->gpio) != expected) {
                ctx->failures++;
                break;
            }
        } else {
            gpio_drive_mode_t expected_mode;
            gpio_drive_mode_t current_mode;
            gpio_pin_value_t  value;

            switch (i % 3) {
            case 0:
                expected_mode = GPIO_DM_INPUT;
                break;
            case 1:
                expected_mode = GPIO_DM_INPUT_PULLUP;
                break;
            default:
                expected_mode = GPIO_DM_INPUT_PULLDOWN;
                break;
            }

            if (drv_gpio_mode_set(ctx->gpio, expected_mode) != 0) {
                ctx->failures++;
                break;
            }

            current_mode = drv_gpio_mode_get(ctx->gpio);
            if (current_mode != expected_mode) {
                ctx->failures++;
                break;
            }

            value = drv_gpio_value_get(ctx->gpio);
            if ((value != GPIO_PV_LOW) && (value != GPIO_PV_HIGH)) {
                ctx->failures++;
                break;
            }
        }

        ctx->completed_loops++;

        if ((i % 64) == 0) {
            usleep(100);
        }
    }

    return NULL;
}

// 测试1：创建和销毁实例
static int test_create_destroy(void)
{
    TEST_START("Create and Destroy Instance");

    drv_gpio_inst_t* gpio = NULL;
    int ret;

    // 配置引脚为GPIO功能
    ret = drv_fpioa_set_pin_func(TEST_GPIO_PIN1, GPIO0 + TEST_GPIO_PIN1);
    TEST_ASSERT(ret == 0, "Set pin function to GPIO");

    // 测试正常创建
    ret = drv_gpio_inst_create(TEST_GPIO_PIN1, &gpio);
    TEST_ASSERT(ret == 0, "Create GPIO instance");
    TEST_ASSERT(gpio != NULL, "GPIO instance not NULL");

    // 测试重复创建（应该先销毁旧的）
    drv_gpio_inst_t* gpio2 = gpio;
    ret = drv_gpio_inst_create(TEST_GPIO_PIN1, &gpio2);
    TEST_ASSERT(ret == 0, "Create GPIO instance again");

    // 销毁实例
    drv_gpio_inst_destroy(&gpio2);
    TEST_ASSERT(gpio2 == NULL, "Instance pointer cleared after destroy");

    // 测试无效引脚号
    ret = drv_gpio_inst_create(TEST_GPIO_PIN_INVALID, &gpio);
    TEST_ASSERT(ret != 0, "Create with invalid pin should fail");

    // 测试引脚功能不是GPIO的情况
    drv_fpioa_set_pin_func(TEST_GPIO_PIN2, UART2_RXD);
    ret = drv_gpio_inst_create(TEST_GPIO_PIN2, &gpio);
    TEST_ASSERT(ret != 0, "Create with non-GPIO function should fail");

    // 测试NULL指针
    drv_gpio_inst_destroy(NULL);  // 不应该崩溃
    TEST_ASSERT(1, "Destroy NULL pointer");

    return 0;
}

// 测试2：GPIO模式设置和获取
static int test_mode_operations(void)
{
    TEST_START("Mode Set and Get");

    drv_gpio_inst_t* gpio = NULL;
    gpio_drive_mode_t mode;
    int ret;

    // 准备GPIO
    drv_fpioa_set_pin_func(TEST_GPIO_PIN1, GPIO0 + TEST_GPIO_PIN1);
    ret = drv_gpio_inst_create(TEST_GPIO_PIN1, &gpio);
    TEST_ASSERT(ret == 0, "Create GPIO instance");

    // 测试设置输出模式
    ret = drv_gpio_mode_set(gpio, GPIO_DM_OUTPUT);
    TEST_ASSERT(ret == 0, "Set output mode");

    mode = drv_gpio_mode_get(gpio);
    TEST_ASSERT(mode == GPIO_DM_OUTPUT, "Get mode returns OUTPUT");

    // 测试设置输入模式
    ret = drv_gpio_mode_set(gpio, GPIO_DM_INPUT);
    TEST_ASSERT(ret == 0, "Set input mode");

    mode = drv_gpio_mode_get(gpio);
    TEST_ASSERT(mode == GPIO_DM_INPUT, "Get mode returns INPUT");

    // 测试相同模式设置（应该直接返回）
    ret = drv_gpio_mode_set(gpio, GPIO_DM_INPUT);
    TEST_ASSERT(ret == 0, "Set same mode again");

    // 测试无效模式
    ret = drv_gpio_mode_set(gpio, GPIO_DM_MAX);
    TEST_ASSERT(ret != 0, "Set invalid mode should fail");

    // 测试NULL指针
    ret = drv_gpio_mode_set(NULL, GPIO_DM_OUTPUT);
    TEST_ASSERT(ret != 0, "Set mode with NULL instance should fail");

    mode = drv_gpio_mode_get(NULL);
    TEST_ASSERT(mode == GPIO_DM_MAX, "Get mode with NULL returns invalid");

    drv_gpio_inst_destroy(&gpio);
    return 0;
}

// 测试3：GPIO值设置和获取
static int test_value_operations(void)
{
    TEST_START("Value Set and Get");

    drv_gpio_inst_t* gpio = NULL;
    gpio_pin_value_t value;
    int ret;

    // 准备GPIO
    drv_fpioa_set_pin_func(TEST_GPIO_PIN1, GPIO0 + TEST_GPIO_PIN1);
    ret = drv_gpio_inst_create(TEST_GPIO_PIN1, &gpio);
    TEST_ASSERT(ret == 0, "Create GPIO instance");

    // 设置为输出模式
    ret = drv_gpio_mode_set(gpio, GPIO_DM_OUTPUT);
    TEST_ASSERT(ret == 0, "Set output mode");

    // 测试输出高电平
    ret = drv_gpio_value_set(gpio, GPIO_PV_HIGH);
    TEST_ASSERT(ret == 0, "Set HIGH value");

    value = drv_gpio_value_get(gpio);
    TEST_ASSERT(value == GPIO_PV_HIGH, "Read back HIGH value");

    // 测试输出低电平
    ret = drv_gpio_value_set(gpio, GPIO_PV_LOW);
    TEST_ASSERT(ret == 0, "Set LOW value");

    value = drv_gpio_value_get(gpio);
    TEST_ASSERT(value == GPIO_PV_LOW, "Read back LOW value");

    // 测试相同值设置（应该直接返回）
    ret = drv_gpio_value_set(gpio, GPIO_PV_LOW);
    TEST_ASSERT(ret == 0, "Set same value again");

    // 测试toggle功能
    ret = drv_gpio_toggle(gpio);
    TEST_ASSERT(ret == 0, "Toggle GPIO");

    value = drv_gpio_value_get(gpio);
    TEST_ASSERT(value == GPIO_PV_HIGH, "Value toggled to HIGH");

    ret = drv_gpio_toggle(gpio);
    TEST_ASSERT(ret == 0, "Toggle GPIO again");

    value = drv_gpio_value_get(gpio);
    TEST_ASSERT(value == GPIO_PV_LOW, "Value toggled to LOW");

    // 测试NULL指针
    ret = drv_gpio_value_set(NULL, GPIO_PV_HIGH);
    TEST_ASSERT(ret != 0, "Set value with NULL should fail");

    value = drv_gpio_value_get(NULL);
    TEST_ASSERT(value == GPIO_PV_LOW, "Get value with NULL returns LOW");

    ret = drv_gpio_toggle(NULL);
    TEST_ASSERT(ret != 0, "Toggle with NULL should fail");

    drv_gpio_inst_destroy(&gpio);
    return 0;
}

// 测试4：获取引脚ID
static int test_get_pin_id(void)
{
    TEST_START("Get Pin ID");

    drv_gpio_inst_t* gpio = NULL;
    int ret, pin_id;

    // 测试不同的引脚
    for (int pin = 0; pin <= TEST_GPIO_PIN_MAX; pin += 10) {
        ret = drv_fpioa_set_pin_func(pin, GPIO0 + pin);
        if (ret != 0) continue;

        ret = drv_gpio_inst_create(pin, &gpio);
        if (ret != 0) continue;

        pin_id = drv_gpio_get_pin_id(gpio);
        TEST_ASSERT(pin_id == pin, "Get correct pin ID");

        drv_gpio_inst_destroy(&gpio);
    }

    // 测试NULL指针
    pin_id = drv_gpio_get_pin_id(NULL);
    TEST_ASSERT(pin_id == -1, "Get pin ID with NULL returns -1");

    return 0;
}

// 测试5：中断功能测试
static int test_interrupt_operations(void)
{
    TEST_START("Interrupt Operations");

    drv_gpio_inst_t* gpio = NULL;
    drv_gpio_inst_t *gpio_out = NULL;
    int ret;

    // 使用支持中断的引脚（0-63）
    drv_fpioa_set_pin_func(TEST_GPIO_PIN_IRQ, GPIO0 + TEST_GPIO_PIN_IRQ);
    ret = drv_gpio_inst_create(TEST_GPIO_PIN_IRQ, &gpio);
    TEST_ASSERT(ret == 0, "Create GPIO instance for IRQ");

    // 设置为输入模式
    ret = drv_gpio_mode_set(gpio, GPIO_DM_INPUT);
    TEST_ASSERT(ret == 0, "Set input mode for IRQ");

    // 测试各种中断模式
    gpio_pin_edge_t irq_modes[] = {
        GPIO_PE_RISING,
        GPIO_PE_FALLING,
        GPIO_PE_BOTH,
        GPIO_PE_HIGH,
        GPIO_PE_LOW
    };
    const char* mode_names[] = {
        "Rising Edge",
        "Falling Edge",
        "Both Edges",
        "High Level",
        "Low Level"
    };

    ret = drv_fpioa_set_pin_func(TEST_GPIO_PIN2, GPIO0 + TEST_GPIO_PIN2);
    TEST_ASSERT(ret == 0, "Set pin2 function");

    ret = drv_gpio_inst_create(TEST_GPIO_PIN2, &gpio_out);
    TEST_ASSERT(ret == 0, "Create output GPIO");

    // 设置模式
    ret = drv_gpio_mode_set(gpio_out, GPIO_DM_OUTPUT);
    TEST_ASSERT(ret == 0, "Set output mode");

    ret = drv_gpio_value_set(gpio_out, GPIO_PV_HIGH);
    TEST_ASSERT(ret == 0, "Output HIGH");

    printf("\n[INFO] Please connect GPIO%d and GPIO%d for irq test\n",
           TEST_GPIO_PIN_IRQ, TEST_GPIO_PIN2);
    printf("Press Enter to continue...\n");

    getchar();

    ret = drv_gpio_register_irq(gpio, irq_modes[1], 50, gpio_irq_handler, &irq_count);
    TEST_ASSERT(ret == 0, mode_names[1]);

    // 使能中断
    ret = drv_gpio_enable_irq(gpio);
    TEST_ASSERT(ret == 0, "Enable IRQ");

    ret = drv_gpio_value_set(gpio_out, GPIO_PV_LOW);
    TEST_ASSERT(ret == 0, "Output LOW");

    usleep(100000);
    TEST_ASSERT(irq_count == 1, "Falling Edge IRQ");

    // 禁用中断
    ret = drv_gpio_disable_irq(gpio);
    TEST_ASSERT(ret == 0, "Disable IRQ");

    ret = drv_gpio_unregister_irq(gpio);
    TEST_ASSERT(ret == 0, "Unregister gpio irq");

    // 测试防抖时间边界
    ret = drv_gpio_register_irq(gpio, GPIO_PE_RISING, 5, gpio_irq_handler, &irq_count);
    TEST_ASSERT(ret == 0, "Register with 5ms debounce (should be adjusted to 10ms)");
    ret = drv_gpio_unregister_irq(gpio);
    TEST_ASSERT(ret == 0, "Unregister gpio irq");

    ret = drv_gpio_register_irq(gpio, GPIO_PE_RISING, 1000, gpio_irq_handler, &irq_count);
    TEST_ASSERT(ret == 0, "Register with 1000ms debounce");
    ret = drv_gpio_unregister_irq(gpio);
    TEST_ASSERT(ret == 0, "Unregister gpio irq");

    // 测试无效中断模式
    ret = drv_gpio_register_irq(gpio, GPIO_PE_MAX, 50, gpio_irq_handler, &irq_count);
    TEST_ASSERT(ret != 0, "Register with invalid mode");
    ret = drv_gpio_unregister_irq(gpio);
    TEST_ASSERT(ret == 0, "Unregister gpio irq");

    // 测试重复注册（应该先注销旧的）
    ret = drv_gpio_register_irq(gpio, GPIO_PE_RISING, 50, gpio_irq_handler, &irq_count);
    TEST_ASSERT(ret == 0, "First register");

    ret = drv_gpio_register_irq(gpio, GPIO_PE_FALLING, 100, gpio_irq_handler, &irq_count);
    TEST_ASSERT(ret == 0, "Register again (should unregister first)");

    ret = drv_gpio_unregister_irq(gpio);
    TEST_ASSERT(ret == 0, "Unregister gpio irq");

    // 测试NULL指针
    ret = drv_gpio_register_irq(NULL, GPIO_PE_RISING, 50, gpio_irq_handler, &irq_count);
    TEST_ASSERT(ret != 0, "Register IRQ with NULL instance should fail");

    ret = drv_gpio_set_irq(NULL, 1);
    TEST_ASSERT(ret != 0, "Set IRQ with NULL instance should fail");

    drv_gpio_inst_destroy(&gpio);

    return 0;
}

// 测试6：边界条件测试
static int test_boundary_conditions(void)
{
    TEST_START("Boundary Conditions");

    drv_gpio_inst_t* gpio = NULL;
    int ret;

    // 测试引脚0（最小值）
    ret = drv_fpioa_set_pin_func(0, GPIO0);
    if (ret == 0) {
        ret = drv_gpio_inst_create(0, &gpio);
        TEST_ASSERT(ret == 0, "Create GPIO with pin 0");
        drv_gpio_inst_destroy(&gpio);
    }

    // 测试引脚71（最大值）
    ret = drv_fpioa_set_pin_func(TEST_GPIO_PIN_MAX, GPIO0 + TEST_GPIO_PIN_MAX);
    if (ret == 0) {
        ret = drv_gpio_inst_create(TEST_GPIO_PIN_MAX, &gpio);
        TEST_ASSERT(ret == 0, "Create GPIO with max pin");
        drv_gpio_inst_destroy(&gpio);
    }

    // 测试引脚72（超出范围）
    ret = drv_gpio_inst_create(TEST_GPIO_PIN_INVALID, &gpio);
    TEST_ASSERT(ret != 0, "Create GPIO with invalid pin should fail");

    // 测试引脚64的中断（超出中断支持范围）
    if (TEST_GPIO_PIN_MAX > 63) {
        ret = drv_fpioa_set_pin_func(64, TEST_PIN0);
        if (ret == 0) {
            ret = drv_gpio_inst_create(64, &gpio);
            if (ret == 0) {
                ret = drv_gpio_register_irq(gpio, GPIO_PE_RISING, 50, gpio_irq_handler, NULL);
                TEST_ASSERT(ret != 0, "Register IRQ on pin 64 should fail");
                drv_gpio_inst_destroy(&gpio);
            }
        }
    }

    return 0;
}

// 测试7：输入输出联动测试
static int test_io_loopback(void)
{
    TEST_START("Input/Output Loopback");

    drv_gpio_inst_t *gpio_out = NULL, *gpio_in = NULL;
    gpio_pin_value_t value;
    int ret;

    // 配置两个GPIO引脚
    ret = drv_fpioa_set_pin_func(TEST_GPIO_PIN1, GPIO0 + TEST_GPIO_PIN1);
    TEST_ASSERT(ret == 0, "Set pin1 function");

    ret = drv_fpioa_set_pin_func(TEST_GPIO_PIN2, GPIO0 + TEST_GPIO_PIN2);
    TEST_ASSERT(ret == 0, "Set pin2 function");

    // 创建实例
    ret = drv_gpio_inst_create(TEST_GPIO_PIN1, &gpio_out);
    TEST_ASSERT(ret == 0, "Create output GPIO");

    ret = drv_gpio_inst_create(TEST_GPIO_PIN2, &gpio_in);
    TEST_ASSERT(ret == 0, "Create input GPIO");

    // 设置模式
    ret = drv_gpio_mode_set(gpio_out, GPIO_DM_OUTPUT);
    TEST_ASSERT(ret == 0, "Set output mode");

    ret = drv_gpio_mode_set(gpio_in, GPIO_DM_INPUT);
    TEST_ASSERT(ret == 0, "Set input mode");

    printf("\n[INFO] Please connect GPIO%d and GPIO%d for loopback test\n",
           TEST_GPIO_PIN1, TEST_GPIO_PIN2);
    printf("Press Enter to continue...");
    getchar();

    // 测试输出高电平
    ret = drv_gpio_value_set(gpio_out, GPIO_PV_HIGH);
    TEST_ASSERT(ret == 0, "Output HIGH");

    usleep(10000);  // 10ms延时
    value = drv_gpio_value_get(gpio_in);
    TEST_ASSERT(value == GPIO_PV_HIGH, "Input reads HIGH");

    // 测试输出低电平
    ret = drv_gpio_value_set(gpio_out, GPIO_PV_LOW);
    TEST_ASSERT(ret == 0, "Output LOW");

    usleep(10000);  // 10ms延时
    value = drv_gpio_value_get(gpio_in);
    TEST_ASSERT(value == GPIO_PV_LOW, "Input reads LOW");

    // 清理
    drv_gpio_inst_destroy(&gpio_out);
    drv_gpio_inst_destroy(&gpio_in);

    return 0;
}

static int test_concurrent_gpio_fpioa_access(void)
{
    TEST_START("Concurrent GPIO/FPIOA Access");

    static const int candidate_pins[] = { 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    struct gpio_concurrency_ctx ctx[TEST_GPIO_CONCURRENCY_THREADS];
    pthread_t                   threads[TEST_GPIO_CONCURRENCY_THREADS];
    volatile int               start_flag = 0;
    int                        prepared = 0;
    int                        created_threads = 0;

    memset(ctx, 0, sizeof(ctx));
    memset(threads, 0, sizeof(threads));

    for (size_t i = 0; i < sizeof(candidate_pins) / sizeof(candidate_pins[0]); i++) {
        drv_gpio_inst_t* gpio = NULL;
        int              pin  = candidate_pins[i];
        int              ret;

        if (prepared >= TEST_GPIO_CONCURRENCY_THREADS) {
            break;
        }

        ret = drv_fpioa_set_pin_func(pin, GPIO0 + pin);
        if (ret != 0) {
            continue;
        }

        ret = drv_gpio_inst_create(pin, &gpio);
        if (ret != 0) {
            continue;
        }

        ctx[prepared].gpio       = gpio;
        ctx[prepared].pin        = pin;
        ctx[prepared].is_output  = ((prepared & 0x01) == 0);
        ctx[prepared].start_flag = &start_flag;

        ret = drv_gpio_mode_set(gpio, ctx[prepared].is_output ? GPIO_DM_OUTPUT : GPIO_DM_INPUT);
        if (ret != 0) {
            drv_gpio_inst_destroy(&gpio);
            memset(&ctx[prepared], 0, sizeof(ctx[prepared]));
            continue;
        }

        if (ctx[prepared].is_output) {
            ret = drv_gpio_value_set(gpio, GPIO_PV_LOW);
            if (ret != 0) {
                drv_gpio_inst_destroy(&gpio);
                memset(&ctx[prepared], 0, sizeof(ctx[prepared]));
                continue;
            }
        }

        prepared++;
    }

    if (prepared & 0x01) {
        drv_gpio_inst_destroy(&ctx[prepared - 1].gpio);
        memset(&ctx[prepared - 1], 0, sizeof(ctx[prepared - 1]));
        prepared--;
    }

    TEST_ASSERT(prepared >= TEST_GPIO_CONCURRENCY_MIN_THREADS, "Prepared enough GPIOs for concurrent stress test");

    printf("[INFO] Starting %d concurrent GPIO workers, %d loops each\n", prepared, TEST_GPIO_CONCURRENCY_LOOPS);

    for (int i = 0; i < prepared; i++) {
        int ret = pthread_create(&threads[i], NULL, gpio_concurrency_worker, &ctx[i]);
        if (ret != 0) {
            printf("[INFO] Failed to create worker for pin %d, ret=%d\n", ctx[i].pin, ret);
            ctx[i].failures++;
            break;
        }
        created_threads++;
    }

    start_flag = 1;

    for (int i = 0; i < created_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < prepared; i++) {
        if (ctx[i].failures != 0 || ctx[i].completed_loops != TEST_GPIO_CONCURRENCY_LOOPS) {
            printf("[INFO] pin=%d output=%d loops=%d failures=%d\n",
                   ctx[i].pin,
                   ctx[i].is_output,
                   ctx[i].completed_loops,
                   ctx[i].failures);
        }
    }

    for (int i = 0; i < prepared; i++) {
        drv_gpio_inst_destroy(&ctx[i].gpio);
    }

    for (int i = 0; i < prepared; i++) {
        TEST_ASSERT(ctx[i].failures == 0, "Concurrent worker completed without GPIO/FPIOA error");
        TEST_ASSERT(ctx[i].completed_loops == TEST_GPIO_CONCURRENCY_LOOPS,
                    "Concurrent worker finished every planned iteration");
    }

    return 0;
}

// 主测试函数
int main(int argc, char* argv[])
{
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--skip-manual")) {
            g_run_manual_tests = 0;
        }
    }

    printf("\n==================================================\n");
    printf("GPIO HAL Comprehensive Test Program\n");
    printf("==================================================\n");

    // 忽略一些信号，避免测试被中断
    signal(SIGPIPE, SIG_IGN);

    // 执行所有测试
    int test_count = 0;

    test_create_destroy();
    test_count++;
    test_mode_operations();
    test_count++;
    test_value_operations();
    test_count++;
    test_get_pin_id();
    test_count++;
    test_boundary_conditions();
    test_count++;
    test_concurrent_gpio_fpioa_access();
    test_count++;

    if (g_run_manual_tests) {
        test_interrupt_operations();
        test_count++;
        test_io_loopback();
        test_count++;
    } else {
        printf("[INFO] Manual IRQ/loopback tests skipped by --skip-manual\n");
    }

    // 统计结果
    printf("\n==================================================\n");
    printf("Test Summary:\n");
    printf("  Total Tests: %d\n", test_count);
    printf("  Passed: %d\n", test_passed);
    printf("  Failed: %d\n", test_failed);
    printf("==================================================\n");

    return (test_failed > 0) ? -1 : 0;
}
