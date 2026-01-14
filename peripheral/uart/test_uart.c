#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "drv_fpioa.h"
#include "drv_uart.h"

#define UART2_TX_PIN    11
#define UART2_RX_PIN    12
#define UART3_TX_PIN    50
#define UART3_RX_PIN    51
#define TEST_ITERATIONS 3
#define TEST_MSG_SIZE   128

typedef struct {
    uint32_t    baud_rate;
    uint8_t     parity;
    uint8_t     stop_bits;
    const char* parity_name;
    const char* stop_bits_name;
    uint16_t    buffer_size;
} uart_test_config;

const uart_test_config test_configs[]
    = { { 9600, PARITY_NONE, STOP_BITS_1, "NONE", "1", 1024 },   { 19200, PARITY_NONE, STOP_BITS_1, "NONE", "1", 0 },
        { 38400, PARITY_NONE, STOP_BITS_1, "NONE", "1", 2048 },  { 57600, PARITY_NONE, STOP_BITS_1, "NONE", "1", 2048 },
        { 115200, PARITY_NONE, STOP_BITS_1, "NONE", "1", 3072 }, { 230400, PARITY_NONE, STOP_BITS_1, "NONE", "1", 3072 },
        { 9600, PARITY_ODD, STOP_BITS_1, "ODD", "1", 0 },        { 19200, PARITY_EVEN, STOP_BITS_1, "EVEN", "1", 0 },
        { 38400, PARITY_ODD, STOP_BITS_2, "ODD", "1.5", 8192 },  { 57600, PARITY_EVEN, STOP_BITS_3, "EVEN", "2", 8192 } };

/**
 * @brief Configure FPIOA pins for UART testing
 */
void configure_fpioa_pins()
{
    // Configure UART2 pins
    drv_fpioa_set_pin_func(UART2_TX_PIN, UART2_TXD);
    drv_fpioa_set_pin_func(UART2_RX_PIN, UART2_RXD);

    // Configure UART3 pins
    drv_fpioa_set_pin_func(UART3_TX_PIN, UART3_TXD);
    drv_fpioa_set_pin_func(UART3_RX_PIN, UART3_RXD);

    // Set pull-up on RX pins
    drv_fpioa_set_pin_pu(UART2_RX_PIN, 1);
    drv_fpioa_set_pin_pu(UART3_RX_PIN, 1);

    // Set appropriate drive strength
    drv_fpioa_set_pin_ds(UART2_TX_PIN, 3); // Medium drive strength
    drv_fpioa_set_pin_ds(UART3_TX_PIN, 3);
}

/**
 * @brief Initialize UART instances with specific configuration
 */
int init_uarts_with_config(drv_uart_inst_t** uart2, drv_uart_inst_t** uart3, const uart_test_config* config)
{
    if (config->buffer_size) {
        drv_uart_configure_buffer_size(2, config->buffer_size);
        drv_uart_configure_buffer_size(3, config->buffer_size);
    }

    // Initialize UART2 (TX -> UART3 RX)
    int ret = drv_uart_inst_create(2, uart2);
    if (ret != 0) {
        printf("UART2 creation failed: %d\n", ret);
        return -1;
    }

    // Initialize UART3 (RX <- UART2 TX)
    ret = drv_uart_inst_create(3, uart3);
    if (ret != 0) {
        printf("UART3 creation failed: %d\n", ret);
        drv_uart_inst_destroy(uart2);
        return -1;
    }

    struct uart_configure curr_cfg;

    if (0x00 != drv_uart_get_config(*uart2, &curr_cfg)) {
        printf("get uart configure failed.\n");

        drv_uart_inst_destroy(uart2);
        drv_uart_inst_destroy(uart3);
        return -1;
    }

    struct uart_configure cfg = { .baud_rate = config->baud_rate,
                                  .data_bits = DATA_BITS_8,
                                  .stop_bits = config->stop_bits,
                                  .parity    = config->parity,
                                  .bit_order = BIT_ORDER_LSB,
                                  .invert    = NRZ_NORMAL,
                                  .bufsz     = curr_cfg.bufsz };

    // Configure both UARTs with same settings
    if (drv_uart_set_config(*uart2, &cfg) != 0 || drv_uart_set_config(*uart3, &cfg) != 0) {
        printf("UART configuration failed for %d baud %s parity %s stop bits\n", config->baud_rate, config->parity_name,
               config->stop_bits_name);
        drv_uart_inst_destroy(uart2);
        drv_uart_inst_destroy(uart3);
        return -1;
    }

    return 0;
}

/**
 * @brief Clear UART buffers
 */
void clear_uart_buffers(drv_uart_inst_t* uart)
{
    while (drv_uart_recv_available(uart) > 0) {
        uint8_t dummy;
        drv_uart_read(uart, &dummy, 1);
    }
}

/**
 * @brief Test basic read/write between UART2 and UART3
 */
void test_basic_io(drv_uart_inst_t* uart2, drv_uart_inst_t* uart3, const uart_test_config* config)
{
    printf("\n=== Basic I/O Tests (%d baud, %s parity, %s stop) ===\n", config->baud_rate, config->parity_name,
           config->stop_bits_name);

    uint8_t      tx_buf[TEST_MSG_SIZE];
    uint8_t      rx_buf[TEST_MSG_SIZE];
    const size_t test_sizes[] = { 1, 16, 64, TEST_MSG_SIZE };

    for (size_t i = 0; i < sizeof(test_sizes) / sizeof(test_sizes[0]); i++) {
        size_t len = test_sizes[i];

        // Fill buffer with test pattern
        for (size_t j = 0; j < len; j++) {
            tx_buf[j] = (uint8_t)(j % 256);
        }

        // Clear receive buffer
        memset(rx_buf, 0, sizeof(rx_buf));

        // Test UART2 -> UART3 direction
        ssize_t written = drv_uart_write(uart2, tx_buf, len);
        if (written != (ssize_t)len) {
            printf("  UART2->UART3 %zu-byte write failed: wrote %zd/%zu\n", len, written, len);
            continue;
        }

        // Calculate transmission time with margin
        int bit_count = (1 + 8 + (config->parity != PARITY_NONE ? 1 : 0)
                         + (config->stop_bits == STOP_BITS_1       ? 1
                                : config->stop_bits == STOP_BITS_2 ? 15
                                                                   : 2))
            * len;
        int delay_us = (bit_count * 1000000) / config->baud_rate + 10000;
        usleep(delay_us);

        // Read data from UART3
        ssize_t read = drv_uart_read(uart3, rx_buf, len);
        if (read != (ssize_t)len) {
            printf("  UART2->UART3 %zu-byte read failed: read %zd/%zu\n", len, read, len);
            clear_uart_buffers(uart3);
            continue;
        }

        // Verify data
        if (memcmp(tx_buf, rx_buf, len) != 0) {
            printf("  UART2->UART3 %zu-byte verify failed: data mismatch\n", len);
        } else {
            printf("  UART2->UART3 %4zu-byte: PASS\n", len);
        }
        clear_uart_buffers(uart3);

        // Test UART3 -> UART2 direction
        written = drv_uart_write(uart3, tx_buf, len);
        if (written != (ssize_t)len) {
            printf("  UART3->UART2 %zu-byte write failed: wrote %zd/%zu\n", len, written, len);
            continue;
        }

        usleep(delay_us);

        read = drv_uart_read(uart2, rx_buf, len);
        if (read != (ssize_t)len) {
            printf("  UART3->UART2 %zu-byte read failed: read %zd/%zu\n", len, read, len);
            clear_uart_buffers(uart2);
            continue;
        }

        if (memcmp(tx_buf, rx_buf, len) != 0) {
            printf("  UART3->UART2 %zu-byte verify failed: data mismatch\n", len);
        } else {
            printf("  UART3->UART2 %4zu-byte: PASS\n", len);
        }
        clear_uart_buffers(uart2);
    }
}

/**
 * @brief Test polling functionality
 */
void test_polling(drv_uart_inst_t* uart2, drv_uart_inst_t* uart3, const uart_test_config* config)
{
    printf("\n=== Polling Tests (%d baud, %s parity, %s stop) ===\n", config->baud_rate, config->parity_name,
           config->stop_bits_name);

    uint8_t         test_byte = 0x55;
    struct timespec start, end;

    // Test 1: Poll with no data (non-blocking)
    int ret = drv_uart_poll(uart3, 0);
    printf("  UART3 poll no data (non-block): %s\n", ret == 0 ? "PASS" : "FAIL");

    // Test 2: Poll with timeout (no data)
    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = drv_uart_poll(uart3, 100); // 100ms timeout
    clock_gettime(CLOCK_MONOTONIC, &end);

    long elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

    printf("  UART3 poll timeout (100ms): %s (actual %ldms)\n",
           (ret == 0 && elapsed_ms >= 90 && elapsed_ms <= 110) ? "PASS" : "FAIL", elapsed_ms);

    // Test 3: Poll with data available (UART2 -> UART3)
    drv_uart_write(uart2, &test_byte, 1);

    // Calculate transmission time with margin
    int bit_count = (1 + 8 + (config->parity != PARITY_NONE ? 1 : 0)
                     + (config->stop_bits == STOP_BITS_1       ? 1
                            : config->stop_bits == STOP_BITS_2 ? 15
                                                               : 2));
    int delay_us  = (bit_count * 1000000) / config->baud_rate + 10000;
    usleep(delay_us);

    ret = drv_uart_poll(uart3, 0);
    printf("  UART3 poll with data: %s\n", ret > 0 ? "PASS" : "FAIL");

    // Clean up
    uint8_t dummy;
    drv_uart_read(uart3, &dummy, 1);

    // Repeat tests for UART2
    ret = drv_uart_poll(uart2, 0);
    printf("  UART2 poll no data (non-block): %s\n", ret == 0 ? "PASS" : "FAIL");

    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = drv_uart_poll(uart2, 100);
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

    printf("  UART2 poll timeout (100ms): %s (actual %ldms)\n",
           (ret == 0 && elapsed_ms >= 90 && elapsed_ms <= 110) ? "PASS" : "FAIL", elapsed_ms);

    drv_uart_write(uart3, &test_byte, 1);
    usleep(delay_us);

    ret = drv_uart_poll(uart2, 0);
    printf("  UART2 poll with data: %s\n", ret > 0 ? "PASS" : "FAIL");
    drv_uart_read(uart2, &dummy, 1);
}

/**
 * @brief Test receive available functionality
 */
void test_receive_available(drv_uart_inst_t* uart2, drv_uart_inst_t* uart3, const uart_test_config* config)
{
    printf("\n=== Receive Available Tests (%d baud, %s parity, %s stop) ===\n", config->baud_rate, config->parity_name,
           config->stop_bits_name);

    uint8_t test_data[16] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF };

    // Test UART3 available bytes
    ssize_t avail = drv_uart_recv_available(uart3);
    printf("  UART3 available empty: %s\n", avail == 0 ? "PASS" : "FAIL");

    // Send data from UART2 to UART3
    drv_uart_write(uart2, test_data, sizeof(test_data));

    // Calculate transmission time
    int bit_count = (1 + 8 + (config->parity != PARITY_NONE ? 1 : 0)
                     + (config->stop_bits == STOP_BITS_1       ? 1
                            : config->stop_bits == STOP_BITS_2 ? 15
                                                               : 2))
        * sizeof(test_data);
    int delay_us = (bit_count * 1000000) / config->baud_rate + 10000;
    usleep(delay_us);

    avail = drv_uart_recv_available(uart3);
    printf("  UART3 available with data: %s (%zd bytes)\n", avail == sizeof(test_data) ? "PASS" : "FAIL", avail);

    // Clean up
    uint8_t dummy[16];
    drv_uart_read(uart3, dummy, sizeof(dummy));

    // Test UART2 available bytes
    avail = drv_uart_recv_available(uart2);
    printf("  UART2 available empty: %s\n", avail == 0 ? "PASS" : "FAIL");

    // Send data from UART3 to UART2
    drv_uart_write(uart3, test_data, sizeof(test_data));
    usleep(delay_us);

    avail = drv_uart_recv_available(uart2);
    printf("  UART2 available with data: %s (%zd bytes)\n", avail == sizeof(test_data) ? "PASS" : "FAIL", avail);
    drv_uart_read(uart2, dummy, sizeof(dummy));
}

/**
 * @brief Test error conditions
 */
void test_error_conditions()
{
    printf("\n=== Error Condition Tests ===\n");

    // Test 1: NULL instance
    ssize_t ret = drv_uart_read(NULL, NULL, 0);
    printf("  NULL instance read: %s\n", ret == -1 ? "PASS" : "FAIL");

    ret = drv_uart_write(NULL, NULL, 0);
    printf("  NULL instance write: %s\n", ret == -1 ? "PASS" : "FAIL");

    // Test 2: Invalid FD
    drv_uart_inst_t dummy_inst = { 0 };
    dummy_inst.fd              = -1;
    ret                        = drv_uart_read(&dummy_inst, NULL, 0);
    printf("  Invalid FD read: %s\n", ret == -1 ? "PASS" : "FAIL");

    ret = drv_uart_write(&dummy_inst, NULL, 0);
    printf("  Invalid FD write: %s\n", ret == -1 ? "PASS" : "FAIL");

    // Test 3: NULL buffers
    drv_uart_inst_t* uart2 = NULL;
    drv_uart_inst_create(2, &uart2);
    if (uart2) {
        ret = drv_uart_read(uart2, NULL, 10);
        printf("  NULL read buffer: %s\n", ret == -1 ? "PASS" : "FAIL");

        ret = drv_uart_write(uart2, NULL, 10);
        printf("  NULL write buffer: %s\n", ret == -1 ? "PASS" : "FAIL");
        drv_uart_inst_destroy(&uart2);
    }

    // Test 4: Zero length
    uint8_t dummy;
    if (uart2) {
        ret = drv_uart_read(uart2, &dummy, 0);
        printf("  Zero-length read: %s\n", ret == 0 ? "PASS" : "FAIL");

        ret = drv_uart_write(uart2, &dummy, 0);
        printf("  Zero-length write: %s\n", ret == 0 ? "PASS" : "FAIL");
    }
}

/**
 * @brief Stress test with random data
 */
void test_stress(drv_uart_inst_t* uart2, drv_uart_inst_t* uart3, const uart_test_config* config)
{
    printf("\n=== Stress Test (%d baud, %s parity, %s stop, %d iterations) ===\n", config->baud_rate, config->parity_name,
           config->stop_bits_name, TEST_ITERATIONS);

    uint8_t tx_buf[TEST_MSG_SIZE];
    uint8_t rx_buf[TEST_MSG_SIZE];
    int     failures = 0;

    srand(time(NULL));

    for (int i = 0; i < TEST_ITERATIONS; i++) {
        // Generate random test data
        size_t len = 1 + (rand() % (TEST_MSG_SIZE - 1));
        for (size_t j = 0; j < len; j++) {
            tx_buf[j] = (uint8_t)rand();
        }

        // Clear receive buffer
        memset(rx_buf, 0, sizeof(rx_buf));

        // Test UART2 -> UART3 direction
        ssize_t written = drv_uart_write(uart2, tx_buf, len);
        if (written != (ssize_t)len) {
            printf("  UART2->UART3 Iter %d: write failed (%zd/%zu)\n", i, written, len);
            failures++;
            continue;
        }

        // Calculate transmission time
        int bit_count = (1 + 8 + (config->parity != PARITY_NONE ? 1 : 0)
                         + (config->stop_bits == STOP_BITS_1       ? 1
                                : config->stop_bits == STOP_BITS_2 ? 15
                                                                   : 2))
            * len;
        int delay_us = (bit_count * 1000000) / config->baud_rate + 10000;
        usleep(delay_us);

        // Read data from UART3
        ssize_t read = drv_uart_read(uart3, rx_buf, len);
        if (read != (ssize_t)len) {
            printf("  UART2->UART3 Iter %d: read failed (%zd/%zu)\n", i, read, len);
            failures++;
            clear_uart_buffers(uart3);
            continue;
        }

        // Verify data
        if (memcmp(tx_buf, rx_buf, len) != 0) {
            printf("  UART2->UART3 Iter %d: data mismatch\n", i);
            failures++;
        }
        clear_uart_buffers(uart3);

        // Test UART3 -> UART2 direction
        written = drv_uart_write(uart3, tx_buf, len);
        if (written != (ssize_t)len) {
            printf("  UART3->UART2 Iter %d: write failed (%zd/%zu)\n", i, written, len);
            failures++;
            continue;
        }

        usleep(delay_us);

        read = drv_uart_read(uart2, rx_buf, len);
        if (read != (ssize_t)len) {
            printf("  UART3->UART2 Iter %d: read failed (%zd/%zu)\n", i, read, len);
            failures++;
            clear_uart_buffers(uart2);
            continue;
        }

        if (memcmp(tx_buf, rx_buf, len) != 0) {
            printf("  UART3->UART2 Iter %d: data mismatch\n", i);
            failures++;
        }
        clear_uart_buffers(uart2);
    }

    printf("  Stress test complete: %d/%d passed\n", TEST_ITERATIONS * 2 - failures, TEST_ITERATIONS * 2);
}
/**
 * @brief Test configuration get/set with proper verification
 */
int test_config_operations(drv_uart_inst_t* uart2, drv_uart_inst_t* uart3, const uart_test_config* current_config)
{
    printf("\n=== Configuration Tests ===\n");
    int                   config_failures = 0;
    struct uart_configure cfg, get_cfg;

    // Test 1: Verify current configuration matches what we set
    if (drv_uart_get_config(uart2, &get_cfg) == 0) {
        if (get_cfg.baud_rate == current_config->baud_rate && get_cfg.parity == current_config->parity
            && get_cfg.stop_bits == current_config->stop_bits) {
            printf("  UART2 current config verify: PASS\n");
        } else {
            printf("  UART2 current config verify: FAIL (mismatch), baud_rate %d != %d, parity %d != %d, stop_bits %d != %d\n",
                   get_cfg.baud_rate, current_config->baud_rate, get_cfg.parity, current_config->parity, get_cfg.stop_bits,
                   current_config->stop_bits);
            config_failures++;
        }
    } else {
        printf("  UART2 get config: FAIL\n");
        config_failures++;
    }

    // Test 2: Set and verify new configuration
    cfg                 = get_cfg;
    uint32_t new_baud   = (cfg.baud_rate == 115200) ? 57600 : 115200;
    uint8_t  new_parity = (cfg.parity == PARITY_NONE) ? PARITY_ODD : PARITY_NONE;
    cfg.baud_rate       = new_baud;
    cfg.parity          = new_parity;

    if (drv_uart_set_config(uart2, &cfg) == 0) {
        if (drv_uart_get_config(uart2, &get_cfg) == 0) {
            if (get_cfg.baud_rate == new_baud && get_cfg.parity == new_parity) {
                printf("  UART2 new config verify: PASS\n");
            } else {
                printf("  UART2 new config verify: FAIL (expected %d/%d, got %d/%d)\n", new_baud, new_parity, get_cfg.baud_rate,
                       get_cfg.parity);
                config_failures++;
            }
        } else {
            printf("  UART2 get new config: FAIL\n");
            config_failures++;
        }
    } else {
        printf("  UART2 set config: FAIL\n");
        config_failures++;
    }

    // Repeat for UART3
    if (drv_uart_get_config(uart3, &get_cfg) == 0) {
        if (get_cfg.baud_rate == current_config->baud_rate && get_cfg.parity == current_config->parity
            && get_cfg.stop_bits == current_config->stop_bits) {
            printf("  UART3 current config verify: PASS\n");
        } else {
            printf("  UART3 current config verify: FAIL (mismatch), baud_rate %d != %d, parity %d != %d, stop_bits %d != %d\n",
                   get_cfg.baud_rate, current_config->baud_rate, get_cfg.parity, current_config->parity, get_cfg.stop_bits,
                   current_config->stop_bits);
            config_failures++;
        }
    } else {
        printf("  UART3 get config: FAIL\n");
        config_failures++;
    }

    if (drv_uart_set_config(uart3, &cfg) == 0) {
        if (drv_uart_get_config(uart3, &get_cfg) == 0) {
            if (get_cfg.baud_rate == new_baud && get_cfg.parity == new_parity) {
                printf("  UART3 new config verify: PASS\n");
            } else {
                printf("  UART3 new config verify: FAIL (expected %d/%d, got %d/%d)\n", new_baud, new_parity, get_cfg.baud_rate,
                       get_cfg.parity);
                config_failures++;
            }
        } else {
            printf("  UART3 get new config: FAIL\n");
            config_failures++;
        }
    } else {
        printf("  UART3 set config: FAIL\n");
        config_failures++;
    }

    // Restore original configuration
    cfg.baud_rate = current_config->baud_rate;
    cfg.parity    = current_config->parity;
    cfg.stop_bits = current_config->stop_bits;
    drv_uart_set_config(uart2, &cfg);
    drv_uart_set_config(uart3, &cfg);

    return config_failures;
}

int main()
{
    printf("Starting Comprehensive UART Cross-Test (UART2 <-> UART3)\n");
    printf("Pin configuration:\n");
    printf("  UART2 TX -> Pin %d\n", UART2_TX_PIN);
    printf("  UART2 RX -> Pin %d\n", UART2_RX_PIN);
    printf("  UART3 TX -> Pin %d\n", UART3_TX_PIN);
    printf("  UART3 RX -> Pin %d\n", UART3_RX_PIN);
    printf("\n");

    // Configure FPIOA pins first
    configure_fpioa_pins();

    // Run error condition tests once (they don't need UART instances)
    test_error_conditions();

    int total_configs  = sizeof(test_configs) / sizeof(test_configs[0]);
    int passed_configs = 0;
    int total_failures = 0;

    for (int i = 0; i < total_configs; i++) {
        printf("\n=== Testing Configuration %d/%d ===\n", i + 1, total_configs);
        printf("Baud Rate: %d, Parity: %s, Stop Bits: %s\n", test_configs[i].baud_rate, test_configs[i].parity_name,
               test_configs[i].stop_bits_name);

        drv_uart_inst_t* uart2 = NULL;
        drv_uart_inst_t* uart3 = NULL;
        if (init_uarts_with_config(&uart2, &uart3, &test_configs[i]) != 0) {
            total_failures++;
            continue;
        }

        int config_failures = 0;

        test_basic_io(uart2, uart3, &test_configs[i]);
        test_polling(uart2, uart3, &test_configs[i]);
        test_receive_available(uart2, uart3, &test_configs[i]);
        config_failures += test_config_operations(uart2, uart3, &test_configs[i]);
        test_stress(uart2, uart3, &test_configs[i]);

        drv_uart_inst_destroy(&uart2);
        drv_uart_inst_destroy(&uart3);

        if (config_failures == 0) {
            passed_configs++;
        } else {
            total_failures += config_failures;
        }
    }

    printf("\nTest Complete: %d/%d configurations passed with %d individual test failures\n", passed_configs, total_configs,
           total_failures);

    return (total_failures == 0) ? 0 : 1;
}
