/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "drv_fpioa.h"
#include "drv_i2c.h"
#include "drv_spi.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define TEST_I2C_BUS        4
#define TEST_I2C_SCL_PIN    46
#define TEST_I2C_SDA_PIN    47
#define TEST_I2C_FREQ       100000U
#define TEST_I2C_TIMEOUT_MS 50U

#define TEST_SPI_BUS       1
#define TEST_SPI_CLK_PIN   15
#define TEST_SPI_D0_PIN    16
#define TEST_SPI_D1_PIN    17
#define TEST_SPI_BAUDRATE  1000000U

static int total_checks;
static int total_failures;

#define CHECK(condition, format, ...)                            \
    do {                                                         \
        int passed = !!(condition);                              \
        total_checks++;                                          \
        if (!passed)                                             \
            total_failures++;                                    \
        printf("%s: " format "\n", passed ? "PASS" : "FAIL", \
               ##__VA_ARGS__);                                   \
    } while (0)

static uint64_t elapsed_ms(const struct timespec* start,
                           const struct timespec* end)
{
    int64_t seconds = end->tv_sec - start->tv_sec;
    int64_t nanoseconds = end->tv_nsec - start->tv_nsec;

    return (uint64_t)(seconds * 1000 + nanoseconds / 1000000);
}

static void test_i2c(void)
{
    drv_i2c_inst_t* inst = NULL;
    uint8_t byte = 0x5a;
    i2c_msg_t msg = {
        .addr = 0x7f,
        .flags = DRV_I2C_WR,
        .len = 1,
        .buf = &byte,
    };
    struct timespec start;
    struct timespec end;
    int ret;
    int lifecycle_ok = 1;

    printf("\n=== I2C API and recovery ===\n");
    CHECK(drv_fpioa_set_pin_func(TEST_I2C_SCL_PIN, IIC4_SCL) == 0 &&
              drv_fpioa_set_pin_func(TEST_I2C_SDA_PIN, IIC4_SDA) == 0 &&
              drv_fpioa_set_pin_pu(TEST_I2C_SCL_PIN, 1) == 0 &&
              drv_fpioa_set_pin_pu(TEST_I2C_SDA_PIN, 1) == 0,
          "configure I2C%d pins %d/%d", TEST_I2C_BUS,
          TEST_I2C_SCL_PIN, TEST_I2C_SDA_PIN);

    drv_i2c_inst_destroy(NULL);
    drv_i2c_inst_destroy(&inst);
    CHECK(inst == NULL, "NULL and empty destroy are safe");
    CHECK(drv_i2c_inst_create(-1, TEST_I2C_FREQ, TEST_I2C_TIMEOUT_MS,
                              0xff, 0xff, &inst) != 0 && inst == NULL,
          "reject negative controller ID");
    CHECK(drv_i2c_inst_create(TEST_I2C_BUS, 0, TEST_I2C_TIMEOUT_MS,
                              0xff, 0xff, &inst) != 0 && inst == NULL,
          "reject zero frequency");
    CHECK(drv_i2c_inst_create(TEST_I2C_BUS, TEST_I2C_FREQ, 0,
                              0xff, 0xff, &inst) != 0 && inst == NULL,
          "reject zero timeout");
    CHECK(drv_i2c_inst_create(TEST_I2C_BUS, TEST_I2C_FREQ,
                              TEST_I2C_TIMEOUT_MS, 0xff, 0xff,
                              NULL) != 0,
          "reject NULL output pointer");

    CHECK(drv_i2c_inst_create(TEST_I2C_BUS, TEST_I2C_FREQ,
                              TEST_I2C_TIMEOUT_MS, 0xff, 0xff,
                              &inst) == 0 && inst != NULL,
          "open and configure I2C%d", TEST_I2C_BUS);
    if (!inst)
        return;

    CHECK(drv_i2c_master_get_freq(inst) == TEST_I2C_FREQ &&
              drv_i2c_master_get_timeout_ms(inst) == TEST_I2C_TIMEOUT_MS,
          "creation applies frequency and timeout");
    CHECK(drv_i2c_set_freq(inst, 0) != 0,
          "reject zero runtime frequency");
    CHECK(drv_i2c_set_timeout(inst, 0) != 0,
          "reject zero runtime timeout");
    CHECK(drv_i2c_transfer(inst, NULL, 0) == 0,
          "zero-message transfer is a no-op");
    CHECK(drv_i2c_transfer(inst, NULL, 1) != 0,
          "reject NULL message array");
    CHECK(drv_i2c_transfer(inst, &msg, -1) != 0,
          "reject negative message count");

    msg.buf = NULL;
    CHECK(drv_i2c_transfer(inst, &msg, 1) != 0,
          "reject NULL non-empty message buffer");
    msg.buf = (uint8_t*)(uintptr_t)1;
    CHECK(drv_i2c_transfer(inst, &msg, 1) != 0,
          "reject inaccessible message buffer");
    msg.buf = &byte;

    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = drv_i2c_transfer(inst, &msg, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    CHECK(ret != 0 && elapsed_ms(&start, &end) < 1000,
          "unaddressed target returns bounded error (%llu ms)",
          (unsigned long long)elapsed_ms(&start, &end));
    CHECK(drv_i2c_set_freq(inst, 400000U) == 0,
          "controller accepts reconfiguration after failed transfer");

    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = drv_i2c_transfer(inst, &msg, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    CHECK(ret != 0 && elapsed_ms(&start, &end) < 1000,
          "controller remains responsive after failure (%llu ms)",
          (unsigned long long)elapsed_ms(&start, &end));

    drv_i2c_inst_destroy(&inst);
    CHECK(inst == NULL, "destroy clears I2C handle");

    for (int i = 0; i < 100; i++) {
        if (drv_i2c_inst_create(TEST_I2C_BUS, TEST_I2C_FREQ,
                                TEST_I2C_TIMEOUT_MS, 0xff, 0xff,
                                &inst) != 0 || inst == NULL) {
            lifecycle_ok = 0;
            break;
        }
        drv_i2c_inst_destroy(&inst);
        if (inst != NULL) {
            lifecycle_ok = 0;
            break;
        }
    }
    CHECK(lifecycle_ok, "100 I2C open/close cycles");
    if (inst)
        drv_i2c_inst_destroy(&inst);

    printf("SKIP: verified I2C payload requires an attached target or "
           "master/slave loopback wiring\n");
}

static void test_spi_loopback(drv_spi_inst_t inst)
{
    static const size_t lengths[] = {
        1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
    };
    uint8_t tx[255];
    uint8_t rx[255];
    int boundary_ok = 1;
    int stress_ok = 1;

    for (size_t test = 0; test < sizeof(lengths) / sizeof(lengths[0]); test++) {
        size_t length = lengths[test];

        for (size_t i = 0; i < length; i++)
            tx[i] = (uint8_t)(test * 29U + i * 17U + 0x5aU);
        memset(rx, 0, length);
        if (drv_spi_transfer(inst, tx, rx, length, true) != (int)length ||
            memcmp(tx, rx, length) != 0) {
            boundary_ok = 0;
            break;
        }
    }
    CHECK(boundary_ok, "SPI loopback boundary lengths 1..255");

    for (int iteration = 0; iteration < 100 && stress_ok; iteration++) {
        for (size_t i = 0; i < sizeof(tx); i++)
            tx[i] = (uint8_t)(iteration * 13 + (int)i * 7);
        memset(rx, 0, sizeof(rx));
        if (drv_spi_transfer(inst, tx, rx, sizeof(tx), true) !=
                (int)sizeof(tx) ||
            memcmp(tx, rx, sizeof(tx)) != 0)
            stress_ok = 0;
    }
    CHECK(stress_ok, "100 verified SPI loopback payloads");
}

static void test_spi(int loopback)
{
    drv_spi_inst_t inst = NULL;
    struct rt_qspi_message empty_msg;
    struct rt_qspi_message invalid_msg;
    uint8_t tx[16];
    int lifecycle_ok = 1;

    printf("\n=== SPI API and recovery ===\n");
    CHECK(drv_fpioa_set_pin_func(TEST_SPI_CLK_PIN, QSPI0_CLK) == 0 &&
              drv_fpioa_set_pin_func(TEST_SPI_D0_PIN, QSPI0_D0) == 0 &&
              (!loopback ||
               drv_fpioa_set_pin_func(TEST_SPI_D1_PIN, QSPI0_D1) == 0),
          "configure SPI%d clock/data pins %d/%d%s", TEST_SPI_BUS,
          TEST_SPI_CLK_PIN, TEST_SPI_D0_PIN,
          loopback ? "/17" : "");

    drv_spi_inst_destroy(NULL);
    drv_spi_inst_destroy(&inst);
    CHECK(inst == NULL, "NULL and empty destroy are safe");
    CHECK(drv_spi_inst_create(-1, true, SPI_HAL_MODE_0,
                              TEST_SPI_BAUDRATE, 8, -1,
                              SPI_HAL_DATA_LINE_1, &inst) != 0 && inst == NULL,
          "reject negative controller ID");
    CHECK(drv_spi_inst_create(TEST_SPI_BUS, true, SPI_HAL_MODE_0,
                              0, 8, -1, SPI_HAL_DATA_LINE_1,
                              &inst) != 0 && inst == NULL,
          "reject zero baudrate");
    CHECK(drv_spi_inst_create(TEST_SPI_BUS, true, SPI_HAL_MODE_0,
                              TEST_SPI_BAUDRATE, 3, -1,
                              SPI_HAL_DATA_LINE_1, &inst) != 0 && inst == NULL,
          "reject invalid data width");
    CHECK(drv_spi_inst_create(TEST_SPI_BUS, true, SPI_HAL_MODE_0,
                              TEST_SPI_BAUDRATE, 8, -1, 3,
                              &inst) != 0 && inst == NULL,
          "reject invalid data-line count");
    CHECK(drv_spi_inst_create(TEST_SPI_BUS, true, SPI_HAL_MODE_0,
                              TEST_SPI_BAUDRATE, 8, -1,
                              SPI_HAL_DATA_LINE_1, NULL) != 0,
          "reject NULL output pointer");

    CHECK(drv_spi_inst_create(TEST_SPI_BUS, true, SPI_HAL_MODE_0,
                              TEST_SPI_BAUDRATE, 8, -1,
                              SPI_HAL_DATA_LINE_1, &inst) == 0 && inst != NULL,
          "open and configure SPI%d", TEST_SPI_BUS);
    if (!inst)
        return;

    CHECK(drv_spi_transfer(inst, NULL, NULL, 0, true) == 0,
          "zero-length transfer is a no-op");
    CHECK(drv_spi_transfer(inst, NULL, NULL, 1, true) < 0,
          "reject transfer without a buffer");
    CHECK(drv_spi_transfer_message(inst, NULL) < 0,
          "reject NULL message pointer");

    memset(&empty_msg, 0, sizeof(empty_msg));
    CHECK(drv_spi_transfer_message(inst, &empty_msg) == 0,
          "zero-length message is a no-op");

    memset(&invalid_msg, 0, sizeof(invalid_msg));
    invalid_msg.parent.send_buf = (void*)(uintptr_t)1;
    invalid_msg.parent.length = 1;
    invalid_msg.parent.cs_take = 1;
    invalid_msg.parent.cs_release = 1;
    CHECK(drv_spi_transfer_message(inst, &invalid_msg) == 0,
          "reject inaccessible transfer buffer without a fault");

    CHECK(drv_spi_set_baudrate(inst, 0) != 0,
          "reject zero runtime baudrate");
    CHECK(drv_spi_set_datamode(inst, 4) != 0,
          "reject invalid runtime mode");
    CHECK(drv_spi_set_data_bits(inst, 3) != 0 &&
              drv_spi_set_data_bits(inst, 33) != 0,
          "reject invalid runtime data widths");
    CHECK(drv_spi_set_cs_mode(inst, true) != 0,
          "reject hardware-managed CS without a CS pin");
    CHECK(drv_spi_set_baudrate(inst, 2000000U) == 0,
          "CS-mode rejection releases the configuration lock");

    for (size_t i = 0; i < sizeof(tx); i++)
        tx[i] = (uint8_t)(0xa0U + i);
    CHECK(drv_spi_write(inst, tx, sizeof(tx), true) == (int)sizeof(tx),
          "normal controller transmit completes");
    CHECK(drv_spi_transfer(inst, NULL, NULL, 1, true) < 0 &&
              drv_spi_write(inst, tx, sizeof(tx), true) == (int)sizeof(tx),
          "controller remains usable after rejected transfer");

    if (loopback)
        test_spi_loopback(inst);
    else
        printf("SKIP: verified SPI receive data requires an attached target or "
               "loopback wiring\n");

    drv_spi_inst_destroy(&inst);
    CHECK(inst == NULL, "destroy clears SPI handle");
    drv_spi_inst_destroy(&inst);

    for (int i = 0; i < 100; i++) {
        if (drv_spi_inst_create(TEST_SPI_BUS, true, SPI_HAL_MODE_0,
                                TEST_SPI_BAUDRATE, 8, -1,
                                SPI_HAL_DATA_LINE_1, &inst) != 0 ||
            inst == NULL) {
            lifecycle_ok = 0;
            break;
        }
        drv_spi_inst_destroy(&inst);
        if (inst != NULL) {
            lifecycle_ok = 0;
            break;
        }
    }
    CHECK(lifecycle_ok, "100 SPI open/close cycles");
    if (inst)
        drv_spi_inst_destroy(&inst);

}

int main(int argc, char** argv)
{
    int spi_loopback = 0;

    if (argc == 2 && strcmp(argv[1], "--spi-loopback") == 0) {
        spi_loopback = 1;
    } else if (argc != 1) {
        printf("Usage: %s [--spi-loopback]\n", argv[0]);
        printf("For loopback, connect IO16 (QSPI0_D0) to IO17 (QSPI0_D1).\n");
        return 2;
    }

    printf("I2C/SPI non-destructive regression test\n");
    test_i2c();
    test_spi(spi_loopback);

    printf("\nChecks: %d passed, %d failed, %d total\n",
           total_checks - total_failures, total_failures, total_checks);
    return total_failures == 0 ? 0 : 1;
}
