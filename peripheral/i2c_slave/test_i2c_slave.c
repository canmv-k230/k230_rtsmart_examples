/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>

#include "drv_fpioa.h"
#include "drv_i2c.h"

#define DEFAULT_MASTER_BUS  4
#define DEFAULT_SLAVE_BUS   2
#define DEFAULT_MASTER_SCL  46
#define DEFAULT_MASTER_SDA  47
#define DEFAULT_SLAVE_SCL   11
#define DEFAULT_SLAVE_SDA   12
#define DEFAULT_SLAVE_ADDR  0x22
#define DEFAULT_ITERATIONS  32
#define DEFAULT_MAX_SPEED   3400000U
#define I2C_TIMEOUT_MS      1000U
#define EEPROM_SIZE         256U
#define MAX_TEST_DATA       (EEPROM_SIZE - 1U)

#define I2C_SLAVE_IOCTL_SET_BUFFER_SIZE 0
#define I2C_SLAVE_IOCTL_SET_ADDR        1

struct test_config {
    int master_bus;
    int slave_bus;
    int master_scl;
    int master_sda;
    int slave_scl;
    int slave_sda;
    int iterations;
    uint32_t max_speed;
    uint8_t slave_addr;
};

static int failures;

static void usage(const char* program)
{
    printf("Usage: %s [options]\n", program);
    printf("  -m, --master-bus N    master controller (default %d)\n", DEFAULT_MASTER_BUS);
    printf("  -s, --slave-bus N     slave controller (default %d)\n", DEFAULT_SLAVE_BUS);
    printf("  -a, --address ADDR    7-bit slave address (default 0x%02x)\n", DEFAULT_SLAVE_ADDR);
    printf("      --master-scl PIN  master SCL pin (default %d)\n", DEFAULT_MASTER_SCL);
    printf("      --master-sda PIN  master SDA pin (default %d)\n", DEFAULT_MASTER_SDA);
    printf("      --slave-scl PIN   slave SCL pin (default %d)\n", DEFAULT_SLAVE_SCL);
    printf("      --slave-sda PIN   slave SDA pin (default %d)\n", DEFAULT_SLAVE_SDA);
    printf("  -n, --iterations N    stress iterations (default %d)\n", DEFAULT_ITERATIONS);
    printf("      --max-speed HZ    highest speed to test (default %u)\n", DEFAULT_MAX_SPEED);
    printf("  -h, --help            show this help\n\n");
    printf("Connect master SCL to slave SCL and master SDA to slave SDA.\n");
    printf("The kernel must configure the selected slave controller in slave mode.\n");
    printf("The speed sweep covers 100 kHz, 400 kHz, 1 MHz, and 3.4 MHz up to --max-speed.\n");
}

static int parse_number(const char* text, long min, long max, int* value)
{
    char* end;
    long parsed;

    errno  = 0;
    parsed = strtol(text, &end, 0);
    if (errno || end == text || *end != '\0' || parsed < min || parsed > max)
        return -1;

    *value = (int)parsed;
    return 0;
}

static int parse_args(int argc, char** argv, struct test_config* cfg)
{
    enum {
        OPT_MASTER_SCL = 1000,
        OPT_MASTER_SDA,
        OPT_SLAVE_SCL,
        OPT_SLAVE_SDA,
        OPT_MAX_SPEED,
    };
    static const struct option options[] = {
        { "master-bus", required_argument, NULL, 'm' },
        { "slave-bus", required_argument, NULL, 's' },
        { "address", required_argument, NULL, 'a' },
        { "iterations", required_argument, NULL, 'n' },
        { "master-scl", required_argument, NULL, OPT_MASTER_SCL },
        { "master-sda", required_argument, NULL, OPT_MASTER_SDA },
        { "slave-scl", required_argument, NULL, OPT_SLAVE_SCL },
        { "slave-sda", required_argument, NULL, OPT_SLAVE_SDA },
        { "max-speed", required_argument, NULL, OPT_MAX_SPEED },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 },
    };
    int opt;
    int value;

    while ((opt = getopt_long(argc, argv, "m:s:a:n:h", options, NULL)) != -1) {
        switch (opt) {
        case 'm':
            if (parse_number(optarg, 0, KD_HARD_I2C_MAX_NUM - 1, &cfg->master_bus))
                return -1;
            break;
        case 's':
            if (parse_number(optarg, 0, KD_HARD_I2C_MAX_NUM - 1, &cfg->slave_bus))
                return -1;
            break;
        case 'a':
            if (parse_number(optarg, 0x08, 0x77, &value))
                return -1;
            cfg->slave_addr = (uint8_t)value;
            break;
        case 'n':
            if (parse_number(optarg, 1, 10000, &cfg->iterations))
                return -1;
            break;
        case OPT_MASTER_SCL:
            if (parse_number(optarg, 0, FPIOA_PIN_MAX_NUM - 1, &cfg->master_scl))
                return -1;
            break;
        case OPT_MASTER_SDA:
            if (parse_number(optarg, 0, FPIOA_PIN_MAX_NUM - 1, &cfg->master_sda))
                return -1;
            break;
        case OPT_SLAVE_SCL:
            if (parse_number(optarg, 0, FPIOA_PIN_MAX_NUM - 1, &cfg->slave_scl))
                return -1;
            break;
        case OPT_SLAVE_SDA:
            if (parse_number(optarg, 0, FPIOA_PIN_MAX_NUM - 1, &cfg->slave_sda))
                return -1;
            break;
        case OPT_MAX_SPEED:
            if (parse_number(optarg, 100000, DEFAULT_MAX_SPEED, &value))
                return -1;
            cfg->max_speed = (uint32_t)value;
            break;
        case 'h':
            usage(argv[0]);
            exit(0);
        default:
            return -1;
        }
    }

    if (optind != argc || cfg->master_bus == cfg->slave_bus)
        return -1;
    return 0;
}

static int configure_pin(int pin, fpioa_func_t function)
{
    if (drv_fpioa_set_pin_func(pin, function) != 0)
        return -1;
    if (drv_fpioa_set_pin_pu(pin, 1) != 0)
        return -1;
    return 0;
}

static int configure_pins(const struct test_config* cfg)
{
    if (configure_pin(cfg->master_scl, (fpioa_func_t)(IIC0_SCL + cfg->master_bus * 2)) ||
        configure_pin(cfg->master_sda, (fpioa_func_t)(IIC0_SDA + cfg->master_bus * 2)) ||
        configure_pin(cfg->slave_scl, (fpioa_func_t)(IIC0_SCL + cfg->slave_bus * 2)) ||
        configure_pin(cfg->slave_sda, (fpioa_func_t)(IIC0_SDA + cfg->slave_bus * 2))) {
        printf("FAIL: requested I2C function is not available on one of the pins\n");
        return -1;
    }
    return 0;
}

static int slave_read_at(int fd, uint8_t offset, uint8_t* data, size_t length)
{
    if (lseek(fd, offset, SEEK_SET) != offset)
        return -1;
    return read(fd, data, length) == (ssize_t)length ? 0 : -1;
}

static int slave_read_wrapped(int fd, uint8_t offset, uint8_t* data,
                              size_t length, size_t buffer_size)
{
    size_t position = offset;

    while (length) {
        size_t chunk = buffer_size - position;

        if (chunk > length)
            chunk = length;
        if (slave_read_at(fd, (uint8_t)position, data, chunk))
            return -1;

        data += chunk;
        length -= chunk;
        position = 0;
    }
    return 0;
}

static int slave_write_at(int fd, uint8_t offset, const uint8_t* data, size_t length)
{
    if (lseek(fd, offset, SEEK_SET) != offset)
        return -1;
    return write(fd, data, length) == (ssize_t)length ? 0 : -1;
}

static int master_write(drv_i2c_inst_t* master, uint8_t address, uint8_t offset,
                        const uint8_t* data, size_t length)
{
    uint8_t buffer[MAX_TEST_DATA + 1];
    i2c_msg_t msg;

    if (length > MAX_TEST_DATA)
        return -1;
    buffer[0] = offset;
    memcpy(buffer + 1, data, length);
    msg.addr  = address;
    msg.flags = DRV_I2C_WR;
    msg.len   = (uint16_t)(length + 1);
    msg.buf   = buffer;
    return drv_i2c_transfer(master, &msg, 1);
}

static int master_read(drv_i2c_inst_t* master, uint8_t address, uint8_t offset,
                       uint8_t* data, size_t length)
{
    i2c_msg_t msgs[2] = {
        { .addr = address, .flags = DRV_I2C_WR, .len = 1, .buf = &offset },
        { .addr = address, .flags = DRV_I2C_RD, .len = (uint16_t)length, .buf = data },
    };

    return drv_i2c_transfer(master, msgs, 2);
}

static void report_compare(const char* name, const uint8_t* expected,
                           const uint8_t* actual, size_t length)
{
    size_t i;

    for (i = 0; i < length; i++) {
        if (expected[i] != actual[i]) {
            printf("FAIL: %-28s byte %zu expected 0x%02x got 0x%02x\n",
                   name, i, expected[i], actual[i]);
            failures++;
            return;
        }
    }
    printf("PASS: %s\n", name);
}

static void test_preload_and_master_read(int slave_fd, drv_i2c_inst_t* master, uint8_t address)
{
    uint8_t expected[EEPROM_SIZE];
    uint8_t actual[EEPROM_SIZE];
    size_t i;

    for (i = 0; i < sizeof(expected); i++)
        expected[i] = (uint8_t)(i ^ 0x5aU);

    if (slave_write_at(slave_fd, 0, expected, sizeof(expected)) ||
        master_read(master, address, 0, actual, sizeof(actual))) {
        printf("FAIL: slave preload/master read transfer\n");
        failures++;
        return;
    }
    report_compare("slave preload -> master read", expected, actual, sizeof(actual));
}

static void test_master_write_and_local_read(int slave_fd, drv_i2c_inst_t* master, uint8_t address)
{
    uint8_t expected[MAX_TEST_DATA];
    uint8_t actual[MAX_TEST_DATA];
    size_t i;

    for (i = 0; i < sizeof(expected); i++)
        expected[i] = (uint8_t)(0xa0U + i);

    if (master_write(master, address, 37, expected, sizeof(expected)) ||
        slave_read_wrapped(slave_fd, 37, actual, sizeof(actual), EEPROM_SIZE)) {
        printf("FAIL: master write/slave local read transfer\n");
        failures++;
        return;
    }
    report_compare("master write -> slave buffer", expected, actual, sizeof(actual));
}

static void test_poll_notification(int slave_fd, drv_i2c_inst_t* master, uint8_t address)
{
    struct timeval clear_timeout = { 0 };
    struct timeval timeout = { .tv_sec = 1, .tv_usec = 0 };
    uint8_t value = 0x3c;
    fd_set readfds;
    int result;

    /* Discard notifications left by earlier test cases. */
    FD_ZERO(&readfds);
    FD_SET(slave_fd, &readfds);
    (void)select(slave_fd + 1, &readfds, NULL, NULL, &clear_timeout);

    FD_ZERO(&readfds);
    FD_SET(slave_fd, &readfds);
    if (master_write(master, address, 3, &value, 1)) {
        printf("FAIL: poll notification master write\n");
        failures++;
        return;
    }

    result = select(slave_fd + 1, &readfds, NULL, NULL, &timeout);
    if (result == 1 && FD_ISSET(slave_fd, &readfds)) {
        printf("PASS: slave write poll notification\n");
    } else {
        printf("FAIL: slave write poll notification (select=%d errno=%d)\n", result, errno);
        failures++;
    }
}

static void test_read_does_not_poll(int slave_fd, drv_i2c_inst_t* master, uint8_t address)
{
    struct timeval clear_timeout = { 0 };
    struct timeval timeout = { .tv_sec = 0, .tv_usec = 100000 };
    uint8_t data[8];
    fd_set readfds;
    int result;

    FD_ZERO(&readfds);
    FD_SET(slave_fd, &readfds);
    (void)select(slave_fd + 1, &readfds, NULL, NULL, &clear_timeout);

    if (master_read(master, address, 0, data, sizeof(data))) {
        printf("FAIL: read-only poll test transfer\n");
        failures++;
        return;
    }

    FD_ZERO(&readfds);
    FD_SET(slave_fd, &readfds);
    result = select(slave_fd + 1, &readfds, NULL, NULL, &timeout);
    if (result == 0) {
        printf("PASS: master read does not signal write poll\n");
    } else {
        printf("FAIL: master read signaled write poll (select=%d errno=%d)\n", result, errno);
        failures++;
    }
}

static void test_wraparound(drv_i2c_inst_t* master, uint8_t address)
{
    uint8_t expected[16];
    uint8_t actual[16];
    size_t i;

    for (i = 0; i < sizeof(expected); i++)
        expected[i] = (uint8_t)(0xf0U + i);

    if (master_write(master, address, 248, expected, sizeof(expected)) ||
        master_read(master, address, 248, actual, sizeof(actual))) {
        printf("FAIL: EEPROM address wraparound transfer\n");
        failures++;
        return;
    }
    report_compare("EEPROM address wraparound", expected, actual, sizeof(actual));
}

static void test_buffer_resize(int slave_fd, drv_i2c_inst_t* master, uint8_t address)
{
    uint32_t small_size = 64;
    uint32_t restore_size = EEPROM_SIZE;
    uint8_t expected[16];
    uint8_t actual[16];
    size_t i;
    int failed = 0;

    for (i = 0; i < sizeof(expected); i++)
        expected[i] = (uint8_t)(0x70U + i);

    if (ioctl(slave_fd, I2C_SLAVE_IOCTL_SET_BUFFER_SIZE, &small_size) ||
        master_write(master, address, 60, expected, sizeof(expected)) ||
        master_read(master, address, 60, actual, sizeof(actual)) ||
        memcmp(expected, actual, sizeof(expected)) != 0) {
        failed = 1;
    }

    if (ioctl(slave_fd, I2C_SLAVE_IOCTL_SET_BUFFER_SIZE, &restore_size)) {
        printf("FAIL: restore slave buffer size to %u\n", restore_size);
        failures++;
        return;
    }

    if (failed) {
        printf("FAIL: runtime buffer resize and wraparound\n");
        failures++;
    } else {
        printf("PASS: runtime buffer resize and wraparound\n");
    }
}

static void test_address_reconfiguration(int slave_fd, drv_i2c_inst_t* master, uint8_t address)
{
    uint8_t alternate = address == 0x30 ? 0x31 : 0x30;
    uint8_t expected[16];
    uint8_t actual[16];
    size_t i;
    int failed = 0;

    for (i = 0; i < sizeof(expected); i++)
        expected[i] = (uint8_t)(0xc3U ^ i);

    if (ioctl(slave_fd, I2C_SLAVE_IOCTL_SET_ADDR, &alternate) ||
        master_write(master, alternate, 91, expected, sizeof(expected)) ||
        master_read(master, alternate, 91, actual, sizeof(actual)) ||
        memcmp(expected, actual, sizeof(expected)) != 0) {
        failed = 1;
    }

    if (ioctl(slave_fd, I2C_SLAVE_IOCTL_SET_ADDR, &address)) {
        printf("FAIL: restore slave address to 0x%02x\n", address);
        failures++;
        return;
    }

    if (failed) {
        printf("FAIL: runtime slave address reconfiguration\n");
        failures++;
    } else {
        printf("PASS: runtime slave address reconfiguration\n");
    }
}

static void test_stress(drv_i2c_inst_t* master, uint8_t address, int iterations)
{
    uint8_t expected[MAX_TEST_DATA];
    uint8_t actual[MAX_TEST_DATA];
    int iteration;

    for (iteration = 0; iteration < iterations; iteration++) {
        size_t length = (size_t)(iteration % MAX_TEST_DATA) + 1;
        uint8_t offset = (uint8_t)((iteration * 29) & 0xff);
        size_t i;

        for (i = 0; i < length; i++)
            expected[i] = (uint8_t)(iteration * 17 + (int)i * 7);
        memset(actual, 0, sizeof(actual));

        if (master_write(master, address, offset, expected, length) ||
            master_read(master, address, offset, actual, length) ||
            memcmp(expected, actual, length) != 0) {
            printf("FAIL: stress iteration %d (offset=%u length=%zu)\n",
                   iteration, offset, length);
            failures++;
            return;
        }
    }
    printf("PASS: %d write/read stress iterations\n", iterations);
}

static uint64_t elapsed_ns(const struct timespec* start, const struct timespec* end)
{
    return (uint64_t)(end->tv_sec - start->tv_sec) * 1000000000ULL +
           (uint64_t)(end->tv_nsec - start->tv_nsec);
}

static int run_speed_case(drv_i2c_inst_t* master, uint8_t address,
                          uint32_t speed, int iterations)
{
    static const uint16_t lengths[] = {
        1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
    };
    uint8_t expected[MAX_TEST_DATA];
    uint8_t actual[MAX_TEST_DATA];
    struct timespec start;
    struct timespec end;
    uint64_t duration;
    uint64_t payload_bytes = 0;
    size_t case_index;
    int iteration;

    if (drv_i2c_set_freq(master, speed)) {
        printf("FAIL: set speed to %u Hz\n", speed);
        return -1;
    }

    for (case_index = 0; case_index < sizeof(lengths) / sizeof(lengths[0]); case_index++) {
        size_t length = lengths[case_index];
        uint8_t offset = (uint8_t)(case_index * 37U + 251U);
        size_t i;

        for (i = 0; i < length; i++)
            expected[i] = (uint8_t)(speed / 100000U + case_index * 19U + i * 11U);
        memset(actual, 0, length);

        if (master_write(master, address, offset, expected, length) ||
            master_read(master, address, offset, actual, length) ||
            memcmp(expected, actual, length) != 0) {
            printf("FAIL: %u Hz boundary length %zu offset %u\n", speed, length, offset);
            return -1;
        }
    }

    for (case_index = 0; case_index < sizeof(expected); case_index++)
        expected[case_index] = (uint8_t)(case_index * 13U + speed / 100000U);

    if (clock_gettime(CLOCK_MONOTONIC, &start)) {
        printf("FAIL: clock_gettime before %u Hz benchmark\n", speed);
        return -1;
    }
    for (iteration = 0; iteration < iterations; iteration++) {
        uint8_t offset = (uint8_t)(iteration * 23);

        if (master_write(master, address, offset, expected, sizeof(expected)) ||
            master_read(master, address, offset, actual, sizeof(actual)) ||
            memcmp(expected, actual, sizeof(expected)) != 0) {
            printf("FAIL: %u Hz benchmark iteration %d\n", speed, iteration);
            return -1;
        }
        payload_bytes += sizeof(expected) * 2U;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end)) {
        printf("FAIL: clock_gettime after %u Hz benchmark\n", speed);
        return -1;
    }

    duration = elapsed_ns(&start, &end);
    printf("PASS: %7u Hz, FIFO/boundary lengths 1..255, payload %llu kbit/s\n",
           speed, duration ? (unsigned long long)(payload_bytes * 8ULL * 1000000ULL / duration) : 0ULL);
    return 0;
}

static void test_speed_sweep(drv_i2c_inst_t* master, uint8_t address,
                             uint32_t max_speed, int iterations)
{
    static const uint32_t speeds[] = { 100000U, 400000U, 1000000U, 3400000U };
    size_t i;

    printf("\nSpeed sweep (reported rate is bidirectional payload throughput):\n");
    for (i = 0; i < sizeof(speeds) / sizeof(speeds[0]); i++) {
        if (speeds[i] > max_speed)
            continue;
        if (run_speed_case(master, address, speeds[i], iterations))
            failures++;
    }
}

int main(int argc, char** argv)
{
    struct test_config cfg = {
        .master_bus = DEFAULT_MASTER_BUS,
        .slave_bus = DEFAULT_SLAVE_BUS,
        .master_scl = DEFAULT_MASTER_SCL,
        .master_sda = DEFAULT_MASTER_SDA,
        .slave_scl = DEFAULT_SLAVE_SCL,
        .slave_sda = DEFAULT_SLAVE_SDA,
        .iterations = DEFAULT_ITERATIONS,
        .max_speed = DEFAULT_MAX_SPEED,
        .slave_addr = DEFAULT_SLAVE_ADDR,
    };
    drv_i2c_inst_t* master = NULL;
    uint32_t buffer_size = EEPROM_SIZE;
    char slave_path[32];
    int slave_fd = -1;
    int status = EXIT_FAILURE;

    if (parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    printf("I2C self-test: master i2c%d pins %d/%d, slave i2c%d pins %d/%d, address 0x%02x\n",
           cfg.master_bus, cfg.master_scl, cfg.master_sda,
           cfg.slave_bus, cfg.slave_scl, cfg.slave_sda, cfg.slave_addr);
    printf("Wiring: pin %d <-> pin %d (SCL), pin %d <-> pin %d (SDA)\n",
           cfg.master_scl, cfg.slave_scl, cfg.master_sda, cfg.slave_sda);

    if (configure_pins(&cfg))
        goto out;

    snprintf(slave_path, sizeof(slave_path), "/dev/i2c%d_slave", cfg.slave_bus);
    slave_fd = open(slave_path, O_RDWR);
    if (slave_fd < 0) {
        printf("FAIL: open %s: %s\n", slave_path, strerror(errno));
        printf("Configure I2C%d_MODE_SLAVE in the RT-Smart kernel.\n", cfg.slave_bus);
        goto out;
    }
    if (ioctl(slave_fd, I2C_SLAVE_IOCTL_SET_BUFFER_SIZE, &buffer_size) ||
        ioctl(slave_fd, I2C_SLAVE_IOCTL_SET_ADDR, &cfg.slave_addr)) {
        printf("FAIL: configure %s: %s\n", slave_path, strerror(errno));
        goto out;
    }

    /* A zero cached frequency makes the first set_freq reach the kernel. */
    if (drv_i2c_inst_create(cfg.master_bus, 0, I2C_TIMEOUT_MS,
                            0xff, 0xff, &master)) {
        printf("FAIL: open master i2c%d\n", cfg.master_bus);
        goto out;
    }
    if (drv_i2c_set_freq(master, 100000U)) {
        printf("FAIL: set initial master speed\n");
        goto out;
    }

    test_preload_and_master_read(slave_fd, master, cfg.slave_addr);
    test_master_write_and_local_read(slave_fd, master, cfg.slave_addr);
    test_poll_notification(slave_fd, master, cfg.slave_addr);
    test_read_does_not_poll(slave_fd, master, cfg.slave_addr);
    test_wraparound(master, cfg.slave_addr);
    test_buffer_resize(slave_fd, master, cfg.slave_addr);
    test_address_reconfiguration(slave_fd, master, cfg.slave_addr);
    test_stress(master, cfg.slave_addr, cfg.iterations);
    test_speed_sweep(master, cfg.slave_addr, cfg.max_speed, cfg.iterations);

    printf("\nI2C slave self-test: %s (%d failure%s)\n",
           failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    status = failures ? EXIT_FAILURE : EXIT_SUCCESS;

out:
    if (master)
        drv_i2c_inst_destroy(&master);
    if (slave_fd >= 0)
        close(slave_fd);
    return status;
}
