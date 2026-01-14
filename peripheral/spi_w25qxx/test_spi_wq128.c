#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include "drv_spi.h"
#include "drv_gpio.h"
#include "drv_fpioa.h"

// W25Q128命令定义
#define W25Q_CMD_WRITE_ENABLE           0x06
#define W25Q_CMD_WRITE_DISABLE          0x04
#define W25Q_CMD_READ_STATUS_REG1       0x05
#define W25Q_CMD_READ_STATUS_REG2       0x35
#define W25Q_CMD_READ_STATUS_REG3       0x15
#define W25Q_CMD_WRITE_STATUS_REG1      0x01
#define W25Q_CMD_WRITE_STATUS_REG2      0x31
#define W25Q_CMD_WRITE_STATUS_REG3      0x11
#define W25Q_CMD_CHIP_ERASE             0xC7
#define W25Q_CMD_ERASE_SECTOR_4K        0x20  // 3字节地址
#define W25Q_CMD_ERASE_SECTOR_4K_4B     0x21  // 4字节地址
#define W25Q_CMD_ERASE_BLOCK_32K        0x52
#define W25Q_CMD_ERASE_BLOCK_64K        0xD8
#define W25Q_CMD_PAGE_PROGRAM           0x02  // 3字节地址
#define W25Q_CMD_PAGE_PROGRAM_4B        0x12  // 4字节地址
#define W25Q_CMD_READ_DATA              0x03  // 3字节地址
#define W25Q_CMD_READ_DATA_4B           0x13  // 4字节地址
#define W25Q_CMD_FAST_READ              0x0B
#define W25Q_CMD_FAST_READ_4B           0x0C
#define W25Q_CMD_READ_JEDEC_ID          0x9F
#define W25Q_CMD_READ_UNIQUE_ID         0x4B
#define W25Q_CMD_ENTER_4BYTE_MODE       0xB7
#define W25Q_CMD_EXIT_4BYTE_MODE        0xE9
#define W25Q_CMD_ENABLE_RESET           0x66
#define W25Q_CMD_RESET                  0x99
#define W25Q_CMD_POWER_DOWN             0xB9
#define W25Q_CMD_RELEASE_POWER_DOWN     0xAB

// 状态寄存器位定义
#define W25Q_SR1_BUSY                   (1 << 0)
#define W25Q_SR1_WEL                    (1 << 1)

// 芯片参数
#define W25Q128_PAGE_SIZE               256
#define W25Q128_SECTOR_SIZE             4096
#define W25Q_CHIP_SIZE                  (16 * 1024 * 1024)  // 16MB
#define W25Q128_JEDEC_ID                0xEF4018  // W25Q128的ID

// Flash驱动结构体
typedef struct {
    drv_spi_inst_t spi_handle;
    bool addr_4byte_mode;
} w25q128_t;

// 延时函数
static void delay_ms(uint32_t ms) {
    usleep(ms * 1000);
}

// CS控制函数,空实现，仅帮助理解 cs_change的用法
static void w25q_cs_select(w25q128_t *flash) {}

static void w25q_cs_deselect(w25q128_t *flash) {}

// 读取状态寄存器1
static uint8_t w25q_read_status_reg1(w25q128_t *flash) {
    uint8_t cmd = W25Q_CMD_READ_STATUS_REG1;
    uint8_t status;

    w25q_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 0);
    drv_spi_read(flash->spi_handle, &status, 1, 1);
    w25q_cs_deselect(flash);

    return status;
}

// 等待忙状态结束
static bool w25q_wait_busy(w25q128_t *flash, uint32_t timeout_ms)
{
    uint32_t wait_time = 0;

    while (wait_time < timeout_ms) {
        uint8_t status = w25q_read_status_reg1(flash);
        if (!(status & W25Q_SR1_BUSY)) {
            return true;
        }
        delay_ms(1);
        wait_time++;
    }

    printf("W25Q128: Timeout waiting for busy\n");
    return false;
}

// 写使能
static bool w25q_write_enable(w25q128_t *flash)
{
    uint8_t cmd = W25Q_CMD_WRITE_ENABLE;

    w25q_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    w25q_cs_deselect(flash);

    // 检查WEL位
    uint8_t status = w25q_read_status_reg1(flash);
    return (status & W25Q_SR1_WEL) != 0;
}

// 写禁止
static void w25q_write_disable(w25q128_t *flash)
{
    uint8_t cmd = W25Q_CMD_WRITE_DISABLE;

    w25q_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    w25q_cs_deselect(flash);
}

// 读取JEDEC ID
static uint32_t w25q_read_jedec_id(w25q128_t *flash)
{
    uint8_t cmd = W25Q_CMD_READ_JEDEC_ID;
    uint8_t id[3];

    w25q_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 0);
    drv_spi_read(flash->spi_handle, id, 3, 1);
    w25q_cs_deselect(flash);

    return (id[0] << 16) | (id[1] << 8) | id[2];
}

// 进入4字节地址模式
static bool w25q_enter_4byte_mode(w25q128_t *flash)
{
    uint8_t cmd = W25Q_CMD_ENTER_4BYTE_MODE;

    w25q_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    w25q_cs_deselect(flash);

    flash->addr_4byte_mode = true;
    return true;
}

// 退出4字节地址模式
static bool w25q_exit_4byte_mode(w25q128_t *flash)
{
    uint8_t cmd = W25Q_CMD_EXIT_4BYTE_MODE;

    w25q_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    w25q_cs_deselect(flash);

    flash->addr_4byte_mode = false;
    return true;
}

// 擦除扇区（4KB）
static bool w25q_erase_sector(w25q128_t *flash, uint32_t addr)
{
    if (!w25q_write_enable(flash)) {
        printf("W25Q128: Failed to enable write\n");
        return false;
    }

    uint8_t cmd_buf[5];

    if (flash->addr_4byte_mode) {
        cmd_buf[0] = W25Q_CMD_ERASE_SECTOR_4K_4B;
        cmd_buf[1] = (addr >> 24) & 0xFF;
        cmd_buf[2] = (addr >> 16) & 0xFF;
        cmd_buf[3] = (addr >> 8) & 0xFF;
        cmd_buf[4] = addr & 0xFF;

        w25q_cs_select(flash);
        drv_spi_write(flash->spi_handle, cmd_buf, 5, 1);
        w25q_cs_deselect(flash);
    } else {
        cmd_buf[0] = W25Q_CMD_ERASE_SECTOR_4K;
        cmd_buf[1] = (addr >> 16) & 0xFF;
        cmd_buf[2] = (addr >> 8) & 0xFF;
        cmd_buf[3] = addr & 0xFF;

        w25q_cs_select(flash);
        drv_spi_write(flash->spi_handle, cmd_buf, 4, 1);
        w25q_cs_deselect(flash);
    }

    return w25q_wait_busy(flash, 400);  // 扇区擦除最长400ms
}

// 页编程（最多256字节）
static bool w25q_page_program(w25q128_t *flash, uint32_t addr, const uint8_t *data, size_t len)
{
    if (len > W25Q128_PAGE_SIZE) {
        printf("W25Q128: Page program size exceed 256 bytes\n");
        return false;
    }

    if (!w25q_write_enable(flash)) {
        printf("W25Q128: Failed to enable write\n");
        return false;
    }

    uint8_t cmd_buf[5];

    w25q_cs_select(flash);

    if (flash->addr_4byte_mode) {
        cmd_buf[0] = W25Q_CMD_PAGE_PROGRAM_4B;
        cmd_buf[1] = (addr >> 24) & 0xFF;
        cmd_buf[2] = (addr >> 16) & 0xFF;
        cmd_buf[3] = (addr >> 8) & 0xFF;
        cmd_buf[4] = addr & 0xFF;
        drv_spi_write(flash->spi_handle, cmd_buf, 5, 0);
    } else {
        cmd_buf[0] = W25Q_CMD_PAGE_PROGRAM;
        cmd_buf[1] = (addr >> 16) & 0xFF;
        cmd_buf[2] = (addr >> 8) & 0xFF;
        cmd_buf[3] = addr & 0xFF;
        drv_spi_write(flash->spi_handle, cmd_buf, 4, 0);
    }

    drv_spi_write(flash->spi_handle, data, len, 1);
    w25q_cs_deselect(flash);

    return w25q_wait_busy(flash, 3);  // 页编程最长3ms
}

// 读取数据
static bool w25q_read_data(w25q128_t *flash, uint32_t addr, uint8_t *data, size_t len)
{
    uint8_t cmd_buf[5];

    w25q_cs_select(flash);

    if (flash->addr_4byte_mode) {
        cmd_buf[0] = W25Q_CMD_READ_DATA_4B;
        cmd_buf[1] = (addr >> 24) & 0xFF;
        cmd_buf[2] = (addr >> 16) & 0xFF;
        cmd_buf[3] = (addr >> 8) & 0xFF;
        cmd_buf[4] = addr & 0xFF;
        drv_spi_write(flash->spi_handle, cmd_buf, 5, 0);
    } else {
        cmd_buf[0] = W25Q_CMD_READ_DATA;
        cmd_buf[1] = (addr >> 16) & 0xFF;
        cmd_buf[2] = (addr >> 8) & 0xFF;
        cmd_buf[3] = addr & 0xFF;
        drv_spi_write(flash->spi_handle, cmd_buf, 4, 0);
    }

    drv_spi_read(flash->spi_handle, data, len, 1);
    w25q_cs_deselect(flash);

    return true;
}

// 写入数据（处理跨页）
static bool w25q_write(w25q128_t *flash, uint32_t addr, const uint8_t *data, size_t len)
{
    size_t offset = 0;

    while (offset < len) {
        size_t page_offset = addr % W25Q128_PAGE_SIZE;
        size_t write_len = W25Q128_PAGE_SIZE - page_offset;

        if (write_len > (len - offset)) {
            write_len = len - offset;
        }

        if (!w25q_page_program(flash, addr, data + offset, write_len)) {
            return false;
        }

        addr += write_len;
        offset += write_len;
    }

    return true;
}

// 创建Flash实例
static w25q128_t* w25q_create(int spi_id, int cs_pin, uint32_t baudrate)
{
    int ret;
    w25q128_t *flash = (w25q128_t*)malloc(sizeof(w25q128_t));
    if (!flash) {
        printf("Failed to allocate flash structure\n");
        return NULL;
    }

    memset(flash, 0, sizeof(w25q128_t));

    // 初始化SPI (Mode 0: CPOL=0, CPHA=0)
    ret = drv_spi_inst_create(spi_id, true, SPI_HAL_MODE_0, baudrate,
                                            8, cs_pin, SPI_HAL_DATA_LINE_1, &flash->spi_handle);

    if (ret != 0) {
        printf("Failed to create SPI instance\n");
        free(flash);
        return NULL;
    }

    flash->addr_4byte_mode = false;

    return flash;
}

// 销毁Flash实例
static void w25q_destroy(w25q128_t *flash) {
    if (!flash) return;

    if (flash->spi_handle) {
        drv_spi_inst_destroy(&flash->spi_handle);
    }

    free(flash);
}

// 测试函数
static void print_hex_dump(const char *prefix, const uint8_t *data, size_t len) {
    printf("%s", prefix);
    for (size_t i = 0; i < len; i++) {
        if (i % 16 == 0) printf("\n  ");
        printf("%02X ", data[i]);
    }
    printf("\n");
}

static int drv_spi_write_then_read(drv_spi_inst_t handle, const void *send_buf, size_t send_length, void *recv_buf, size_t recv_length)
{
    int ret = 0;
    struct rt_qspi_message msg;
    unsigned char *ptr = (unsigned char *)send_buf;
    size_t count = 0;

    msg.instruction.content = ptr[0];
    msg.instruction.qspi_lines = 1;
    msg.instruction.size = 8;
    count++;

    /* get address */
    if (send_length > 1) {
        if (send_length >= 5) {
            /* medium size greater than 16Mb, address size is 4 Byte */
            msg.address.content = (ptr[1] << 24) | (ptr[2] << 16) | (ptr[3] << 8) | (ptr[4]);
            msg.address.size = 32;
            count += 4;
        }
        else if (send_length >= 4) {
            /* address size is 3 Byte */
            msg.address.content = (ptr[1] << 16) | (ptr[2] << 8) | (ptr[3]);
            msg.address.size = 24;
            count += 3;
        }
        else {
            ret = -1;
            goto out;
        }
        msg.address.qspi_lines = 1;
    } else {
        /* no address stage */
        msg.address.content = 0 ;
        msg.address.qspi_lines = 0;
        msg.address.size = 0;
    }

    msg.alternate_bytes.content = 0;
    msg.alternate_bytes.size = 0;
    msg.alternate_bytes.qspi_lines = 0;

    /* set dummy cycles */
    if (count != send_length) {
        msg.dummy_cycles = (send_length - count) * 8;

    } else {
        msg.dummy_cycles = 0;
    }

    /* set recv buf and recv size */
    msg.parent.recv_buf = recv_buf;
    msg.parent.send_buf = NULL;
    msg.parent.length = recv_length;
    msg.parent.cs_take = 1;
    msg.parent.cs_release = 1;

    msg.parent.next = NULL;

    ret = drv_spi_transfer_message(handle, &msg);
    if (ret != (int)recv_length) {
        printf("spi write then read fail: ret: %d\n", ret);
    }
out:
    return ret;
}

static bool test_read_id(w25q128_t *flash)
{
    uint8_t id[3];
    uint32_t jedec_id;
    uint8_t cmd = W25Q_CMD_READ_JEDEC_ID;

    // 1. 读取JEDEC ID
    w25q_cs_select(flash);
    drv_spi_write_then_read(flash->spi_handle, &cmd, 1, &id[0], 3);
    w25q_cs_deselect(flash);

    jedec_id =  (id[0] << 16) | (id[1] << 8) | id[2];
    printf("1. JEDEC ID: 0x%06X (Expected: 0x%06X)\n", jedec_id, W25Q128_JEDEC_ID);
    if (jedec_id != W25Q128_JEDEC_ID) {
        printf("   FAILED: Incorrect JEDEC ID\n");
        return false;
    }
    printf("   PASSED\n");

    return true;
}

static bool test_basic_functions(w25q128_t *flash)
{
    printf("\n=== Basic Function Test ===\n");

    // 1. 读取JEDEC ID
    uint32_t jedec_id = w25q_read_jedec_id(flash);
    printf("1. JEDEC ID: 0x%06X (Expected: 0x%06X)\n", jedec_id);
    if (jedec_id != W25Q128_JEDEC_ID) {
        printf("   FAILED: Incorrect JEDEC ID\n");
        return false;
    }
    printf("   PASSED\n");

    // 2. 读取状态寄存器
    uint8_t status = w25q_read_status_reg1(flash);
    printf("2. Status Register 1: 0x%02X\n", status);
    printf("   BUSY: %d, WEL: %d\n",
           (status & W25Q_SR1_BUSY) ? 1 : 0,
           (status & W25Q_SR1_WEL) ? 1 : 0);
    printf("   PASSED\n");

    // 3. 测试写使能/禁止
    printf("3. Write Enable/Disable Test\n");
    if (!w25q_write_enable(flash)) {
        printf("   FAILED: Write enable failed\n");
        return false;
    }
    status = w25q_read_status_reg1(flash);
    if (!(status & W25Q_SR1_WEL)) {
        printf("   FAILED: WEL bit not set\n");
        return false;
    }
    printf("   Write enabled successfully\n");

    w25q_write_disable(flash);
    status = w25q_read_status_reg1(flash);
    if (status & W25Q_SR1_WEL) {
        printf("   FAILED: WEL bit still set\n");
        return false;
    }
    printf("   Write disabled successfully\n");
    printf("   PASSED\n");

    return true;
}

static bool test_read_write(w25q128_t *flash)
{
    printf("\n=== Read/Write Test ===\n");

    const uint32_t test_addr = 0x100000;  // 1MB位置
    const size_t test_size = 512;
    uint8_t *write_buf = malloc(test_size);
    uint8_t *read_buf = malloc(test_size);

    if (!write_buf || !read_buf) {
        printf("Failed to allocate test buffers\n");
        if (write_buf) free(write_buf);
        if (read_buf) free(read_buf);
        return false;
    }

    // 生成测试数据
    for (size_t i = 0; i < test_size; i++) {
        write_buf[i] = (uint8_t)(i & 0xFF);
    }

    // 1. 擦除扇区
    printf("1. Erasing sector at 0x%06X...\n", test_addr);
    if (!w25q_erase_sector(flash, test_addr)) {
        printf("   FAILED: Sector erase failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Sector erased successfully\n");

    // 2. 验证擦除（应该全是0xFF）
    printf("2. Verifying erase...\n");
    if (!w25q_read_data(flash, test_addr, read_buf, test_size)) {
        printf("   FAILED: Read failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    bool erase_ok = true;
    for (size_t i = 0; i < test_size; i++) {
        if (read_buf[i] != 0xFF) {
            erase_ok = false;
            break;
        }
    }
    if (!erase_ok) {
        printf("   FAILED: Erase verification failed\n");
        print_hex_dump("   Read data:", read_buf, 32);
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Erase verified successfully\n");

    // 3. 写入数据
    printf("3. Writing %zu bytes to 0x%06X...\n", test_size, test_addr);
    if (!w25q_write(flash, test_addr, write_buf, test_size)) {
        printf("   FAILED: Write failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Write completed successfully\n");

    // 4. 读取并验证数据
    printf("4. Reading and verifying data...\n");
    memset(read_buf, 0, test_size);
    if (!w25q_read_data(flash, test_addr, read_buf, test_size)) {
        printf("   FAILED: Read failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    if (memcmp(write_buf, read_buf, test_size) != 0) {
        printf("   FAILED: Data verification failed\n");
        print_hex_dump("   Written:", write_buf, 32);
        print_hex_dump("   Read:", read_buf, 32);
        free(write_buf);
        free(read_buf);
        return false;
    }

    printf("   Data verified successfully\n");
    print_hex_dump("   Sample data:", read_buf, 32);
    printf("   PASSED\n");

    free(write_buf);
    free(read_buf);
    return true;
}

#if 0
static bool test_4byte_mode(w25q128_t *flash)
{
    printf("\n=== 4-Byte Address Mode Test ===\n");

    // 测试大于16MB的地址
    const uint32_t test_addr = 0x1000000;  // 16MB位置
    const size_t test_size = 256;
    uint8_t *write_buf = malloc(test_size);
    uint8_t *read_buf = malloc(test_size);

    if (!write_buf || !read_buf) {
        printf("Failed to allocate test buffers\n");
        if (write_buf) free(write_buf);
        if (read_buf) free(read_buf);
        return false;
    }

    // 生成测试数据
    for (size_t i = 0; i < test_size; i++) {
        write_buf[i] = (uint8_t)(0xA0 + (i & 0x0F));
    }

    // 1. 进入4字节地址模式
    printf("1. Entering 4-byte address mode...\n");
    if (!w25q_enter_4byte_mode(flash)) {
        printf("   FAILED: Enter 4-byte mode failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Entered 4-byte mode successfully\n");

    // 2. 擦除扇区
    printf("2. Erasing sector at 0x%08X...\n", test_addr);
    if (!w25q_erase_sector(flash, test_addr)) {
        printf("   FAILED: Sector erase failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Sector erased successfully\n");

    // 3. 写入数据
    printf("3. Writing %zu bytes to 0x%08X...\n", test_size, test_addr);
    if (!w25q_write(flash, test_addr, write_buf, test_size)) {
        printf("   FAILED: Write failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Write completed successfully\n");

    // 4. 读取并验证数据
    printf("4. Reading and verifying data...\n");
    memset(read_buf, 0, test_size);
    if (!w25q_read_data(flash, test_addr, read_buf, test_size)) {
        printf("   FAILED: Read failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    if (memcmp(write_buf, read_buf, test_size) != 0) {
        printf("   FAILED: Data verification failed\n");
        print_hex_dump("   Written:", write_buf, 32);
        print_hex_dump("   Read:", read_buf, 32);
        free(write_buf);
        free(read_buf);
        return false;
    }

    printf("   Data verified successfully\n");
    print_hex_dump("   Sample data:", read_buf, 32);

    // 5. 退出4字节地址模式
    printf("5. Exiting 4-byte address mode...\n");
    if (!w25q_exit_4byte_mode(flash)) {
        printf("   FAILED: Exit 4-byte mode failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Exited 4-byte mode successfully\n");
    printf("   PASSED\n");

    free(write_buf);
    free(read_buf);
    return true;
}
#endif

// 性能测试
static bool test_performance(w25q128_t *flash)
{
    printf("\n=== Performance Test ===\n");

    const size_t test_size = 64 * 1024;  // 64KB
    const uint32_t test_addr = 0x200000;  // 2MB位置
    uint8_t *buffer = malloc(test_size);

    if (!buffer) {
        printf("Failed to allocate test buffer\n");
        return false;
    }

    // 生成随机数据
    for (size_t i = 0; i < test_size; i++) {
        buffer[i] = (uint8_t)(rand() & 0xFF);
    }

    // 1. 擦除时间测试
    printf("1. Erase performance (64KB block):\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 擦除16个4KB扇区 = 64KB
    for (int i = 0; i < 16; i++) {
        if (!w25q_erase_sector(flash, test_addr + i * W25Q128_SECTOR_SIZE)) {
            printf("   FAILED: Erase failed\n");
            free(buffer);
            return false;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double erase_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("   Erased 64KB in %.3f seconds\n", erase_time);

    // 2. 写入时间测试
    printf("2. Write performance:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (!w25q_write(flash, test_addr, buffer, test_size)) {
        printf("   FAILED: Write failed\n");
        free(buffer);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double write_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double write_speed = (test_size / 1024.0) / write_time;
    printf("   Wrote %zu KB in %.3f seconds (%.1f KB/s)\n",
           test_size / 1024, write_time, write_speed);

    // 3. 读取时间测试
    printf("3. Read performance:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (!w25q_read_data(flash, test_addr, buffer, test_size)) {
        printf("   FAILED: Read failed\n");
        free(buffer);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double read_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double read_speed = (test_size / 1024.0) / read_time;
    printf("   Read %zu KB in %.3f seconds (%.1f KB/s)\n",
           test_size / 1024, read_time, read_speed);

    printf("   PASSED\n");

    free(buffer);
    return true;
}

// 主函数
int main(int argc, char *argv[])
{
    printf("W25Q128 SPI Flash Test Program\n");

    // 默认参数
    int spi_id = 2;

    // 解析命令行参数
    if (argc >= 2) {
        spi_id = atoi(argv[1]);
    }

    if (spi_id == 0) {
        drv_fpioa_set_pin_func(15, OSPI_CLK);
        drv_fpioa_set_pin_func(16, OSPI_D0);
        drv_fpioa_set_pin_func(17, OSPI_D1);
        printf("Using OSPI\n");
    } else if (spi_id == 1) {
        drv_fpioa_set_pin_func(15, QSPI0_CLK);
        drv_fpioa_set_pin_func(16, QSPI0_D0);
        drv_fpioa_set_pin_func(17, QSPI0_D1);
        printf("Using QSPI0\n");
    } else if (spi_id == 2) {
        drv_fpioa_set_pin_func(21, QSPI1_CLK);
        drv_fpioa_set_pin_func(40, QSPI1_D0);
        drv_fpioa_set_pin_func(41, QSPI1_D1);
        printf("Using QSPI1\n");
    }

    // 创建Flash实例1
    w25q128_t *inst_1 = w25q_create(spi_id, 11, 1000 * 1000);
    if (!inst_1) {
        printf("Failed to create W25Q128 instance\n");
        return -1;
    }

    bool all_passed = true;

    // 运行测试
    if (!test_basic_functions(inst_1)) {
        all_passed = false;
    }

    // 创建Flash实例2,使用不同CS和baudrate,模拟接多个设备的情况，睡眠5秒给换CS pin争取时间。
    usleep(5000000);
    w25q128_t *inst_2 = w25q_create(spi_id, 14, 1000 * 1000 * 10);
    if (!inst_2) {
        printf("Failed to create W25Q128 instance\n");
        w25q_destroy(inst_1);
        return -1;
    }

    if (!test_read_write(inst_2)) {
        all_passed = false;
    }

    if (!test_performance(inst_2)) {
        all_passed = false;
    }

    if (!test_read_id(inst_2)) {
        all_passed = false;
    }

    // 总结
    printf("\n=== Test Summary ===\n");
    if (all_passed) {
        printf("All tests PASSED!\n");
    } else {
        printf("Some tests FAILED!\n");
    }

    // 清理资源
    w25q_destroy(inst_1);
    w25q_destroy(inst_2);

    return all_passed ? 0 : -1;
}
