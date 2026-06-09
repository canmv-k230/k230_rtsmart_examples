#include "drv_fpioa.h"
#include "drv_gpio.h"
#include "drv_spi.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>

#define W25Q_TEST_ALIGN 64

// Flash device database
typedef struct {
    uint32_t    jedec_id;
    const char* name;
    uint32_t    size; // Total size in bytes
    uint16_t    page_size;
    uint32_t    sector_size;
    uint32_t    block_32k_size;
    uint32_t    block_64k_size;
    uint32_t    max_erase_time_ms; // Max sector erase time
    uint32_t    max_program_time_ms; // Max page program time
    bool        support_4byte_addr;
    bool        support_dual_page_program;
    bool        support_quad_read;
    bool        support_quad_page_program;
} flash_device_info_t;

// Flash device database
static const flash_device_info_t flash_devices[] = {
    // Winbond
    { 0xEF4014, "W25Q80", 1 * 1024 * 1024, 256, 4096, 32768, 65536, 300, 3, false },
    { 0xEF4015, "W25Q16", 2 * 1024 * 1024, 256, 4096, 32768, 65536, 300, 3, false },
    { 0xEF4016, "W25Q32", 4 * 1024 * 1024, 256, 4096, 32768, 65536, 300, 3, false, false, true, true },
    { 0xEF4017, "W25Q64", 8 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true, false, true, true },
    { 0xEF4018, "W25Q128", 16 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true, false, true, true },
    { 0xEF4019, "W25Q256", 32 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true, false, true, true },
    { 0xEF6019, "W25Q256JW", 32 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true, false, true, true },

    // Macronix
    { 0xC22018, "MX25L128", 16 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },
    { 0xC22019, "MX25L256", 32 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },

    // Gigadevice
    { 0xC84016, "GD25Q32", 4 * 1024 * 1024, 256, 4096, 32768, 65536, 300, 3, false },
    { 0xC84017, "GD25Q64", 8 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },
    { 0xC84018, "GD25Q128", 16 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },

    // Puya
    { 0x85C017, "PY25Q64", 8 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },
    { 0x85C018, "PY25Q128", 16 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },

    // XMC
    { 0x207017, "XM25Q64", 8 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },
    { 0x207018, "XM25Q128", 16 * 1024 * 1024, 256, 4096, 32768, 65536, 400, 3, true },

    { 0, NULL, 0, 0, 0, 0, 0, 0, 0, false, false, false, false } // Terminator
};

// Flash命令定义 (保持不变)
#define W25Q_CMD_WRITE_ENABLE       0x06
#define W25Q_CMD_WRITE_DISABLE      0x04
#define W25Q_CMD_READ_STATUS_REG1   0x05
#define W25Q_CMD_READ_STATUS_REG2   0x35
#define W25Q_CMD_READ_STATUS_REG3   0x15
#define W25Q_CMD_WRITE_STATUS_REG1  0x01
#define W25Q_CMD_WRITE_STATUS_REG2  0x31
#define W25Q_CMD_WRITE_STATUS_REG3  0x11
#define W25Q_CMD_CHIP_ERASE         0xC7
#define W25Q_CMD_ERASE_SECTOR_4K    0x20 // 3字节地址
#define W25Q_CMD_ERASE_SECTOR_4K_4B 0x21 // 4字节地址
#define W25Q_CMD_ERASE_BLOCK_32K    0x52
#define W25Q_CMD_ERASE_BLOCK_64K    0xD8
#define W25Q_CMD_PAGE_PROGRAM       0x02 // 3字节地址
#define W25Q_CMD_PAGE_PROGRAM_4B    0x12 // 4字节地址
#define W25Q_CMD_DUAL_PAGE_PROGRAM  0xA2
#define W25Q_CMD_QUAD_PAGE_PROGRAM  0x32
#define W25Q_CMD_QUAD_PAGE_PROGRAM_4B 0x34
#define W25Q_CMD_READ_DATA          0x03 // 3字节地址
#define W25Q_CMD_READ_DATA_4B       0x13 // 4字节地址
#define W25Q_CMD_FAST_READ          0x0B
#define W25Q_CMD_FAST_READ_4B       0x0C
#define W25Q_CMD_DUAL_FAST_READ     0x3B
#define W25Q_CMD_DUAL_FAST_READ_4B  0x3C
#define W25Q_CMD_QUAD_FAST_READ     0x6B
#define W25Q_CMD_QUAD_FAST_READ_4B  0x6C
#define W25Q_CMD_READ_JEDEC_ID      0x9F
#define W25Q_CMD_READ_UNIQUE_ID     0x4B
#define W25Q_CMD_ENTER_4BYTE_MODE   0xB7
#define W25Q_CMD_EXIT_4BYTE_MODE    0xE9
#define W25Q_CMD_ENABLE_RESET       0x66
#define W25Q_CMD_RESET              0x99
#define W25Q_CMD_POWER_DOWN         0xB9
#define W25Q_CMD_RELEASE_POWER_DOWN 0xAB

#define W25Q_MULTI_READ_CHUNK_SIZE  (32 * 1024)

// 状态寄存器位定义
#define W25Q_SR1_BUSY (1 << 0)
#define W25Q_SR1_WEL  (1 << 1)
#define W25Q_SR2_QE   (1 << 1)

// Flash驱动结构体
typedef struct {
    drv_spi_inst_t      spi_handle;
    drv_spi_inst_t      qspi2_handle;
    drv_spi_inst_t      qspi4_handle;
    bool                addr_4byte_mode;
    uint8_t             data_lines;
    flash_device_info_t device_info; // 存储检测到的设备信息
    uint32_t            jedec_id; // 实际读取的JEDEC ID
} flash_device_t;

// 延时函数
static void delay_ms(uint32_t ms) { usleep(ms * 1000); }

// CS控制函数
static void flash_cs_select(flash_device_t* flash) { }
static void flash_cs_deselect(flash_device_t* flash) { }

static const char* flash_data_line_name(uint8_t data_lines)
{
    if (data_lines == 4)
        return "4-line";
    if (data_lines == 2)
        return "2-line";

    return "1-line";
}

static void flash_set_data_lines(flash_device_t* flash, uint8_t data_lines)
{
    if (data_lines == 4)
        flash->data_lines = 4;
    else if (data_lines == 2)
        flash->data_lines = 2;
    else
        flash->data_lines = 1;
}

static bool flash_supports_4line(const flash_device_t* flash)
{
    return flash->device_info.support_quad_read && flash->qspi4_handle != NULL;
}

static drv_spi_inst_t flash_get_data_handle(flash_device_t* flash)
{
    if (flash->data_lines == 4)
        return flash->qspi4_handle;
    if (flash->data_lines == 2)
        return flash->qspi2_handle;

    return flash->spi_handle;
}

static uint32_t flash_infer_size_from_jedec(uint32_t jedec_id)
{
    uint8_t capacity = jedec_id & 0xFF;

    if (capacity >= 0x14 && capacity <= 0x1F)
        return 1UL << capacity;

    return 0;
}

static int flash_multi_data_xfer(flash_device_t* flash, uint8_t cmd, uint32_t addr, const void* tx_buf, void* rx_buf, size_t len,
                                 uint32_t dummy_cycles)
{
    struct rt_qspi_message msg;
    drv_spi_inst_t         handle = flash_get_data_handle(flash);

    if (handle == NULL) {
        printf("%s: No SPI handle for %s transfers\n", flash->device_info.name, flash_data_line_name(flash->data_lines));
        return -1;
    }

    memset(&msg, 0, sizeof(msg));

    msg.instruction.content    = cmd;
    msg.instruction.qspi_lines = 1;
    msg.instruction.size       = 8;

    msg.address.content    = addr;
    msg.address.qspi_lines = 1;
    msg.address.size       = flash->addr_4byte_mode ? 32 : 24;

    msg.dummy_cycles = dummy_cycles;

    msg.parent.send_buf   = tx_buf;
    msg.parent.recv_buf   = rx_buf;
    msg.parent.length     = len;
    msg.parent.cs_take    = 1;
    msg.parent.cs_release = 1;
    msg.parent.next       = NULL;
    msg.qspi_data_lines   = flash->data_lines;

    return drv_spi_transfer_message(handle, &msg);
}

static uint8_t flash_get_dual_read_cmd(const flash_device_t* flash)
{
    return flash->addr_4byte_mode ? W25Q_CMD_DUAL_FAST_READ_4B : W25Q_CMD_DUAL_FAST_READ;
}

static uint8_t flash_get_quad_read_cmd(const flash_device_t* flash)
{
    return flash->addr_4byte_mode ? W25Q_CMD_QUAD_FAST_READ_4B : W25Q_CMD_QUAD_FAST_READ;
}

static uint8_t flash_get_quad_program_cmd(const flash_device_t* flash)
{
    return flash->addr_4byte_mode ? W25Q_CMD_QUAD_PAGE_PROGRAM_4B : W25Q_CMD_QUAD_PAGE_PROGRAM;
}

// 读取状态寄存器1
static uint8_t flash_read_status_reg1(flash_device_t* flash)
{
    uint8_t cmd = W25Q_CMD_READ_STATUS_REG1;
    uint8_t status;

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 0);
    drv_spi_read(flash->spi_handle, &status, 1, 1);
    flash_cs_deselect(flash);

    return status;
}

static uint8_t flash_read_status_reg2(flash_device_t* flash)
{
    uint8_t cmd = W25Q_CMD_READ_STATUS_REG2;
    uint8_t status;

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 0);
    drv_spi_read(flash->spi_handle, &status, 1, 1);
    flash_cs_deselect(flash);

    return status;
}

// 等待忙状态结束
static bool flash_wait_busy(flash_device_t* flash, uint32_t timeout_ms)
{
    uint32_t wait_time = 0;

    while (wait_time < timeout_ms) {
        uint8_t status = flash_read_status_reg1(flash);
        if (!(status & W25Q_SR1_BUSY)) {
            return true;
        }
        delay_ms(1);
        wait_time++;
    }

    printf("%s: Timeout waiting for busy\n", flash->device_info.name);
    return false;
}

// 写使能
static bool flash_write_enable(flash_device_t* flash)
{
    uint8_t cmd = W25Q_CMD_WRITE_ENABLE;

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    flash_cs_deselect(flash);

    uint8_t status = flash_read_status_reg1(flash);
    return (status & W25Q_SR1_WEL) != 0;
}

// 写禁止
static void flash_write_disable(flash_device_t* flash)
{
    uint8_t cmd = W25Q_CMD_WRITE_DISABLE;

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    flash_cs_deselect(flash);
}

static bool flash_write_status_reg2(flash_device_t* flash, uint8_t value)
{
    uint8_t cmd_buf[2] = { W25Q_CMD_WRITE_STATUS_REG2, value };

    if (!flash_write_enable(flash)) {
        printf("%s: Failed to enable write before SR2 update\n", flash->device_info.name);
        return false;
    }

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, cmd_buf, sizeof(cmd_buf), 1);
    flash_cs_deselect(flash);

    return flash_wait_busy(flash, flash->device_info.max_program_time_ms);
}

// 读取JEDEC ID
static uint32_t flash_read_jedec_id(flash_device_t* flash)
{
    uint8_t cmd = W25Q_CMD_READ_JEDEC_ID;
    uint8_t id[3];

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 0);
    drv_spi_read(flash->spi_handle, id, 3, 1);
    flash_cs_deselect(flash);

    return (id[0] << 16) | (id[1] << 8) | id[2];
}

// 识别Flash设备
static bool flash_detect_device(flash_device_t* flash)
{
    uint32_t jedec_id = flash_read_jedec_id(flash);
    flash->jedec_id   = jedec_id;

    printf("Detected JEDEC ID: 0x%06X\n", jedec_id);

    // 在数据库中查找匹配的设备
    const flash_device_info_t* dev = flash_devices;
    while (dev->jedec_id != 0) {
        if (dev->jedec_id == jedec_id) {
            memcpy(&flash->device_info, dev, sizeof(flash_device_info_t));
            printf("Found matching device: %s, Size: %u MB\n", dev->name, dev->size / (1024 * 1024));
            return true;
        }
        dev++;
    }

    // 未找到匹配，使用默认参数
    printf("Warning: Unknown Flash device, using default parameters\n");
    uint32_t inferred_size = flash_infer_size_from_jedec(jedec_id);

    if (inferred_size) {
        printf("Inferred flash size from JEDEC capacity 0x%02X: %u MB\n", jedec_id & 0xFF, inferred_size / (1024 * 1024));
    } else {
        inferred_size = 16 * 1024 * 1024; // fallback only when capacity code is not recognized
    }

    flash->device_info.jedec_id            = jedec_id;
    flash->device_info.name                = "Unknown";
    flash->device_info.size                = inferred_size;
    flash->device_info.page_size           = 256;
    flash->device_info.sector_size         = 4096;
    flash->device_info.block_32k_size      = 32768;
    flash->device_info.block_64k_size      = 65536;
    flash->device_info.max_erase_time_ms   = 400;
    flash->device_info.max_program_time_ms = 3;
    flash->device_info.support_4byte_addr  = (jedec_id == 0xEF4019 || jedec_id == 0xC22019); // >16MB
    flash->device_info.support_dual_page_program = false;
    flash->device_info.support_quad_read = false;
    flash->device_info.support_quad_page_program = false;

    return false;
}

static bool flash_enable_quad_mode(flash_device_t* flash)
{
    uint8_t status2;

    if (!flash_supports_4line(flash)) {
        return true;
    }

    status2 = flash_read_status_reg2(flash);
    if (status2 & W25Q_SR2_QE) {
        return true;
    }

    if (!flash_write_status_reg2(flash, status2 | W25Q_SR2_QE)) {
        printf("%s: Failed to set QE bit\n", flash->device_info.name);
        return false;
    }

    status2 = flash_read_status_reg2(flash);
    if (!(status2 & W25Q_SR2_QE)) {
        printf("%s: QE bit did not latch\n", flash->device_info.name);
        return false;
    }

    return true;
}

// 进入4字节地址模式（仅当支持且需要时）
static bool flash_enter_4byte_mode(flash_device_t* flash)
{
    if (!flash->device_info.support_4byte_addr) {
        return true; // 不支持4字节模式，直接返回成功
    }

    uint8_t cmd = W25Q_CMD_ENTER_4BYTE_MODE;

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    flash_cs_deselect(flash);

    flash->addr_4byte_mode = true;
    return true;
}

// 退出4字节地址模式
static bool flash_exit_4byte_mode(flash_device_t* flash)
{
    if (!flash->device_info.support_4byte_addr) {
        return true;
    }

    uint8_t cmd = W25Q_CMD_EXIT_4BYTE_MODE;

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    flash_cs_deselect(flash);

    flash->addr_4byte_mode = false;
    return true;
}

// 擦除扇区（4KB）
static bool flash_erase_sector(flash_device_t* flash, uint32_t addr)
{
    // 检查地址是否超出范围
    if (addr >= flash->device_info.size) {
        printf("%s: Address 0x%06X out of range (max: 0x%06X)\n", flash->device_info.name, addr, flash->device_info.size - 1);
        return false;
    }

    if (!flash_write_enable(flash)) {
        printf("%s: Failed to enable write\n", flash->device_info.name);
        return false;
    }

    uint8_t cmd_buf[5];

    if (flash->addr_4byte_mode) {
        cmd_buf[0] = W25Q_CMD_ERASE_SECTOR_4K_4B;
        cmd_buf[1] = (addr >> 24) & 0xFF;
        cmd_buf[2] = (addr >> 16) & 0xFF;
        cmd_buf[3] = (addr >> 8) & 0xFF;
        cmd_buf[4] = addr & 0xFF;

        flash_cs_select(flash);
        drv_spi_write(flash->spi_handle, cmd_buf, 5, 1);
        flash_cs_deselect(flash);
    } else {
        cmd_buf[0] = W25Q_CMD_ERASE_SECTOR_4K;
        cmd_buf[1] = (addr >> 16) & 0xFF;
        cmd_buf[2] = (addr >> 8) & 0xFF;
        cmd_buf[3] = addr & 0xFF;

        flash_cs_select(flash);
        drv_spi_write(flash->spi_handle, cmd_buf, 4, 1);
        flash_cs_deselect(flash);
    }

    return flash_wait_busy(flash, flash->device_info.max_erase_time_ms);
}

// 页编程
static bool flash_page_program(flash_device_t* flash, uint32_t addr, const uint8_t* data, size_t len)
{
    if (len > flash->device_info.page_size) {
        printf("%s: Page program size exceed %u bytes\n", flash->device_info.name, flash->device_info.page_size);
        return false;
    }

    if (addr >= flash->device_info.size) {
        printf("%s: Address 0x%06X out of range\n", flash->device_info.name, addr);
        return false;
    }

    if (!flash_write_enable(flash)) {
        printf("%s: Failed to enable write\n", flash->device_info.name);
        return false;
    }

    if (flash->data_lines == 4 && flash->device_info.support_quad_page_program) {
        if (flash_multi_data_xfer(flash, flash_get_quad_program_cmd(flash), addr, data, NULL, len, 0) != (int)len) {
            printf("%s: Quad page program failed\n", flash->device_info.name);
            return false;
        }
    } else if (flash->data_lines == 2 && flash->device_info.support_dual_page_program) {
        if (flash_multi_data_xfer(flash, W25Q_CMD_DUAL_PAGE_PROGRAM, addr, data, NULL, len, 0) != (int)len) {
            printf("%s: Dual page program failed\n", flash->device_info.name);
            return false;
        }
    } else {
        uint8_t cmd_buf[5];

        flash_cs_select(flash);

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
        flash_cs_deselect(flash);
    }

    return flash_wait_busy(flash, flash->device_info.max_program_time_ms);
}

// 读取数据
static bool flash_read_data(flash_device_t* flash, uint32_t addr, uint8_t* data, size_t len)
{
    if (addr + len > flash->device_info.size) {
        printf("%s: Read range 0x%06X-0x%06X out of range\n", flash->device_info.name, addr, addr + len - 1);
        return false;
    }

    if (flash->data_lines == 4) {
        size_t offset = 0;

        while (offset < len) {
            size_t chunk = len - offset;
            if (chunk > W25Q_MULTI_READ_CHUNK_SIZE)
                chunk = W25Q_MULTI_READ_CHUNK_SIZE;

            if (flash_multi_data_xfer(flash, flash_get_quad_read_cmd(flash), addr + offset, NULL, data + offset, chunk, 8) != (int)chunk) {
                printf("%s: Quad fast read failed at 0x%06X\n", flash->device_info.name, addr + (uint32_t)offset);
                return false;
            }

            offset += chunk;
        }
    } else if (flash->data_lines == 2) {
        size_t offset = 0;

        while (offset < len) {
            size_t chunk = len - offset;
            if (chunk > W25Q_MULTI_READ_CHUNK_SIZE)
                chunk = W25Q_MULTI_READ_CHUNK_SIZE;

            if (flash_multi_data_xfer(flash, flash_get_dual_read_cmd(flash), addr + offset, NULL, data + offset, chunk, 8) != (int)chunk) {
                printf("%s: Dual fast read failed at 0x%06X\n", flash->device_info.name, addr + (uint32_t)offset);
                return false;
            }

            offset += chunk;
        }
    } else {
        uint8_t cmd_buf[5];

        flash_cs_select(flash);

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
        flash_cs_deselect(flash);
    }

    return true;
}

// 写入数据（处理跨页）
static bool flash_write(flash_device_t* flash, uint32_t addr, const uint8_t* data, size_t len)
{
    if (addr + len > flash->device_info.size) {
        printf("%s: Write range 0x%06X-0x%06X out of range\n", flash->device_info.name, addr, addr + len - 1);
        return false;
    }

    size_t offset = 0;

    while (offset < len) {
        size_t page_offset = addr % flash->device_info.page_size;
        size_t write_len   = flash->device_info.page_size - page_offset;

        if (write_len > (len - offset)) {
            write_len = len - offset;
        }

        if (!flash_page_program(flash, addr, data + offset, write_len)) {
            return false;
        }

        addr += write_len;
        offset += write_len;
    }

    return true;
}

// 芯片擦除
static bool flash_chip_erase(flash_device_t* flash)
{
    printf("%s: Performing chip erase...\n", flash->device_info.name);

    if (!flash_write_enable(flash)) {
        printf("%s: Failed to enable write\n", flash->device_info.name);
        return false;
    }

    uint8_t cmd = W25Q_CMD_CHIP_ERASE;

    flash_cs_select(flash);
    drv_spi_write(flash->spi_handle, &cmd, 1, 1);
    flash_cs_deselect(flash);

    // 芯片擦除可能需要较长时间（几十秒）
    bool result = flash_wait_busy(flash, 30000); // 30秒超时
    if (result) {
        printf("%s: Chip erase completed\n", flash->device_info.name);
    } else {
        printf("%s: Chip erase timeout\n", flash->device_info.name);
    }

    return result;
}

// 创建Flash实例
static flash_device_t* flash_create(int spi_id, int cs_pin, uint32_t baudrate)
{
    int             ret;
    flash_device_t* flash = (flash_device_t*)malloc(sizeof(flash_device_t));
    if (!flash) {
        printf("Failed to allocate flash structure\n");
        return NULL;
    }

    memset(flash, 0, sizeof(flash_device_t));

    ret = drv_spi_inst_create(spi_id, true, SPI_HAL_MODE_0, baudrate, 8, cs_pin, SPI_HAL_DATA_LINE_1, &flash->spi_handle);
    if (ret != 0) {
        printf("Failed to create 1-line SPI instance\n");
        free(flash);
        return NULL;
    }

    ret = drv_spi_inst_create(spi_id, true, SPI_HAL_MODE_0, baudrate, 8, cs_pin, SPI_HAL_DATA_LINE_2, &flash->qspi2_handle);
    if (ret != 0) {
        printf("Failed to create 2-line SPI instance\n");
        drv_spi_inst_destroy(&flash->spi_handle);
        free(flash);
        return NULL;
    }

    ret = drv_spi_inst_create(spi_id, true, SPI_HAL_MODE_0, baudrate, 8, cs_pin, SPI_HAL_DATA_LINE_4, &flash->qspi4_handle);
    if (ret != 0) {
        printf("Warning: Failed to create 4-line SPI instance, quad mode will be skipped\n");
        flash->qspi4_handle = NULL;
    }

    flash_set_data_lines(flash, 1);

    // 检测设备
    if (!flash_detect_device(flash)) {
        printf("Warning: Using default device parameters\n");
    }

    // 根据设备大小决定是否使用4字节地址模式
    flash->addr_4byte_mode = false;
    if (flash->device_info.size > 16 * 1024 * 1024) {
        printf("Device size >16MB, entering 4-byte address mode\n");
        flash_enter_4byte_mode(flash);
    }

    if (flash_supports_4line(flash) && !flash_enable_quad_mode(flash)) {
        printf("Warning: Failed to enable quad mode, 4-line tests will be skipped\n");
        flash->device_info.support_quad_read = false;
        flash->device_info.support_quad_page_program = false;
    }

    return flash;
}

// 销毁Flash实例
static void flash_destroy(flash_device_t* flash)
{
    if (!flash)
        return;

    if (flash->addr_4byte_mode) {
        flash_exit_4byte_mode(flash);
    }

    if (flash->spi_handle) {
        drv_spi_inst_destroy(&flash->spi_handle);
    }

    if (flash->qspi2_handle) {
        drv_spi_inst_destroy(&flash->qspi2_handle);
    }

    if (flash->qspi4_handle) {
        drv_spi_inst_destroy(&flash->qspi4_handle);
    }

    free(flash);
}

// 辅助函数：打印十六进制数据
static void print_hex_dump(const char* prefix, const uint8_t* data, size_t len)
{
    printf("%s", prefix);
    for (size_t i = 0; i < len; i++) {
        if (i % 16 == 0)
            printf("\n  ");
        printf("%02X ", data[i]);
    }
    printf("\n");
}

static bool flash_buffer_is_erased(const uint8_t* data, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        if (data[i] != 0xFF)
            return false;
    }

    return true;
}

static uint8_t* flash_alloc_aligned_buffer(size_t size)
{
    void* ptr = NULL;

    if (posix_memalign(&ptr, W25Q_TEST_ALIGN, size) != 0)
        return NULL;

    return (uint8_t*)ptr;
}

static bool flash_confirm_erase_with_single_read(flash_device_t* flash, uint32_t addr, uint8_t* data, size_t len)
{
    uint8_t saved_data_lines = flash->data_lines;
    bool    result;

    flash_set_data_lines(flash, 1);
    result = flash_read_data(flash, addr, data, len) && flash_buffer_is_erased(data, len);
    flash_set_data_lines(flash, saved_data_lines);

    return result;
}

// 测试用例1: 读取设备信息
static bool test_device_info(flash_device_t* flash)
{
    printf("\n=== Device Information Test ===\n");

    printf("Device Name: %s\n", flash->device_info.name);
    printf("JEDEC ID: 0x%06X\n", flash->jedec_id);
    printf("Size: %u bytes (%.2f MB)\n", flash->device_info.size, flash->device_info.size / (1024.0 * 1024.0));
    printf("Page Size: %u bytes\n", flash->device_info.page_size);
    printf("Sector Size: %u bytes\n", flash->device_info.sector_size);
    printf("Block Size (64K): %u bytes\n", flash->device_info.block_64k_size);
    printf("4-Byte Address Mode Support: %s\n", flash->device_info.support_4byte_addr ? "Yes" : "No");
    printf("Dual Page Program Support: %s\n", flash->device_info.support_dual_page_program ? "Yes" : "No");
    printf("Quad Fast Read Support: %s\n", flash->device_info.support_quad_read ? "Yes" : "No");
    printf("Quad Page Program Support: %s\n", flash->device_info.support_quad_page_program ? "Yes" : "No");
    printf("Current Address Mode: %s\n", flash->addr_4byte_mode ? "4-Byte" : "3-Byte");

    // 读取状态寄存器
    uint8_t status = flash_read_status_reg1(flash);
    printf("Status Register 1: 0x%02X\n", status);
    printf("  BUSY: %d, WEL: %d\n", (status & W25Q_SR1_BUSY) ? 1 : 0, (status & W25Q_SR1_WEL) ? 1 : 0);
    printf("Status Register 2: 0x%02X\n", flash_read_status_reg2(flash));

    return true;
}

// 测试用例2: 基本功能测试
static bool test_basic_functions(flash_device_t* flash)
{
    printf("\n=== Basic Function Test ===\n");

    // 1. 验证JEDEC ID
    uint32_t jedec_id = flash_read_jedec_id(flash);
    printf("1. JEDEC ID: 0x%06X", jedec_id);
    if (jedec_id == flash->jedec_id) {
        printf(" (Matches detected device)\n");
    } else {
        printf(" (WARNING: Different from detected!)\n");
    }

    // 2. 测试写使能/禁止
    printf("2. Write Enable/Disable Test\n");
    if (!flash_write_enable(flash)) {
        printf("   FAILED: Write enable failed\n");
        return false;
    }
    uint8_t status = flash_read_status_reg1(flash);
    if (!(status & W25Q_SR1_WEL)) {
        printf("   FAILED: WEL bit not set\n");
        return false;
    }
    printf("   Write enabled successfully\n");

    flash_write_disable(flash);
    status = flash_read_status_reg1(flash);
    if (status & W25Q_SR1_WEL) {
        printf("   FAILED: WEL bit still set\n");
        return false;
    }
    printf("   Write disabled successfully\n");
    printf("   PASSED\n");

    return true;
}

// 测试用例3: 读写测试
static bool test_read_write(flash_device_t* flash)
{
    printf("\n=== Read/Write Test ===\n");

    // 选择一个安全的测试地址（靠近中间位置）
    uint32_t test_addr = flash->device_info.size / 2;
    // 对齐到扇区边界
    test_addr = (test_addr / flash->device_info.sector_size) * flash->device_info.sector_size;

    size_t test_size = flash->device_info.page_size * 2; // 2页
    if (test_size > 1024)
        test_size = 1024; // 限制最大1KB

    uint8_t* write_buf = malloc(test_size);
    uint8_t* read_buf  = malloc(test_size);

    if (!write_buf || !read_buf) {
        printf("Failed to allocate test buffers\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    // 生成测试数据
    for (size_t i = 0; i < test_size; i++) {
        write_buf[i] = (uint8_t)(i & 0xFF);
    }

    printf("Testing at address 0x%06X, size %zu bytes\n", test_addr, test_size);

    // 1. 擦除扇区
    printf("1. Erasing sector at 0x%06X...\n", test_addr);
    if (!flash_erase_sector(flash, test_addr)) {
        printf("   FAILED: Sector erase failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Sector erased successfully\n");

    // 2. 验证擦除
    printf("2. Verifying erase...\n");
    if (!flash_read_data(flash, test_addr, read_buf, test_size)) {
        printf("   FAILED: Read failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    bool erase_ok = flash_buffer_is_erased(read_buf, test_size);
    if (!erase_ok) {
        printf("   FAILED: Erase verification failed\n");
        print_hex_dump("   First 32 bytes:", read_buf, 32);

        if (flash->data_lines > 1) {
            printf("   Re-checking erase with 1-line read...\n");
            if (flash_confirm_erase_with_single_read(flash, test_addr, read_buf, test_size)) {
                printf("   1-line read confirms the area is erased\n");
                printf("   Root cause: %s fast-read path returned incorrect data, erase itself did not fail\n",
                       flash_data_line_name(flash->data_lines));
            } else {
                printf("   1-line read also failed erase verification\n");
            }
        }

        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Erase verified successfully\n");

    // 3. 写入数据
    printf("3. Writing %zu bytes...\n", test_size);
    if (!flash_write(flash, test_addr, write_buf, test_size)) {
        printf("   FAILED: Write failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Write completed successfully\n");

    // 4. 读取并验证
    printf("4. Reading and verifying data...\n");
    memset(read_buf, 0, test_size);
    if (!flash_read_data(flash, test_addr, read_buf, test_size)) {
        printf("   FAILED: Read failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    if (memcmp(write_buf, read_buf, test_size) != 0) {
        printf("   FAILED: Data verification failed\n");
        print_hex_dump("   Written (first 32):", write_buf, 32);
        print_hex_dump("   Read (first 32):", read_buf, 32);
        free(write_buf);
        free(read_buf);
        return false;
    }

    printf("   Data verified successfully\n");
    printf("   PASSED\n");

    free(write_buf);
    free(read_buf);
    return true;
}

// 测试用例4: 边界测试
static bool test_boundary(flash_device_t* flash)
{
    printf("\n=== Boundary Test ===\n");

    bool    all_passed      = true;
    uint8_t write_pattern[] = { 0xAA, 0x55, 0x00, 0xFF };
    uint8_t read_buf[sizeof(write_pattern)];
    uint8_t single_read_buf[sizeof(write_pattern)];

    // 测试起始地址
    uint32_t boundaries[] = {
        0, // 起始
        flash->device_info.size - flash->device_info.page_size, // 结束边界
        flash->device_info.sector_size - 4, // 扇区内边界
        flash->device_info.sector_size + 4, // 跨扇区边界
    };

    for (int i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); i++) {
        uint32_t addr = boundaries[i];
        if (addr >= flash->device_info.size)
            continue;

        printf("Testing boundary at 0x%06X\n", addr);

        // 擦除所在扇区
        uint32_t sector_addr = (addr / flash->device_info.sector_size) * flash->device_info.sector_size;
        if (!flash_erase_sector(flash, sector_addr)) {
            printf("  Failed to erase sector at 0x%06X\n", sector_addr);
            all_passed = false;
            continue;
        }

        // 写入数据
        if (!flash_write(flash, addr, write_pattern, sizeof(write_pattern))) {
            printf("  Failed to write at boundary\n");
            all_passed = false;
            continue;
        }

        // 读取验证
        memset(read_buf, 0, sizeof(read_buf));
        if (!flash_read_data(flash, addr, read_buf, sizeof(read_buf))) {
            printf("  Failed to read at boundary\n");
            all_passed = false;
            continue;
        }

        if (memcmp(write_pattern, read_buf, sizeof(write_pattern)) != 0) {
            printf("  Data mismatch at boundary\n");
            print_hex_dump("  Expected:", write_pattern, sizeof(write_pattern));
            print_hex_dump("  Readback:", read_buf, sizeof(read_buf));

            if (flash->data_lines > 1) {
                uint8_t saved_data_lines = flash->data_lines;

                memset(single_read_buf, 0, sizeof(single_read_buf));
                flash_set_data_lines(flash, 1);
                if (flash_read_data(flash, addr, single_read_buf, sizeof(single_read_buf))) {
                    print_hex_dump("  1-line readback:", single_read_buf, sizeof(single_read_buf));
                } else {
                    printf("  1-line readback also failed\n");
                }
                flash_set_data_lines(flash, saved_data_lines);
            }

            all_passed = false;
        }
    }

    printf("Boundary test %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed;
}

// 测试用例5: 性能测试
static bool test_performance(flash_device_t* flash)
{
    printf("\n=== Performance Test ===\n");

    // 根据设备大小动态调整测试大小
    size_t test_size = flash->device_info.sector_size * 4; // 4个扇区
    if (test_size > 64 * 1024)
        test_size = 64 * 1024; // 最大64KB

    uint32_t test_addr = flash->device_info.size / 4; // 使用1/4位置

    uint8_t* buffer = malloc(test_size);
    if (!buffer) {
        printf("Failed to allocate test buffer\n");
        return false;
    }

    // 生成随机数据
    for (size_t i = 0; i < test_size; i++) {
        buffer[i] = (uint8_t)(rand() & 0xFF);
    }

    struct timespec start, end;

    // 1. 擦除性能
    printf("1. Erase performance (%zu KB):\n", test_size / 1024);
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t sectors_to_erase = test_size / flash->device_info.sector_size;
    for (uint32_t i = 0; i < sectors_to_erase; i++) {
        if (!flash_erase_sector(flash, test_addr + i * flash->device_info.sector_size)) {
            printf("   FAILED: Erase failed\n");
            free(buffer);
            return false;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double erase_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("   Erased %zu KB in %.3f seconds\n", test_size / 1024, erase_time);

    // 2. 写入性能
    printf("2. Write performance:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (!flash_write(flash, test_addr, buffer, test_size)) {
        printf("   FAILED: Write failed\n");
        free(buffer);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double write_time  = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double write_speed = (test_size / 1024.0) / write_time;
    printf("   Wrote %zu KB in %.3f seconds (%.1f KB/s)\n", test_size / 1024, write_time, write_speed);

    // 3. 读取性能
    printf("3. Read performance:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (!flash_read_data(flash, test_addr, buffer, test_size)) {
        printf("   FAILED: Read failed\n");
        free(buffer);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double read_time  = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double read_speed = (test_size / 1024.0) / read_time;
    printf("   Read %zu KB in %.3f seconds (%.1f KB/s)\n", test_size / 1024, read_time, read_speed);

    printf("   PASSED\n");

    free(buffer);
    return true;
}

// 测试用例6: 压力测试（可选，耗时较长）
static bool test_stress(flash_device_t* flash)
{
    printf("\n=== Stress Test (Optional) ===\n");
    printf("This test may take a long time. Skip? (y/n): ");

    // 简单跳过，实际使用时可添加用户输入
    printf("Skipping stress test\n");
    return true;

    /* 压力测试代码可以在这里实现
    // 多次擦写测试
    // 随机地址测试
    // 电源波动模拟等
    */
}

// SPI传输辅助函数
static int drv_spi_write_then_read(drv_spi_inst_t handle, const void* send_buf, size_t send_length, void* recv_buf,
                                   size_t recv_length)
{
    int                    ret = 0;
    struct rt_qspi_message msg;
    unsigned char*         ptr   = (unsigned char*)send_buf;
    size_t                 count = 0;

    msg.instruction.content    = ptr[0];
    msg.instruction.qspi_lines = 1;
    msg.instruction.size       = 8;
    count++;

    if (send_length > 1) {
        if (send_length >= 5) {
            msg.address.content = (ptr[1] << 24) | (ptr[2] << 16) | (ptr[3] << 8) | (ptr[4]);
            msg.address.size    = 32;
            count += 4;
        } else if (send_length >= 4) {
            msg.address.content = (ptr[1] << 16) | (ptr[2] << 8) | (ptr[3]);
            msg.address.size    = 24;
            count += 3;
        } else {
            ret = -1;
            goto out;
        }
        msg.address.qspi_lines = 1;
    } else {
        msg.address.content    = 0;
        msg.address.qspi_lines = 0;
        msg.address.size       = 0;
    }

    msg.alternate_bytes.content    = 0;
    msg.alternate_bytes.size       = 0;
    msg.alternate_bytes.qspi_lines = 0;

    if (count != send_length) {
        msg.dummy_cycles = (send_length - count) * 8;
    } else {
        msg.dummy_cycles = 0;
    }

    msg.parent.recv_buf   = recv_buf;
    msg.parent.send_buf   = NULL;
    msg.parent.length     = recv_length;
    msg.parent.cs_take    = 1;
    msg.parent.cs_release = 1;
    msg.parent.next       = NULL;

    ret = drv_spi_transfer_message(handle, &msg);
    if (ret != (int)recv_length) {
        printf("spi write then read fail: ret: %d\n", ret);
    }
out:
    return ret;
}

// 带参数的读写测试
static bool test_read_write_with_params(flash_device_t* flash, uint32_t test_addr, size_t test_size)
{
    printf("\n=== Read/Write Test (%s, Addr:0x%06X, Size:%zu KB) ===\n", flash_data_line_name(flash->data_lines), test_addr,
           test_size / 1024);

    uint8_t* write_buf = malloc(test_size);
    uint8_t* read_buf  = flash_alloc_aligned_buffer(test_size);

    if (!write_buf || !read_buf) {
        printf("Failed to allocate test buffers\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    // 生成测试数据
    for (size_t i = 0; i < test_size; i++) {
        write_buf[i] = (uint8_t)(i & 0xFF);
    }

    // 1. 擦除扇区
    printf("1. Erasing sectors...\n");
    uint32_t sectors_to_erase = (test_size + flash->device_info.sector_size - 1) / flash->device_info.sector_size;

    for (uint32_t i = 0; i < sectors_to_erase; i++) {
        uint32_t sector_addr = test_addr + i * flash->device_info.sector_size;
        if (!flash_erase_sector(flash, sector_addr)) {
            printf("   FAILED: Sector erase at 0x%06X\n", sector_addr);
            free(write_buf);
            free(read_buf);
            return false;
        }
    }
    printf("   Erased %u sectors successfully\n", sectors_to_erase);

    // 2. 验证擦除
    printf("2. Verifying erase...\n");
    if (!flash_read_data(flash, test_addr, read_buf, test_size)) {
        printf("   FAILED: Read failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    bool erase_ok = true;
    erase_ok = flash_buffer_is_erased(read_buf, test_size);
    if (!erase_ok) {
        printf("   FAILED: Erase verification failed\n");
        print_hex_dump("   First 32 bytes:", read_buf, 32);

        if (flash->data_lines > 1) {
            printf("   Re-checking erase with 1-line read...\n");
            if (flash_confirm_erase_with_single_read(flash, test_addr, read_buf, test_size)) {
                printf("   1-line read confirms the area is erased\n");
                printf("   Root cause: %s fast-read path returned incorrect data, erase itself did not fail\n",
                       flash_data_line_name(flash->data_lines));
            } else {
                printf("   1-line read also failed erase verification\n");
            }
        }

        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Erase verified successfully\n");

    // 3. 写入数据
    printf("3. Writing %zu bytes...\n", test_size);
    if (!flash_write(flash, test_addr, write_buf, test_size)) {
        printf("   FAILED: Write failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }
    printf("   Write completed successfully\n");

    // 4. 读取并验证
    printf("4. Reading and verifying data...\n");
    memset(read_buf, 0, test_size);
    if (!flash_read_data(flash, test_addr, read_buf, test_size)) {
        printf("   FAILED: Read failed\n");
        free(write_buf);
        free(read_buf);
        return false;
    }

    if (memcmp(write_buf, read_buf, test_size) != 0) {
        printf("   FAILED: Data verification failed\n");
        print_hex_dump("   Written (first 32):", write_buf, 32);
        print_hex_dump("   Read (first 32):", read_buf, 32);
        free(write_buf);
        free(read_buf);
        return false;
    }

    printf("   Data verified successfully\n");
    printf("   PASSED\n");

    free(write_buf);
    free(read_buf);
    return true;
}

// 带参数的性能测试
static bool test_performance_with_params(flash_device_t* flash, uint32_t test_addr, size_t test_size)
{
    printf("\n=== Performance Test (%s, Addr:0x%06X, Size:%zu KB) ===\n", flash_data_line_name(flash->data_lines), test_addr,
           test_size / 1024);

    uint8_t* buffer = malloc(test_size);
    if (!buffer) {
        printf("Failed to allocate test buffer\n");
        return false;
    }

    // 生成随机数据
    for (size_t i = 0; i < test_size; i++) {
        buffer[i] = (uint8_t)(rand() & 0xFF);
    }

    struct timespec start, end;

    // 1. 擦除性能
    printf("1. Erase performance:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t sectors_to_erase = test_size / flash->device_info.sector_size;
    for (uint32_t i = 0; i < sectors_to_erase; i++) {
        if (!flash_erase_sector(flash, test_addr + i * flash->device_info.sector_size)) {
            printf("   FAILED: Erase failed\n");
            free(buffer);
            return false;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double erase_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("   Erased %zu KB in %.3f seconds\n", test_size / 1024, erase_time);

    // 2. 写入性能
    printf("2. Write performance:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (!flash_write(flash, test_addr, buffer, test_size)) {
        printf("   FAILED: Write failed\n");
        free(buffer);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double write_time  = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double write_speed = (test_size / 1024.0) / write_time;
    printf("   Wrote %zu KB in %.3f seconds (%.1f KB/s)\n", test_size / 1024, write_time, write_speed);

    // 3. 读取性能
    printf("3. Read performance:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (!flash_read_data(flash, test_addr, buffer, test_size)) {
        printf("   FAILED: Read failed\n");
        free(buffer);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double read_time  = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double read_speed = (test_size / 1024.0) / read_time;
    printf("   Read %zu KB in %.3f seconds (%.1f KB/s)\n", test_size / 1024, read_time, read_speed);

    printf("   PASSED\n");

    free(buffer);
    return true;
}

// 打印帮助信息
static void print_help(const char* program_name)
{
    printf("\nSPI Flash Test Program\n");
    printf("======================\n\n");
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -i, --spi-id <id>       SPI controller ID (0: OSPI, 1: QSPI0, 2: QSPI1) [default: 2]\n");
    printf("  -c, --cs-pin <pin>      CS GPIO pin number [default: 11]\n");
    printf("  -b, --baudrate <mhz>    SPI baudrate in MHz [default: 10]\n");
    printf("  -t, --test <testname>    Run specific test (info,basic,rw, boundary,perf,all) [default: all]\n");
    printf("  -a, --address <hex>     Test start address in hex [default: auto]\n");
    printf("  -s, --size <kb>         Test size in KB [default: auto]\n");
    printf("  -4, --check-4line       Enable 4-line tests and treat failures as real failures [default: disabled]\n");
    printf("\nExamples:\n");
    printf("  %s                       # Run all tests on QSPI1, CS11, 10MHz\n", program_name);
    printf("  %s -i 1 -c 14 -b 5       # Use QSPI0, CS14, 5MHz\n", program_name);
    printf("  %s --spi-id 0 --cs-pin 8 --baudrate 20  # Use OSPI, CS8, 20MHz\n", program_name);
    printf("  %s -t perf -a 0x100000 -s 64  # Run performance test at 1MB, 64KB size\n", program_name);
    printf("  %s --check-4line        # Run 4-line tests and make their failures fail the testcase\n", program_name);
    printf("\nAvailable tests:\n");
    printf("  info      - Display device information\n");
    printf("  basic     - Test basic functions (JEDEC ID, write enable/disable)\n");
    printf("  rw        - Read/write test\n");
    printf("  boundary  - Boundary test (address edges)\n");
    printf("  perf      - Performance test\n");
    printf("  all       - Run all tests (default)\n");
    printf("\n");
}

// 测试选择函数
typedef struct {
    bool run_info;
    bool run_basic;
    bool run_rw;
    bool run_boundary;
    bool run_perf;
} test_config_t;

// 解析测试名称
static bool parse_test_name(const char* testname, test_config_t* config)
{
    if (strcmp(testname, "all") == 0) {
        config->run_info     = true;
        config->run_basic    = true;
        config->run_rw       = true;
        config->run_boundary = true;
        config->run_perf     = true;
    } else if (strcmp(testname, "info") == 0) {
        config->run_info = true;
    } else if (strcmp(testname, "basic") == 0) {
        config->run_basic = true;
    } else if (strcmp(testname, "rw") == 0) {
        config->run_rw = true;
    } else if (strcmp(testname, "boundary") == 0) {
        config->run_boundary = true;
    } else if (strcmp(testname, "perf") == 0) {
        config->run_perf = true;
    } else {
        printf("Unknown test: %s\n", testname);
        return false;
    }
    return true;
}

// 主函数
int main(int argc, char* argv[])
{
    // 默认参数
    int           spi_id       = 2;
    int           cs_pin       = 11;
    uint32_t      baudrate_mhz = 10;
    uint32_t      baudrate;
    test_config_t test_config  = { false, false, false, false, false };
    uint32_t      test_addr    = 0; // 0表示自动选择
    uint32_t      test_size_kb = 0; // 0表示自动选择
    bool          check_4line  = false;

    // 解析命令行参数
    static struct option long_options[] = { { "help", no_argument, 0, 'h' },         { "spi-id", required_argument, 0, 'i' },
                                            { "cs-pin", required_argument, 0, 'c' }, { "baudrate", required_argument, 0, 'b' },
                                            { "test", required_argument, 0, 't' },   { "address", required_argument, 0, 'a' },
                                            { "size", required_argument, 0, 's' },   { "check-4line", no_argument, 0, '4' },
                                            { 0, 0, 0, 0 } };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "hi:c:b:t:a:s:4", long_options, &option_index)) != -1) {
        switch (opt) {
        case 'h':
            print_help(argv[0]);
            return 0;

        case 'i':
            spi_id = atoi(optarg);
            if (spi_id < 0 || spi_id > 2) {
                printf("Error: SPI ID must be 0, 1, or 2\n");
                return -1;
            }
            break;

        case 'c':
            cs_pin = atoi(optarg);
            if (cs_pin < 0 || cs_pin > 47) {
                printf("Error: CS pin must be 0-47\n");
                return -1;
            }
            break;

        case 'b':
            baudrate_mhz = atoi(optarg);
            if (baudrate_mhz < 1 || baudrate_mhz > 50) {
                printf("Error: Baudrate must be 1-50 MHz\n");
                return -1;
            }
            break;

        case 't':
            if (!parse_test_name(optarg, &test_config)) {
                print_help(argv[0]);
                return -1;
            }
            break;

        case 'a':
            test_addr = strtoul(optarg, NULL, 16);
            break;

        case 's':
            test_size_kb = atoi(optarg);
            break;

        case '4':
            check_4line = true;
            break;

        default:
            print_help(argv[0]);
            return -1;
        }
    }

    // 如果没有指定测试，默认运行所有测试
    if (!test_config.run_info && !test_config.run_basic && !test_config.run_rw && !test_config.run_boundary
        && !test_config.run_perf) {
        test_config.run_info     = true;
        test_config.run_basic    = true;
        test_config.run_rw       = true;
        test_config.run_boundary = true;
        test_config.run_perf     = true;
    }

    // 转换波特率
    baudrate = baudrate_mhz * 1000 * 1000;

    printf("\n========================================\n");
    printf("SPI Flash Test Program\n");
    printf("========================================\n");

    // 显示配置
    printf("\nTest Configuration:\n");
    printf("  SPI ID:      %d ", spi_id);
    switch (spi_id) {
    case 0:
        printf("(OSPI)\n");
        break;
    case 1:
        printf("(QSPI0)\n");
        break;
    case 2:
        printf("(QSPI1)\n");
        break;
    }
    printf("  CS Pin:      %d\n", cs_pin);
    printf("  Baudrate:    %u MHz\n", baudrate_mhz);
    printf("  Test Addr:   %s\n", test_addr ? "0x%06X" : "Auto");
    if (test_addr)
        printf("               0x%06X\n", test_addr);
    printf("  Test Size:   %s\n", test_size_kb ? "%u KB" : "Auto");
    if (test_size_kb)
        printf("               %u KB\n", test_size_kb);
    printf("  Check 4-line:%s\n", check_4line ? " Yes" : " No");
    printf("  Tests:       ");
    if (test_config.run_info)
        printf("info ");
    if (test_config.run_basic)
        printf("basic ");
    if (test_config.run_rw)
        printf("rw ");
    if (test_config.run_boundary)
        printf("boundary ");
    if (test_config.run_perf)
        printf("perf ");
    printf("\n");

    // 配置引脚
    printf("\nConfiguring SPI pins...\n");
    if (spi_id == 0) {
        drv_fpioa_set_pin_func(15, OSPI_CLK);
        drv_fpioa_set_pin_func(16, OSPI_D0);
        drv_fpioa_set_pin_func(17, OSPI_D1);
        drv_fpioa_set_pin_func(18, OSPI_D2);
        drv_fpioa_set_pin_func(19, OSPI_D3);
        printf("  Using OSPI (CLK:15, D0:16, D1:17, D2:18, D3:19)\n");
    } else if (spi_id == 1) {
        drv_fpioa_set_pin_func(15, QSPI0_CLK);
        drv_fpioa_set_pin_func(16, QSPI0_D0);
        drv_fpioa_set_pin_func(17, QSPI0_D1);
        drv_fpioa_set_pin_func(18, QSPI0_D2);
        drv_fpioa_set_pin_func(19, QSPI0_D3);
        printf("  Using QSPI0 (CLK:15, D0:16, D1:17, D2:18, D3:19)\n");
    } else if (spi_id == 2) {
        drv_fpioa_set_pin_func(21, QSPI1_CLK);
        drv_fpioa_set_pin_func(40, QSPI1_D0);
        drv_fpioa_set_pin_func(41, QSPI1_D1);
        drv_fpioa_set_pin_func(42, QSPI1_D2);
        drv_fpioa_set_pin_func(43, QSPI1_D3);
        printf("  Using QSPI1 (CLK:21, D0:40, D1:41, D2:42, D3:43)\n");
    }

    // 创建Flash实例
    printf("\nInitializing Flash device...\n");
    flash_device_t* flash = flash_create(spi_id, cs_pin, baudrate);
    if (!flash) {
        printf("ERROR: Failed to create flash device\n");
        return -1;
    }

    printf("  Device:      %s\n", flash->device_info.name);
    printf("  JEDEC ID:    0x%06X\n", flash->jedec_id);
    printf("  Size:        %u MB (%.2f MB)\n", flash->device_info.size / (1024 * 1024),
           flash->device_info.size / (1024.0 * 1024.0));

    bool all_passed = true;

    // 自动选择测试地址（如果未指定）
    if (test_addr == 0) {
        // 选择中间位置，并对齐到扇区
        test_addr = (flash->device_info.size / 2);
        test_addr = (test_addr / flash->device_info.sector_size) * flash->device_info.sector_size;
        printf("  Auto Addr:   0x%06X\n", test_addr);
    }

    // 自动选择测试大小（如果未指定）
    if (test_size_kb == 0) {
        test_size_kb = 4; // 默认4KB
        if (flash->device_info.size >= 16 * 1024 * 1024) {
            test_size_kb = 64; // 对于16MB+设备，测试64KB
        }
        printf("  Auto Size:   %u KB\n", test_size_kb);
    }

    size_t test_size_bytes = test_size_kb * 1024;

    printf("\n========================================\n");
    printf("Running Tests on %s\n", flash->device_info.name);
    printf("========================================\n");

    // 运行选择的测试
    if (test_config.run_info) {
        if (!test_device_info(flash))
            all_passed = false;
    }

    if (test_config.run_basic) {
        if (!test_basic_functions(flash))
            all_passed = false;
    }

    if (test_config.run_rw || test_config.run_boundary || test_config.run_perf) {
        const uint8_t data_line_modes[] = { 1, 2, 4 };

        if (!check_4line && flash_supports_4line(flash))
            printf("\nNote: skipping 4-line tests by default; use --check-4line to enable them\n");

        for (size_t i = 0; i < sizeof(data_line_modes) / sizeof(data_line_modes[0]); i++) {
            if (data_line_modes[i] == 4 && (!flash_supports_4line(flash) || !check_4line))
                continue;

            flash_set_data_lines(flash, data_line_modes[i]);

            printf("\n========================================\n");
            printf("Data Line Mode: %s\n", flash_data_line_name(flash->data_lines));
            if (flash->data_lines == 4 && !flash->device_info.support_quad_page_program)
                printf("Program: 1-line page program, Read: 4-line fast read\n");
            else if (flash->data_lines == 4)
                printf("Program: 4-line page program, Read: 4-line fast read\n");
            else if (flash->data_lines == 2 && !flash->device_info.support_dual_page_program)
                printf("Program: 1-line page program, Read: 2-line fast read\n");
            if (flash->data_lines == 4)
                printf("Note: 4-line test failures affect the overall result\n");
            printf("========================================\n");

            if (test_config.run_rw) {
                // 为读写测试传递参数
                if (!test_read_write_with_params(flash, test_addr, test_size_bytes)) {
                    all_passed = false;
                }
            }

            if (test_config.run_boundary) {
                if (!test_boundary(flash))
                    all_passed = false;
            }

            if (test_config.run_perf) {
                if (!test_performance_with_params(flash, test_addr, test_size_bytes)) {
                    all_passed = false;
                }
            }
        }
    }

    // 总结
    printf("\n========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Device: %s\n", flash->device_info.name);
    printf("JEDEC ID: 0x%06X\n", flash->jedec_id);
    printf("Size: %u bytes (%.2f MB)\n", flash->device_info.size, flash->device_info.size / (1024.0 * 1024.0));
    printf("\nOverall Result: %s\n", all_passed ? "✓ ALL TESTS PASSED!" : "✗ SOME TESTS FAILED!");

    // 清理资源
    flash_destroy(flash);

    return all_passed ? 0 : -1;
}
