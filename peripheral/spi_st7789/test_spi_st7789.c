#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include "drv_spi.h"
#include "drv_gpio.h"
#include "drv_fpioa.h"  // 需要包含fpioa头文件

// LCD类型定义
#define LCD_TYPE_ST7789     1

// 颜色定义
#define COLOR_BLACK         0x0000
#define COLOR_WHITE         0xFFFF
#define COLOR_RED           0xF800
#define COLOR_GREEN         0x07E0
#define COLOR_BLUE          0x001F

// 简单的字体数据（8x16 ASCII字体）
static const uint8_t font8x16[][16] = {
    // 空格 (0x20)
    [0x20] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // '!' (0x21)
    [0x21] = {0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x00,0x00},
    // ',' (0x2C)
    [0x2C] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x18,0x30,0x00,0x00,0x00},
    // 'R' (0x52)
    [0x52] = {0x00,0x7C,0x66,0x66,0x66,0x7C,0x6C,0x66,0x66,0x66,0x66,0x00,0x00,0x00,0x00,0x00},
    // 'E' (0x45)
    [0x45] = {0x00,0x7E,0x60,0x60,0x60,0x7C,0x60,0x60,0x60,0x60,0x7E,0x00,0x00,0x00,0x00,0x00},
    // 'D' (0x44)
    [0x44] = {0x00,0x7C,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x7C,0x00,0x00,0x00,0x00,0x00},
    // 'H' (0x48)
    [0x48] = {0x00,0x66,0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x66,0x66,0x00,0x00,0x00,0x00,0x00},
    // 'e' (0x65)
    [0x65] = {0x00,0x00,0x00,0x00,0x3C,0x66,0x66,0x7E,0x60,0x60,0x3E,0x00,0x00,0x00,0x00,0x00},
    // 'l' (0x6C)
    [0x6C] = {0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00,0x00},
    // 'o' (0x6F)
    [0x6F] = {0x00,0x00,0x00,0x00,0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00,0x00},
    // 'W' (0x57)
    [0x57] = {0x00,0x66,0x66,0x66,0x66,0x66,0x66,0x7E,0x3C,0x24,0x24,0x00,0x00,0x00,0x00,0x00},
    // 'r' (0x72)
    [0x72] = {0x00,0x00,0x00,0x00,0x6C,0x76,0x60,0x60,0x60,0x60,0x60,0x00,0x00,0x00,0x00,0x00},
    // 'd' (0x64)
    [0x64] = {0x00,0x00,0x00,0x00,0x1E,0x36,0x66,0x66,0x66,0x66,0x3E,0x06,0x06,0x00,0x00,0x00},
    // 'B' (0x42)
    [0x42] = {0x00,0x7C,0x66,0x66,0x66,0x7C,0x66,0x66,0x66,0x66,0x7C,0x00,0x00,0x00,0x00,0x00},
    // 'L' (0x4C)
    [0x4C] = {0x00,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x7E,0x00,0x00,0x00,0x00,0x00},
    // 'U' (0x55)
    [0x55] = {0x00,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00,0x00,0x00,0x00,0x00},
    // 'G' (0x47)
    [0x47] = {0x00,0x3C,0x66,0x60,0x60,0x60,0x6E,0x66,0x66,0x66,0x3E,0x00,0x00,0x00,0x00,0x00},
    // 'V' (0x56)
    [0x56] = {0x00,0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x3C,0x18,0x18,0x00,0x00,0x00,0x00,0x00},
    // 'A' (0x41)
    [0x41] = {0x00,0x18,0x3C,0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x66,0x00,0x00,0x00,0x00,0x00},
};

// LCD驱动结构体
typedef struct {
    drv_spi_inst_t spi_inst;

    // GPIO实例
    drv_gpio_inst_t *gpio_dc;
    drv_gpio_inst_t *gpio_rst;
    drv_gpio_inst_t *gpio_bl;

    int type;
    int width;
    int height;
    int direction;
    bool hmirror;
    bool vflip;
    bool bgr;

    uint16_t *framebuffer;
} lcd_st7789_t;

// SPI写命令
static void lcd_write_command(lcd_st7789_t *lcd, uint8_t cmd, const uint8_t *data, size_t len) {
    // DC = 0 (命令模式)
    drv_gpio_value_set(lcd->gpio_dc, GPIO_PV_LOW);

    // CS = 0 (选中)

    // 发送命令
    drv_spi_write(lcd->spi_inst, &cmd, 1, 0);

    // DC = 1 (数据模式)
    drv_gpio_value_set(lcd->gpio_dc, GPIO_PV_HIGH);

    // 发送数据
    if (len > 0 && data != NULL) {
        drv_spi_write(lcd->spi_inst, data, len, 1);
    }

    // CS = 1 (取消选中)
}

// SPI写数据
static void lcd_write_data(lcd_st7789_t *lcd, const uint8_t *data, size_t len) {
    if (len == 0) return;

    // DC = 1 (数据模式)
    drv_gpio_value_set(lcd->gpio_dc, GPIO_PV_HIGH);

    // CS = 0 (选中)

    drv_spi_write(lcd->spi_inst, data, len, 1);

    // CS = 1 (取消选中)
}

// 发送命令序列
static void lcd_send_cmd_seq(lcd_st7789_t *lcd, const uint8_t *seq, size_t size) {
    const uint8_t *p = seq;
    const uint8_t *end = seq + size;

    while (p < end) {
        if (p[0] == 0x00) {
            // 延时命令
            usleep(p[1] * 1000);
            p += 2;
        } else {
            // 普通命令
            lcd_write_command(lcd, p[0], &p[2], p[1]);
            p += p[1] + 2;
        }
    }
}

// ST7789初始化
static void lcd_init_st7789(lcd_st7789_t *lcd) {
    const uint8_t init_seq[] = {
        0x11, 0,              // Sleep Out
        0x00, 5,              // Delay 5ms
        0x11, 0,              // Sleep Out (send twice)
        0x00, 30,             // Delay 30ms
        0x36, 1, 0x00,        // Memory Access Control
        0x3A, 1, 0x65,        // Interface Pixel Format
        0xB2, 5, 0x0C, 0x0C, 0x00, 0x33, 0x33, // Porch Setting
        0xB7, 1, 0x75,        // Gate Control
        0xBB, 1, 0x1A,        // VCOM Setting
        0xC0, 1, 0x2C,        // LCM Control
        0xC2, 1, 0x01,        // VDV and VRH Command Enable
        0xC3, 1, 0x13,        // VRH Set
        0xC4, 1, 0x20,        // VDV Set
        0xC6, 1, 0x0F,        // Frame Rate Control
        0xD0, 2, 0xA4, 0xA1,  // Power Control 1
        0xD6, 1, 0xA1,        // Power Control 2
        0xE0, 14, 0xD0, 0x0D, 0x14, 0x0D, 0x0D, 0x09, 0x38, 0x44, 0x4E, 0x3A, 0x17, 0x18, 0x2F, 0x30,
        0xE1, 14, 0xD0, 0x09, 0x0F, 0x08, 0x07, 0x14, 0x37, 0x44, 0x4D, 0x38, 0x15, 0x16, 0x2C, 0x2E,
        0x20, 0,              // Inversion Off
        0x29, 0,              // Display On
    };

    lcd_send_cmd_seq(lcd, init_seq, sizeof(init_seq));
}

// 设置显示区域
static void lcd_set_area(lcd_st7789_t *lcd, int x, int y, int width, int height) {
    uint8_t cmd_seq[14];

    // Column Address Set (0x2A)
    cmd_seq[0] = 0x2A;
    cmd_seq[1] = 4;
    cmd_seq[2] = (x >> 8) & 0xFF;
    cmd_seq[3] = x & 0xFF;
    cmd_seq[4] = ((x + width - 1) >> 8) & 0xFF;
    cmd_seq[5] = (x + width - 1) & 0xFF;

    // Row Address Set (0x2B)
    cmd_seq[6] = 0x2B;
    cmd_seq[7] = 4;
    cmd_seq[8] = (y >> 8) & 0xFF;
    cmd_seq[9] = y & 0xFF;
    cmd_seq[10] = ((y + height - 1) >> 8) & 0xFF;
    cmd_seq[11] = (y + height - 1) & 0xFF;

    // Write Memory Start (0x2C)
    cmd_seq[12] = 0x2C;
    cmd_seq[13] = 0;

    lcd_send_cmd_seq(lcd, cmd_seq, sizeof(cmd_seq));
}

// 应用配置
static void lcd_apply_config(lcd_st7789_t *lcd) {
    uint8_t direction = 0;

    if (lcd->vflip) {
        direction |= (1 << 7);  // MY bit
    }

    if (lcd->hmirror) {
        direction |= (1 << 6);  // MX bit
    }

    if (lcd->width > lcd->height) {
        direction |= (1 << 5);  // MV bit (landscape)
    }

    if (lcd->bgr) {
        direction |= (1 << 3);  // BGR bit
    }

    uint8_t cmd[3] = {0x36, 1, direction};
    lcd_send_cmd_seq(lcd, cmd, sizeof(cmd));

    lcd->direction = direction;
}

// 创建LCD实例
lcd_st7789_t* lcd_create(int spi_id, int pin_dc, int pin_cs, int pin_rst, int pin_bl) {
    lcd_st7789_t *lcd = (lcd_st7789_t*)malloc(sizeof(lcd_st7789_t));
    if (!lcd) {
        printf("Failed to allocate LCD structure\n");
        return NULL;
    }

    memset(lcd, 0, sizeof(lcd_st7789_t));

    int ret;
    // 初始化SPI (50MHz, Mode 3: CPOL=1, CPHA=1)
    ret = drv_spi_inst_create(spi_id, true, SPI_HAL_MODE_3, 50000000,
                                         8, pin_cs, SPI_HAL_DATA_LINE_1, &lcd->spi_inst);
    if (ret != 0) {
        printf("Failed to create SPI instance\n");
        free(lcd);
        return NULL;
    }

    // 创建GPIO实例 - DC引脚（必须）
    if (drv_gpio_inst_create(pin_dc, &lcd->gpio_dc) != 0) {
        printf("Failed to create GPIO instance for DC pin\n");
        drv_spi_inst_destroy(&lcd->spi_inst);
        free(lcd);
        return NULL;
    }
    drv_gpio_mode_set(lcd->gpio_dc, GPIO_DM_OUTPUT);
    drv_gpio_value_set(lcd->gpio_dc, GPIO_PV_HIGH);  // DC默认高电平

    // 创建GPIO实例 - RST引脚（可选）
    if (pin_rst >= 0) {
        if (drv_gpio_inst_create(pin_rst, &lcd->gpio_rst) != 0) {
            printf("Failed to create GPIO instance for RST pin\n");
            drv_gpio_inst_destroy(&lcd->gpio_dc);
            drv_spi_inst_destroy(&lcd->spi_inst);
            free(lcd);
            return NULL;
        }
        drv_gpio_mode_set(lcd->gpio_rst, GPIO_DM_OUTPUT);
    }

    // 创建GPIO实例 - BL引脚（可选）
    if (pin_bl >= 0) {
        if (drv_gpio_inst_create(pin_bl, &lcd->gpio_bl) != 0) {
            printf("Failed to create GPIO instance for BL pin\n");
            if (lcd->gpio_rst) drv_gpio_inst_destroy(&lcd->gpio_rst);
            drv_gpio_inst_destroy(&lcd->gpio_dc);
            drv_spi_inst_destroy(&lcd->spi_inst);
            free(lcd);
            return NULL;
        }
        drv_gpio_mode_set(lcd->gpio_bl, GPIO_DM_OUTPUT);
    }

    lcd->type = LCD_TYPE_ST7789;

    return lcd;
}

// 配置LCD
void lcd_configure(lcd_st7789_t *lcd, int width, int height, bool hmirror, bool vflip, bool bgr) {
    lcd->width = width;
    lcd->height = height;
    lcd->hmirror = hmirror;
    lcd->vflip = vflip;
    lcd->bgr = bgr;

    // 分配帧缓冲
    if (lcd->framebuffer) {
        free(lcd->framebuffer);
    }
    lcd->framebuffer = (uint16_t*)malloc(width * height * sizeof(uint16_t));
    if (!lcd->framebuffer) {
        printf("Failed to allocate framebuffer\n");
    }
}

// 初始化LCD
void lcd_init(lcd_st7789_t *lcd) {
    // 硬件复位
    if (lcd->gpio_rst) {
        drv_gpio_value_set(lcd->gpio_rst, GPIO_PV_LOW);
        usleep(100000);  // 100ms
        drv_gpio_value_set(lcd->gpio_rst, GPIO_PV_HIGH);
        usleep(100000);  // 100ms
    }

    // 初始化屏幕
    lcd_init_st7789(lcd);

    // 应用配置
    lcd_apply_config(lcd);

    // 打开背光
    if (lcd->gpio_bl) {
        drv_gpio_value_set(lcd->gpio_bl, GPIO_PV_HIGH);
    }

    // 清屏
    if (lcd->framebuffer) {
        memset(lcd->framebuffer, 0, lcd->width * lcd->height * sizeof(uint16_t));
    }
}

// RGB888转RGB565
static uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

// 填充颜色
void lcd_fill(lcd_st7789_t *lcd, uint16_t color) {
    if (!lcd->framebuffer) return;

    for (int i = 0; i < lcd->width * lcd->height; i++) {
        lcd->framebuffer[i] = color;
    }
}

// 画像素
void lcd_pixel(lcd_st7789_t *lcd, int x, int y, uint16_t color) {
    if (!lcd->framebuffer) return;
    if (x < 0 || x >= lcd->width || y < 0 || y >= lcd->height) return;

    lcd->framebuffer[y * lcd->width + x] = color;
}

// 绘制字符
void lcd_draw_char(lcd_st7789_t *lcd, int x, int y, char c, uint16_t color, int scale) {
    if (!lcd->framebuffer) return;
    if (c < 0 || c > 127) return;

    const uint8_t *char_data = font8x16[(int)c];

    for (int row = 0; row < 16; row++) {
        uint8_t line = char_data[row];
        for (int col = 0; col < 8; col++) {
            if (line & (0x80 >> col)) {
                // 根据scale放大字符
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        lcd_pixel(lcd, x + col * scale + sx, y + row * scale + sy, color);
                    }
                }
            }
        }
    }
}

// 绘制字符串
void lcd_draw_string(lcd_st7789_t *lcd, int x, int y, const char *str, uint16_t color, int scale) {
    if (!lcd->framebuffer || !str) return;

    int char_width = 8 * scale;
    int char_x = x;

    while (*str) {
        lcd_draw_char(lcd, char_x, y, *str, color, scale);
        char_x += char_width;
        str++;
    }
}

// 显示缓冲区内容
void lcd_show(lcd_st7789_t *lcd) {
    if (!lcd->framebuffer) return;

    // 设置全屏区域
    lcd_set_area(lcd, 0, 0, lcd->width, lcd->height);

    // 转换字节序并发送数据
    uint8_t *buffer = (uint8_t*)lcd->framebuffer;
    size_t size = lcd->width * lcd->height * 2;

    // ST7789需要大端字节序，转换数据
    uint8_t *temp = (uint8_t*)malloc(size);
    if (temp) {
        for (size_t i = 0; i < size; i += 2) {
            temp[i] = buffer[i + 1];
            temp[i + 1] = buffer[i];
        }
        lcd_write_data(lcd, temp, size);
        free(temp);
    }
}

// 销毁LCD实例
void lcd_destroy(lcd_st7789_t *lcd) {
    if (!lcd) return;

    if (lcd->framebuffer) {
        free(lcd->framebuffer);
    }

    // 销毁GPIO实例
    if (lcd->gpio_bl) {
        drv_gpio_inst_destroy(&lcd->gpio_bl);
    }
    if (lcd->gpio_rst) {
        drv_gpio_inst_destroy(&lcd->gpio_rst);
    }
    if (lcd->gpio_dc) {
        drv_gpio_inst_destroy(&lcd->gpio_dc);
    }

    if (lcd->spi_inst) {
        drv_spi_inst_destroy(&lcd->spi_inst);
    }

    free(lcd);
}

// 主函数示例
int main(void)
{
    // 首先需要配置引脚功能为GPIO
    // 假设使用的引脚如下（根据实际硬件调整）
    int pin_dc = 20;
    int pin_cs = 19;
    int pin_rst = 12;
    int pin_bl = -1;
    int spi_id = 1;

    // 配置引脚功能为GPIO（需要使用fpioa配置）
    drv_fpioa_set_pin_func(pin_dc, GPIO0 + pin_dc);
    drv_fpioa_set_pin_func(pin_rst, GPIO0 + pin_rst);
    drv_fpioa_set_pin_func(pin_bl, GPIO0 + pin_bl);

    // 配置SPI功能
    drv_fpioa_set_pin_func(15, QSPI0_CLK);
    drv_fpioa_set_pin_func(16, QSPI0_D0);

    // 创建LCD实例
    lcd_st7789_t *lcd = lcd_create(spi_id, pin_dc, pin_cs, pin_rst, pin_bl);
    if (!lcd) {
        printf("Failed to create LCD\n");
        return -1;
    }

    // 配置LCD (320x240, vflip=true)
    lcd_configure(lcd, 320, 240, false, true, false);

    // 初始化LCD
    lcd_init(lcd);

    // 清屏
    lcd_fill(lcd, COLOR_BLACK);

    // 绘制三行文字
    lcd_draw_string(lcd, 0, 0, "RED, Hello World!", COLOR_RED, 2);
    lcd_draw_string(lcd, 0, 40, "GREEN, Hello World!", COLOR_GREEN, 2);
    lcd_draw_string(lcd, 0, 80, "BLUE, Hello World!", COLOR_BLUE, 2);

    // 显示到LCD
    lcd_show(lcd);

    printf("LCD display completed!\n");

    // 保持显示5秒
    sleep(5);

    // 清理资源
    lcd_destroy(lcd);

    return 0;
}
