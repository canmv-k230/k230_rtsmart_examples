#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "drv_fpioa.h"
#include "drv_i2c.h"

// SSD1306 Configuration
#define SSD1306_I2C_ADDR 0x3C // Default I2C address (may be 0x3D)
#define SSD1306_WIDTH    128
#define SSD1306_HEIGHT   64

// SSD1306 Command Constants
#define SSD1306_SETCONTRAST      0x81
#define SSD1306_DISPLAYON        0xAF
#define SSD1306_DISPLAYOFF       0xAE
#define SSD1306_NORMALDISPLAY    0xA6
#define SSD1306_INVERTDISPLAY    0xA7
#define SSD1306_SETPAGESTART     0xB0 // Page start address (0-7)
#define SSD1306_SETCOLSTART_LOW  0x00
#define SSD1306_SETCOLSTART_HIGH 0x10
#define SSD1306_SETSTARTLINE     0x40

// Buffer for OLED screen (1bpp, 128x64)
static uint8_t oled_buffer[SSD1306_HEIGHT / 8][SSD1306_WIDTH];

static int ssd1306_refresh(drv_i2c_inst_t* i2c);

// Send a command to SSD1306
static int ssd1306_write_cmd(drv_i2c_inst_t* i2c, uint8_t cmd)
{
    uint8_t   buf[2] = { 0x00, cmd }; // Control byte (0x00 = command)
    i2c_msg_t msg    = { .addr = SSD1306_I2C_ADDR, .flags = DRV_I2C_WR, .len = 2, .buf = buf };
    return drv_i2c_transfer(i2c, &msg, 1);
}

// Send data to SSD1306
static int ssd1306_write_data(drv_i2c_inst_t* i2c, uint8_t* data, uint16_t len)
{
    uint8_t* buf = malloc(len + 1);
    buf[0]       = 0x40; // Control byte (0x40 = data)
    memcpy(buf + 1, data, len);

    i2c_msg_t msg = { .addr = SSD1306_I2C_ADDR, .flags = DRV_I2C_WR, .len = len + 1, .buf = buf };
    int       ret = drv_i2c_transfer(i2c, &msg, 1);
    free(buf);
    return ret;
}

// Initialize SSD1306
static int ssd1306_init(drv_i2c_inst_t* i2c)
{
    // Turn off display
    if (ssd1306_write_cmd(i2c, SSD1306_DISPLAYOFF) != 0) {
        return -1;
    }

    // Fundamental commands
    uint8_t init_seq[] = {
        0xD5,
        0x80, // Set display clock divide ratio/oscillator frequency
        0xA8,
        0x3F, // Set multiplex ratio (1:64)
        0xD3,
        0x00, // Set display offset
        0x40, // Set start line (0)
        0x8D,
        0x14, // Charge pump (enable)
        0x20,
        0x00, // Memory mode (horizontal)
        0xA1, // Segment remap (column 127 mapped to SEG0)
        0xC8, // COM output scan direction (remapped mode)
        0xDA,
        0x12, // COM pins hardware configuration
        SSD1306_SETCONTRAST,
        0x7F, // Contrast
        0xA4, // Resume to RAM content display
        SSD1306_NORMALDISPLAY,
    };

    for (size_t i = 0; i < sizeof(init_seq); i++) {
        if (ssd1306_write_cmd(i2c, init_seq[i]) != 0) {
            return -1;
        }
    }

    // Clear screen
    memset(oled_buffer, 0, sizeof(oled_buffer));
    if (ssd1306_refresh(i2c) != 0) {
        return -1;
    }

    // Turn on display
    return ssd1306_write_cmd(i2c, SSD1306_DISPLAYON);
}

// Refresh entire OLED from buffer
static int ssd1306_refresh(drv_i2c_inst_t* i2c)
{
    for (uint8_t page = 0; page < 8; page++) {
        ssd1306_write_cmd(i2c, SSD1306_SETPAGESTART + page);
        ssd1306_write_cmd(i2c, SSD1306_SETCOLSTART_LOW);
        ssd1306_write_cmd(i2c, SSD1306_SETCOLSTART_HIGH);

        if (ssd1306_write_data(i2c, oled_buffer[page], SSD1306_WIDTH) != 0) {
            return -1;
        }
    }
    return 0;
}

// Set a pixel in the buffer (1=on, 0=off)
static void ssd1306_set_pixel(int x, int y, int on)
{
    if (x >= 0 && x < SSD1306_WIDTH && y >= 0 && y < SSD1306_HEIGHT) {
        if (on) {
            oled_buffer[y / 8][x] |= (1 << (y % 8));
        } else {
            oled_buffer[y / 8][x] &= ~(1 << (y % 8));
        }
    }
}

// Test pattern: Draw a checkerboard
static void draw_test_pattern(void)
{
    for (int y = 0; y < SSD1306_HEIGHT; y++) {
        for (int x = 0; x < SSD1306_WIDTH; x++) {
            ssd1306_set_pixel(x, y, (x + y) % 8 < 4);
        }
    }
}

int main()
{
    drv_i2c_inst_t* i2c       = NULL;
    uint32_t        i2c_clock = 400 * 1000;

    drv_fpioa_set_pin_func(11, IIC2_SCL);
    drv_fpioa_set_pin_func(12, IIC2_SDA);

    printf("K230 OLED Test Configuration:\n");
    printf("  PIN11 -> SCL\n");
    printf("  PIN12 -> SDA\n");
    printf("  I2C Bus: %d, Frequency: %d Hz\n", 2, i2c_clock);

    // Initialize I2C (adjust I2C ID/freq per your hardware)
    if (drv_i2c_inst_create(2, i2c_clock, 1000, 0xff, 0xff, &i2c) < 0) { // 400kHz, 1s timeout
        printf("I2C init failed!\n");
        return -1;
    }

    // Initialize SSD1306
    if (ssd1306_init(i2c) != 0) {
        printf("SSD1306 init failed!\n");
        drv_i2c_inst_destroy(&i2c);
        return -1;
    }

    // Draw test pattern
    draw_test_pattern();
    if (ssd1306_refresh(i2c) != 0) {
        printf("Refresh failed!\n");
    }

    printf("OLED test complete. Check your screen!\n");

    // Cleanup
    drv_i2c_inst_destroy(&i2c);

    return 0;
}
