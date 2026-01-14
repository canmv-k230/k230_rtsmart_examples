#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

#include "drv_fpioa.h"
#include "drv_gpio.h"
#include "drv_sbus.h"

#define KEY_PIN_NUM (21)
#define UART3_TX (50)
#define UART3_RX (51)

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s:%d - %s\n", __func__, __LINE__, msg); \
        return -1; \
    } \
} while(0)

static bool exit_flag;

static void sig_handler(int sig_no) {

    exit_flag = true;

    printf("exit sig = %d\n", sig_no);
}

int main(int argc, char *argv[])
{
    int ret = 0;
    sbus_dev_t dev;
    drv_gpio_inst_t *gpio_in = NULL;

    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    ret = drv_fpioa_set_pin_func(KEY_PIN_NUM, GPIO0 + KEY_PIN_NUM);
    TEST_ASSERT(ret == 0, "drv_fpioa_set_pin_func GPIO");
    ret = drv_gpio_inst_create(KEY_PIN_NUM, &gpio_in);
    TEST_ASSERT(ret == 0, "drv_gpio_inst_create");
    ret = drv_gpio_mode_set(gpio_in, GPIO_DM_INPUT);
    TEST_ASSERT(ret == 0, "drv_gpio_mode_set");
    ret = drv_fpioa_set_pin_func(UART3_RX, UART3_RXD);
    TEST_ASSERT(ret == 0, "drv_fpioa_set_pin_func UART3_RXD");
    ret = drv_fpioa_set_pin_func(UART3_TX, UART3_TXD);
    TEST_ASSERT(ret == 0, "drv_fpioa_set_pin_func UART3_TXD");

    dev = sbus_create(3);
    if (!dev) {
        ret = -1;
        goto out_1;
    }


    for (int i = 0; i < 16; i++)
        sbus_set_channel(dev, i, 172);

    for (int i = 0; i < 50; i++) {
        sbus_send_frame(dev);
        usleep(10 * 1000);
    }

    int value[10] = {456, 678, 890, 1023, 1234, 1456, 1678, 1890, 2000, 2047};
    int idx = 0, ch = 0;

    while (!exit_flag) {
        if (drv_gpio_value_get(gpio_in) == GPIO_PV_LOW) {

            for (int i = 0; i < 16; i++)
                sbus_set_channel(dev, i, 172);

            sbus_set_channel(dev, ch, value[idx]);
            printf("set ch[%d] = %d\n", ch, value[idx]);
            idx = (++idx == (sizeof(value)/sizeof(value[0])))? 0 : idx;

            if (idx == 0) {
                ch = (++ch == 16) ? 0 : ch;
            }
            usleep(200 * 1000);
        }

        sbus_send_frame(dev);
        usleep(10 * 1000);
    }

    sbus_destroy(dev);
out_1:
    drv_gpio_inst_destroy(&gpio_in);

    return ret;
}
