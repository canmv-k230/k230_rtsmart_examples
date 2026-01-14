#include <stdio.h>
#include <signal.h>
#include <stdbool.h>
#include <assert.h>

#include "drv_tsensor.h"

static bool exit_flag;

static void sig_handler(int sig_no) {

    exit_flag = true;

    printf("exit sig = %d\n", sig_no);
}

int main(int argc, char **argv)
{
    int ret, cnt = 0;
    double temp;
    uint8_t mode, trim;

    exit_flag = false;

    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    mode = RT_DEVICE_TS_CTRL_MODE_CONTINUUOS;
    trim = 0;

    while (!exit_flag) {
        assert(0 == drv_tsensor_set_trim(trim));
        assert(0 == drv_tsensor_set_mode(mode));

        assert(0 == drv_tsensor_get_mode(&mode));
        assert(0 == drv_tsensor_get_trim(&trim));

        printf("mode = %d, trim = %d\n", mode, trim);

        while (!exit_flag) {
            ret = drv_tsensor_read_temperature(&temp);
            if (ret) {
                printf("drv_tsensor_read_temperature fail = %d\n", ret);
                break;
            }
            printf("temperature = %f\n", temp);
            usleep(1000000);
            if (++cnt >= 3) {
                break;
            }
        }

        trim = ++trim % (RT_DEVICE_TS_CTRL_MAX_TRIM + 1);
        cnt = 0;

        if (trim == RT_DEVICE_TS_CTRL_MAX_TRIM) {
            if (mode == RT_DEVICE_TS_CTRL_MODE_CONTINUUOS) {
                mode = RT_DEVICE_TS_CTRL_MODE_SINGLE;
            } else {
                mode = RT_DEVICE_TS_CTRL_MODE_CONTINUUOS;
            }
        }
    }

    return 0;
}
