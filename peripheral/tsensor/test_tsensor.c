#include <stdio.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>

#include "drv_tsensor.h"

static volatile sig_atomic_t exit_flag;

static void sig_handler(int sig_no) {
    (void)sig_no;
    exit_flag = 1;
}

int main(int argc, char **argv)
{
    int ret;
    double temp;
    uint8_t mode, trim;

    (void)argc;
    (void)argv;

    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    if (drv_tsensor_get_mode(&mode) || drv_tsensor_get_trim(&trim)) {
        return 1;
    }

    printf("mode = %u, trim = %u\n", (unsigned)mode, (unsigned)trim);

    while (!exit_flag) {
        ret = drv_tsensor_read_temperature(&temp);
        if (ret) {
            printf("drv_tsensor_read_temperature fail = %d\n", ret);
        } else {
            printf("temperature = %f\n", temp);
        }
        usleep(1000000);
    }

    return 0;
}
