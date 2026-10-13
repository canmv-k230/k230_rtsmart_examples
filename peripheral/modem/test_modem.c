/*
 * hal_modem 示例:验证 K230 侧的 ML307H 4G 唤醒链路。
 *
 * 用法: modem.elf [mode] [gpio=NN] [at=/dev/ttyUSBX]
 *   mode:
 *     up    (默认) init -> wake,然后退出(快速联网检查)
 *     hold        init -> wake,保持 REQ_READY 高,然后退出
 *     cycle       init -> wake -> enter_edrx(联网 + 关机全流程)
 *     edrx        init -> enter_edrx(只关机,不联网)
 *     imei        init -> 读 IMEI,然后退出(会话已建立时运行)
 *   gpio=NN          覆盖 REQ_READY GPIO(默认 26)
 *   at=/dev/ttyUSBX  覆盖 AT 口(默认 /dev/ttyUSB2)
 *
 * IMEI 特意不放在联网路径里(会多一次 AT 往返),需要时用 imei 模式。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hal_modem.h"

enum { MODE_UP, MODE_HOLD, MODE_CYCLE, MODE_EDRX, MODE_IMEI };

int main(int argc, char **argv)
{
    modem_cfg_t cfg;
    int         mode = MODE_UP;
    int         i;

    memset(&cfg, 0, sizeof(cfg)); /* 全 0 => 用默认值 */

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "up"))          mode = MODE_UP;
        else if (!strcmp(argv[i], "hold"))   mode = MODE_HOLD;
        else if (!strcmp(argv[i], "cycle"))  mode = MODE_CYCLE;
        else if (!strcmp(argv[i], "edrx"))   mode = MODE_EDRX;
        else if (!strcmp(argv[i], "imei"))   mode = MODE_IMEI;
        else if (!strncmp(argv[i], "gpio=", 5)) cfg.req_ready_gpio = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "at=", 3))   cfg.at_dev = argv[i] + 3;
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') cfg.req_ready_gpio = atoi(argv[i]);
        else printf("ignoring unknown arg: %s\n", argv[i]);
    }

    printf("=== [MODEM DEMO] mode=%s ===\n",
           mode == MODE_UP ? "up" : mode == MODE_HOLD ? "hold" :
           mode == MODE_CYCLE ? "cycle" : mode == MODE_EDRX ? "edrx" : "imei");

    if (modem_init(&cfg) != 0) {
        printf("modem_init failed\n");
        return 1;
    }

    const char *ws = board_get_wakeup_source();
    printf("wakeup source: %s\n", ws ? ws : "unknown");

    /* 只关机:不联网。 */
    if (mode == MODE_EDRX) {
        modem_enter_edrx();      /* 成功不返回 */
        printf("modem_enter_edrx returned -> shutdown failed\n");
        modem_deinit();
        return 1;
    }

    /* 只读 IMEI:趁会话已建立(如 hold 之后),不联网、不动 REQ_READY。 */
    if (mode == MODE_IMEI) {
        char imei[MODEM_IMEI_LEN + 1] = { 0 };

        if (modem_get_imei(imei, sizeof(imei)) == 0)
            printf("IMEI: %s\n", imei);
        else
            printf("IMEI: <unavailable> (先建立会话,如 hold)\n");
        return 0;
    }

    if (modem_wake() != 0) {
        printf("modem_wake failed\n");
        modem_deinit();
        return 1;
    }
    printf("online\n");

    if (mode == MODE_CYCLE) {
        modem_enter_edrx();      /* 成功不返回 */
        printf("modem_enter_edrx returned -> shutdown failed\n");
        modem_deinit();
        return 1;
    }

    /* MODE_UP / MODE_HOLD:已联网,退出(hold 不 deinit,保持 REQ_READY 高)。 */
    printf("=== [DONE] ===\n");
    return 0;
}
