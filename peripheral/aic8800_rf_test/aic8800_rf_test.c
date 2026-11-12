/*
 * Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "aic8800_rf_test.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define RELEASE_DATE "2024_0701"
#define CHIP_AIC8800D AIC8800_RF_CHIP_AIC8800D
#define CHIP_AIC8800DCDW AIC8800_RF_CHIP_AIC8800DCDW
#define CHIP_AIC8800D80 AIC8800_RF_CHIP_AIC8800D80
#define CHIP_AIC8800D80X2 AIC8800_RF_CHIP_AIC8800D80X2
#define CHIP_SELECT_AUTO (-2)

enum result_format
{
    RESULT_NONE = 0,
    RESULT_HEX,
    RESULT_WORD,
    RESULT_RX,
    RESULT_MAC,
    RESULT_FREQ,
    RESULT_RSSI,
    RESULT_NOISE,
    RESULT_COB,
};

enum argument_format
{
    ARGUMENT_NONE = 0,
    ARGUMENT_SET_TX,
    ARGUMENT_SET_TXTONE,
    ARGUMENT_SET_RX,
    ARGUMENT_SET_POWER,
    ARGUMENT_DECIMAL_BYTE,
    ARGUMENT_DECIMAL_SIGNED_BYTE,
    ARGUMENT_HEX_BYTE,
    ARGUMENT_MAC,
    ARGUMENT_PWRMM,
    ARGUMENT_PWRIDX,
    ARGUMENT_PWRLVL,
    ARGUMENT_PWROFST,
    ARGUMENT_PWROFST_FINE,
    ARGUMENT_FUNC_HEX_BYTE,
    ARGUMENT_EFUSE_USRDATA,
    ARGUMENT_USB_VID_PID,
    ARGUMENT_PLL_TEST,
    ARGUMENT_BT_PWROFST,
    ARGUMENT_PWRADD2X,
    ARGUMENT_COB,
    ARGUMENT_DO_COB,
    ARGUMENT_RAW,
};

struct command_spec
{
    const char *name;
    uint32_t command;
    enum argument_format arguments;
    enum result_format result;
    const char *usage;
};

struct encoded_command
{
    struct aic8800_rf_command request;
    struct aic8800_rf_command prefix;
    int has_prefix;
    unsigned int delay_ms;
};

static const struct command_spec commands[] = {
    {"set_tx", AIC8800_RF_SET_TX, ARGUMENT_SET_TX, RESULT_NONE,
     "<chan> <bw> <mode> <rate> <length> [interval] [power]"},
    {"set_txstop", AIC8800_RF_SET_TXSTOP, ARGUMENT_NONE, RESULT_NONE, ""},
    {"set_txtone", AIC8800_RF_SET_TXTONE, ARGUMENT_SET_TXTONE,
     RESULT_NONE, "<func> [freq]"},
    {"set_rx", AIC8800_RF_SET_RX, ARGUMENT_SET_RX, RESULT_NONE,
     "<chan> <bw>"},
    {"get_rx_result", AIC8800_RF_GET_RX_RESULT, ARGUMENT_NONE, RESULT_RX,
     ""},
    {"set_rxstop", AIC8800_RF_SET_RXSTOP, ARGUMENT_NONE, RESULT_NONE, ""},
    {"set_rx_meter", AIC8800_RF_SET_RX_METER,
     ARGUMENT_DECIMAL_SIGNED_BYTE, RESULT_NONE, "<freq>"},
    {"set_power", AIC8800_RF_SET_POWER, ARGUMENT_SET_POWER, RESULT_NONE,
     "<dec-power> | <hex-analog> <hex-digital>"},
    {"set_xtal_cap", AIC8800_RF_SET_XTAL_CAP,
     ARGUMENT_DECIMAL_SIGNED_BYTE, RESULT_WORD, "<signed-dec-delta>"},
    {"set_xtal_cap_fine", AIC8800_RF_SET_XTAL_CAP_FINE,
     ARGUMENT_DECIMAL_SIGNED_BYTE, RESULT_WORD, "<signed-dec-delta>"},
    {"get_efuse_block", AIC8800_RF_GET_EFUSE_BLOCK,
     ARGUMENT_DECIMAL_BYTE, RESULT_WORD, "<dec-block>"},
    {"set_freq_cal", AIC8800_RF_SET_FREQ_CAL, ARGUMENT_HEX_BYTE,
     RESULT_WORD, "<hex-val>"},
    {"set_freq_cal_fine", AIC8800_RF_SET_FREQ_CAL_FINE,
     ARGUMENT_HEX_BYTE, RESULT_WORD, "<hex-val>"},
    {"get_freq_cal", AIC8800_RF_GET_FREQ_CAL, ARGUMENT_NONE, RESULT_FREQ,
     ""},
    {"set_mac_addr", AIC8800_RF_SET_MAC_ADDR, ARGUMENT_MAC, RESULT_NONE,
     "<hex-byte> <hex-byte> <hex-byte> <hex-byte> <hex-byte> <hex-byte>"},
    {"get_mac_addr", AIC8800_RF_GET_MAC_ADDR, ARGUMENT_NONE, RESULT_MAC,
     ""},
    {"set_bt_mac_addr", AIC8800_RF_SET_BT_MAC_ADDR, ARGUMENT_MAC,
     RESULT_NONE,
     "<hex-byte> <hex-byte> <hex-byte> <hex-byte> <hex-byte> <hex-byte>"},
    {"get_bt_mac_addr", AIC8800_RF_GET_BT_MAC_ADDR, ARGUMENT_NONE,
     RESULT_MAC, ""},
    {"set_vendor_info", AIC8800_RF_SET_VENDOR_INFO, ARGUMENT_HEX_BYTE,
     RESULT_HEX, "<hex-val>"},
    {"get_vendor_info", AIC8800_RF_GET_VENDOR_INFO, ARGUMENT_NONE,
     RESULT_HEX, ""},
    {"rdwr_pwrmm", AIC8800_RF_RDWR_PWRMM, ARGUMENT_PWRMM, RESULT_WORD,
     "[hex-val]"},
    {"rdwr_pwridx", AIC8800_RF_RDWR_PWRIDX, ARGUMENT_PWRIDX, RESULT_HEX,
     "[<band> <mod> <idx>]"},
    {"rdwr_pwrlvl", AIC8800_RF_RDWR_PWRLVL, ARGUMENT_PWRLVL, RESULT_HEX,
     "[<band> <group> <idx> <level>]"},
    {"rdwr_pwrofst", AIC8800_RF_RDWR_PWROFST, ARGUMENT_PWROFST,
     RESULT_HEX, "[<band> [type] <channel-group> <offset>]"},
    {"rdwr_pwrofstfine", AIC8800_RF_RDWR_PWROFSTFINE,
     ARGUMENT_PWROFST_FINE, RESULT_HEX,
     "[<band> <channel-group> <offset>]"},
    {"rdwr_drvibit", AIC8800_RF_RDWR_DRVIBIT, ARGUMENT_FUNC_HEX_BYTE,
     RESULT_HEX, "[<func> <hex-val>]"},
    {"rdwr_efuse_pwrofst", AIC8800_RF_RDWR_EFUSE_PWROFST,
     ARGUMENT_PWROFST, RESULT_HEX,
     "[<band> [type] <channel-group> <offset>]"},
    {"rdwr_efuse_pwrofstfine", AIC8800_RF_RDWR_EFUSE_PWROFSTFINE,
     ARGUMENT_PWROFST_FINE, RESULT_HEX,
     "[<band> <channel-group> <offset>]"},
    {"rdwr_efuse_drvibit", AIC8800_RF_RDWR_EFUSE_DRVIBIT,
     ARGUMENT_FUNC_HEX_BYTE, RESULT_HEX, "[<func> <hex-val>]"},
    {"rdwr_efuse_usrdata", AIC8800_RF_RDWR_EFUSE_USRDATA,
     ARGUMENT_EFUSE_USRDATA, RESULT_HEX,
     "[<func> <count> [hex-word ...]]"},
    {"rdwr_efuse_sdiocfg", AIC8800_RF_RDWR_EFUSE_SDIOCFG,
     ARGUMENT_FUNC_HEX_BYTE, RESULT_HEX, "[<func> <hex-val>]"},
    {"rdwr_efuse_usbvidpid", AIC8800_RF_RDWR_EFUSE_USBVIDPID,
     ARGUMENT_USB_VID_PID, RESULT_HEX, "[<func> <hex-vidpid>]"},
    {"rdwr_efuse_he_off", AIC8800_RF_RDWR_EFUSE_HE_OFF,
     ARGUMENT_DECIMAL_BYTE, RESULT_HEX, "<func>"},
    {"set_papr", AIC8800_RF_SET_PAPR, ARGUMENT_DECIMAL_BYTE, RESULT_NONE,
     "<val>"},
    {"set_cal_xtal", AIC8800_RF_SET_CAL_XTAL, ARGUMENT_NONE, RESULT_NONE,
     ""},
    {"get_cal_xtal_res", AIC8800_RF_GET_CAL_XTAL_RES, ARGUMENT_NONE,
     RESULT_FREQ, ""},
    {"set_cob_cal", AIC8800_RF_SET_COB_CAL, ARGUMENT_COB, RESULT_NONE,
     "<dutid> <chip-num> <disable-xtal>"},
    {"get_cob_cal_res", AIC8800_RF_GET_COB_CAL_RES, ARGUMENT_NONE,
     RESULT_COB, ""},
    {"do_cob_test", AIC8800_RF_GET_COB_CAL_RES, ARGUMENT_DO_COB,
     RESULT_COB, "[disable-xtal]"},
    {"set_notch", AIC8800_RF_SET_NOTCH, ARGUMENT_DECIMAL_BYTE,
     RESULT_NONE, "<val>"},
    {"set_srrc", AIC8800_RF_SET_SRRC, ARGUMENT_DECIMAL_BYTE, RESULT_NONE,
     "<func>"},
    {"set_fss", AIC8800_RF_SET_FSS, ARGUMENT_DECIMAL_BYTE, RESULT_NONE,
     "<func>"},
    {"set_usb_off", AIC8800_RF_SET_USB_OFF, ARGUMENT_NONE, RESULT_NONE, ""},
    {"set_pll_test", AIC8800_RF_SET_PLL_TEST, ARGUMENT_PLL_TEST,
     RESULT_NONE, "<func> [freq tx-power]"},
    {"set_ant", AIC8800_RF_SET_ANT_MODE, ARGUMENT_DECIMAL_BYTE,
     RESULT_NONE, "<0|1|2>"},
    {"get_noise", AIC8800_RF_GET_NOISE, ARGUMENT_NONE, RESULT_NOISE, ""},
    {"rdwr_bt_efuse_pwrofst", AIC8800_RF_RDWR_BT_EFUSE_PWROFST,
     ARGUMENT_BT_PWROFST, RESULT_HEX, "[power-offset]"},
    {"exec_flash_oper", AIC8800_RF_EXEC_FLASH_OPER, ARGUMENT_DECIMAL_BYTE,
     RESULT_NONE, "<operation>"},
    {"rdwr_pwradd2x", AIC8800_RF_RDWR_PWRADD2X, ARGUMENT_PWRADD2X,
     RESULT_HEX, "[<band> <value>]"},
    {"rdwr_efuse_pwradd2x", AIC8800_RF_RDWR_EFUSE_PWRADD2X,
     ARGUMENT_PWRADD2X, RESULT_HEX, "[<band> <value>]"},
    {"get_rssi", AIC8800_RF_GET_RSSI, ARGUMENT_NONE, RESULT_RSSI, ""},
};

static uint32_t get_le32(const uint8_t *data)
{
    return (uint32_t)data[0] | (uint32_t)data[1] << 8 |
           (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
}

static void put_le16(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)value;
    data[1] = (uint8_t)(value >> 8);
}

static void put_le32(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)value;
    data[1] = (uint8_t)(value >> 8);
    data[2] = (uint8_t)(value >> 16);
    data[3] = (uint8_t)(value >> 24);
}

static int detect_chip_type(int descriptor)
{
    uint32_t chip;

    if (ioctl(descriptor, AIC8800_RF_IOCTL_GET_CHIP_TYPE, &chip) < 0)
    {
        fprintf(stderr, "cannot detect AIC8800 chip type: %s\n",
                strerror(errno));
        return -1;
    }
    if (chip > CHIP_AIC8800D80X2)
    {
        fprintf(stderr, "invalid AIC8800 chip type from driver: %u\n", chip);
        return -1;
    }
    return (int)chip;
}

static int detect_product_id(int descriptor, uint32_t *product)
{
    if (ioctl(descriptor, AIC8800_RF_IOCTL_GET_PRODUCT_ID, product) < 0)
    {
        fprintf(stderr, "cannot detect AIC8800 product: %s\n",
                strerror(errno));
        return -1;
    }
    return 0;
}

static int parse_number(const char *text, int base, long minimum,
                        unsigned long maximum, long *value)
{
    char *end;
    long parsed;

    if (!text || !text[0] || !value)
    {
        return -1;
    }
    errno = 0;
    parsed = strtol(text, &end, base);
    if (errno || *end || parsed < minimum ||
        (parsed >= 0 && (unsigned long)parsed > maximum))
    {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int parse_byte(const char *text, int base, int is_signed, uint8_t *value)
{
    long parsed;

    if (parse_number(text, base, is_signed ? -128 : 0,
                     is_signed ? 127U : 255U, &parsed))
    {
        return -1;
    }
    *value = (uint8_t)parsed;
    return 0;
}

static int require_count(const struct command_spec *spec, int argc,
                         int minimum, int maximum)
{
    if (argc >= minimum && argc <= maximum)
    {
        return 0;
    }
    fprintf(stderr, "usage: wifi_test <interface> %s %s\n", spec->name,
            spec->usage);
    return -1;
}

static int encode_set_tx(const struct command_spec *spec,
                         struct encoded_command *encoded, uint32_t product,
                         int argc, char **argv)
{
    long values[7];
    int index;

    if (require_count(spec, argc, 5, 7))
    {
        return -1;
    }
    for (index = 0; index < argc; index++)
    {
        if (parse_number(argv[index], 10, 0,
                         index >= 4 && index <= 5 ? 65535U : 255U,
                         &values[index]))
        {
            fprintf(stderr, "invalid SET_TX argument: %s\n", argv[index]);
            return -1;
        }
    }
    encoded->request.arguments[0] = (uint8_t)values[0];
    encoded->request.arguments[1] = (uint8_t)values[1];
    encoded->request.arguments[2] = (uint8_t)values[2];
    encoded->request.arguments[3] = (uint8_t)values[3];
    put_le16(encoded->request.arguments + 4, (uint16_t)values[4]);
    put_le16(encoded->request.arguments + 6,
             (uint16_t)(argc >= 6 ? values[5] : 10000));
    encoded->request.arguments[8] = 127;
    encoded->request.argument_length = 10;

    if (argc == 7 && values[6] != 255)
    {
        uint8_t mode = (uint8_t)values[2];
        uint8_t rate = (uint8_t)values[3];

        if (product == AIC8800_RF_PRODUCT_AIC8801)
        {
            fprintf(stderr,
                    "SET_TX power override is unsupported on AIC8801\n");
            return -1;
        }

        encoded->has_prefix = 1;
        encoded->prefix.command = AIC8800_RF_RDWR_PWRLVL;
        encoded->prefix.argument_length = 4;
        encoded->prefix.arguments[0] = values[0] >= 36 ? 2 : 1;
        encoded->prefix.arguments[1] = mode == 0 ? 0 :
                                       (mode == 2 || mode == 4 ? 1 : 2);
        encoded->prefix.arguments[2] = mode >= 4 ? rate & 0x0fU :
                                       (mode >= 2 ? rate & 0x07U : rate);
        encoded->prefix.arguments[3] = (uint8_t)values[6];
    }
    return 0;
}

static int encode_func_hex_byte(const struct command_spec *spec,
                                struct aic8800_rf_command *request,
                                int argc, char **argv)
{
    if (argc == 0 || (argc == 1 && !strcmp(argv[0], "0")))
    {
        request->argument_length = 0;
        return 0;
    }
    if (require_count(spec, argc, 2, 2) ||
        parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
        request->arguments[0] != 1 ||
        parse_byte(argv[1], 16, 0, &request->arguments[1]))
    {
        fprintf(stderr, "invalid read/write arguments\n");
        return -1;
    }
    request->argument_length = 2;
    return 0;
}

static int encode_pwrofst(const struct command_spec *spec,
                          struct aic8800_rf_command *request, int chip,
                          int argc, char **argv)
{
    int newer = chip == CHIP_AIC8800D80 || chip == CHIP_AIC8800D80X2;
    int expected = newer ? 4 : 3;

    if (argc == 0 || (argc == 1 && !strcmp(argv[0], "0")))
    {
        return 0;
    }
    if (require_count(spec, argc, expected, expected) ||
        parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
        request->arguments[0] < 1 || request->arguments[0] > 4)
    {
        return -1;
    }
    if (newer)
    {
        if (parse_byte(argv[1], 16, 0, &request->arguments[1]) ||
            parse_byte(argv[2], 16, 0, &request->arguments[2]) ||
            parse_byte(argv[3], 10, 1, &request->arguments[3]))
        {
            return -1;
        }
    }
    else if (parse_byte(argv[1], 16, 0, &request->arguments[1]) ||
             parse_byte(argv[2], 10, 1, &request->arguments[2]))
    {
        return -1;
    }
    request->argument_length = (uint32_t)expected;
    return 0;
}

static int encode_arguments(const struct command_spec *spec,
                            struct encoded_command *encoded, int chip,
                            uint32_t product, int argc, char **argv)
{
    struct aic8800_rf_command *request = &encoded->request;
    long value;
    int index;

    switch (spec->arguments)
    {
    case ARGUMENT_NONE:
        return require_count(spec, argc, 0, 0);
    case ARGUMENT_SET_TX:
        return encode_set_tx(spec, encoded, product, argc, argv);
    case ARGUMENT_SET_TXTONE:
        if (require_count(spec, argc, 1, 2) ||
            parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
            (argc == 2 && parse_byte(argv[1], 10, 1,
                                     &request->arguments[1])))
        {
            return -1;
        }
        request->argument_length = (uint32_t)argc;
        return 0;
    case ARGUMENT_SET_RX:
        if (require_count(spec, argc, 2, 2) ||
            parse_byte(argv[0], 10, 0, &request->arguments[0]) ||
            parse_byte(argv[1], 10, 0, &request->arguments[1]))
        {
            return -1;
        }
        request->argument_length = 2;
        return 0;
    case ARGUMENT_SET_POWER:
        if (product == AIC8800_RF_PRODUCT_AIC8801)
        {
            uint8_t analog;
            uint8_t digital;

            if (require_count(spec, argc, 2, 2) ||
                parse_byte(argv[0], 16, 0, &analog) || analog > 0x0f ||
                parse_byte(argv[1], 16, 0, &digital) || digital > 0x0f)
            {
                return -1;
            }
            request->arguments[0] = (uint8_t)(analog << 4 | digital);
        }
        else
        {
            long power;

            if (require_count(spec, argc, 1, 1) ||
                parse_number(argv[0], 10, 0, 0x1eU, &power))
            {
                return -1;
            }
            request->arguments[0] = (uint8_t)power;
        }
        request->argument_length = 1;
        return 0;
    case ARGUMENT_DECIMAL_BYTE:
    case ARGUMENT_DECIMAL_SIGNED_BYTE:
    case ARGUMENT_HEX_BYTE:
        if (require_count(spec, argc, 1, 1) ||
            parse_byte(argv[0], spec->arguments == ARGUMENT_HEX_BYTE ? 16 : 10,
                       spec->arguments == ARGUMENT_DECIMAL_SIGNED_BYTE,
                       &request->arguments[0]))
        {
            return -1;
        }
        request->argument_length = 1;
        return 0;
    case ARGUMENT_MAC:
        if (require_count(spec, argc, 6, 6))
        {
            return -1;
        }
        for (index = 0; index < 6; index++)
        {
            if (parse_byte(argv[index], 16, 0,
                           &request->arguments[5 - index]))
            {
                return -1;
            }
        }
        request->argument_length = 6;
        return 0;
    case ARGUMENT_PWRMM:
        if (require_count(spec, argc, 0, 1))
        {
            return -1;
        }
        if (argc == 1)
        {
            if (parse_byte(argv[0], 16, 0, &request->arguments[0]))
            {
                return -1;
            }
            request->arguments[0] = request->arguments[0] ? 1 : 0;
            request->argument_length = 1;
        }
        return 0;
    case ARGUMENT_PWRIDX:
        if (product != AIC8800_RF_PRODUCT_AIC8801)
        {
            fprintf(stderr, "RDWR_PWRIDX is supported only on AIC8801\n");
            return -1;
        }
        if (argc == 0 || (argc == 1 && !strcmp(argv[0], "0")))
        {
            return 0;
        }
        if (require_count(spec, argc, 3, 3) ||
            parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
            request->arguments[0] < 1 || request->arguments[0] > 2 ||
            parse_byte(argv[1], 16, 0, &request->arguments[1]) ||
            parse_byte(argv[2], 10, 0, &request->arguments[2]))
        {
            return -1;
        }
        request->argument_length = 3;
        return 0;
    case ARGUMENT_PWRLVL:
        if (product == AIC8800_RF_PRODUCT_AIC8801)
        {
            fprintf(stderr, "RDWR_PWRLVL is unsupported on AIC8801\n");
            return -1;
        }
        if (argc == 0 || (argc == 1 && !strcmp(argv[0], "0")))
        {
            return 0;
        }
        if (argc != 4 && argc != 12 && argc != 14)
        {
            return require_count(spec, argc, 4, 4);
        }
        if (parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
            request->arguments[0] < 1 || request->arguments[0] > 2 ||
            parse_byte(argv[1], 16, 0, &request->arguments[1]))
        {
            return -1;
        }
        if (argc == 4)
        {
            if (parse_byte(argv[2], 10, 0, &request->arguments[2]) ||
                parse_byte(argv[3], 10, 1, &request->arguments[3]))
            {
                return -1;
            }
            request->argument_length = 4;
            return 0;
        }
        if ((request->arguments[1] == 1 && argc != 12) ||
            (request->arguments[1] != 1 && argc != 14))
        {
            fprintf(stderr, "wrong power-level group size\n");
            return -1;
        }
        for (index = 2; index < argc; index++)
        {
            if (parse_byte(argv[index], 10, 1,
                           &request->arguments[index]))
            {
                return -1;
            }
        }
        request->argument_length = (uint32_t)argc;
        return 0;
    case ARGUMENT_PWROFST:
        return encode_pwrofst(spec, request, chip, argc, argv);
    case ARGUMENT_PWROFST_FINE:
        if (argc == 0 || (argc == 1 && !strcmp(argv[0], "0")))
        {
            return 0;
        }
        if (require_count(spec, argc, 3, 3) ||
            parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
            request->arguments[0] < 1 || request->arguments[0] > 2 ||
            parse_byte(argv[1], 16, 0, &request->arguments[1]) ||
            parse_byte(argv[2], 10, 1, &request->arguments[2]))
        {
            return -1;
        }
        request->argument_length = 3;
        return 0;
    case ARGUMENT_FUNC_HEX_BYTE:
        return encode_func_hex_byte(spec, request, argc, argv);
    case ARGUMENT_EFUSE_USRDATA:
        request->argument_length = 16;
        request->arguments[1] = 3;
        if (argc == 0)
        {
            return 0;
        }
        if (argc < 2 || parse_byte(argv[0], 10, 0,
                                   &request->arguments[0]) ||
            parse_byte(argv[1], 10, 0, &request->arguments[1]) ||
            request->arguments[1] > 3)
        {
            return -1;
        }
        if (request->arguments[0] == 1)
        {
            if (argc != request->arguments[1] + 2)
            {
                return -1;
            }
            for (index = 0; index < request->arguments[1]; index++)
            {
                if (parse_number(argv[index + 2], 16, 0, UINT32_MAX,
                                 &value))
                {
                    return -1;
                }
                put_le32(request->arguments + 4 + index * 4,
                         (uint32_t)value);
            }
        }
        return 0;
    case ARGUMENT_USB_VID_PID:
        if (argc == 0 || (argc == 1 && !strcmp(argv[0], "0")))
        {
            return 0;
        }
        if (require_count(spec, argc, 2, 2) ||
            parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
            request->arguments[0] != 1 ||
            parse_number(argv[1], 16, 0, UINT32_MAX, &value))
        {
            return -1;
        }
        put_le32(request->arguments + 1, (uint32_t)value);
        request->argument_length = 5;
        return 0;
    case ARGUMENT_PLL_TEST:
        if (argc != 1 && argc != 3)
        {
            return require_count(spec, argc, 1, 1);
        }
        if (parse_byte(argv[0], 16, 0, &request->arguments[0]) ||
            request->arguments[0] > 1)
        {
            return -1;
        }
        request->arguments[2] = 0x0c;
        if (argc == 3 &&
            (parse_byte(argv[1], 10, 1, &request->arguments[1]) ||
             parse_byte(argv[2], 16, 0, &request->arguments[2])))
        {
            return -1;
        }
        request->argument_length = 3;
        return 0;
    case ARGUMENT_BT_PWROFST:
        if (require_count(spec, argc, 0, 1))
        {
            return -1;
        }
        if (argc == 1)
        {
            if (parse_number(argv[0], 10, -7, 7, &value))
            {
                return -1;
            }
            request->arguments[0] = (uint8_t)value;
            request->argument_length = 1;
        }
        return 0;
    case ARGUMENT_PWRADD2X:
        if (argc == 0 || (argc == 1 && !strcmp(argv[0], "0")))
        {
            return 0;
        }
        if (require_count(spec, argc, 2, 2) ||
            parse_byte(argv[0], 10, 0, &request->arguments[0]) ||
            request->arguments[0] < 1 || request->arguments[0] > 2 ||
            parse_number(argv[1], 10, -15, 15, &value))
        {
            return -1;
        }
        request->arguments[1] = (uint8_t)value;
        request->argument_length = 2;
        return 0;
    case ARGUMENT_COB:
        if (require_count(spec, argc, 3, 3))
        {
            return -1;
        }
        for (index = 0; index < 3; index++)
        {
            if (parse_byte(argv[index], 10, 0, &request->arguments[index]))
            {
                return -1;
            }
        }
        request->argument_length = 3;
        return 0;
    case ARGUMENT_DO_COB:
        if (require_count(spec, argc, 0, 1))
        {
            return -1;
        }
        encoded->has_prefix = 1;
        encoded->prefix.command = AIC8800_RF_SET_COB_CAL;
        encoded->prefix.argument_length = 3;
        encoded->prefix.arguments[0] = 1;
        encoded->prefix.arguments[1] = 1;
        if (argc == 1 &&
            parse_byte(argv[0], 10, 0, &encoded->prefix.arguments[2]))
        {
            return -1;
        }
        encoded->delay_ms = 2000;
        return 0;
    case ARGUMENT_RAW:
        if ((unsigned int)argc > AIC8800_RF_ARGUMENT_SIZE)
        {
            fprintf(stderr, "too many raw arguments\n");
            return -1;
        }
        for (index = 0; index < argc; index++)
        {
            if (parse_byte(argv[index], 0, 0, &request->arguments[index]))
            {
                return -1;
            }
        }
        request->argument_length = (uint32_t)argc;
        return 0;
    }
    return -1;
}

static const struct command_spec *find_command(const char *name)
{
    size_t index;

    for (index = 0; index < sizeof(commands) / sizeof(commands[0]); index++)
    {
        if (!strcasecmp(name, commands[index].name))
        {
            return &commands[index];
        }
    }
    return NULL;
}

static uint32_t expected_hex_length(const struct command_spec *spec, int chip)
{
    if (spec->command == AIC8800_RF_SET_VENDOR_INFO ||
        spec->command == AIC8800_RF_GET_VENDOR_INFO)
    {
        return chip == CHIP_AIC8800DCDW ? 2 : 1;
    }
    if (spec->arguments == ARGUMENT_PWRIDX)
    {
        return 9;
    }
    if (spec->arguments == ARGUMENT_PWRLVL)
    {
        return chip >= CHIP_AIC8800D80 ? 72 : 36;
    }
    switch (spec->command)
    {
    case AIC8800_RF_RDWR_PWROFST:
        return chip == CHIP_AIC8800DCDW ? 3 :
               (chip == CHIP_AIC8800D80 ? 27 :
                (chip == CHIP_AIC8800D80X2 ? 54 : 7));
    case AIC8800_RF_RDWR_EFUSE_PWROFST:
        return chip == CHIP_AIC8800DCDW ? 6 :
               (chip == CHIP_AIC8800D80 ? 54 :
                (chip == CHIP_AIC8800D80X2 ? 108 : 7));
    case AIC8800_RF_RDWR_PWROFSTFINE:
    case AIC8800_RF_RDWR_EFUSE_PWROFSTFINE: return 7;
    case AIC8800_RF_RDWR_DRVIBIT: return 16;
    case AIC8800_RF_RDWR_EFUSE_DRVIBIT:
    case AIC8800_RF_RDWR_EFUSE_SDIOCFG:
    case AIC8800_RF_RDWR_EFUSE_USBVIDPID:
    case AIC8800_RF_RDWR_EFUSE_HE_OFF: return 4;
    case AIC8800_RF_RDWR_EFUSE_USRDATA: return 12;
    case AIC8800_RF_RDWR_BT_EFUSE_PWROFST:
    case AIC8800_RF_RDWR_PWRADD2X: return 2;
    case AIC8800_RF_RDWR_EFUSE_PWRADD2X: return 3;
    default: return AIC8800_RF_RESULT_SIZE;
    }
}

static void print_hex(const uint8_t *data, uint32_t length)
{
    uint32_t index;

    for (index = 0; index < length; index++)
    {
        printf("%02x%s", data[index], index + 1 == length ? "\n" : " ");
    }
}

static int print_result(const struct command_spec *spec,
                        const struct aic8800_rf_command *request, int chip)
{
    const uint8_t *result = request->result;
    uint32_t length = request->result_length;
    uint32_t needed;

    switch (spec->result)
    {
    case RESULT_NONE:
        printf("done\n");
        return 0;
    case RESULT_WORD:
        needed = 4;
        if (length >= needed)
        {
            printf("done: 0x%08x\n", get_le32(result));
            return 0;
        }
        break;
    case RESULT_RX:
        needed = 8;
        if (length >= needed)
        {
            printf("done: getrx fcsok=%u, total=%u\n", get_le32(result),
                   get_le32(result + 4));
            return 0;
        }
        break;
    case RESULT_MAC:
        needed = 6;
        if (length >= needed)
        {
            printf("done: %02x : %02x : %02x : %02x : %02x : %02x\n",
                   result[5], result[4], result[3], result[2], result[1],
                   result[0]);
            return 0;
        }
        break;
    case RESULT_FREQ:
        needed = 4;
        if (length >= needed)
        {
            uint32_t value = get_le32(result);

            if (chip == CHIP_AIC8800D)
            {
                printf("done: xtal_cap=0x%x, xtal_cap_fine=0x%x\n",
                       value & 0xffU, value >> 8 & 0xffU);
            }
            else if (chip == CHIP_AIC8800DCDW)
            {
                printf("done: xtal_cap=0x%x (remain:%x), "
                       "xtal_cap_fine=0x%x (remain:%x)\n",
                       value & 0xffU, value >> 8 & 0xffU,
                       value >> 16 & 0xffU, value >> 24 & 0xffU);
            }
            else
            {
                printf("done: xtal_cap=0x%x (remain:%x), "
                       "xtal_cap_fine=0x%x (remain:%x)\n",
                       value & 0xffU, value >> 16 & 0xffU,
                       value >> 8 & 0xffU, value >> 24 & 0xffU);
            }
            return 0;
        }
        break;
    case RESULT_RSSI:
        needed = 1;
        if (length >= needed)
        {
            printf("rssi=%d\n", (int8_t)result[0]);
            return 0;
        }
        break;
    case RESULT_NOISE:
        needed = 2;
        if (length >= needed)
        {
            printf("rssi=%d,%d\n", (int8_t)result[0], (int8_t)result[1]);
            return 0;
        }
        break;
    case RESULT_COB:
        needed = 16;
        if (length >= needed)
        {
            if (result[2])
            {
                fprintf(stderr, "cob not idle\n");
                return -1;
            }
            printf("cap=0x%x, cap_fine=0x%x, freq_ofst=%d Hz\n",
                   result[0], result[1], (int32_t)get_le32(result + 4));
            print_hex(result, needed);
            return 0;
        }
        break;
    case RESULT_HEX:
        needed = expected_hex_length(spec, chip);
        if (spec->arguments != ARGUMENT_RAW && length < needed)
        {
            break;
        }
        if (needed > length)
        {
            needed = length;
        }
        printf("done:\n");
        print_hex(result, needed);
        return 0;
    }
    fprintf(stderr, "short firmware response: %u byte(s)\n", length);
    return -1;
}

static void print_usage(const char *program, int chip)
{
    size_t index;

    printf("Please enter parameters!\n");
    printf("Usage:\n");
    printf("  %s <interface> <command> [args]\n", program);
    printf("  %s version - Show version\n", program);
    printf("  Available chip types:\n");
    printf("  0 - AIC8800D\n");
    printf("  1 - AIC8800DCDW\n");
    printf("  2 - AIC8800D80\n");
    printf("  3 - AIC8800D80X2\n");
    if (chip == CHIP_SELECT_AUTO)
    {
        printf("  Current chip selection: auto (queried from driver)\n\n");
    }
    else
    {
        printf("  Current chip selection: %d\n\n", chip);
    }
    printf("Commands:\n");
    for (index = 0; index < sizeof(commands) / sizeof(commands[0]); index++)
    {
        printf("  %-25s %s\n", commands[index].name, commands[index].usage);
    }
    printf("  %-25s %s\n", "raw", "<command-id> [byte ...]");
}

static int resolve_device(const char *interface, char *path, size_t path_size)
{
    static const char device_prefix[] = "/dev/aicrf";
    const char *cursor;
    char *end;
    unsigned long index = 0;

    if (!strncmp(interface, "/dev/", 5))
    {
        if (!strncmp(interface, device_prefix, sizeof(device_prefix) - 1U))
        {
            cursor = interface + sizeof(device_prefix) - 1U;
            errno = 0;
            index = strtoul(cursor, &end, 10);
            if (errno || end == cursor || *end ||
                index >= AIC8800_RF_DEVICE_COUNT)
            {
                return -1;
            }
        }
        if (snprintf(path, path_size, "%s", interface) >= (int)path_size)
        {
            return -1;
        }
        return 0;
    }
    cursor = interface + strlen(interface);
    while (cursor > interface && cursor[-1] >= '0' && cursor[-1] <= '9')
    {
        cursor--;
    }
    if (*cursor)
    {
        errno = 0;
        index = strtoul(cursor, &end, 10);
        if (errno || *end || index >= AIC8800_RF_DEVICE_COUNT)
        {
            return -1;
        }
    }
    if (snprintf(path, path_size, "/dev/aicrf%lu", index) >= (int)path_size)
    {
        return -1;
    }
    return 0;
}

static int execute_command(int descriptor, struct aic8800_rf_command *request,
                           const char *name)
{
    if (ioctl(descriptor, AIC8800_RF_IOCTL_COMMAND, request) < 0)
    {
        fprintf(stderr, "RF command %s failed: %s\n", name, strerror(errno));
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const struct command_spec *spec;
    struct command_spec raw_spec;
    struct encoded_command encoded;
    char device_path[32];
    long raw_command;
    int command_argument;
    int descriptor;
    int chip;
    int result;
    uint32_t product;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    chip = CHIP_SELECT_AUTO;
    if (argc == 2 &&
        (!strcasecmp(argv[1], "help") || !strcasecmp(argv[1], "-h") ||
         !strcasecmp(argv[1], "--help")))
    {
        print_usage(argv[0], chip);
        return EXIT_SUCCESS;
    }
    if (argc == 2 &&
        (!strcasecmp(argv[1], "-v") || !strcasecmp(argv[1], "version")))
    {
        printf("wifi_test %s\n", RELEASE_DATE);
        if (chip == CHIP_SELECT_AUTO)
        {
            printf("Current chip selection: auto\n");
        }
        else
        {
            printf("Current chip selection: %d\n", chip);
        }
        return EXIT_SUCCESS;
    }
    if (argc < 3)
    {
        print_usage(argv[0], chip);
        return EXIT_FAILURE;
    }
    if (resolve_device(argv[1], device_path, sizeof(device_path)))
    {
        fprintf(stderr, "invalid interface: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    if (!strcasecmp(argv[2], "help") || !strcasecmp(argv[2], "-h") ||
        !strcasecmp(argv[2], "--help"))
    {
        print_usage(argv[0], chip);
        return EXIT_SUCCESS;
    }

    memset(&encoded, 0, sizeof(encoded));
    command_argument = 3;
    if (!strcasecmp(argv[2], "raw"))
    {
        if (argc < 4 ||
            parse_number(argv[3], 0, 0, UINT32_MAX, &raw_command))
        {
            fprintf(stderr, "raw requires a numeric command ID\n");
            return EXIT_FAILURE;
        }
        memset(&raw_spec, 0, sizeof(raw_spec));
        raw_spec.name = "raw";
        raw_spec.command = (uint32_t)raw_command;
        raw_spec.arguments = ARGUMENT_RAW;
        raw_spec.result = RESULT_HEX;
        raw_spec.usage = "<command-id> [byte ...]";
        spec = &raw_spec;
        command_argument = 4;
    }
    else
    {
        spec = find_command(argv[2]);
        if (!spec)
        {
            fprintf(stderr, "unknown command: %s\n", argv[2]);
            return EXIT_FAILURE;
        }
    }
    descriptor = open(device_path, O_RDWR);
    if (descriptor < 0)
    {
        fprintf(stderr, "open %s: %s\n", device_path, strerror(errno));
        return EXIT_FAILURE;
    }
    if (chip == CHIP_SELECT_AUTO)
    {
        chip = detect_chip_type(descriptor);
        if (chip < 0)
        {
            close(descriptor);
            return EXIT_FAILURE;
        }
    }
    if (detect_product_id(descriptor, &product))
    {
        close(descriptor);
        return EXIT_FAILURE;
    }
    encoded.request.command = spec->command;
    if (encode_arguments(spec, &encoded, chip, product,
                         argc - command_argument,
                         argv + command_argument))
    {
        close(descriptor);
        return EXIT_FAILURE;
    }
    if (encoded.has_prefix &&
        execute_command(descriptor, &encoded.prefix, spec->name))
    {
        close(descriptor);
        return EXIT_FAILURE;
    }
    if (encoded.delay_ms)
    {
        usleep(encoded.delay_ms * 1000U);
    }
    result = execute_command(descriptor, &encoded.request, spec->name);
    close(descriptor);
    if (result)
    {
        return EXIT_FAILURE;
    }
    return print_result(spec, &encoded.request, chip) ?
           EXIT_FAILURE : EXIT_SUCCESS;
}
