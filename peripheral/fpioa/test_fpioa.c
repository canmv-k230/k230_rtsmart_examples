#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "drv_fpioa.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "drv_fpioa.h"

bool should_ignore_pin(int pin)
{
    /* List of pins to ignore for voltage bank consistency checks */
    const int    ignore_pins[]   = { 9, 42, 43, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58 };
    const size_t num_ignore_pins = sizeof(ignore_pins) / sizeof(ignore_pins[0]);

    for (size_t i = 0; i < num_ignore_pins; i++) {
        if (pin == ignore_pins[i]) {
            return true;
        }
    }
    return false;
}

void generate_voltage_bank_defines(void)
{
    fpioa_iomux_cfg_t cfg;
    struct {
        int         first_pin;
        int         last_pin;
        const char* name;
        bool        checked;
        int         msc_value;
    } banks[] = {
        { 0, 1, "VOL_BANK_IO0_1", false, -1 },      { 2, 13, "VOL_BANK0_IO2_13", false, -1 },
        { 14, 25, "VOL_BANK1_IO14_25", false, -1 }, { 26, 37, "VOL_BANK2_IO26_37", false, -1 },
        { 38, 49, "VOL_BANK3_IO38_49", false, -1 }, { 50, 61, "VOL_BANK4_IO50_61", false, -1 },
        { 62, 63, "VOL_BANK5_IO62_63", false, -1 },
    };

    /* Determine bank voltages */
    for (size_t i = 0; i < sizeof(banks) / sizeof(banks[0]); i++) {
        bool     consistent = true;
        uint32_t first_msc  = 0;
        int      valid_pins = 0;

        for (int pin = banks[i].first_pin; pin <= banks[i].last_pin; pin++) {
            if (should_ignore_pin(pin)) {
                continue; // Skip this pin for consistency check
            }

            uint32_t reg_value;
            if (drv_fpioa_get_pin_cfg(pin, &reg_value) != 0) {
                fprintf(stderr, "Error reading pin %d\n", pin);
                continue;
            }

            cfg.u.value = reg_value;
            valid_pins++;

            if (valid_pins == 1) {
                first_msc = cfg.u.bit.msc;
            } else if (cfg.u.bit.msc != first_msc) {
                consistent = false;
                fprintf(stderr, "Warning: Pin %d in bank %s has different msc (%d) than first valid pin (%d)\n", pin,
                        banks[i].name, cfg.u.bit.msc, first_msc);
            }
        }

        if (consistent && valid_pins > 0) {
            /* Find first non-ignored pin to get msc value */
            for (int pin = banks[i].first_pin; pin <= banks[i].last_pin; pin++) {
                if (should_ignore_pin(pin)) {
                    continue;
                }

                uint32_t reg_value;
                if (drv_fpioa_get_pin_cfg(pin, &reg_value) == 0) {
                    cfg.u.value        = reg_value;
                    banks[i].msc_value = cfg.u.bit.msc;
                    banks[i].checked   = true;
                    break;
                }
            }
        } else if (valid_pins == 0) {
            fprintf(stderr, "Warning: No valid pins found for bank %s\n", banks[i].name);
        }
    }

    /* Generate macros */
    printf("#include \"drv_fpioa.h\"\n\n");

    for (size_t i = 0; i < sizeof(banks) / sizeof(banks[0]); i++) {
        if (banks[i].checked) {
            printf("#define %s %s\n", banks[i].name, banks[i].msc_value ? "BANK_VOL_1V8_MSC" : "BANK_VOL_3V3_MSC");
        } else {
            printf("/* Could not determine voltage for %s (no valid non-ignored pins) */\n", banks[i].name);
        }
    }
    printf("\n");
}

const char* get_pin_function_name(int pin)
{
    static char func_name[64];
    if (drv_fpioa_get_pin_func_name(pin, func_name, sizeof(func_name))) {
        snprintf(func_name, sizeof(func_name), "IO%d", pin);
    }
    return func_name;
}

void generate_pin_config_array(void)
{
    fpioa_iomux_cfg_t cfg;
    const char*       vol_bank_names[] = { "VOL_BANK_IO0_1",    "VOL_BANK0_IO2_13",  "VOL_BANK1_IO14_25", "VOL_BANK2_IO26_37",
                                           "VOL_BANK3_IO38_49", "VOL_BANK4_IO50_61", "VOL_BANK5_IO62_63" };

    generate_voltage_bank_defines();

    printf("/* clang-format off */\n");
    printf("const struct st_iomux_reg_t board_pinmux_cfg[K230_PIN_COUNT] = {\n");

    for (int pin = 0; pin < 64; pin++) {
        uint32_t reg_value;
        if (drv_fpioa_get_pin_cfg(pin, &reg_value) != 0) {
            fprintf(stderr, "Error reading pin %d\n", pin);
            continue;
        }

        cfg.u.value = reg_value;

        const char* vol_bank;
        if (pin <= 1)
            vol_bank = vol_bank_names[0];
        else if (pin <= 13)
            vol_bank = vol_bank_names[1];
        else if (pin <= 25)
            vol_bank = vol_bank_names[2];
        else if (pin <= 37)
            vol_bank = vol_bank_names[3];
        else if (pin <= 49)
            vol_bank = vol_bank_names[4];
        else if (pin <= 61)
            vol_bank = vol_bank_names[5];
        else
            vol_bank = vol_bank_names[6];

        if (pin == 0)
            printf("    /* BOOT IO */\n");
        else if (pin == 2)
            printf("\n    /* BANK0 */\n");
        else if (pin == 14)
            printf("\n    /* BANK1 */\n");
        else if (pin == 26)
            printf("\n    /* BANK2 */\n");
        else if (pin == 38)
            printf("\n    /* BANK3 */\n");
        else if (pin == 50)
            printf("\n    /* BANK4 */\n");
        else if (pin == 62)
            printf("\n    /* BANK5 */\n");

        printf("    [%d] = { .u.bit = { .st = %d, .ds = %d, .pd = %d, .pu = %d, "
               ".oe = %d, .ie = %d, .msc = %s, .io_sel = %d } }, // %s\n",
               pin, cfg.u.bit.st, cfg.u.bit.ds, cfg.u.bit.pd, cfg.u.bit.pu, cfg.u.bit.oe, cfg.u.bit.ie, vol_bank,
               cfg.u.bit.io_sel, get_pin_function_name(pin));
    }
    printf("};\n");

    printf("/* clang-format on */\n\n");
}

int main(void)
{
    generate_pin_config_array();
    return 0;
}
