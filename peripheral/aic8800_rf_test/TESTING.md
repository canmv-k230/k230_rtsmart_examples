# AIC8800 Wi-Fi RF test guide

This guide describes how to run the AIC `wifi_test` manufacturing commands on
RT-Smart. It adapts the AIC reference manuals to this driver and keeps the
official command name and argument order:

```text
wifi_test.elf <interface> <command> [args]
```

The AIC Linux instructions for `insmod`, `testmode=1`, and private network
ioctls do not apply to RT-Smart. RT-Smart loads the selected test firmware at
boot and exposes the same command protocol through `/dev/aicrfN`.

## Safety

- Run transmit tests in a shield box or through a conducted 50-ohm RF path
  with suitable attenuation. Follow local spectrum regulations.
- Start with a conservative transmit power and always issue `SET_TXSTOP` or
  `SET_TXTONE 0` before disconnecting the DUT.
- Do not run write commands until the measured value and the target storage
  are independently verified. eFuse writes are limited or irreversible.
- Keep a normal-Wi-Fi image available. RF mode does not provide a station,
  SoftAP, IP traffic, or normal WLAN control interface.

## Firmware model

Normal operation uses the AIC FMAC/full-MAC runtime. The host driver provides
an offload interface while the firmware implements the Wi-Fi MAC. RF test mode
uses a dedicated manufacturing runtime and only exposes `DBG_RFTEST` commands.

The newer AIC packages name that runtime `lmacfw_rf_*.bin`; the original
AIC8800 package names it `fmacfw_rf.bin`. Therefore the D80/D80X2/DC test files
are LMAC test firmware, not normal soft-MAC networking firmware. The RT-Smart
RF driver is compatible with both naming families because it uses their common
test request/confirmation protocol.

| Module family | Driver chip ID | Test firmware | Transport |
| --- | ---: | --- | --- |
| AIC8800/AIC8801 | `0` | `fmacfw_rf.bin` | USB, SDIO |
| AIC8800DC/DW/DL | `1` | `lmacfw_rf_8800dc.bin` | USB, SDIO |
| AIC8800D80/D40 U02-compatible | `2` | `lmacfw_rf_8800d80_u02.bin` | USB, SDIO |
| AIC8800D80X2 | `3` | `lmacfw_rf_8800d80x2.bin` | USB |

The chip ID is exchanged between the tool and driver. It is not a shell
setting and does not change the AIC command line.

The packaged firmware has no D80 U01 RF image. The driver rejects that
revision instead of silently loading normal WLAN firmware.

## Build and install

From the SDK root, open the configuration and enable the following options:

```text
CONFIG_RT_USING_AIC8800_WIFI=y
CONFIG_AIC8800_WIFI_TRANSPORT_USB=y       # and/or SDIO
CONFIG_AIC8800_WIFI_RF_TEST_MODE=y
CONFIG_RTSMART_ENABLE_BUILD_EXAMPLES=y
CONFIG_RTSMART_ENABLE_PERIPHERAL_EXAMPLES=y
CONFIG_RTSMART_PERIPHERAL_ENABLE_AIC8800_RF_TEST=y
```

Also select at least one matching firmware family:

```text
CONFIG_AIC8800_WIFI_FIRMWARE_AIC8800=y
CONFIG_AIC8800_WIFI_FIRMWARE_D80=y
CONFIG_AIC8800_WIFI_FIRMWARE_D80X2=y
CONFIG_AIC8800_WIFI_FIRMWARE_DC_DW=y
```

Only the families needed by the image have to be selected. D80X2 requires USB.

```sh
make menuconfig
make rtsmart
```

The application is built with the official basename:

```text
src/rtsmart/examples/elf/peripheral/wifi_test.elf
```

The RT-Smart image installs it as:

```text
/sdcard/app/examples/peripheral/wifi_test.elf
```

Selected firmware is installed below `/bin/firmware`. A single-transport
image contains the family directories directly. A dual-transport image keeps
them below `/bin/firmware/usb` and `/bin/firmware/sdio` so same-named firmware
files cannot overwrite one another.

Flash or deploy the rebuilt RT-Smart image, connect the module, and power-cycle
the module. Switching between normal and RF firmware requires a real module
reset; restarting only the application is not sufficient.

## Boot check

The kernel log must contain a line like:

```text
RF test firmware control=/dev/aicrf0
```

Confirm the device and tool before transmitting:

```sh
list_device
/sdcard/app/examples/peripheral/wifi_test.elf version
```

Look for `aicrf0` with type `Miscellaneous Device` in `list_device`. RT-Smart
devfs can open `/dev/aicrfN` even when `ls /dev/aicrf*` does not show a
directory entry. The tool queries the family from the device; no `export`
command is needed or expected in RT-Smart `msh`.

`wlan0` remains the official command-line selector and maps to `/dev/aicrf0`.
It is not a network interface in RF mode. Either form is accepted:

```sh
wifi_test.elf wlan0 GET_RSSI
wifi_test.elf /dev/aicrf0 GET_RSSI
```

Command names are case-insensitive. Run `wifi_test.elf` without arguments to
print every supported named command and its argument form.

The named command surface is grouped below. Availability and valid limits can
differ by firmware family and silicon revision.

| Group | Commands |
| --- | --- |
| TX/RX | `SET_TX`, `SET_TXSTOP`, `SET_RX`, `SET_RXSTOP`, `GET_RX_RESULT` |
| Tone/measurement | `SET_TXTONE`, `SET_RX_METER`, `GET_RSSI`, `GET_NOISE` |
| Crystal | `SET_XTAL_CAP`, `SET_XTAL_CAP_FINE`, `SET_FREQ_CAL`, `SET_FREQ_CAL_FINE`, `GET_FREQ_CAL` |
| Identity | `SET_MAC_ADDR`, `GET_MAC_ADDR`, `SET_BT_MAC_ADDR`, `GET_BT_MAC_ADDR`, `SET_VENDOR_INFO`, `GET_VENDOR_INFO` |
| Power | `SET_POWER`, `RDWR_PWRMM`, `RDWR_PWRIDX`, `RDWR_PWRLVL`, `RDWR_PWROFST`, `RDWR_PWROFSTFINE` |
| Persistent power | `RDWR_EFUSE_PWROFST`, `RDWR_EFUSE_PWROFSTFINE`, `RDWR_EFUSE_DRVIBIT`, `RDWR_EFUSE_PWRADD2X` |
| eFuse | `GET_EFUSE_BLOCK`, `RDWR_EFUSE_USRDATA`, `RDWR_EFUSE_SDIOCFG`, `RDWR_EFUSE_USBVIDPID`, `RDWR_EFUSE_HE_OFF` |
| Calibration helpers | `SET_CAL_XTAL`, `GET_CAL_XTAL_RES`, `SET_COB_CAL`, `GET_COB_CAL_RES`, `DO_COB_TEST` |
| RF controls | `RDWR_DRVIBIT`, `SET_PAPR`, `SET_NOTCH`, `SET_SRRC`, `SET_FSS`, `SET_PLL_TEST`, `SET_ANT` |
| Platform-specific | `SET_USB_OFF`, `EXEC_FLASH_OPER`, `RDWR_PWRADD2X`, `RDWR_BT_EFUSE_PWROFST` |

The exact syntax printed by the executable is authoritative for this port.
Commands containing `EFUSE`, identity writes, frequency-calibration writes,
and flash operations must be treated as permanent programming.

## Receiver smoke test

An RX test does not intentionally transmit and is the preferred first test.
Configure a signal generator or Wi-Fi tester for channel 1, 20 MHz, then run:

```sh
wifi_test.elf wlan0 SET_RX 1 0
sleep 1
wifi_test.elf wlan0 SET_RXSTOP
wifi_test.elf wlan0 GET_RX_RESULT
```

Firmware revisions that implement the optional monitor commands can also run:

```sh
wifi_test.elf wlan0 GET_RSSI
wifi_test.elf wlan0 GET_NOISE
```

`GET_RX_RESULT` reports `fcsok` and `total` accumulated between `SET_RX` and
`SET_RXSTOP`. Confirm that `total` increases with the generator on, `fcsok` is
not greater than `total`, and RSSI changes plausibly when input power changes.

### Verified antenna-only check

This path was exercised on the `k230-2` 01Studio K230 board with only an
antenna and with two AIC8800D80 NICs attached. The SDIO `c8a1:0082` NIC
registered `/dev/aicrf0`; the USB `a69c:8d81` NIC registered `/dev/aicrf1`.
Both loaded `lmacfw_rf_8800d80_u02.bin`. The following non-transmit checks
passed on both device paths:

```sh
wifi_test.elf /dev/aicrfN GET_FREQ_CAL
wifi_test.elf /dev/aicrfN GET_MAC_ADDR
wifi_test.elf /dev/aicrfN RDWR_PWRLVL 0
wifi_test.elf /dev/aicrfN SET_RX 1 0
wifi_test.elf /dev/aicrfN SET_RXSTOP
wifi_test.elf /dev/aicrfN GET_RX_RESULT
```

The one-second ambient receive runs reported `fcsok=3715, total=3835` for SDIO
and `fcsok=1347, total=1420` for USB. All twelve commands returned success.
Those values only prove command transport and receiver operation in that RF
environment; they are not production acceptance limits. No `SET_TX`,
`SET_TXTONE`, calibration write, eFuse write, or flash operation was run
without conducted test equipment.

## Conducted transmit smoke test

Connect the DUT through attenuation to a spectrum analyzer or Wi-Fi tester.
The following official form starts HT20 MCS7 traffic on channel 1. Omitting
the optional power argument keeps the firmware's current power table.

```sh
wifi_test.elf wlan0 SET_TX 1 0 2 7 4000 1000
```

Verify center frequency, occupied bandwidth, packet modulation, and power on
the instrument, then stop transmission:

```sh
wifi_test.elf wlan0 SET_TXSTOP
```

For firmware that supports per-command power selection, append power in dBm.
The interval must be present when power is present:

```sh
wifi_test.elf wlan0 SET_TX 1 0 2 7 4000 1000 10
```

Use `255` for the power argument to retain the existing/default power:

```sh
wifi_test.elf wlan0 SET_TX 1 0 2 7 4000 1000 255
```

## `SET_TX` arguments

```text
SET_TX <channel> <bandwidth> <mode> <rate> <length> [interval] [power]
```

| Argument | Meaning |
| --- | --- |
| `channel` | 2.4 GHz channel 1-13 or a supported 5 GHz channel 36-165 |
| `bandwidth` | `0` = 20 MHz, `1` = 40 MHz, `2` = 80 MHz |
| `mode` | `0` = non-HT, `2` = HT, `4` = VHT, `5` = HE SU |
| `rate` | Rate index from the tables below |
| `length` | Payload length in bytes |
| `interval` | Packet interval in microseconds; this port uses 10000 when omitted |
| `power` | Optional target in dBm; `255` keeps the existing/default value |

Only select bands, bandwidths, modes, and spatial streams supported by the
actual module. D80X2 is the dual-stream family. Channel availability is also
subject to the module variant and local regulations.

For `mode=0`, rate indices are:

| Index | Rate | Index | Rate |
| ---: | --- | ---: | --- |
| `0` | 1 Mbps DSSS | `6` | 12 Mbps OFDM |
| `1` | 2 Mbps DSSS | `7` | 18 Mbps OFDM |
| `2` | 5.5 Mbps DSSS | `8` | 24 Mbps OFDM |
| `3` | 11 Mbps DSSS | `9` | 36 Mbps OFDM |
| `4` | 6 Mbps OFDM | `10` | 48 Mbps OFDM |
| `5` | 9 Mbps OFDM | `11` | 54 Mbps OFDM |

For HT/VHT/HE, the index identifies the MCS and stream count:

| Mode | SISO indices | D80X2 MIMO indices |
| --- | --- | --- |
| HT (`2`) | `0`-`7` = MCS0-MCS7 | `8`-`15` = MCS8-MCS15 |
| VHT (`4`) | `0`-`9` = MCS0-MCS9 | `16`-`25` = MCS0-MCS9 |
| HE SU (`5`) | `0`-`11` = MCS0-MCS11 | `16`-`27` = MCS0-MCS11 |

AIC's D80/D80X2 manuals use these practical payload sizes:

| Test | Suggested length |
| --- | ---: |
| non-HT/B | 1000 |
| HT/VHT/HE, 20 MHz, high MCS | 4000 |
| HT/VHT/HE, 40 MHz, high MCS | 8000 |
| VHT/HE, 80 MHz, high MCS | 16000 |
| MCS0 | 1000 |

The original AIC8800 manual uses 1500 in its HT example, while the DC/DW
manual uses 4096. Use the value specified for the selected family when
reproducing a vendor test case.

## Antenna and single-tone tests

D80X2 antenna selection uses:

```text
SET_ANT 0    # ANT0 only
SET_ANT 1    # ANT1 only
SET_ANT 2    # ANT0 and ANT1 enabled
```

Start a single tone at the center frequency, inspect it, and stop it:

```sh
wifi_test.elf wlan0 SET_TXTONE 1 0
wifi_test.elf wlan0 SET_TXTONE 0
```

The tone offset is a signed value from `-20` through `19`. AIC documents the
SRRC controls as `SET_SRRC 0|1` and, on D80, additional filter selections may
be firmware-specific. Do not leave a tone or continuous TX active.

## Crystal calibration

The temporary adjustment commands take signed decimal deltas and return the
resulting value. They do not use the same number format as the permanent
commands:

```sh
wifi_test.elf wlan0 SET_XTAL_CAP -2
wifi_test.elf wlan0 SET_XTAL_CAP_FINE 10
wifi_test.elf wlan0 GET_FREQ_CAL
```

Measure frequency error with calibrated RF equipment and adjust toward the
target. `SET_FREQ_CAL` and `SET_FREQ_CAL_FINE` take absolute hexadecimal
values and write calibration storage:

```text
SET_FREQ_CAL <hex-absolute-value>
SET_FREQ_CAL_FINE <hex-absolute-value>
```

Treat both as permanent-write operations. Do not derive an absolute value from
the signed delta without checking the returned register value and the manual
for the exact silicon revision.

## Power controls

`SET_POWER` follows the vendor utility's product-specific syntax. AIC8801
uses two hexadecimal nibbles (`<analog> <digital>`); DC/DW, D80, and D80X2
use one decimal value from `0` through `30`.

Read current tables before making temporary changes:

```sh
wifi_test.elf wlan0 RDWR_PWRIDX 0     # AIC8801
wifi_test.elf wlan0 RDWR_PWRLVL 0     # DC/DW, D80, D80X2
wifi_test.elf wlan0 RDWR_PWROFST 0
```

The AIC8801 command is:

```text
RDWR_PWRIDX <band> <modulation-group> <index>
RDWR_PWROFST <band> <channel-group> <signed-offset>
```

For `RDWR_PWRIDX`, `band=1` is 2.4 GHz and `band=2` is 5 GHz. Modulation
groups are `0` low-rate OFDM, `1` 64-QAM, `2` 256-QAM, `3` 1024-QAM, and `4`
DSSS. AIC documents the original power offset step as approximately 0.375 dB.

DC/DW, D80, and D80X2 use power levels:

```text
RDWR_PWRLVL <band> <group> <rate-index> <signed-level>
```

`band=1` is 2.4 GHz and `band=2` is 5 GHz. Groups are `0` legacy, `1` HT/VHT,
and `2` HE. The tool also accepts the AIC whole-group form: after `band` and
`group`, provide all 10 values for group 1 or all 12 values for groups 0/2.

D80 and D80X2 use the four-field power-offset form selected automatically for
those families:

```text
RDWR_PWROFST <band> <rate-type> <channel-group> <signed-offset>
```

DC/DW uses the older three-field form selected automatically for that family:

```text
RDWR_PWROFST <band> <channel-group> <signed-offset>
```

For D80X2, the `band` field also selects the antenna: `1` = 2.4 GHz ANT0,
`2` = 2.4 GHz ANT1, `3` = 5 GHz ANT0, and `4` = 5 GHz ANT1. Its offset step is
0.5 dB. Family limits differ, so use the applicable AIC manual before writing.

Commands containing `EFUSE` persist their values and are not substitutes for
the temporary `RDWR_*` commands during characterization.

## Permanent and destructive commands

The following command families can consume eFuse bits, alter identity, change
boot behavior, or modify flash:

```text
SET_FREQ_CAL
SET_FREQ_CAL_FINE
SET_MAC_ADDR
SET_BT_MAC_ADDR
SET_VENDOR_INFO
RDWR_EFUSE_*
EXEC_FLASH_OPER
```

Use read forms first where available: `GET_FREQ_CAL`, `GET_MAC_ADDR`,
`GET_BT_MAC_ADDR`, `GET_VENDOR_INFO`, and `RDWR_EFUSE_* 0`.

On D80X2, AIC documents `EXEC_FLASH_OPER` values as:

| Value | Operation |
| ---: | --- |
| `0` | Read flash write-protection status |
| `1` | Recover/unprotect flash |
| `2` | Protect flash |
| `3` | Read the power-on calibration flag |
| `4` | Erase the stored calibration result |

These operations depend on the fitted flash and firmware revision. Values
`1`, `2`, and especially `4` must not be part of a smoke test.

## Compatibility limits

- The command name, argument order, decimal/hexadecimal conventions, and
  optional arguments match AIC's `wifi_test` utility.
- RT-Smart replaces the Linux socket/private-ioctl transport with
  `/dev/aicrfN`; `wlanN` is retained only as a selector.
- AIC `country_set`, `aic_userconfig.txt`, and `powerlimit.txt` are normal
  Linux driver configuration paths and are not RF device commands here.
- AIC Bluetooth RF commands use a separate HCI test path. They are not carried
  by this Wi-Fi RF device. USB Bluetooth can use `/dev/hciN` when enabled;
  UART and SDIO Bluetooth manufacturing transports are outside this tool.
- `raw <command-id> [byte ...]` is an RT-Smart extension for bounded firmware
  commands not represented by a named form. Use it only with a verified
  command ID and payload for the exact firmware revision.

## Return to normal Wi-Fi

Disable RF mode and rebuild the image:

```text
# CONFIG_AIC8800_WIFI_RF_TEST_MODE is not set
```

Then flash/deploy the normal image and power-cycle the module. The driver will
load FMAC firmware and register normal `wlanN`, `wlanNap`, and `/dev/wlanctlN`
interfaces. A normal WLAN interface appearing in an RF image, or `/dev/aicrfN`
appearing in a normal image, indicates that the wrong kernel/image is running.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No `aicrfN` in `list_device` | Check RF mode, transport, family, firmware, log, and power cycle |
| D80 U01 rejection | No compatible RF image is packaged; do not substitute the U02 image |
| `open /dev/aicrfN` fails | Use the logged index or pass the device path directly |
| Command timeout | Check the RF image, transport, and module reset |
| Chip detection fails | Confirm the tool and kernel were built from the same source |
| No normal `wlanN` | Expected in RF mode; rebuild without RF mode for networking |
| Unexpected RF result | Stop TX; recheck the setup and all command parameters |

## AIC source manuals

This guide was checked against the following files in the AIC source checkout
under `src/AIC_Docs`:

- `AIC8800_wifi_test_*.pdf` (original AIC8800 manual)
- `AIC8800DCDW_wifi_test_*_3.0.pdf` (DC/DW v3.0 manual)
- `AIC8800D80_RF_wifi_test_v60.pdf`
- `AIC8800D80X2_RF_wifi_test_v40_en.pdf`

The PDF filenames containing non-ASCII text are abbreviated above. The vendor
manual for the exact chip and revision remains authoritative for calibration
limits, regulatory setup, and permanent programming. This guide intentionally
does not reproduce the AIC Linux module-loading procedure because it is not
valid for RT-Smart.
