# AIC8800 RF test

This example controls the AIC manufacturing RF firmware through `/dev/aicrfN`.
The kernel must be built with `AIC8800_WIFI_RF_TEST_MODE=y`; normal Wi-Fi is
disabled in that image. Enable
`RTSMART_PERIPHERAL_ENABLE_AIC8800_RF_TEST=y` to install `wifi_test.elf` in the
peripheral examples output.

The normal runtime is an FMAC/full-MAC offload image. RF mode deliberately
loads the vendor manufacturing image instead: AIC8800/AIC8801 calls its image
`fmacfw_rf.bin`, while D80, D80X2, and DC/DW use `lmacfw_rf_*.bin`. The filename
does not change the userspace interface. RF mode supports the AIC test command
protocol only and cannot be used for station or SoftAP networking.

The executable keeps the AIC reference tool command line:

```text
wifi_test.elf <interface> <command> [args]
wifi_test.elf version
```

RF mode has no normal WLAN network interface. For command-line compatibility,
`wlanN` selects `/dev/aicrfN`; a `/dev/aicrfN` path can also be supplied
directly. Command names are case-insensitive and their numeric bases and
argument layouts match the AIC reference driver.

Common commands:

```text
wifi_test.elf wlan0 SET_TX 1 0 0 0 1024 10000
wifi_test.elf wlan0 SET_TXSTOP
wifi_test.elf wlan0 SET_RX 1 0
wifi_test.elf wlan0 SET_RXSTOP
wifi_test.elf wlan0 GET_RX_RESULT
wifi_test.elf wlan0 GET_RSSI
```

The tool queries the chip family from `/dev/aicrfN`, so RT-Smart `msh` does not
need `export` or environment-variable support. The driver reports AIC8800D,
AIC8800DC/DW, AIC8800D80, or AIC8800D80X2 automatically; the official command
arguments remain unchanged.

`raw <command-id> [byte ...]` is retained as a CanMV extension for firmware
commands not covered by a named form. Raw values use C-style numeric notation
and represent exact firmware payload bytes.

Commands whose names contain `efuse`, and `SET_MAC_ADDR`, `SET_BT_MAC_ADDR`,
`SET_VENDOR_INFO`, and `EXEC_FLASH_OPER`, may permanently alter the module.

See [TESTING.md](TESTING.md) for firmware selection, build and deployment,
conducted RF test procedures, command arguments, and recovery to normal Wi-Fi.
