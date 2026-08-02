# NimBLE BLE peripheral

This sample runs the RT-Smart NimBLE host stack against an H:4 controller
device and exposes one readable and writable GATT characteristic.

The default controller is `/dev/hci0`. An alternative path can be supplied as
the first argument:

```sh
ble_peripheral.elf /dev/hci0
```

Scan for `K230-NimBLE`. The service UUID is
`6e400001-b5a3-f393-e0a9-e50e24dcca9e`; its value characteristic UUID is
`6e400002-b5a3-f393-e0a9-e50e24dcca9e`.
