# K230 LVGL Benchmark

The benchmark runs LVGL's upstream benchmark demo and emits JSON records on the
RT-Smart console. A full run takes about one minute.

Build the dedicated 01Studio images from the SDK root:

```sh
make k230_rtos_01studio_lvgl_benchmark_vglite_defconfig
make

make k230_rtos_01studio_lvgl_benchmark_cpu_defconfig
make
```

Run the installed application from the RT-Smart shell:

```sh
/sdcard/app/examples/3rd_party/lvgl_benchmark.elf --list-connectors
/sdcard/app/examples/3rd_party/lvgl_benchmark.elf \
    --connector st7701-480x800 --format argb8888
/sdcard/app/examples/3rd_party/lvgl_benchmark.elf \
    --connector st7701-480x800 --rotate 90 --format argb8888
```

The `LVGL_BENCHMARK_BEGIN`, `LVGL_BENCHMARK_RESULT`, and
`LVGL_BENCHMARK_END` prefixes are stable and intended for serial-log parsers.
The renderer is compiled into the firmware and is reported as `vg_lite+sw` or
`sw`.

The 01Studio hardware results are recorded in
[`REPORT_01STUDIO_2026-09-29.md`](REPORT_01STUDIO_2026-09-29.md).
