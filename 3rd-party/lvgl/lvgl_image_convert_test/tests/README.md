# Wait regression checks

Run the host test from `lvgl_image_convert_test`:

```sh
cc -std=gnu99 -Wall -Wextra -Werror -pthread tests/test_thread.c -o /tmp/lvgl-test-thread
/tmp/lvgl-test-thread
```

It forces an interrupt to arrive before the blocking call, checks normal
completion and worker reuse, and runs stuck workers in child processes. The
two `join failed: 110` messages are expected: the children must exit with
`EXIT_FAILURE`, rather than return while workers still reference their stacks.
The host uses a 200 ms deadline; target builds use the production 2000 ms.

With the SDK configured and libraries built, compile the target checks from
the same directory:

```sh
make C_SRCS=tests/test_thread.c BUILD=/tmp/lvgl-thread-target BIN=/tmp/lvgl-thread-test.elf
make C_SRCS=tests/test_wait_regression.c BUILD=/tmp/lvgl-wait-target BIN=/tmp/lvgl-wait-test.elf
make -C ../lvgl_sensor C_SRCS=tests/test_rotation.c BUILD=/tmp/lvgl-rotation-target BIN=/tmp/lvgl-rotation-test.elf
```

Run the thread executable with `--worker-check` for the positive cases and
separately with `--timeout` and `--timeout-interrupt` for the expected failure
cases. The default host mode uses `fork`, so use these options on RT-Smart.

The wait executable uses the actual regression functions with a simulated
VO/WBC backend. It verifies native priority 10, timeout boundaries, delayed
mutex acquisition, and detection of cleanup that incorrectly takes the busy
mutex. Its final `VO contention cleanup: FAIL` is deliberately injected and
must be followed by `VO regression fixture: PASS`. Signal-aware polling models
driver interruption; this is not a test of real VO ioctls.

The rotation executable tests the actual sensor argument parser: the default
plus four angles through each of `-r`, `--rotate`, and `-rotate` (13 cases).
These isolated executables do not configure the display or camera. The real
display regression executable still needs exclusive use of those resources.

For real camera/display integration on the ST7701 480x800 panel and GC2093
on CSI2, build the bounded sensor application from the sensor directory:

```sh
make -C ../lvgl_sensor C_SRCS=tests/test_sensor_hardware.c BUILD=/tmp/lvgl-sensor-hardware BIN=/tmp/lvgl-sensor-hardware.elf
```

Run it without arguments, with `--rotate 90`, and with `--rotate 270`.
Each run uses the application's actual capture/render/cleanup path, stops
after 120 image submissions (or a 10-second loop deadline), and requires at
least 10 display flushes. It checks the rotation read back from the VO driver,
the logical resolution, and released application mappings. It does not
inspect the physical panel orientation. Stop other media applications first;
CanMV also keeps VB initialized at its REPL, so the shared VB backend must
be released before a native executable can configure its own pools.
