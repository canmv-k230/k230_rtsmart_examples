# 01Studio K230 LVGL Benchmark Report

Date: 2026-09-29

## Test setup

- Board: K230 01Studio (`k230-1` in `k230_lab`)
- LVGL: 9.4.0, upstream revision `c210a4efa2f474d0223d3e91c79963e1ae4ac0bc`
- Color format: ARGB8888
- Benchmark: upstream `lv_demo_benchmark`
- Accelerated image: `k230_rtos_01studio_lvgl_benchmark_vglite_defconfig`
- CPU image: `k230_rtos_01studio_lvgl_benchmark_cpu_defconfig`
- Accelerated renderer reported by the application: `vg_lite+sw`
- CPU renderer reported by the application: `sw`

Both complete SD-card images built successfully and were flashed with
`k230_lab`. Each run completed all 16 benchmark scenes and printed
`LVGL_BENCHMARK_END status=PASS`.

## Summary results

| Connector | Logical resolution / refresh | Rotation | Renderer | CPU | FPS | Render | Flush | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| ST7701 DSI | 480x800 @ 59 Hz | 0 | VG-Lite + SW | 57% | 55 | 5 ms | 9 ms | PASS |
| ST7701 DSI | 480x800 @ 59 Hz | 0 | CPU | 58% | 55 | 4 ms | 11 ms | PASS |
| ST7701 DSI | 800x480 @ 59 Hz | 90 | VG-Lite + SW | 42% | 56 | 9 ms | 0 ms | PASS |
| ST7701 DSI | 800x480 @ 59 Hz | 90 | CPU | 44% | 58 | 7 ms | 1 ms | PASS |
| HX8399 DSI | 1080x1920 @ 55 Hz | 0 | VG-Lite + SW | 58% | 38 | 33 ms | 8 ms | PASS |
| HX8399 DSI | 1080x1920 @ 55 Hz | 0 | CPU | 71% | 32 | 43 ms | 10 ms | PASS |
| LT9611 HDMI | 1920x1080 @ 30 Hz | 0 | VG-Lite + SW | 61% | 22 | 37 ms | 17 ms | PASS |
| LT9611 HDMI | 1920x1080 @ 30 Hz | 0 | CPU | 74% | 21 | 41 ms | 20 ms | PASS |
| LT9611 HDMI | 1280x720 @ 30 Hz | 0 | VG-Lite + SW | 60% | 27 | 13 ms | 19 ms | PASS |
| LT9611 HDMI | 1280x720 @ 30 Hz | 0 | CPU | 68% | 27 | 13 ms | 19 ms | PASS |
| LT9611 HDMI | 1280x960 @ 30 Hz | 0 | VG-Lite + SW | 60% | 27 | 18 ms | 19 ms | PASS |
| LT9611 HDMI | 1280x960 @ 30 Hz | 0 | CPU | 68% | 26 | 18 ms | 20 ms | PASS |

The supported HDMI VG-Lite envelope was tested at 720p30 and 960p30. Both
modes completed successfully. VG-Lite reduced average CPU usage by eight
percentage points at both resolutions. The 720p result was scanout-limited,
while 960p gained one average FPS and reduced average flush time by 1 ms.

The earlier 1080p30 result is retained as an out-of-envelope diagnostic run;
it is not used as the supported VG-Lite HDMI maximum. The ST7701 result is
mostly limited by its 59 Hz scanout and framebuffer synchronization.

## Connector coverage

The tests cover all connector families enabled for the 01Studio target:

- ST7701 DSI using `st7701-480x800`
- HX8399 DSI using `hx8399-1080x1920`
- LT9611 HDMI using `lt9611-1280x720-30` and `lt9611-1280x960-30`

The benchmark command also exposes every configured timing through
`--list-connectors`: four ST7701 modes, one HX8399 mode, and thirteen LT9611
custom/VESA modes. One representative mode was benchmarked for each physical
connector driver; this report does not claim physical display validation of
every timing variant.

## Display correctness

The initial ST7701 run exposed tearing/flicker and stale dirty regions. The
LVGL K230 display port was corrected to:

- wait until VO releases the previous framebuffer before LVGL reuses it;
- present only the final flush of a direct-render frame.

After these changes, repeated ST7701 VG-Lite runs passed and the attached
480x800 panel was visually checked without the earlier artifacts.

The ST7701 90-degree path was also run with both renderers. The application
reported the expected 800x480 logical resolution, completed all 16 scenes,
and returned `status=PASS`. Quarter-turn output uses full-frame GDMA rotation;
the VG-Lite image is left flashed on the board for visual inspection.

HX8399 and LT9611 connector initialization and benchmark execution were
validated through the serial results. The VG-Lite 1280x960@30 image is left
flashed and selected on `k230-1` for HDMI visual inspection.

## Notes

Some serial logs contained USB host `urb timeout` messages. They were emitted
by the unrelated USB host subsystem and did not interrupt any benchmark run.
The benchmark's machine-readable result records remained complete.
