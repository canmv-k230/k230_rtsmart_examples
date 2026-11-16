# thermal_stress

Loads every K230 engine that can run without external input, all at once, and
logs temperature, clocks, power-domain state and per-module performance once a
second. Use it to check thermal protection and how much performance the chip
keeps under sustained full load.

| Module | Power domain | Load | Performance metric |
|---|---|---|---|
| `cpu` | CPU1 (big core) | integer, FP and RVV memory-copy chunks, duty-cycled | chunks/s, as % of a 2 s cool baseline |
| `kpu` | AI | `yolov8n_640.kmodel` inference back to back | inferences/s and latency |
| `ai2d` | AI | 1920x1080 RGB planar letterboxed to 640x640, back to back alongside the KPU | resizes/s |
| `venc` | VPU | H.265 1920x1080 CBR 4 Mbps from synthetic frames | encoded frames/s |
| `vdec` | VPU | decodes a one-GOP clip from `venc` in a loop | decoded frames/s |
| `display` | DISP | colour bars and a moving white band on the HX8399 DSI LCD, paced to the panel refresh | frames shown per second |
| `gpu2d` | DISP | VGLite renders 1920x1080 RGBA off-screen frames of 96 rotating, alpha-blended stars with 16x anti-aliasing | frames/s |
| `isp` | ISP | IMX335 on CSI0 streaming 1080p NV12 | camera frames/s |
| `gsdma` | none (DDR) | GDMA rotates a 1920x1080 NV12 frame by 90 degrees, back to back | frames/s |
| `nonai2d` | none (DDR) | 1920x1080 NV12 to RGB888 planar colour conversion on a free nonai_2d channel | frames/s |

Missing hardware is skipped, not failed: no camera ("sensor not detected on
CSI0"), no model file, a build without the HX8399 panel, or a build without
VGLite (`CONFIG_RTSMART_HAL_ENABLE_VG_LITE`, or no `/dev/vg_lite`). `venc` and
`vdec` share one VPU, so each runs slower while both are active. `gsdma` and
`nonai2d` have no power domain of their own; they add the DDR traffic of a
camera and AI pipeline, and the DDR ports they use are the ones the kernel's
thermal caps throttle along with the VPU and DPU.

Not covered: the DPU needs calibration parameter files, and audio and the
low-speed peripherals add almost no heat.

## Build

Enabled by `CONFIG_RTSMART_ENABLE_THERMAL_STRESS_TEST`, on by default in
`k230_rtos_evb_defconfig`. It installs to
`/sdcard/app/examples/test/thermal_stress/` with the kmodel next to the program.

## Run

Stop anything else using the KPU, VPU, display or GPU first. On this board the
fast-boot `smart_ipc` application uses the KPU, VPU and display.

```sh
cd /sdcard/app/examples/test/thermal_stress
./thermal_stress.elf -t 600 -c 120 -o /sdcard/thermal.csv
```

Common variations:

```sh
./thermal_stress.elf -m cpu,kpu -t 300            # CPU and KPU only
./thermal_stress.elf -l 50 -K 50                  # half load on CPU and KPU
./thermal_stress.elf -T 95                        # stop the load at 95 C
./thermal_stress.elf -m gpu2d,gsdma,nonai2d -t 300  # the newer engines only
./thermal_stress.elf --min-cpu-perf 60 --min-kpu-fps 15 --min-venc-fps 20
```

Options: `-t` load seconds (600), `-c` monitoring seconds after the load (120),
`-i` log interval ms (1000), `-m` module list (`cpu,kpu,ai2d,vpu,display,gpu2d,
isp,gsdma,nonai2d` or `all`), `-l` CPU duty %, `-j` CPU threads,
`-w int|fp|mem|mix`, `-k` kmodel (`none` disables the KPU), `-K` KPU and AI2D
duty %, `-r WxH` encoder size, `-T` stop
temperature, `-o` CSV file, `-V MHz` pin the VPU profile (800/600/400/300/200;
thermal caps still apply). On exit the default (highest) VPU request is restored.
`--gpu-cmdbuf KB` (512, as `vglite_tiger`), `--gpu-stars N` (96) and
`--gpu-quality high|medium|low` (high, 16x anti-aliasing) shape the GPU2D load.
A VGLite library without the 512-row tessellation strip fix hangs the GPU at
`high` within seconds; the tool reports it as `HUNG` and a reboot is needed
afterwards. `--gpu-sweep --gpu-scale 2.4` is the single-path reproducer.
Pass criteria: `--min-cpu-perf`, `--min-kpu-fps`, `--min-venc-fps`,
`--min-vdec-fps`, `--min-disp-fps`, `--min-ai2d-fps`, `--min-gpu2d-fps`,
`--min-gsdma-fps` and `--min-nonai2d-fps`.

Numeric arguments must be complete, finite values within their allowed ranges;
negative values are accepted only for signed options such as GPU scene offsets
and sweep angles. Invalid arguments or a selection with no available modules
exit with status 2. Module initialization or worker startup failures exit with
status 1, including when none of the selected modules can start.

When `-o FILE` is given, the CSV is opened before hardware setup or worker
startup; an open failure exits immediately with status 2. Each run truncates an
existing file. Use a different filename to preserve a log from an earlier run,
including one interrupted by a shutdown.

To measure how much heat the VPU clock saves, compare runs such as
`./thermal_stress.elf -m vpu -V 800 -t 900 -o /sdcard/vpu800.csv` and the same
with `-V 400`: temperature, `venc`/`vdec` fps and the `vpu_mhz` column.

## Reading the output

```
[   42s load]  78.4C reg    | 1188MHz vpu 600 cpu  71% | kpu 800MHz 24.8fps 38ms | venc 41.2 | vdec 38.9 | display 55.8 | pd kpu+ vpu+ dpu- disp+
```

- **Zone** is estimated from the kernel's critical shutdown temperature and the
  level step: regulation from critical −5 steps, WARM −3, HOT −2, MIN −1. The
  kernel calculates the step as `min(5°C, (critical - target) / 4)`. Lower the
  normal `K230_PM_THERMAL_CRITICAL_C` and `K230_PM_THERMAL_TARGET_C` settings
  to test at lower temperatures. The tool infers the step from the default
  per-domain threshold, which is the MIN stage rounded down to whole degrees.
  Fractional steps therefore make the displayed zones approximate; the boot
  log reports the exact stage thresholds in millicelsius. A custom target
  more than 20°C below critical also makes the estimated regulation zone differ.
- **`cpu NN%`** is the CPU loader's throughput relative to its 2 s baseline,
  measured alone before the other modules start. It falls when thermal control
  lowers the clock, and also because the single big core is shared with the
  KPU, codec and display threads. The summary separates the two.
- **`pd`** shows each domain's power state: `+` on, `-` off.

At the end, a summary lists peak temperature, time spent above each threshold,
and for each module its baseline, average, last-30-seconds and minimum rate,
perf % and errors. For the CPU and KPU it also compares the performance loss
with the clock reduction. A loss much larger than the clock explains is flagged:
it points at memory or bus limits or contention rather than thermal throttling.

`result: PASS` / `FAIL` and exit status 0 / 1 apply the `--min-*` criteria to
the last 30 seconds of load. A module that failed to start also fails the run.
This includes shared video-buffer initialization and worker thread creation.
Each active module must complete successful work during the load; repeated
errors do not satisfy that requirement. An explicit zero-duty load is exempt.
Exit status 2 means bad options.

The CSV is flushed every line and synced every five, so it survives a thermal
shutdown. If the board shuts down, the last CSV lines show the temperature and
clocks just before it.

## Limits

- **Display.** The rate should equal the panel refresh, 55.8 Hz for the HX8399,
  computed from its timing. A lower rate or errors mean the display pipeline
  stalled. The screen should show eight vertical bars (white, yellow, cyan,
  green, magenta, red, blue, black, left to right) with a white band scrolling
  smoothly down. Wrong colours, tearing or a frozen band point at a display
  problem.
- **Thread priorities.** The monitor runs at RT-Thread priority 21, display 22,
  encoder and decoder 23, GPU2D, GSDMA and nonai_2d 24, KPU and AI2D 25 and the
  CPU loaders 25. CPU, KPU and AI2D share time slices so a continuously runnable
  AI feeder cannot starve the CPU load. All of these sit below the
  kernel thermal threads (8 and 12) and the shell (20).
- **Stopping.** Ctrl+C stops the load, waits for each module and cleans up.
  A module whose driver call does not return within 5 s (a hung engine) is
  reported as `HUNG`; the run fails and exits without shared teardown or C++
  destructors, keeping worker state alive until the process is stopped. A
  reboot may be needed to recover the hardware. During the load, a module with
  no successful work for 5 s ends as `STALLED` unless it recovers, even if it
  keeps reporting errors. Zero-duty pauses do not count toward this timeout.
  A second Ctrl+C exits at once without cleanup.
- **Media memory.** With every module running the tool takes about 180 MB of
  the 256 MB MMZ. A module that cannot get its buffers reports "MMZ exhausted?"
  and fails; free the memory or leave that module out with `-m`.
- **Estimated zones.** Thermal stage and caps are not exposed to userspace, so
  zones are estimated. Use the kernel's `k230_pm_thermal_dump` on the console
  for the exact stage.
- **Kernel CPU usage** is written to the CSV (`kernel_usage`) but not the
  console. Its sampler runs at a very low priority and goes stale while the core
  is saturated.
