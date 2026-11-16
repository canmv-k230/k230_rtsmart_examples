/* SPDX-License-Identifier: BSD-2-Clause */
/* thermal_stress: load every engine that can run without external input and
 * log temperature, clocks and per-module performance. */
#ifndef THERMAL_STRESS_H
#define THERMAL_STRESS_H

#include <atomic>
#include <cstdint>
#include <string>

enum class cpu_workload { integer, fp, mem, mix };

struct stress_options {
    unsigned load_s = 600;        /* load phase */
    unsigned cooldown_s = 120;    /* monitoring after the load stops */
    unsigned interval_ms = 1000;  /* log period */
    unsigned cpu_duty = 100;      /* percent of each 100 ms window */
    unsigned cpu_threads = 1;
    cpu_workload workload = cpu_workload::mix;
    bool cpu = true, kpu = true, vpu = true, display = true, isp = true;
    bool ai2d = true, gpu2d = true, gsdma = true, nonai2d = true;
    unsigned kpu_duty = 100;
    std::string kmodel;           /* empty: next to the program */
    unsigned venc_width = 1920, venc_height = 1080;
    double stop_c = 0;            /* stop the load early at this temperature; 0 = off */
    std::string csv;
    double min_cpu_perf = 0, min_kpu_fps = 0, min_venc_fps = 0, min_vdec_fps = 0;
    double min_disp_fps = 0, min_ai2d_fps = 0, min_gpu2d_fps = 0;
    double min_gsdma_fps = 0, min_nonai2d_fps = 0;
    unsigned vpu_mhz = 0;         /* pin the VPU profile; 0 leaves it to DVFS */
    unsigned gpu_cmdbuf_kb = 512; /* VGLite command buffer, as vglite_tiger */
    unsigned gpu_stars = 96;      /* stars drawn per GPU2D frame */
    double gpu_scale = 1.0;       /* star size; below about 0.4 every star stays inside the target */
    bool gpu_upload = true;       /* draw the path from GPU memory (vg_lite_upload_path) */
    bool gpu_scissor = false;     /* full-target scissor: per-path tessellation windows */
    unsigned gpu_height = 1080;   /* GPU2D target (and tessellation) height in rows */
    int gpu_offset_y = 0;         /* shift every star down (or up, negative) by this many pixels */
    bool gpu_sweep = false;       /* one star whose lowest row walks down from 960, printed as it goes */
    unsigned gpu_sweep_stars = 1; /* sweep: 1 star at x=960, or 2 at x=480 and 1440 */
    double gpu_sweep_angle = 0;   /* sweep: rotation in degrees */
    unsigned gpu_sweep_alpha = 255; /* sweep: colour alpha; below 255 the GPU blends with the target */
    unsigned gpu_sweep_from = 960;  /* sweep: first lowest row; rows are printed one by one when set */
    bool gpu_sweep_hold = false;    /* sweep: keep the star at gpu_sweep_from */
    unsigned gpu_frames = 0;        /* stop drawing after this many GPU2D frames; 0 = no limit */
    unsigned gpu_quality = 0;       /* 0 high (16x AA), 1 medium (4x), 2 low (none) */
};

/* Counters a module updates from its own thread; the monitor takes deltas. */
struct module_counters {
    std::atomic<uint64_t> units{0};      /* chunks, inferences or frames */
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> latency_us{0}; /* sum over units */
    std::atomic<uint64_t> latency_max_us{0};

    void add(uint64_t latency)
    {
        units.fetch_add(1, std::memory_order_relaxed);
        latency_us.fetch_add(latency, std::memory_order_relaxed);
        uint64_t max = latency_max_us.load(std::memory_order_relaxed);
        while (latency > max &&
            !latency_max_us.compare_exchange_weak(max, latency, std::memory_order_relaxed))
            ;
    }
};

class stress_module {
public:
    enum class state { off, skipped, failed, running };

    virtual ~stress_module() = default;
    virtual const char *name() const = 0;
    /* Unit label for the log, e.g. "fps". */
    virtual const char *unit() const { return "fps"; }
    /* Acquire hardware. Return false with why set; skipped marks absent hardware. */
    virtual bool setup(const stress_options &options, std::string &why, bool &skipped) = 0;
    /* Runs on the module's own thread until stop is set. */
    virtual void run(const std::atomic<bool> &stop) = 0;
    /* Release hardware after run() returned. */
    virtual void teardown() = 0;
    /* RT-Thread priority (lower = more urgent). CPU loaders share time
     * slices with KPU/AI2D feeders; display and media feeders run above
     * them. All stay below the thermal threads (8/12) and shell (20). */
    virtual int priority() const { return 25; }
    /* Optional duty change during the test (load phase on/off). */
    virtual void set_duty(unsigned percent) { duty.store(percent); }

    module_counters counters;
    std::atomic<unsigned> duty{100};
    state status = state::off;
    std::string detail;
};

uint64_t now_us();
void sleep_us(uint64_t us);
/* Set the calling thread's RT-Thread priority; returns false if refused. */
bool set_thread_priority(int priority);

stress_module *make_cpu_module(unsigned index);
stress_module *make_kpu_module();
stress_module *make_ai2d_module();
stress_module *make_gpu2d_module();
stress_module *make_venc_module();
stress_module *make_vdec_module();
stress_module *make_display_module();
stress_module *make_isp_module();
stress_module *make_gsdma_module();
stress_module *make_nonai2d_module();

/* Process-wide MPP video-buffer setup shared by the media modules. */
bool mpp_vb_start(std::string &why);
void mpp_vb_stop();

#endif
