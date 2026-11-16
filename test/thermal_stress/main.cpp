/* SPDX-License-Identifier: BSD-2-Clause */
/* thermal_stress: load the CPU, KPU and AI2D, VPU (encode + decode), display,
 * GPU2D, GSDMA, nonai_2d and, when fitted, the camera ISP together, and log
 * temperature, clocks, power-domain state and per-module performance once per
 * interval. Absent hardware is skipped. The CSV log is flushed every line so it
 * survives a thermal shutdown.
 * Exit status: 0 pass, 1 fail, 2 usage or setup error. */
#include <cerrno>
#include <cctype>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <getopt.h>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "canmv_misc.h"
#include "drv_tsensor.h"
#include "mpi_pm_api.h"
#include "stress.h"

#define DEFAULT_KMODEL "yolov8n_640.kmodel"
#define INSTALL_DIR "/sdcard/app/examples/test/thermal_stress"

uint64_t now_us()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

void sleep_us(uint64_t us)
{
    while (us) {
        unsigned chunk = us > 500000 ? 500000 : (unsigned)us;
        usleep(chunk);
        us -= chunk;
    }
}

bool set_thread_priority(int priority)
{
    struct sched_param param;
    memset(&param, 0, sizeof(param));
    param.sched_priority = priority;
    return pthread_setschedparam(pthread_self(), SCHED_RR, &param) == 0;
}

namespace {

volatile sig_atomic_t interrupted;

/* The first Ctrl+C stops the load and cleans up; a second one quits at once,
 * for when a driver call never returns and cleanup cannot finish. */
void on_signal(int)
{
    if (interrupted) {
        static const char msg[] = "\nsecond interrupt: exiting without cleanup\n";
        (void)!write(STDOUT_FILENO, msg, sizeof(msg) - 1);
        _exit(130);
    }
    interrupted = 1;
}

/* Sent to a load thread that does not stop: interrupts a blocking driver wait
 * (the VGLite and nncase waits are interruptible) so the thread can see stop. */
void on_wake(int) {}

struct domain_clock {
    k_pm_domain domain;
    std::vector<k_pm_profile> profiles;

    void load()
    {
        uint32_t count = 0;
        if (kd_mpi_pm_get_profiles(domain, &count, nullptr) || !count)
            return;
        if (count > K_PM_MAX_PROFILE_COUNT)
            count = K_PM_MAX_PROFILE_COUNT;
        profiles.resize(count);
        if (kd_mpi_pm_get_profiles(domain, &count, profiles.data()))
            profiles.clear();
        else
            profiles.resize(count);
    }
    /* Read-back profile in MHz, or -1. */
    int mhz() const
    {
        int32_t index = -1;
        if (kd_mpi_pm_get_profile(domain, &index) || index < 0 ||
            (size_t)index >= profiles.size())
            return -1;
        return profiles[index].freq / 1000000;
    }
};

struct pd_probe {
    const char *name;
    k_pm_domain domain;
};
const pd_probe pd_probes[] = {
    {"kpu", PM_DOMAIN_KPU}, {"vpu", PM_DOMAIN_VPU}, {"dpu", PM_DOMAIN_DPU},
    {"disp", PM_DOMAIN_DISPLAY},
};

/* Per-module results kept by the monitor. */
struct tracked {
    std::unique_ptr<stress_module> module;
    pthread_t thread{};
    bool started = false;
    std::atomic<bool> stop{false};
    std::atomic<bool> finished{false};  /* run() returned */
    bool hung = false;                  /* run() never returned; hardware left as is */
    double idle_s = 0;                  /* load seconds without successful work */
    bool stall_reported = false;
    uint64_t units = 0, latency = 0, errors = 0;
    uint64_t units_at_load_start = 0;
    double rate = 0, latency_ms = 0, latency_max_ms = 0;
    double baseline = 0, sum = 0, minimum = -1, last_sum = 0;
    unsigned samples = 0, baseline_samples = 0, last_samples = 0;
    double mhz_at_baseline = 0, mhz_last_sum = 0;
};

void *module_thread(void *arg)
{
    auto *t = static_cast<tracked *>(arg);
    if (!set_thread_priority(t->module->priority()))
        printf("%s: cannot set priority %d\n", t->module->name(), t->module->priority());
    t->module->run(t->stop);
    t->finished.store(true);
    return nullptr;
}

bool start_thread(tracked &t)
{
    pthread_attr_t attr;
    int ret = pthread_attr_init(&attr);
    if (ret != 0)
        return false;
    ret = pthread_attr_setstacksize(&attr, 256 * 1024);
    if (ret == 0)
        ret = pthread_create(&t.thread, &attr, module_thread, &t);
    t.started = ret == 0;
    pthread_attr_destroy(&attr);
    return t.started;
}

bool wait_finished(tracked &t, unsigned ms)
{
    for (unsigned waited = 0; !t.finished.load() && waited < ms; waited += 10)
        sleep_us(10000);
    return t.finished.load();
}

/* Stop a module thread without trusting its driver to return: a call that
 * never comes back (a hung engine) must not keep Ctrl+C from working. Returns
 * false if the thread is still blocked; it is then left running and its
 * hardware is not torn down underneath it. */
bool stop_thread(tracked &t)
{
    t.stop.store(true);
    if (!t.started)
        return true;
    if (!wait_finished(t, 3000)) {
        pthread_kill(t.thread, SIGUSR1);
        if (!wait_finished(t, 2000)) {
            printf("%s: still blocked in its driver after 5 s; left running, not torn down\n",
                t.module->name());
            t.hung = true;
            t.started = false;
            pthread_detach(t.thread);
            return false;
        }
    }
    pthread_join(t.thread, nullptr);
    t.started = false;
    return true;
}

/* Estimate levels below critical: MIN -1, HOT -2, WARM -3, target -4, start -5.
 * The PM API rounds MIN to whole degrees, so fractional stage steps and custom
 * lower regulation targets cannot be represented exactly here. */
const char *zone(double temp, int critical, int step)
{
    if (std::isnan(temp) || critical <= 0)
        return "?";
    if (temp >= critical) return "CRIT";
    if (temp >= critical - step) return "MIN";
    if (temp >= critical - 2 * step) return "HOT";
    if (temp >= critical - 3 * step) return "WARM";
    if (temp >= critical - 5 * step) return "reg";
    return "normal";
}

bool parse_u32(const char *text, unsigned &value, int base = 0)
{
    if (!*text || *text == '-' || std::isspace(static_cast<unsigned char>(*text)))
        return false;
    char *end = nullptr;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, base);
    if (errno || end == text || *end || parsed > UINT_MAX)
        return false;
    value = static_cast<unsigned>(parsed);
    return true;
}

bool parse_i32(const char *text, int &value)
{
    if (!*text || std::isspace(static_cast<unsigned char>(*text)))
        return false;
    char *end = nullptr;
    errno = 0;
    long parsed = strtol(text, &end, 0);
    if (errno || end == text || *end || parsed < INT_MIN || parsed > INT_MAX)
        return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parse_double(const char *text, double &value, bool allow_negative = false)
{
    if (!*text || (!allow_negative && *text == '-') ||
        std::isspace(static_cast<unsigned char>(*text)))
        return false;
    char *end = nullptr;
    errno = 0;
    double parsed = strtod(text, &end);
    if (errno || end == text || *end || !std::isfinite(parsed))
        return false;
    value = parsed;
    return true;
}

bool parse_resolution(const char *text, unsigned &width, unsigned &height)
{
    const char *separator = strchr(text, 'x');
    if (!separator)
        return false;
    std::string first(text, separator - text);
    return parse_u32(first.c_str(), width, 10) && parse_u32(separator + 1, height, 10);
}

void usage(const char *prog)
{
    printf("Usage: %s [options]\n"
        "  -t SEC      load duration (default 600)\n"
        "  -c SEC      monitoring after the load stops (default 120)\n"
        "  -i MS       log interval (default 1000)\n"
        "  -m LIST     modules: cpu,kpu,ai2d,vpu,display,gpu2d,isp,gsdma,nonai2d or all\n"
        "              (default all)\n"
        "  -l PCT      CPU duty per 100 ms window (default 100)\n"
        "  -j N        CPU load threads 1..4 (default 1)\n"
        "  -w TYPE     CPU workload int|fp|mem|mix (default mix)\n"
        "  -k FILE     kmodel for the KPU load (default %s next to the program)\n"
        "  -K PCT      KPU and AI2D duty (default 100)\n"
        "  -r WxH      encoder resolution (default 1920x1080)\n"
        "  -V MHz      pin the VPU profile (800/600/400/300/200); thermal caps still apply\n"
        "  --gpu-cmdbuf KB  --gpu-stars N  --gpu-quality high|medium|low  --gpu-scale F\n"
        "              GPU2D load shape (default 512 KB, 96 stars, high, 1.0)\n"
        "  --gpu-no-upload  send the path inline instead of from GPU memory\n"
        "  --gpu-scissor    full-target scissor (per-path tessellation windows)\n"
        "  --gpu-offset-y N shift the GPU2D scene down by N pixels (negative: up)\n"
        "  --gpu-height N   GPU2D target height in rows (default 1080)\n"
        "  --gpu-sweep      find the row that breaks the GPU: one star walking down\n"
        "  --gpu-sweep-stars 1|2  --gpu-sweep-angle DEG  --gpu-sweep-alpha 0..255\n"
        "  --gpu-sweep-from ROW (print every row)  --gpu-sweep-hold (stay at ROW)\n"
        "  --gpu-frames N   stop drawing after N GPU2D frames\n"
        "  -T DEGC     stop the load early at this temperature (default off)\n"
        "  -o FILE     CSV log, flushed every line\n"
        "  --min-cpu-perf PCT  --min-kpu-fps N  --min-venc-fps N\n"
        "  --min-vdec-fps N    --min-disp-fps N   --min-ai2d-fps N\n"
        "  --min-gpu2d-fps N   --min-gsdma-fps N  --min-nonai2d-fps N\n"
        "              pass criteria on the last 30 s\n"
        "Absent hardware (camera, LCD, model file) is reported and skipped.\n",
        prog, DEFAULT_KMODEL);
}

bool parse_modules(const char *list, stress_options &o)
{
    std::string s(list);
    if (s == "all")
        return true;
    o.cpu = o.kpu = o.vpu = o.display = o.isp = false;
    o.ai2d = o.gpu2d = o.gsdma = o.nonai2d = false;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(',', pos);
        std::string m = s.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (m == "cpu") o.cpu = true;
        else if (m == "kpu") o.kpu = true;
        else if (m == "vpu" || m == "venc" || m == "vdec") o.vpu = true;
        else if (m == "display" || m == "disp") o.display = true;
        else if (m == "isp" || m == "sensor" || m == "camera") o.isp = true;
        else if (m == "ai2d") o.ai2d = true;
        else if (m == "gpu2d" || m == "gpu") o.gpu2d = true;
        else if (m == "gsdma" || m == "gdma") o.gsdma = true;
        else if (m == "nonai2d" || m == "2d" || m == "csc") o.nonai2d = true;
        else return false;
        if (end == std::string::npos)
            break;
        pos = end + 1;
    }
    return true;
}

std::string default_kmodel(const char *argv0)
{
    std::string dir(argv0);
    size_t slash = dir.find_last_of('/');
    std::string candidate = (slash == std::string::npos ? std::string(".") :
        dir.substr(0, slash)) + "/" DEFAULT_KMODEL;
    if (access(candidate.c_str(), R_OK) == 0)
        return candidate;
    return INSTALL_DIR "/" DEFAULT_KMODEL;
}

} // namespace

int main(int argc, char **argv)
{
    stress_options o;
    enum { OPT_MIN_CPU = 256, OPT_MIN_KPU, OPT_MIN_VENC, OPT_MIN_VDEC, OPT_MIN_DISP,
        OPT_MIN_AI2D, OPT_MIN_GPU2D, OPT_MIN_GSDMA, OPT_MIN_NONAI2D,
        OPT_GPU_CMDBUF, OPT_GPU_STARS, OPT_GPU_QUALITY, OPT_GPU_SCALE, OPT_GPU_NO_UPLOAD,
        OPT_GPU_SCISSOR, OPT_GPU_OFFSET_Y, OPT_GPU_HEIGHT, OPT_GPU_SWEEP, OPT_GPU_SWEEP_STARS,
        OPT_GPU_SWEEP_ANGLE, OPT_GPU_SWEEP_ALPHA, OPT_GPU_SWEEP_FROM, OPT_GPU_SWEEP_HOLD,
        OPT_GPU_FRAMES };
    static const struct option long_options[] = {
        {"min-cpu-perf", required_argument, nullptr, OPT_MIN_CPU},
        {"min-kpu-fps", required_argument, nullptr, OPT_MIN_KPU},
        {"min-venc-fps", required_argument, nullptr, OPT_MIN_VENC},
        {"min-vdec-fps", required_argument, nullptr, OPT_MIN_VDEC},
        {"min-disp-fps", required_argument, nullptr, OPT_MIN_DISP},
        {"min-ai2d-fps", required_argument, nullptr, OPT_MIN_AI2D},
        {"min-gpu2d-fps", required_argument, nullptr, OPT_MIN_GPU2D},
        {"min-gsdma-fps", required_argument, nullptr, OPT_MIN_GSDMA},
        {"min-nonai2d-fps", required_argument, nullptr, OPT_MIN_NONAI2D},
        {"gpu-cmdbuf", required_argument, nullptr, OPT_GPU_CMDBUF},
        {"gpu-stars", required_argument, nullptr, OPT_GPU_STARS},
        {"gpu-quality", required_argument, nullptr, OPT_GPU_QUALITY},
        {"gpu-scale", required_argument, nullptr, OPT_GPU_SCALE},
        {"gpu-no-upload", no_argument, nullptr, OPT_GPU_NO_UPLOAD},
        {"gpu-scissor", no_argument, nullptr, OPT_GPU_SCISSOR},
        {"gpu-offset-y", required_argument, nullptr, OPT_GPU_OFFSET_Y},
        {"gpu-height", required_argument, nullptr, OPT_GPU_HEIGHT},
        {"gpu-sweep", no_argument, nullptr, OPT_GPU_SWEEP},
        {"gpu-sweep-stars", required_argument, nullptr, OPT_GPU_SWEEP_STARS},
        {"gpu-sweep-angle", required_argument, nullptr, OPT_GPU_SWEEP_ANGLE},
        {"gpu-sweep-alpha", required_argument, nullptr, OPT_GPU_SWEEP_ALPHA},
        {"gpu-sweep-from", required_argument, nullptr, OPT_GPU_SWEEP_FROM},
        {"gpu-sweep-hold", no_argument, nullptr, OPT_GPU_SWEEP_HOLD},
        {"gpu-frames", required_argument, nullptr, OPT_GPU_FRAMES},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int c;
    while ((c = getopt_long(argc, argv, "t:c:i:m:l:j:w:k:K:r:V:T:o:h", long_options, nullptr)) != -1) {
        bool valid = true;
        switch (c) {
        case 't': valid = parse_u32(optarg, o.load_s); break;
        case 'c': valid = parse_u32(optarg, o.cooldown_s); break;
        case 'i': valid = parse_u32(optarg, o.interval_ms); break;
        case 'm':
            if (!parse_modules(optarg, o)) { usage(argv[0]); return 2; }
            break;
        case 'l': valid = parse_u32(optarg, o.cpu_duty); break;
        case 'j': valid = parse_u32(optarg, o.cpu_threads); break;
        case 'w':
            if (!strcmp(optarg, "int")) o.workload = cpu_workload::integer;
            else if (!strcmp(optarg, "fp")) o.workload = cpu_workload::fp;
            else if (!strcmp(optarg, "mem")) o.workload = cpu_workload::mem;
            else if (!strcmp(optarg, "mix")) o.workload = cpu_workload::mix;
            else { usage(argv[0]); return 2; }
            break;
        case 'k': o.kmodel = optarg; break;
        case 'K': valid = parse_u32(optarg, o.kpu_duty); break;
        case 'r': valid = parse_resolution(optarg, o.venc_width, o.venc_height); break;
        case 'V': valid = parse_u32(optarg, o.vpu_mhz); break;
        case 'T': valid = parse_double(optarg, o.stop_c); break;
        case 'o': o.csv = optarg; break;
        case OPT_MIN_CPU: valid = parse_double(optarg, o.min_cpu_perf); break;
        case OPT_MIN_KPU: valid = parse_double(optarg, o.min_kpu_fps); break;
        case OPT_MIN_VENC: valid = parse_double(optarg, o.min_venc_fps); break;
        case OPT_MIN_VDEC: valid = parse_double(optarg, o.min_vdec_fps); break;
        case OPT_MIN_DISP: valid = parse_double(optarg, o.min_disp_fps); break;
        case OPT_MIN_AI2D: valid = parse_double(optarg, o.min_ai2d_fps); break;
        case OPT_MIN_GPU2D: valid = parse_double(optarg, o.min_gpu2d_fps); break;
        case OPT_MIN_GSDMA: valid = parse_double(optarg, o.min_gsdma_fps); break;
        case OPT_MIN_NONAI2D: valid = parse_double(optarg, o.min_nonai2d_fps); break;
        case OPT_GPU_CMDBUF: valid = parse_u32(optarg, o.gpu_cmdbuf_kb); break;
        case OPT_GPU_STARS: valid = parse_u32(optarg, o.gpu_stars); break;
        case OPT_GPU_SCALE: valid = parse_double(optarg, o.gpu_scale); break;
        case OPT_GPU_NO_UPLOAD: o.gpu_upload = false; break;
        case OPT_GPU_SCISSOR: o.gpu_scissor = true; break;
        case OPT_GPU_OFFSET_Y: valid = parse_i32(optarg, o.gpu_offset_y); break;
        case OPT_GPU_HEIGHT: valid = parse_u32(optarg, o.gpu_height); break;
        case OPT_GPU_SWEEP: o.gpu_sweep = true; break;
        case OPT_GPU_SWEEP_STARS: valid = parse_u32(optarg, o.gpu_sweep_stars); break;
        case OPT_GPU_SWEEP_ANGLE: valid = parse_double(optarg, o.gpu_sweep_angle, true); break;
        case OPT_GPU_SWEEP_ALPHA: valid = parse_u32(optarg, o.gpu_sweep_alpha); break;
        case OPT_GPU_SWEEP_FROM: valid = parse_u32(optarg, o.gpu_sweep_from); break;
        case OPT_GPU_SWEEP_HOLD: o.gpu_sweep_hold = true; break;
        case OPT_GPU_FRAMES: valid = parse_u32(optarg, o.gpu_frames); break;
        case OPT_GPU_QUALITY:
            if (!strcmp(optarg, "high")) o.gpu_quality = 0;
            else if (!strcmp(optarg, "medium")) o.gpu_quality = 1;
            else if (!strcmp(optarg, "low")) o.gpu_quality = 2;
            else { usage(argv[0]); return 2; }
            break;
        default: usage(argv[0]); return c == 'h' ? 0 : 2;
        }
        if (!valid) {
            fprintf(stderr, "invalid numeric value: %s\n", optarg);
            usage(argv[0]);
            return 2;
        }
    }
    if (optind != argc || !o.load_s || !o.interval_ms || o.cpu_duty > 100 || o.kpu_duty > 100 ||
        !o.cpu_threads || o.cpu_threads > 4 || o.venc_width < 64 || o.venc_height < 64 ||
        o.gpu_cmdbuf_kb < 16 || o.gpu_cmdbuf_kb > 4096 || !o.gpu_stars || o.gpu_stars > 1000 ||
        !(o.gpu_scale > 0.01 && o.gpu_scale <= 4) || o.gpu_sweep_from < 400 || o.gpu_sweep_from >= 2400 ||
        o.gpu_height < 16 || o.gpu_height > 4096 || o.gpu_sweep_alpha > 255 ||
        (o.gpu_sweep_stars != 1 && o.gpu_sweep_stars != 2)) {
        usage(argv[0]);
        return 2;
    }
    if (o.kmodel.empty())
        o.kmodel = default_kmodel(argv[0]);
    if (o.kmodel == "none")
        o.kpu = false;
    if (!(o.cpu || o.kpu || o.ai2d || o.gpu2d || o.vpu || o.display || o.isp || o.gsdma || o.nonai2d)) {
        fprintf(stderr, "no modules selected\n");
        usage(argv[0]);
        return 2;
    }

    /* Validate the output before taking power votes or starting workers.
     * Ownership also closes the file on subsequent setup errors. */
    std::unique_ptr<FILE, decltype(&fclose)> csv_file(nullptr, fclose);
    if (!o.csv.empty()) {
        csv_file.reset(fopen(o.csv.c_str(), "w"));
        if (!csv_file) {
            fprintf(stderr, "cannot open %s: %s\n", o.csv.c_str(), strerror(errno));
            return 2;
        }
    }
    FILE *csv = csv_file.get();
    /* The monitor must log on time even while every load runs. */
    set_thread_priority(21);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    {
        struct sigaction wake;
        memset(&wake, 0, sizeof(wake));
        wake.sa_handler = on_wake;  /* no SA_RESTART: blocking calls return */
        sigaction(SIGUSR1, &wake, nullptr);
    }
    signal(SIGPIPE, SIG_IGN);

    /* Board state before loading anything. */
    domain_clock cpu_clock{PM_DOMAIN_CPU, {}}, kpu_clock{PM_DOMAIN_KPU, {}};
    domain_clock vpu_clock{PM_DOMAIN_VPU, {}};
    cpu_clock.load();
    kpu_clock.load();
    vpu_clock.load();
    int32_t critical = 0;
    if (kd_mpi_pm_get_thermal_shutdown(&critical))
        critical = 0;
    /* The default per-domain threshold is the MIN stage, one step below
     * critical; fall back to the production 5 C if it was changed. */
    int step = 5;
    int32_t protect_temp = 0, protect_index = 0;
    if (critical && !kd_mpi_pm_get_thermal_protect(PM_DOMAIN_CPU, &protect_temp, &protect_index) &&
        critical - protect_temp >= 1 && critical - protect_temp <= 5)
        step = critical - protect_temp;
    printf("== thermal_stress ==\n");
    printf("critical shutdown %d C, level step %d C; estimated zones: reg>=%d target %d WARM>=%d HOT>=%d MIN>=%d\n",
        critical, step, critical - 5 * step, critical - 4 * step, critical - 3 * step,
        critical - 2 * step, critical - step);
    printf("cpu profiles:");
    for (auto &p : cpu_clock.profiles) printf(" %d", p.freq / 1000000);
    printf(" MHz\nkpu profiles:");
    for (auto &p : kpu_clock.profiles) printf(" %d", p.freq / 1000000);
    printf(" MHz\nvpu profiles:");
    for (auto &p : vpu_clock.profiles) printf(" %d", p.freq / 1000000);
    printf(" MHz\n");

    /* Optional VPU pin for before/after measurements. The PM API reports the
     * read-back profile, which a thermal cap may already have lowered, so on
     * exit the default (highest) request is restored rather than that. */
    bool vpu_pinned = false;
    if (o.vpu_mhz) {
        int32_t index = -1;
        for (size_t i = 0; i < vpu_clock.profiles.size(); ++i)
            if ((unsigned)(vpu_clock.profiles[i].freq / 1000000) == o.vpu_mhz)
                index = (int32_t)i;
        if (index < 0) {
            printf("-V %u: no such VPU profile\n", o.vpu_mhz);
            return 2;
        }
        if (kd_mpi_pm_set_profile(PM_DOMAIN_VPU, index)) {
            printf("-V %u: cannot set the VPU profile (governor not manual?)\n", o.vpu_mhz);
            return 2;
        }
        vpu_pinned = true;
        printf("vpu pinned to %u MHz\n", o.vpu_mhz);
    }

    /* Acquire every requested module; absent hardware is skipped. */
    std::vector<std::unique_ptr<tracked>> mods;
    auto add = [&](stress_module *m) {
        auto t = std::make_unique<tracked>();
        t->module.reset(m);
        mods.push_back(std::move(t));
    };
    if (o.cpu)
        for (unsigned i = 0; i < o.cpu_threads; ++i) add(make_cpu_module(i));
    if (o.kpu) add(make_kpu_module());
    if (o.ai2d) add(make_ai2d_module());
    if (o.gpu2d) add(make_gpu2d_module());
    bool media = o.vpu || o.display || o.isp || o.gsdma || o.nonai2d;
    bool vb_ok = false;
    std::string why;
    if (media) {
        vb_ok = mpp_vb_start(why);
        if (!vb_ok)
            printf("media: FAILED: video buffers unavailable (%s)\n",
                why.c_str());
    }
    if (vb_ok) {
        if (o.vpu) { add(make_venc_module()); add(make_vdec_module()); }
        if (o.display) add(make_display_module());
        if (o.isp) add(make_isp_module());
        if (o.gsdma) add(make_gsdma_module());
        if (o.nonai2d) add(make_nonai2d_module());
    }
    bool setup_failed = media && !vb_ok;
    for (auto &t : mods) {
        bool skipped = false;
        why.clear();
        if (t->module->setup(o, why, skipped)) {
            t->module->status = stress_module::state::running;
        } else {
            t->module->status = skipped ? stress_module::state::skipped : stress_module::state::failed;
            setup_failed |= !skipped;
        }
        printf("%-8s %s%s%s\n", t->module->name(),
            t->module->status == stress_module::state::running ? "ready" :
            t->module->status == stress_module::state::skipped ? "skipped" : "FAILED",
            why.empty() && t->module->detail.empty() ? "" : ": ",
            why.empty() ? t->module->detail.c_str() : why.c_str());
    }

    auto running = [](tracked &t) { return t.module->status == stress_module::state::running; };
    auto is_cpu = [](tracked &t) { return !strncmp(t.module->name(), "cpu", 3); };
    // Called only before workers start or after every worker has stopped.
    auto cleanup = [&] {
        for (auto &t : mods)
            if (running(*t))
                t->module->teardown();
        mods.clear();
        if (vb_ok)
            mpp_vb_stop();
        if (vpu_pinned && kd_mpi_pm_set_profile(PM_DOMAIN_VPU, 0) == 0)
            printf("vpu request restored to the default (highest) profile\n");
    };
    auto any_running = [&] {
        for (auto &t : mods)
            if (running(*t))
                return true;
        return false;
    };
    if (!any_running()) {
        fprintf(stderr, "no modules available to run\n");
        cleanup();
        if (!setup_failed)
            usage(argv[0]);
        return setup_failed ? 1 : 2;
    }
    auto start_module = [&](tracked &t) {
        if (!start_thread(t)) {
            t.module->status = stress_module::state::failed;
            t.module->detail = "thread creation failed";
            setup_failed = true;
            printf("%s: FAILED: thread creation failed\n", t.module->name());
        }
    };

    /* CPU baseline alone, before the other engines add heat or contention. */
    for (auto &t : mods)
        if (running(*t) && is_cpu(*t)) start_module(*t);
    double cpu_baseline_mhz = cpu_clock.mhz();
    sleep_us(2000000);
    for (auto &t : mods)
        if (running(*t) && is_cpu(*t)) {
            t->units = t->module->counters.units.load();
            t->baseline = t->units / 2.0;
            t->baseline_samples = 1;
            t->mhz_at_baseline = cpu_baseline_mhz;
        }
    for (auto &t : mods)
        if (running(*t)) {
            t->units = t->module->counters.units.load();
            t->latency = t->module->counters.latency_us.load();
            t->units_at_load_start = t->units;
            if (!is_cpu(*t))
                start_module(*t);
        }

    /* Thread startup can fail, so choose CSV columns only after the set of
     * running modules is final. Rows use that same set throughout the run. */
    if (!any_running()) {
        fprintf(stderr, "no worker threads available to run\n");
        cleanup();
        return 1;
    }
    if (csv) {
        fprintf(csv, "time_s,phase,temp_c,zone,kernel_usage,cpu_mhz,kpu_mhz,vpu_mhz");
        for (auto &p : pd_probes) fprintf(csv, ",pd_%s", p.name);
        for (auto &t : mods)
            if (running(*t))
                fprintf(csv, ",%s_rate,%s_lat_ms,%s_lat_max_ms,%s_err", t->module->name(),
                    t->module->name(), t->module->name(), t->module->name());
        fprintf(csv, "\n");
        fflush(csv);
    }

    const uint64_t start = now_us();
    uint64_t last = start;
    const double interval = o.interval_ms / 1000.0;
    bool loading = true, stopped_hot = false;
    double peak = -1000, peak_t = 0, temp_end_load = NAN, temp_start = NAN;
    double above[4] = {0, 0, 0, 0};
    unsigned lines = 0;
    for (;;) {
        sleep_us(o.interval_ms * 1000ull);
        uint64_t now = now_us();
        double t_s = (now - start) / 1e6, dt = (now - last) / 1e6;
        last = now;
        if (loading && (t_s >= o.load_s || interrupted)) {
            loading = false;
            /* Signal everything before joining any thread, then release
             * hardware feeders before the CPU loaders. */
            for (auto &t : mods)
                t->stop.store(true);
            for (int pass = 0; pass < 2; ++pass)
                for (auto &t : mods) {
                    if (is_cpu(*t) != (pass == 1))
                        continue;
                    if (stop_thread(*t) && running(*t))
                        t->module->teardown();
                }
            printf("-- load stopped at %.0f s%s; monitoring for %u s --\n", t_s,
                interrupted ? " (interrupted)" : "", interrupted ? 0 : o.cooldown_s);
        }
        double temp = NAN;
        if (drv_tsensor_read_temperature(&temp))
            temp = NAN;
        if (std::isnan(temp_start)) temp_start = temp;
        int usage = -1;
        if (canmv_misc_get_cpu_usage(&usage) || usage < 0 || usage > 100)
            usage = -1;
        int cpu_mhz = cpu_clock.mhz(), kpu_mhz = kpu_clock.mhz(), vpu_mhz = vpu_clock.mhz();
        int pd[sizeof(pd_probes) / sizeof(pd_probes[0])];
        for (size_t i = 0; i < sizeof(pd_probes) / sizeof(pd_probes[0]); ++i)
            pd[i] = kd_mpi_pm_get_stat(pd_probes[i].domain);
        if (!std::isnan(temp)) {
            if (temp > peak) { peak = temp; peak_t = t_s; }
            for (int z = 0; z < 4; ++z)
                if (loading && critical && temp >= critical - 4 * step + step * z) above[z] += dt;
            if (loading) temp_end_load = temp;
        }

        /* Per-module deltas for this interval. */
        for (auto &t : mods) {
            if (!running(*t)) continue;
            auto &k = t->module->counters;
            uint64_t u = k.units.load(), l = k.latency_us.load(), e = k.errors.load();
            uint64_t du = u - t->units, dl = l - t->latency;
            t->rate = loading ? du / dt : 0;
            t->latency_ms = du ? dl / 1000.0 / du : 0;
            t->latency_max_ms = k.latency_max_us.exchange(0) / 1000.0;
            bool work_expected = t->module->duty.load() != 0;
            bool progressed = du != 0;
            t->units = u; t->latency = l; t->errors = e;
            if (!loading) continue;
            /* Repeated failures are not successful work. An intentional
             * zero-duty pause must not trip the no-success timeout. */
            t->idle_s = progressed || !work_expected ? 0 : t->idle_s + dt;
            if ((progressed || !work_expected) && t->stall_reported) {
                printf("%s: running again\n", t->module->name());
                t->stall_reported = false;
            } else if (t->idle_s >= 5 && !t->stall_reported) {
                printf("%s: no successful work for 5 s (errors, starvation or blocked call)\n",
                    t->module->name());
                t->stall_reported = true;
            }
            /* Non-CPU baselines: first five seconds of steady running. */
            if (!is_cpu(*t) && t_s > 1.5 && t->baseline_samples < (unsigned)(5 / interval + 0.5)) {
                t->baseline = (t->baseline * t->baseline_samples + t->rate) / (t->baseline_samples + 1);
                t->baseline_samples++;
                if (!strcmp(t->module->name(), "kpu")) t->mhz_at_baseline = kpu_mhz;
            }
            t->sum += t->rate;
            t->samples++;
            if (t->minimum < 0 || t->rate < t->minimum) t->minimum = t->rate;
            if (t_s > o.load_s - 30) {
                t->last_sum += t->rate;
                t->last_samples++;
                t->mhz_last_sum += is_cpu(*t) ? cpu_mhz : kpu_mhz;
            }
        }

        /* Console line. */
        char line[768];
        /* The kernel usage figure is left to the CSV: its sampler runs at a
         * very low priority and goes stale while the core is saturated. */
        int n = snprintf(line, sizeof(line), "[%5.0fs %s] %5.1fC %-6s | %4dMHz vpu %3d",
            t_s, loading ? "load" : "cool", temp, zone(temp, critical, step), cpu_mhz, vpu_mhz);
        for (auto &t : mods) {
            if (!running(*t) || n >= (int)sizeof(line) - 48) continue;
            if (is_cpu(*t))
                n += snprintf(line + n, sizeof(line) - n, " %s %3.0f%%", t->module->name(),
                    t->baseline > 0 ? 100.0 * t->rate / t->baseline : 0.0);
            else if (!strcmp(t->module->name(), "kpu"))
                n += snprintf(line + n, sizeof(line) - n, " | kpu %dMHz %.1ffps %.0fms", kpu_mhz,
                    t->rate, t->latency_ms);
            else
                n += snprintf(line + n, sizeof(line) - n, " | %s %.1f", t->module->name(), t->rate);
            if (t->errors && n < (int)sizeof(line) - 16)
                n += snprintf(line + n, sizeof(line) - n, " err%llu", (unsigned long long)t->errors);
        }
        if (n < (int)sizeof(line) - 40) {
            n += snprintf(line + n, sizeof(line) - n, " | pd");
            for (size_t i = 0; i < sizeof(pd_probes) / sizeof(pd_probes[0]); ++i)
                n += snprintf(line + n, sizeof(line) - n, " %s%s", pd_probes[i].name,
                    pd[i] == 1 ? "+" : pd[i] == 0 ? "-" : "?");
        }
        puts(line);

        if (csv) {
            fprintf(csv, "%.1f,%s,%.2f,%s,%d,%d,%d,%d", t_s, loading ? "load" : "cool", temp,
                zone(temp, critical, step), usage, cpu_mhz, kpu_mhz, vpu_mhz);
            for (size_t i = 0; i < sizeof(pd_probes) / sizeof(pd_probes[0]); ++i)
                fprintf(csv, ",%d", pd[i]);
            for (auto &t : mods)
                if (running(*t))
                    fprintf(csv, ",%.2f,%.2f,%.2f,%llu", t->rate, t->latency_ms, t->latency_max_ms,
                        (unsigned long long)t->errors);
            fprintf(csv, "\n");
            fflush(csv);
            /* The file must survive a thermal shutdown: sync every few lines. */
            if (++lines % 5 == 0)
                fsync(fileno(csv));
        }

        if (loading && o.stop_c > 0 && !std::isnan(temp) && temp >= o.stop_c) {
            printf("-- temperature %.1f C reached the -T limit %.1f C --\n", temp, o.stop_c);
            stopped_hot = true;
            o.load_s = (unsigned)t_s;
        }
        if (!loading && (interrupted || t_s >= o.load_s + o.cooldown_s))
            break;
    }
    if (csv) {
        fsync(fileno(csv));
        csv_file.reset();
    }

    /* Summary and pass/fail. */
    bool pass = !setup_failed;
    printf("\n== summary ==\n");
    printf("temperature: start %.1f C, peak %.1f C at %.0f s, end of load %.1f C\n",
        temp_start, peak, peak_t, temp_end_load);
    if (critical)
        printf("time at >= %d/%d/%d/%d C (regulation target/WARM/HOT/MIN): %.0f/%.0f/%.0f/%.0f s\n",
            critical - 4 * step, critical - 3 * step, critical - 2 * step, critical - step,
            above[0], above[1], above[2], above[3]);
    if (stopped_hot)
        printf("load stopped early by -T\n");
    printf("%-8s %-8s %10s %10s %10s %10s %7s %8s\n", "module", "status", "baseline",
        "average", "last30s", "minimum", "perf%", "errors");
    for (auto &t : mods) {
        const char *st = t->hung ? "HUNG" :
            t->module->status == stress_module::state::running ? (t->stall_reported ? "STALLED" : "ran") :
            t->module->status == stress_module::state::skipped ? "skipped" : "FAILED";
        if (t->hung || t->stall_reported || t->module->status == stress_module::state::failed)
            pass = false;
        if (!running(*t)) {
            printf("%-8s %-8s %s\n", t->module->name(), st, t->module->detail.c_str());
            continue;
        }
        /* Also catch error-only runs shorter than the stall timeout. CPU
         * baseline work does not count toward the requested load phase. */
        if (t->module->duty.load() != 0 && t->units == t->units_at_load_start) {
            printf("%s: no successful work during the load\n", t->module->name());
            pass = false;
        }
        double avg = t->samples ? t->sum / t->samples : 0;
        double recent = t->last_samples ? t->last_sum / t->last_samples : avg;
        double perf = t->baseline > 0 ? 100.0 * recent / t->baseline : 0;
        printf("%-8s %-8s %10.1f %10.1f %10.1f %10.1f %6.0f%% %8llu  %s\n", t->module->name(), st,
            t->baseline, avg, recent, t->minimum < 0 ? 0 : t->minimum, perf,
            (unsigned long long)t->errors, t->module->unit());
        /* Performance should follow the clock; a larger loss points at memory or
         * bus throttling, contention, or a clock readback problem. */
        if ((is_cpu(*t) || !strcmp(t->module->name(), "kpu")) && t->mhz_at_baseline > 0 &&
            t->last_samples && t->mhz_last_sum > 0) {
            double mhz = t->mhz_last_sum / t->last_samples;
            double expected = 100.0 * mhz / t->mhz_at_baseline;
            printf("         clock %0.f -> %.0f MHz, clock-expected perf %.0f%%%s\n",
                t->mhz_at_baseline, mhz, expected,
                perf < expected * 0.85 ?
                    (is_cpu(*t) ? "  <-- below the clock: CPU time shared with the KPU/VPU/display threads,"
                        " or memory/bus limits" : "  <-- below the clock: memory/bus limits or contention") : "");
        }
        const char *nm = t->module->name();
        if (is_cpu(*t) && o.min_cpu_perf > 0 && perf < o.min_cpu_perf) pass = false;
        if (!strcmp(nm, "kpu") && o.min_kpu_fps > 0 && recent < o.min_kpu_fps) pass = false;
        if (!strcmp(nm, "venc") && o.min_venc_fps > 0 && recent < o.min_venc_fps) pass = false;
        if (!strcmp(nm, "vdec") && o.min_vdec_fps > 0 && recent < o.min_vdec_fps) pass = false;
        if (!strcmp(nm, "display") && o.min_disp_fps > 0 && recent < o.min_disp_fps) pass = false;
        if (!strcmp(nm, "ai2d") && o.min_ai2d_fps > 0 && recent < o.min_ai2d_fps) pass = false;
        if (!strcmp(nm, "gpu2d") && o.min_gpu2d_fps > 0 && recent < o.min_gpu2d_fps) pass = false;
        if (!strcmp(nm, "gsdma") && o.min_gsdma_fps > 0 && recent < o.min_gsdma_fps) pass = false;
        if (!strcmp(nm, "nonai2d") && o.min_nonai2d_fps > 0 && recent < o.min_nonai2d_fps) pass = false;
    }
    for (auto &t : mods)
        if (t->hung) {
            /* A detached worker still uses tracked::stop/finished and may
             * use shared media state. Keep all of it alive until the OS
             * stops the process; stack/global destructors and VB shutdown
             * must not run underneath that worker. */
            printf("a module hung in its driver; exiting without shared cleanup; a reboot may be needed\n");
            printf("result: FAIL\n");
            fflush(nullptr);
            _exit(1);
        }
    cleanup();
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
