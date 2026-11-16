/* SPDX-License-Identifier: BSD-2-Clause */
/* GPU2D load: the VGLite vector GPU in the display power domain renders
 * off-screen frames back to back, the way an LVGL or vector UI would. Each
 * frame clears a 1920x1080 RGBA8888 target and draws overlapping, rotating
 * stars with 16x anti-aliasing and alpha blending. One unit is one finished
 * frame. Pattern from examples/peripheral/vglite_tiger. */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "stress.h"
#ifdef CONFIG_RTSMART_HAL_ENABLE_VG_LITE
#include "vg_lite.h"
#endif

namespace {

#ifdef CONFIG_RTSMART_HAL_ENABLE_VG_LITE

class gpu2d_module : public stress_module {
public:
    ~gpu2d_module() override { teardown(); }
    const char *name() const override { return "gpu2d"; }
    int priority() const override { return 24; }

    bool setup(const stress_options &o, std::string &why, bool &skipped) override
    {
        skipped = false;
        stars_ = o.gpu_stars;
        height = o.gpu_height;
        scale_ = (float)o.gpu_scale;
        offset_y_ = (float)o.gpu_offset_y;
        sweep_ = o.gpu_sweep;
        sweep_stars_ = o.gpu_sweep_stars;
        sweep_angle_ = (float)o.gpu_sweep_angle;
        sweep_alpha_ = o.gpu_sweep_alpha;
        sweep_from_ = o.gpu_sweep_from;
        sweep_fine_ = o.gpu_sweep_from != 960;
        sweep_hold_ = o.gpu_sweep_hold;
        max_frames_ = o.gpu_frames;
        if (access("/dev/vg_lite", R_OK | W_OK) != 0) {
            skipped = true;
            why = "no /dev/vg_lite (GPU driver not in this kernel)";
            return false;
        }
        if (vg_lite_set_command_buffer_size(o.gpu_cmdbuf_kb * 1024) != VG_LITE_SUCCESS ||
            vg_lite_init(width, height) != VG_LITE_SUCCESS) {
            why = "vg_lite_init failed";
            return false;
        }
        inited_ = true;
        memset(&target_, 0, sizeof(target_));
        target_.width = width;
        target_.height = height;
        target_.format = VG_LITE_RGBA8888;
        if (vg_lite_allocate(&target_) != VG_LITE_SUCCESS) {
            why = "vg_lite_allocate failed (GPU memory exhausted?)";
            return false;
        }
        allocated_ = true;
        /* A scissor makes the driver tessellate each path in a window around
         * its bounds instead of over the whole target. */
        if (o.gpu_scissor && (vg_lite_set_scissor(0, 0, width, height) != VG_LITE_SUCCESS ||
                              vg_lite_enable_scissor() != VG_LITE_SUCCESS)) {
            why = "cannot set the scissor";
            return false;
        }
        build_star();
        memset(&path_, 0, sizeof(path_));
        static const vg_lite_quality_t qualities[] = {VG_LITE_HIGH, VG_LITE_MEDIUM, VG_LITE_LOW};
        static const char *const quality_names[] = {"16x AA", "4x AA", "no AA"};
        if (vg_lite_init_path(&path_, VG_LITE_S32, qualities[o.gpu_quality], star_.size() * sizeof(int32_t),
                star_.data(), -outer, -outer, outer, outer) != VG_LITE_SUCCESS ||
            (o.gpu_upload && vg_lite_upload_path(&path_) != VG_LITE_SUCCESS)) {
            why = "cannot upload the star path";
            return false;
        }
        path_ready_ = true;
        char buf[128];
        snprintf(buf, sizeof(buf), "%ux%u RGBA, %u stars per frame, %s, %u KB command buffer%s%s",
            width, height, stars_, quality_names[o.gpu_quality], o.gpu_cmdbuf_kb,
            o.gpu_upload ? "" : ", paths not uploaded", o.gpu_scissor ? ", scissor" : "");
        detail = buf;
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        uint64_t last = now_us();
        unsigned frame = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (max_frames_ && frame >= max_frames_) {
                /* Frame limit reached (keeps a command dump short). */
                sleep_us(100000);
                continue;
            }
            if (!duty.load(std::memory_order_relaxed)) {
                sleep_us(100000);
                last = now_us();
                continue;
            }
            const char *call = "vg_lite_draw";
            vg_lite_error_t err = render(frame++);
            if (err == VG_LITE_SUCCESS) {
                call = "vg_lite_finish";
                err = vg_lite_finish();
            }
            if (err == VG_LITE_SUCCESS) {
                uint64_t t = now_us();
                counters.add(t - last);  /* latency here is the frame interval */
                last = t;
            } else {
                /* A failed frame leaves the VGLite context unusable (a timed
                 * out wait means the GPU did not go idle), so stop rendering;
                 * the monitor then reports the module as stalled and the run
                 * fails. */
                counters.errors.fetch_add(1, std::memory_order_relaxed);
                printf("gpu2d: %s failed with VGLite error %d at frame %u; stopping the GPU load\n",
                    call, (int)err, frame);
                while (!stop.load(std::memory_order_relaxed))
                    sleep_us(100000);
            }
        }
    }

    void teardown() override
    {
        if (path_ready_)
            vg_lite_clear_path(&path_);
        if (allocated_)
            vg_lite_free(&target_);
        if (inited_)
            vg_lite_close();
        path_ready_ = allocated_ = inited_ = false;
    }

private:
    static constexpr unsigned width = 1920, points = 12;
    unsigned height = 1080;
    static constexpr int32_t outer = 160, inner = 70;

    /* VG_LITE_S32 path data: each opcode takes one 32-bit slot, then its
     * coordinates. A closed star with alternating outer and inner points. */
    void build_star()
    {
        star_.clear();
        for (unsigned i = 0; i < 2 * points; ++i) {
            double a = M_PI * i / points;
            int32_t r = (i & 1) ? inner : outer;
            star_.push_back(i ? VLC_OP_LINE : VLC_OP_MOVE);
            star_.push_back((int32_t)lround(r * sin(a)));
            star_.push_back((int32_t)lround(-r * cos(a)));
        }
        star_.push_back(VLC_OP_CLOSE);
        star_.push_back(VLC_OP_END);
    }

    /* Queue one frame: a clear, then the stars on a grid, each turning and
     * pulsing at its own rate so no two frames are alike. */
    vg_lite_error_t render(unsigned frame)
    {
        vg_lite_error_t err = vg_lite_clear(&target_, nullptr, 0xff202020);
        if (err != VG_LITE_SUCCESS)
            return err;
        if (sweep_)
            return render_sweep(frame);
        const unsigned cols = stars_ < 12 ? stars_ : 12, rows = (stars_ + cols - 1) / cols;
        for (unsigned s = 0; s < stars_; ++s) {
            vg_lite_matrix_t m;
            vg_lite_identity(&m);
            vg_lite_translate((s % cols + 0.5f) * width / cols,
                (s / cols + 0.5f) * height / rows + offset_y_, &m);
            vg_lite_rotate((float)((frame * (3 + s % 7) + s * 15) % 360), &m);
            float scale = scale_ * (0.8f + 0.4f * (float)sin((frame + s * 11) * 0.05));
            vg_lite_scale(scale, scale, &m);
            /* 0xAABBGGRR, half transparent so overlaps blend. */
            vg_lite_color_t colour = 0x80000000u | ((s * 0x2f) & 0xff) << 16 |
                ((frame * 3 + s * 0x51) & 0xff) << 8 | ((s * 0x93) & 0xff);
            err = vg_lite_draw(&target_, &path_, VG_LITE_FILL_NON_ZERO, &m, VG_LITE_BLEND_SRC_OVER, colour);
            if (err != VG_LITE_SUCCESS)
                return err;
        }
        return VG_LITE_SUCCESS;
    }

    unsigned stars_ = 96;
    float scale_ = 1.0f, offset_y_ = 0.0f;
    /* Threshold search: one unrotated star at a fixed size, its lowest point
     * (the star tip straight down) moving down one row every 4 frames from
     * row 960 to 2400, then starting over. The last row printed before the
     * GPU fails brackets the row that triggers it. */
    vg_lite_error_t render_sweep(unsigned frame)
    {
        /* Lowest point of the unrotated star; with --gpu-sweep-angle the tip
         * rises slightly, so the printed row is an upper bound. */
        const float radius = outer * scale_;
        const unsigned first = sweep_from_, last = 2400;
        unsigned lowest = sweep_hold_ ? first : first + (frame / 4) % (last - first);
        if (sweep_hold_ ? frame % 1000 == 0 :
            (frame % 4 == 0 && (sweep_fine_ || lowest % 8 == 0)))
            printf("gpu2d sweep: lowest row %u (frame %u)\n", lowest, frame);
        const vg_lite_color_t colour = (vg_lite_color_t)sweep_alpha_ << 24 | 0x40c0ffu;
        for (unsigned s = 0; s < sweep_stars_; ++s) {
            vg_lite_matrix_t m;
            vg_lite_identity(&m);
            vg_lite_translate(sweep_stars_ == 1 ? width / 2.0f : (s ? 1440.0f : 480.0f), lowest - radius, &m);
            vg_lite_rotate(sweep_angle_, &m);
            vg_lite_scale(scale_, scale_, &m);
            vg_lite_error_t err = vg_lite_draw(&target_, &path_, VG_LITE_FILL_NON_ZERO, &m,
                VG_LITE_BLEND_SRC_OVER, colour);
            if (err != VG_LITE_SUCCESS)
                return err;
        }
        return VG_LITE_SUCCESS;
    }

    bool sweep_ = false;
    unsigned sweep_stars_ = 1, sweep_alpha_ = 255, sweep_from_ = 960;
    bool sweep_fine_ = false, sweep_hold_ = false;
    unsigned max_frames_ = 0;
    float sweep_angle_ = 0;
    std::vector<int32_t> star_;
    vg_lite_buffer_t target_;
    vg_lite_path_t path_;
    bool inited_ = false, allocated_ = false, path_ready_ = false;
};

#else

class gpu2d_module : public stress_module {
public:
    const char *name() const override { return "gpu2d"; }
    bool setup(const stress_options &, std::string &why, bool &skipped) override
    {
        skipped = true;
        why = "VGLite not in this build (CONFIG_RTSMART_HAL_ENABLE_VG_LITE)";
        return false;
    }
    void run(const std::atomic<bool> &) override {}
    void teardown() override {}
};

#endif

} // namespace

stress_module *make_gpu2d_module()
{
    return new gpu2d_module();
}
