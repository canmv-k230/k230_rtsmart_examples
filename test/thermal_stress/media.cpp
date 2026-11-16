/* SPDX-License-Identifier: BSD-2-Clause */
/* Media loads: VPU encode and decode (one shared VPU), the DSI display, the
 * camera ISP, the GDMA rotation engine and the nonai_2d colour converter.
 * Patterns follow examples/mpp sample_venc, sample_vdec, sample_vo_video,
 * sample_vicap_sensor, sample_gsdma and sample_csc. Inputs are synthetic, so no
 * camera or files are needed except for the ISP, which is probed and skipped
 * when no sensor was detected at boot. */
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <vector>

#include "k_connector_comm.h"
#include "k_gsdma_comm.h"
#include "k_module.h"
#include "k_nonai_2d_comm.h"
#include "k_sensor_comm.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_vdec_comm.h"
#include "k_venc_comm.h"
#include "k_vicap_comm.h"
#include "k_video_comm.h"
#include "k_vo_comm.h"
#include "kd_display.h"
#include "mpi_connector_api.h"
#include "mpi_gsdma_api.h"
#include "mpi_nonai_2d_api.h"
#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vdec_api.h"
#include "mpi_venc_api.h"
#include "mpi_vicap_api.h"
#include "mpi_vo_api.h"
#include "stress.h"

namespace {

bool vb_configured;

/* Encoded access units captured by the encoder for the decoder to replay. */
std::vector<std::vector<uint8_t>> clip;

/* A moving diagonal gradient with per-frame noise keeps motion estimation and
 * entropy coding busy. Written once per block; blocks are then cycled. */
void fill_nv12(uint8_t *y, uint8_t *uv, unsigned w, unsigned h, unsigned frame)
{
    uint32_t seed = 0x9e3779b9u * (frame + 1);
    for (unsigned r = 0; r < h; ++r) {
        uint8_t *row = y + (size_t)r * w;
        for (unsigned c = 0; c < w; ++c) {
            seed = seed * 1664525u + 1013904223u;
            row[c] = (uint8_t)((r + c + frame * 24) & 0xff) ^ (uint8_t)((seed >> 24) & 0x1f);
        }
    }
    for (unsigned r = 0; r < h / 2; ++r) {
        uint8_t *row = uv + (size_t)r * w;
        for (unsigned c = 0; c < w; c += 2) {
            row[c] = (uint8_t)(128 + ((c + frame * 16) & 0x3f));
            row[c + 1] = (uint8_t)(128 + ((r + frame * 8) & 0x3f));
        }
    }
}

/* Display test card: eight BT.601 colour bars (white, yellow, cyan, green,
 * magenta, red, blue, black, left to right) and a white band moving down.
 * One luma and one chroma row of each are rendered once; frames are painted
 * by copying whole rows, because byte stores to the uncached frame buffers
 * are slow and would take CPU time from the other loads. */
struct yuv { uint8_t y, u, v; };
const yuv bar_colours[8] = {
    {235, 128, 128}, {210, 16, 146}, {170, 166, 16}, {145, 54, 34},
    {106, 202, 222}, {81, 90, 240}, {41, 240, 110}, {16, 128, 128},
};
const yuv band_colour = {235, 128, 128};

struct row_template {
    std::vector<uint8_t> y, uv;

    void build(unsigned w, const yuv *solid)
    {
        y.resize(w);
        uv.resize(w);
        for (unsigned c = 0; c < w; ++c) {
            const yuv &p = solid ? *solid : bar_colours[c * 8 / w];
            y[c] = p.y;
            uv[c] = (c & 1) ? p.v : p.u;
        }
    }
};

/* Copy the template into luma rows [row0, row1) and their chroma rows. */
void paint_rows(uint8_t *y, uint8_t *uv, unsigned w, unsigned row0, unsigned row1,
    const row_template &t)
{
    for (unsigned r = row0; r < row1; ++r)
        memcpy(y + (size_t)r * w, t.y.data(), w);
    for (unsigned r = row0 / 2; r < (row1 + 1) / 2; ++r)
        memcpy(uv + (size_t)r * w, t.uv.data(), w);
}

struct vb_frame {
    k_vb_blk_handle handle = VB_INVALID_HANDLE;
    k_u64 phys = 0;
    uint8_t *virt = nullptr;
    k_u32 size = 0;
};

bool get_frame_block(k_u32 pool, k_u32 size, vb_frame &f, bool cached)
{
    f.handle = kd_mpi_vb_get_block(pool, size, NULL);
    if (f.handle == VB_INVALID_HANDLE)
        return false;
    f.phys = kd_mpi_vb_handle_to_phyaddr(f.handle);
    f.size = size;
    f.virt = static_cast<uint8_t *>(cached ? kd_mpi_sys_mmap_cached(f.phys, size) :
        kd_mpi_sys_mmap(f.phys, size));
    if (!f.virt) {
        kd_mpi_vb_release_block(f.handle);
        f.handle = VB_INVALID_HANDLE;
        return false;
    }
    return true;
}

void put_frame_block(vb_frame &f)
{
    if (f.virt)
        kd_mpi_sys_munmap(f.virt, f.size);
    if (f.handle != VB_INVALID_HANDLE)
        kd_mpi_vb_release_block(f.handle);
    f = vb_frame();
}

/* ---------------------------------------------------------------- VENC -- */

class venc_module : public stress_module {
public:
    ~venc_module() override { teardown(); }
    const char *name() const override { return "venc"; }
    int priority() const override { return 23; }

    bool setup(const stress_options &o, std::string &why, bool &skipped) override
    {
        skipped = false;
        w_ = o.venc_width & ~15u;
        h_ = o.venc_height & ~1u;
        uv_off_ = VB_ALIGN_UP(w_ * h_, 4096);
        in_blk_ = VB_ALIGN_UP(uv_off_ + w_ * h_ / 2, 4096);
        k_u32 out_blk = VB_ALIGN_UP(w_ * h_ / 2, 4096);
        out_pool_ = kd_mpi_vb_create_pool_ex(out_blk, 15, VB_REMAP_MODE_NOCACHE);
        in_pool_ = kd_mpi_vb_create_pool_ex(in_blk_, frames, VB_REMAP_MODE_NOCACHE);
        if (out_pool_ == VB_INVALID_POOLID || in_pool_ == VB_INVALID_POOLID) {
            why = "cannot create encoder pools (MMZ exhausted?)";
            return false;
        }
        for (unsigned i = 0; i < frames; ++i) {
            if (!get_frame_block(in_pool_, in_blk_, in_[i], true)) {
                why = "cannot map encoder input blocks";
                return false;
            }
            fill_nv12(in_[i].virt, in_[i].virt + uv_off_, w_, h_, i);
            kd_mpi_sys_mmz_flush_cache(in_[i].phys, in_[i].virt, in_blk_);
        }
        /* The output pool must be attached before the channel is created. */
        if (kd_mpi_venc_attach_vb_pool(ch, out_pool_)) {
            why = "kd_mpi_venc_attach_vb_pool failed";
            return false;
        }
        attached_ = true;
        k_venc_chn_attr a;
        memset(&a, 0, sizeof(a));
        a.venc_attr.type = K_PT_H265;
        a.venc_attr.profile = VENC_PROFILE_H265_MAIN;
        a.venc_attr.pic_width = w_;
        a.venc_attr.pic_height = h_;
        a.rc_attr.rc_mode = K_VENC_RC_MODE_CBR;
        a.rc_attr.cbr.gop = gop;
        a.rc_attr.cbr.src_frame_rate = 30;
        a.rc_attr.cbr.dst_frame_rate = 30;
        a.rc_attr.cbr.bit_rate = 4000;  /* kbps */
        if (kd_mpi_venc_create_chn(ch, &a)) {
            why = "kd_mpi_venc_create_chn failed";
            return false;
        }
        created_ = true;
        /* Repeat VPS/SPS/PPS on every I frame so the clip replays cleanly. */
        kd_mpi_venc_enable_idr(ch, K_TRUE);
        if (kd_mpi_venc_start_chn(ch)) {
            why = "kd_mpi_venc_start_chn failed";
            return false;
        }
        started_ = true;
        /* One GOP becomes the decoder's looping clip. */
        clip.clear();
        for (unsigned n = 0; n < gop; ++n) {
            std::vector<uint8_t> au;
            if (!encode(n, &au)) {
                why = "encoding the decoder clip failed";
                clip.clear();
                return false;
            }
            clip.push_back(std::move(au));
        }
        frame_ = gop;
        char buf[64];
        snprintf(buf, sizeof(buf), "H.265 %ux%u CBR 4 Mbps", w_, h_);
        detail = buf;
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        uint64_t last = now_us();
        while (!stop.load(std::memory_order_relaxed)) {
            if (!duty.load(std::memory_order_relaxed)) {
                sleep_us(100000);
                continue;
            }
            /* Keep two frames queued so the VPU never idles between frames,
             * but never more: a slow or failed get_stream must not let the
             * input queue grow until it overflows. With four input blocks and
             * at most two in flight, a block is reused only after its frame
             * has been encoded. */
            if (in_flight_ < max_in_flight) {
                if (send(frame_)) {
                    ++frame_;
                    ++in_flight_;
                } else {
                    counters.errors.fetch_add(1, std::memory_order_relaxed);
                    sleep_us(1000);
                }
                if (in_flight_ < max_in_flight)
                    continue;
            }
            if (receive(nullptr)) {
                --in_flight_;
                uint64_t t = now_us();
                counters.add(t - last);  /* latency here is the frame interval */
                last = t;
            } else {
                counters.errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void teardown() override
    {
        if (started_) {
            drain();
            kd_mpi_venc_stop_chn(ch);
        }
        if (attached_)
            kd_mpi_venc_detach_vb_pool(ch);
        if (created_)
            kd_mpi_venc_destroy_chn(ch);
        for (auto &f : in_)
            put_frame_block(f);
        if (in_pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(in_pool_);
        if (out_pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(out_pool_);
        if (started_ || created_ || attached_)
            kd_mpi_venc_close_fd();
        started_ = created_ = attached_ = false;
        in_pool_ = out_pool_ = VB_INVALID_POOLID;
    }

private:
    static constexpr k_u32 ch = 0;
    static constexpr unsigned frames = 4, gop = 30;

    /* Synchronous: used for the decoder clip, before the pipelined loop. */
    bool encode(unsigned n, std::vector<uint8_t> *au)
    {
        return send(n) && receive(au);
    }

    bool send(unsigned n)
    {
        vb_frame &in = in_[n % frames];
        k_video_frame_info f;
        memset(&f, 0, sizeof(f));
        f.mod_id = K_ID_VENC;
        f.pool_id = in_pool_;
        f.v_frame.width = w_;
        f.v_frame.height = h_;
        f.v_frame.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        f.v_frame.stride[0] = w_;
        f.v_frame.stride[1] = w_;
        f.v_frame.phys_addr[0] = in.phys;
        f.v_frame.phys_addr[1] = in.phys + uv_off_;
        f.v_frame.pts = (k_u64)n * 33333;
        return kd_mpi_venc_send_frame(ch, &f, 1000) == 0;
    }

    /* Receive one encoded frame. With repeated headers enabled, the
     * VPS/SPS/PPS before each I frame can arrive as a stream of its own;
     * counting that as a frame left one frame queued per GOP until the input
     * queue overflowed. Keep reading until a frame pack arrives, so the
     * headers and their I frame form one access unit for the decoder. */
    bool receive(std::vector<uint8_t> *au)
    {
        for (int tries = 0; tries < 4; ++tries) {
            k_venc_chn_status st;
            if (kd_mpi_venc_query_status(ch, &st))
                return false;
            k_venc_pack packs[16];
            k_venc_stream s;
            memset(&s, 0, sizeof(s));
            s.pack = packs;
            s.pack_cnt = st.cur_packs ? (st.cur_packs > 16 ? 16 : st.cur_packs) : 1;
            if (kd_mpi_venc_get_stream(ch, &s, 1000))
                return false;
            bool frame = false;
            for (k_u32 i = 0; i < s.pack_cnt; ++i) {
                frame |= packs[i].type == K_VENC_I_FRAME || packs[i].type == K_VENC_P_FRAME;
                if (!au)
                    continue;
                void *p = kd_mpi_sys_mmap(packs[i].phys_addr, packs[i].len);
                if (p) {
                    auto *b = static_cast<uint8_t *>(p);
                    au->insert(au->end(), b, b + packs[i].len);
                    kd_mpi_sys_munmap(p, packs[i].len);
                }
            }
            kd_mpi_venc_release_stream(ch, &s);
            if (frame)
                return true;
        }
        return false;
    }

    void drain()
    {
        for (int i = 0; i < 32; ++i) {
            k_venc_chn_status st;
            if (kd_mpi_venc_query_status(ch, &st) || !st.cur_packs)
                return;
            k_venc_pack packs[16];
            k_venc_stream s;
            memset(&s, 0, sizeof(s));
            s.pack = packs;
            s.pack_cnt = st.cur_packs > 16 ? 16 : st.cur_packs;
            if (kd_mpi_venc_get_stream(ch, &s, 100))
                return;
            kd_mpi_venc_release_stream(ch, &s);
        }
    }

    static constexpr unsigned max_in_flight = 2;
    unsigned w_ = 0, h_ = 0, frame_ = 0, in_flight_ = 0;
    k_u32 uv_off_ = 0, in_blk_ = 0;
    k_u32 in_pool_ = VB_INVALID_POOLID, out_pool_ = VB_INVALID_POOLID;
    vb_frame in_[frames];
    bool attached_ = false, created_ = false, started_ = false;
};

/* ---------------------------------------------------------------- VDEC -- */

class vdec_module : public stress_module {
public:
    ~vdec_module() override { teardown(); }
    const char *name() const override { return "vdec"; }
    int priority() const override { return 23; }

    bool setup(const stress_options &o, std::string &why, bool &skipped) override
    {
        skipped = false;
        if (clip.empty()) {
            skipped = true;
            why = "no encoded clip (encoder unavailable)";
            return false;
        }
        size_t largest = 0;
        for (auto &au : clip)
            largest = au.size() > largest ? au.size() : largest;
        stream_blk_ = VB_ALIGN_UP(largest + 4096 > (1u << 20) ? largest + 4096 : (1u << 20), 4096);
        k_u32 vw = VB_ALIGN_UP(o.venc_width & ~15u, 16), vh = VB_ALIGN_UP(o.venc_height & ~1u, 16);
        in_pool_ = kd_mpi_vb_create_pool_ex(stream_blk_, 4, VB_REMAP_MODE_NOCACHE);
        out_pool_ = kd_mpi_vb_create_pool_ex(vw * vh * 2, 6, VB_REMAP_MODE_NOCACHE);
        if (in_pool_ == VB_INVALID_POOLID || out_pool_ == VB_INVALID_POOLID) {
            why = "cannot create decoder pools (MMZ exhausted?)";
            return false;
        }
        if (kd_mpi_vdec_attach_vb_pool(ch, out_pool_)) {
            why = "kd_mpi_vdec_attach_vb_pool failed";
            return false;
        }
        attached_ = true;
        k_vdec_chn_attr a;
        memset(&a, 0, sizeof(a));
        a.type = K_PT_H265;
        a.pic_width = vw;
        a.pic_height = vh;
        a.stream_buf_size = stream_blk_;
        if (kd_mpi_vdec_create_chn(ch, &a)) {
            why = "kd_mpi_vdec_create_chn failed";
            return false;
        }
        created_ = true;
        if (kd_mpi_vdec_start_chn(ch)) {
            why = "kd_mpi_vdec_start_chn failed";
            return false;
        }
        started_ = true;
        char buf[64];
        snprintf(buf, sizeof(buf), "looping a %zu-frame H.265 clip", clip.size());
        detail = buf;
        return true;
    }

    /* Feeds the clip; a helper thread drains decoded frames and counts them. */
    void run(const std::atomic<bool> &stop) override
    {
        stop_ = &stop;
        pthread_t consumer;
        bool consuming = pthread_create(&consumer, nullptr, consume, this) == 0;
        size_t next = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            if (!duty.load(std::memory_order_relaxed)) {
                sleep_us(100000);
                continue;
            }
            const auto &au = clip[next];
            vb_frame b;
            if (!get_frame_block(in_pool_, stream_blk_, b, true)) {
                sleep_us(1000);
                continue;
            }
            memcpy(b.virt, au.data(), au.size());
            kd_mpi_sys_mmz_flush_cache(b.phys, b.virt, au.size());
            k_vdec_stream s;
            memset(&s, 0, sizeof(s));
            s.phy_addr = b.phys;
            s.len = au.size();
            s.end_of_stream = K_FALSE;
            k_s32 ret;
            do {
                ret = kd_mpi_vdec_send_stream(ch, &s, 200);
            } while (ret == K_ERR_VDEC_BUSY && !stop.load(std::memory_order_relaxed));
            if (ret && ret != K_ERR_VDEC_BUSY)
                counters.errors.fetch_add(1, std::memory_order_relaxed);
            put_frame_block(b);  /* the decoder keeps its own reference */
            next = (next + 1) % clip.size();
        }
        if (consuming)
            pthread_join(consumer, nullptr);
    }

    void teardown() override
    {
        if (started_)
            kd_mpi_vdec_stop_chn(ch);
        if (created_)
            kd_mpi_vdec_destroy_chn(ch);
        if (attached_)
            kd_mpi_vdec_detach_vb_pool(ch);  /* only valid after destroy */
        if (in_pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(in_pool_);
        if (out_pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(out_pool_);
        if (started_ || created_ || attached_)
            kd_mpi_vdec_close_fd();
        started_ = created_ = attached_ = false;
        in_pool_ = out_pool_ = VB_INVALID_POOLID;
    }

private:
    static constexpr k_u32 ch = 0;

    static void *consume(void *arg)
    {
        auto *self = static_cast<vdec_module *>(arg);
        set_thread_priority(self->priority());
        uint64_t last = now_us();
        while (!self->stop_->load(std::memory_order_relaxed)) {
            k_video_frame_info f;
            k_vdec_supplement_info sup;
            memset(&f, 0, sizeof(f));
            memset(&sup, 0, sizeof(sup));
            if (kd_mpi_vdec_get_frame(ch, &f, &sup, 100))
                continue;
            if (sup.is_valid_frame) {
                uint64_t t = now_us();
                self->counters.add(t - last);  /* latency here is the frame interval */
                last = t;
            }
            /* Unbound output must be released in order or decoding stalls. */
            kd_mpi_vdec_release_frame(ch, &f);
        }
        return nullptr;
    }

    const std::atomic<bool> *stop_ = nullptr;
    k_u32 stream_blk_ = 0;
    k_u32 in_pool_ = VB_INVALID_POOLID, out_pool_ = VB_INVALID_POOLID;
    bool attached_ = false, created_ = false, started_ = false;
};

/* ------------------------------------------------------------- display -- */

class display_module : public stress_module {
public:
    ~display_module() override { teardown(); }
    const char *name() const override { return "display"; }
    /* Highest of the loads: it must hand the panel a frame every 18 ms. */
    int priority() const override { return 22; }

    bool setup(const stress_options &, std::string &why, bool &skipped) override
    {
        skipped = false;
#ifndef CONFIG_MPP_DSI_ENABLE_LCD_HX8399
        skipped = true;
        why = "no HX8399 panel in this build";
        return false;
#else
        if (kd_display_init_ex(HX8399_1080_1920_DSI_V1, 0, 0, GDMA_ROTATE_DEGREE_0, 60)) {
            why = "kd_display_init failed (panel not connected?)";
            return false;
        }
        inited_ = true;
        if (kd_display_get_resolution(&w_, &h_) || !w_ || !h_ || (w_ & 7)) {
            why = "unusable display resolution";
            return false;
        }
        if (kd_display_layer_configure_ex(layer, PIXEL_FORMAT_YUV_SEMIPLANAR_420, w_, h_, 0, 0, 0xff,
                GDMA_ROTATE_DEGREE_0, 0, 0) || kd_display_layer_enable(layer)) {
            why = "cannot configure the video layer";
            return false;
        }
        enabled_ = true;
        /* Refresh from the panel timing; frames are paced to it, because
         * insert_frame replaces a pending frame and a replaced frame is
         * released without ever being shown. */
        k_connector_info ci;
        memset(&ci, 0, sizeof(ci));
        if (kd_mpi_get_connector_info(HX8399_1080_1920_DSI_V1, &ci) == 0) {
            const k_vo_timing &t = ci.timing;
            uint64_t total = (uint64_t)(t.hactive + t.hsync_len + t.hback_porch + t.hfront_porch) *
                (t.vactive + t.vsync_len + t.vback_porch + t.vfront_porch);
            if (t.pclk_khz && total)
                refresh_hz_ = t.pclk_khz * 1000.0 / total;
        }
        if (refresh_hz_ < 10 || refresh_hz_ > 240)
            refresh_hz_ = 60;
        bars_.build(w_, nullptr);
        band_.build(w_, &band_colour);
        uv_off_ = VB_ALIGN_UP(w_ * h_, 4096);
        k_u32 size = VB_ALIGN_UP(uv_off_ + w_ * h_ / 2, 4096);
        pool_ = kd_mpi_vb_create_pool_ex(size, ring, VB_REMAP_MODE_NONE);
        if (pool_ == VB_INVALID_POOLID) {
            why = "cannot create display pool (MMZ exhausted?)";
            return false;
        }
        for (unsigned i = 0; i < ring; ++i) {
            if (!get_frame_block(pool_, size, fb_[i], false)) {
                why = "cannot map display blocks";
                return false;
            }
            /* Uncached mapping: the display sees CPU writes without a flush. */
            paint_rows(fb_[i].virt, fb_[i].virt + uv_off_, w_, 0, h_, bars_);
            band_y_[i] = h_;  /* no band drawn yet */
            memset(&vf_[i], 0, sizeof(vf_[i]));
            vf_[i].mod_id = K_ID_VO;
            vf_[i].pool_id = pool_;
            vf_[i].v_frame.width = w_;
            vf_[i].v_frame.height = h_;
            vf_[i].v_frame.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            vf_[i].v_frame.stride[0] = w_;
            vf_[i].v_frame.stride[1] = w_;
            vf_[i].v_frame.phys_addr[0] = fb_[i].phys;
            vf_[i].v_frame.phys_addr[1] = fb_[i].phys + uv_off_;
        }
        char buf[96];
        snprintf(buf, sizeof(buf), "HX8399 %ux%u @%.1f Hz: colour bars + moving white band",
            w_, h_, refresh_hz_);
        detail = buf;
        return true;
#endif
    }

    /* One frame per refresh period. A frame counts once the display has
     * released the block it reuses; a stall shows up as a lower rate and as
     * errors (release timeouts). Only the band rows are repainted. */
    void run(const std::atomic<bool> &stop) override
    {
        const uint64_t period = (uint64_t)(1e6 / refresh_hz_);
        uint64_t n = 0, last = now_us(), next = last;
        while (!stop.load(std::memory_order_relaxed)) {
            if (!duty.load(std::memory_order_relaxed)) {
                sleep_us(100000);
                next = now_us();
                continue;
            }
            unsigned i = n % ring;
            if (n >= ring && kd_mpi_vo_wait_frame_release(layer, &vf_[i], 1000)) {
                counters.errors.fetch_add(1, std::memory_order_relaxed);
                next = now_us();
                continue;
            }
            unsigned y = (unsigned)((n * band_step) % (h_ - band_rows));
            uint8_t *luma = fb_[i].virt, *chroma = fb_[i].virt + uv_off_;
            if (band_y_[i] < h_)
                paint_rows(luma, chroma, w_, band_y_[i], band_y_[i] + band_rows, bars_);
            paint_rows(luma, chroma, w_, y, y + band_rows, band_);
            band_y_[i] = y;
            uint64_t now = now_us();
            if (now < next)
                sleep_us(next - now);
            /* Fell more than a frame behind: restart the schedule. */
            next = now > next + period ? now + period : next + period;
            if (kd_mpi_vo_insert_frame(layer, &vf_[i])) {
                counters.errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            uint64_t t = now_us();
            if (n >= ring)
                counters.add(t - last);
            last = t;
            ++n;
        }
    }

    void teardown() override
    {
        if (enabled_)
            kd_display_layer_disable(layer);
        for (auto &f : fb_)
            put_frame_block(f);
        if (pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(pool_);
        if (inited_)
            kd_display_deinit();
        enabled_ = inited_ = false;
        pool_ = VB_INVALID_POOLID;
    }

private:
    static constexpr k_vo_layer_id layer = K_VO_LAYER_VIDEO1;
    static constexpr unsigned ring = 6;  /* VIDEO1 can hold up to four frames */
    static constexpr unsigned band_rows = 80, band_step = 8;
    k_u32 w_ = 0, h_ = 0, uv_off_ = 0;
    double refresh_hz_ = 0;
    unsigned band_y_[ring] = {};
    row_template bars_, band_;
    k_u32 pool_ = VB_INVALID_POOLID;
    vb_frame fb_[ring];
    k_video_frame_info vf_[ring];
    bool inited_ = false, enabled_ = false;
};

/* ----------------------------------------------------------------- ISP -- */

class isp_module : public stress_module {
public:
    ~isp_module() override { teardown(); }
    const char *name() const override { return "isp"; }

    bool setup(const stress_options &, std::string &why, bool &skipped) override
    {
        skipped = false;
        k_vicap_probe_config pc;
        k_vicap_sensor_info si;
        memset(&pc, 0, sizeof(pc));
        memset(&si, 0, sizeof(si));
        pc.csi_num = 0;  /* CSI0 */
        pc.width = 1920;
        pc.height = 1080;
        pc.fps = 30;
        /* Only opens /dev/sensor_*, registered at boot if the chip ID read
         * succeeded: no I2C traffic, cannot hang without a sensor. */
        if (kd_mpi_sensor_adapt_get_ex(&pc, &si, VICAP_MIPI_LANE_PREF_ANY) != 0) {
            skipped = true;
            why = "sensor not detected on CSI0";
            return false;
        }
        if (kd_mpi_vicap_get_sensor_info(si.sensor_type, &si)) {
            why = "kd_mpi_vicap_get_sensor_info failed";
            return false;
        }
        k_vicap_dev_attr d;
        memset(&d, 0, sizeof(d));
        d.acq_win.width = si.width;
        d.acq_win.height = si.height;
        d.mode = VICAP_WORK_ONLINE_MODE;
        d.pipe_ctrl.data = 0xFFFFFFFF;
        d.pipe_ctrl.bits.ahdr_enable = K_FALSE;
        d.pipe_ctrl.bits.dnr3_enable = K_FALSE;
        d.buffer_num = 6;
        d.buffer_size = VB_ALIGN_UP(si.width * si.height * 2, 1024);
        d.buffer_pool_id = VB_INVALID_POOLID;
        memcpy(&d.sensor_info, &si, sizeof(si));
        if (kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, d)) {
            why = "kd_mpi_vicap_set_dev_attr failed";
            return false;
        }
        k_vicap_chn_attr c;
        memset(&c, 0, sizeof(c));
        c.out_win.width = 1920;
        c.out_win.height = 1080;
        c.crop_win = d.acq_win;
        c.scale_win = c.out_win;
        c.crop_enable = K_FALSE;
        c.scale_enable = (si.width != 1920 || si.height != 1080) ? K_TRUE : K_FALSE;
        c.chn_enable = K_TRUE;
        c.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        c.buffer_num = 6;
        c.buffer_size = VB_ALIGN_UP(1920 * 1080 * 3 / 2, 4096);
        c.alignment = 12;
        c.buffer_pool_id = VB_INVALID_POOLID;
        if (kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, c)) {
            why = "kd_mpi_vicap_set_chn_attr failed";
            return false;
        }
        if (kd_mpi_vicap_init(VICAP_DEV_ID_0)) {
            why = "kd_mpi_vicap_init failed";
            return false;
        }
        inited_ = true;
        if (kd_mpi_vicap_start_stream(VICAP_DEV_ID_0)) {
            why = "kd_mpi_vicap_start_stream failed";
            return false;
        }
        started_ = true;
        char buf[64];
        snprintf(buf, sizeof(buf), "sensor %ux%u -> 1920x1080 NV12", si.width, si.height);
        detail = buf;
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        uint64_t last = now_us();
        while (!stop.load(std::memory_order_relaxed)) {
            k_video_frame_info vf;
            memset(&vf, 0, sizeof(vf));
            if (kd_mpi_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, VICAP_DUMP_YUV, &vf, 1000)) {
                counters.errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            uint64_t t = now_us();
            counters.add(t - last);
            last = t;
            kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &vf);
        }
    }

    void teardown() override
    {
        if (started_)
            kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
        if (inited_)
            kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
        started_ = inited_ = false;
    }

private:
    bool inited_ = false, started_ = false;
};

/* --------------------------------------------------------------- GSDMA -- */

/* 2D DMA: rotate a 1080p NV12 frame by 90 degrees, as for a portrait panel.
 * Each frame reads and writes about 3 MB, so this is mostly DDR traffic. */
class gsdma_module : public stress_module {
public:
    ~gsdma_module() override { teardown(); }
    const char *name() const override { return "gsdma"; }
    int priority() const override { return 24; }

    bool setup(const stress_options &, std::string &why, bool &skipped) override
    {
        skipped = false;
        blk_ = VB_ALIGN_UP(w * h * 3 / 2, 4096);
        pool_ = kd_mpi_vb_create_pool_ex(blk_, 2 * ring, VB_REMAP_MODE_NOCACHE);
        if (pool_ == VB_INVALID_POOLID) {
            why = "cannot create GSDMA pool (MMZ exhausted?)";
            return false;
        }
        for (unsigned i = 0; i < ring; ++i) {
            if (!get_frame_block(pool_, blk_, src_[i], true) ||
                !get_frame_block(pool_, blk_, dst_[i], false)) {
                why = "cannot map GSDMA blocks";
                return false;
            }
            fill_nv12(src_[i].virt, src_[i].virt + w * h, w, h, i);
            kd_mpi_sys_mmz_flush_cache(src_[i].phys, src_[i].virt, blk_);
        }
        if (kd_mpi_gsdma_init()) {
            why = "kd_mpi_gsdma_init failed";
            return false;
        }
        inited_ = true;
        /* Device attributes from sample_gsdma. */
        k_gsdma_dev_attr_t a;
        memset(&a, 0, sizeof(a));
        a.outstanding = 7;
        a.ckg_bypass = 0xff;
        a.arbitration_weight = 0x444210;
        if (kd_mpi_gsdma_set_dev_attr(&a)) {
            why = "kd_mpi_gsdma_set_dev_attr failed";
            return false;
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%ux%u NV12 rotate 90", w, h);
        detail = buf;
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        for (unsigned n = 0; !stop.load(std::memory_order_relaxed); ++n) {
            if (!duty.load(std::memory_order_relaxed)) {
                sleep_us(100000);
                continue;
            }
            vb_frame &src = src_[n % ring];
            k_video_frame_info in, out;
            memset(&in, 0, sizeof(in));
            memset(&out, 0, sizeof(out));
            in.pool_id = pool_;
            in.v_frame.width = w;
            in.v_frame.height = h;
            in.v_frame.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            in.v_frame.stride[0] = w;
            in.v_frame.stride[1] = w;
            in.v_frame.phys_addr[0] = src.phys;
            in.v_frame.phys_addr[1] = src.phys + w * h;
            k_gdma_chn_cfg_t cfg;
            memset(&cfg, 0, sizeof(cfg));
            cfg.rotation = GDMA_ROTATE_DEGREE_90;
            uint64_t t0 = now_us();
            if (kd_mpi_gsdma_send_frame(&cfg, &in, &out, dst_[n % ring].handle, 1000) == 0)
                counters.add(now_us() - t0);
            else
                counters.errors.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void teardown() override
    {
        if (inited_)
            kd_mpi_gsdma_deinit();
        for (unsigned i = 0; i < ring; ++i) {
            put_frame_block(src_[i]);
            put_frame_block(dst_[i]);
        }
        if (pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(pool_);
        inited_ = false;
        pool_ = VB_INVALID_POOLID;
    }

private:
    static constexpr unsigned w = 1920, h = 1080, ring = 2;
    k_u32 blk_ = 0, pool_ = VB_INVALID_POOLID;
    vb_frame src_[ring], dst_[ring];
    bool inited_ = false;
};

/* ------------------------------------------------------------ nonai_2d -- */

/* Colour conversion: 1080p NV12 to planar RGB888, the usual step between a
 * camera frame and AI pre-processing. */
class nonai2d_module : public stress_module {
public:
    ~nonai2d_module() override { teardown(); }
    const char *name() const override { return "nonai2d"; }
    int priority() const override { return 24; }

    bool setup(const stress_options &, std::string &why, bool &skipped) override
    {
        skipped = false;
        in_blk_ = VB_ALIGN_UP(w * h * 3 / 2, 4096);
        in_pool_ = kd_mpi_vb_create_pool_ex(in_blk_, ring, VB_REMAP_MODE_NOCACHE);
        out_pool_ = kd_mpi_vb_create_pool_ex(VB_ALIGN_UP(w * h * 3, 4096), 3, VB_REMAP_MODE_NOCACHE);
        if (in_pool_ == VB_INVALID_POOLID || out_pool_ == VB_INVALID_POOLID) {
            why = "cannot create nonai_2d pools (MMZ exhausted?)";
            return false;
        }
        for (unsigned i = 0; i < ring; ++i) {
            if (!get_frame_block(in_pool_, in_blk_, in_[i], true)) {
                why = "cannot map nonai_2d input blocks";
                return false;
            }
            fill_nv12(in_[i].virt, in_[i].virt + w * h, w, h, i);
            kd_mpi_sys_mmz_flush_cache(in_[i].phys, in_[i].virt, in_blk_);
        }
        /* A free channel, so other users of the CSC channels are left alone. */
        if (kd_mpi_nonai_2d_request_chn(&ch_)) {
            why = "no free nonai_2d channel";
            return false;
        }
        requested_ = true;
        if (kd_mpi_nonai_2d_attach_vb_pool(ch_, out_pool_)) {
            why = "kd_mpi_nonai_2d_attach_vb_pool failed";
            return false;
        }
        attached_ = true;
        k_nonai_2d_chn_attr a;
        memset(&a, 0, sizeof(a));
        a.mode = K_NONAI_2D_CALC_MODE_CSC;
        a.dst_fmt = PIXEL_FORMAT_RGB_888_PLANAR;
        if (kd_mpi_nonai_2d_create_chn(ch_, &a)) {
            why = "kd_mpi_nonai_2d_create_chn failed";
            return false;
        }
        created_ = true;
        if (kd_mpi_nonai_2d_start_chn(ch_)) {
            why = "kd_mpi_nonai_2d_start_chn failed";
            return false;
        }
        started_ = true;
        char buf[64];
        snprintf(buf, sizeof(buf), "channel %u: %ux%u NV12 -> RGB888 planar", ch_, w, h);
        detail = buf;
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        for (unsigned n = 0; !stop.load(std::memory_order_relaxed); ++n) {
            if (!duty.load(std::memory_order_relaxed)) {
                sleep_us(100000);
                continue;
            }
            vb_frame &src = in_[n % ring];
            k_video_frame_info in, out;
            memset(&in, 0, sizeof(in));
            memset(&out, 0, sizeof(out));
            in.pool_id = in_pool_;
            in.v_frame.width = w;
            in.v_frame.height = h;
            in.v_frame.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            in.v_frame.stride[0] = w;
            in.v_frame.stride[1] = w;
            in.v_frame.phys_addr[0] = src.phys;
            in.v_frame.phys_addr[1] = src.phys + w * h;
            uint64_t t0 = now_us();
            if (kd_mpi_nonai_2d_send_frame(ch_, &in, 1000) ||
                kd_mpi_nonai_2d_get_frame(ch_, &out, 1000)) {
                counters.errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            counters.add(now_us() - t0);
            kd_mpi_nonai_2d_release_frame(ch_, &out);
        }
    }

    void teardown() override
    {
        if (started_)
            kd_mpi_nonai_2d_stop_chn(ch_);
        if (created_)
            kd_mpi_nonai_2d_destroy_chn(ch_);
        if (attached_)
            kd_mpi_nonai_2d_detach_vb_pool(ch_);
        if (requested_)
            kd_mpi_nonai_2d_release_chn(ch_);
        for (auto &f : in_)
            put_frame_block(f);
        if (in_pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(in_pool_);
        if (out_pool_ != VB_INVALID_POOLID)
            kd_mpi_vb_destory_pool(out_pool_);
        if (requested_)
            kd_mpi_nonai_2d_close();
        started_ = created_ = attached_ = requested_ = false;
        in_pool_ = out_pool_ = VB_INVALID_POOLID;
    }

private:
    static constexpr unsigned w = 1920, h = 1080, ring = 2;
    k_u32 ch_ = 0, in_blk_ = 0;
    k_u32 in_pool_ = VB_INVALID_POOLID, out_pool_ = VB_INVALID_POOLID;
    vb_frame in_[ring];
    bool requested_ = false, attached_ = false, created_ = false, started_ = false;
};

} // namespace

bool mpp_vb_start(std::string &why)
{
    k_vb_config c;
    memset(&c, 0, sizeof(c));
    c.max_pool_cnt = 64;  /* private pools only, including VICAP's own */
    k_s32 ret = kd_mpi_vb_set_config(&c);
    if (ret && ret != K_ERR_VB_BUSY) {
        why = "kd_mpi_vb_set_config failed";
        return false;
    }
    /* BUSY: another process already configured VB; reuse it. */
    vb_configured = ret == 0;
    if (kd_mpi_vb_init()) {
        why = "kd_mpi_vb_init failed";
        return false;
    }
    return true;
}

void mpp_vb_stop()
{
    if (vb_configured)
        kd_mpi_vb_exit();
    vb_configured = false;
}

stress_module *make_venc_module() { return new venc_module(); }
stress_module *make_vdec_module() { return new vdec_module(); }
stress_module *make_display_module() { return new display_module(); }
stress_module *make_isp_module() { return new isp_module(); }
stress_module *make_gsdma_module() { return new gsdma_module(); }
stress_module *make_nonai2d_module() { return new nonai2d_module(); }
