/* SPDX-License-Identifier: BSD-2-Clause */
/* AI domain loads. KPU: run one kmodel back to back on a fixed synthetic input;
 * one unit is one inference, its latency the interpreter run() time. AI2D: the
 * pre-processing most AI applications run before the KPU, a 1080p pad-resize to
 * the 640x640 model input, back to back on its own thread so both engines of
 * the AI domain are busy. Every nncase result is checked: a refused run (for
 * example while thermal protection blocks the accelerators) is counted and
 * retried instead of aborting the program. */
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>
#include <nncase/functional/ai2d/ai2d_builder.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
#include "stress.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::F::k230;

namespace {

/* Fill a host tensor with one byte value and write it back to memory. */
void fill_tensor(runtime_tensor &t, gsl::byte value)
{
    auto host = t.impl()->to_host();
    if (host.is_ok()) {
        auto mapped = host.unwrap()->buffer().as_host().unwrap().map(map_access_::map_write);
        if (mapped.is_ok()) {
            auto span = mapped.unwrap().buffer();
            std::fill(span.begin(), span.end(), value);
        }
    }
    (void)hrt::sync(t, sync_op_t::sync_write_back, true);
}

class kpu_module : public stress_module {
public:
    const char *name() const override { return "kpu"; }

    bool setup(const stress_options &options, std::string &why, bool &skipped) override
    {
        skipped = false;
        duty.store(options.kpu_duty);
        std::ifstream file(options.kmodel, std::ios::binary);
        if (!file) {
            skipped = true;
            why = "kmodel not found: " + options.kmodel;
            return false;
        }
        model_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        auto loaded = interp_.load_model({reinterpret_cast<const gsl::byte *>(model_.data()),
            model_.size()});
        if (!loaded.is_ok()) {
            why = "load_model failed: " + loaded.unwrap_err().message();
            return false;
        }
        for (size_t i = 0; i < interp_.inputs_size(); ++i) {
            if (!bind(true, i, why))
                return false;
        }
        for (size_t i = 0; i < interp_.outputs_size(); ++i) {
            if (!bind(false, i, why))
                return false;
        }
        detail = options.kmodel.substr(options.kmodel.find_last_of('/') + 1);
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        while (!stop.load(std::memory_order_relaxed)) {
            unsigned d = duty.load(std::memory_order_relaxed);
            if (!d) {
                sleep_us(100000);
                continue;
            }
            uint64_t t0 = now_us();
            auto result = interp_.run();
            uint64_t latency = now_us() - t0;
            if (result.is_ok()) {
                counters.add(latency);
            } else {
                counters.errors.fetch_add(1, std::memory_order_relaxed);
                sleep_us(100000);
                continue;
            }
            /* Idle for the remainder so that busy time is d percent. */
            if (d < 100)
                sleep_us(latency * (100 - d) / d);
        }
    }

    void teardown() override {}

private:
    bool bind(bool input, size_t i, std::string &why)
    {
        auto desc = input ? interp_.input_desc(i) : interp_.output_desc(i);
        auto shape = input ? interp_.input_shape(i) : interp_.output_shape(i);
        auto tensor = host_runtime_tensor::create(desc.datatype, shape, hrt::pool_shared);
        if (!tensor.is_ok()) {
            why = "cannot create tensor: " + tensor.unwrap_err().message();
            return false;
        }
        runtime_tensor t = tensor.unwrap();
        /* A fixed mid-grey pattern; the content does not change KPU work. */
        if (input)
            fill_tensor(t, gsl::byte{0x80});
        auto set = input ? interp_.input_tensor(i, t) : interp_.output_tensor(i, t);
        if (!set.is_ok()) {
            why = "cannot bind tensor: " + set.unwrap_err().message();
            return false;
        }
        return true;
    }

    std::vector<char> model_;  /* load_model keeps a view of this buffer */
    interpreter interp_;
};

class ai2d_module : public stress_module {
public:
    const char *name() const override { return "ai2d"; }

    bool setup(const stress_options &options, std::string &why, bool &skipped) override
    {
        skipped = false;
        duty.store(options.kpu_duty);
        dims_t in_shape{1, 3, in_h, in_w}, out_shape{1, 3, out_size, out_size};
        auto in = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared);
        auto out = host_runtime_tensor::create(typecode_t::dt_uint8, out_shape, hrt::pool_shared);
        if (!in.is_ok() || !out.is_ok()) {
            why = "cannot create AI2D tensors";
            return false;
        }
        in_ = in.unwrap();
        out_ = out.unwrap();
        fill_tensor(in_, gsl::byte{0x80});
        /* Letterbox as the YOLO demos do: scale by 1/3 to 640x360, pad below. */
        const int32_t pad_bottom = (int32_t)out_size - (int32_t)(in_h * out_size / in_w);
        ai2d_datatype_t dtype{ai2d_format::NCHW_FMT, ai2d_format::NCHW_FMT,
            typecode_t::dt_uint8, typecode_t::dt_uint8};
        ai2d_crop_param_t crop{false, 0, 0, 0, 0};
        ai2d_shift_param_t shift{false, 0};
        ai2d_pad_param_t pad{true, {{0, 0}, {0, 0}, {0, pad_bottom}, {0, 0}},
            ai2d_pad_mode::constant, {114, 114, 114}};
        ai2d_resize_param_t resize{true, ai2d_interp_method::tf_bilinear, ai2d_interp_mode::half_pixel};
        ai2d_affine_param_t affine{false, ai2d_interp_method::cv2_bilinear, 0, 0, 127, 1,
            {0.5, 0.1, 0.0, 0.1, 0.5, 0.0}};
        builder_.reset(new ai2d_builder(in_shape, out_shape, dtype, crop, shift, pad, resize, affine));
        auto built = builder_->build_schedule();
        if (!built.is_ok()) {
            why = "ai2d build_schedule failed: " + built.unwrap_err().message();
            return false;
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%ux%u RGB planar -> %ux%u pad-resize", in_w, in_h, out_size, out_size);
        detail = buf;
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        while (!stop.load(std::memory_order_relaxed)) {
            unsigned d = duty.load(std::memory_order_relaxed);
            if (!d) {
                sleep_us(100000);
                continue;
            }
            uint64_t t0 = now_us();
            auto result = builder_->invoke(in_, out_);
            uint64_t latency = now_us() - t0;
            if (!result.is_ok()) {
                counters.errors.fetch_add(1, std::memory_order_relaxed);
                sleep_us(100000);
                continue;
            }
            counters.add(latency);
            if (d < 100)
                sleep_us(latency * (100 - d) / d);
        }
    }

    void teardown() override { builder_.reset(); }

private:
    static constexpr unsigned in_w = 1920, in_h = 1080, out_size = 640;
    runtime_tensor in_, out_;
    std::unique_ptr<ai2d_builder> builder_;
};

} // namespace

stress_module *make_kpu_module()
{
    return new kpu_module();
}

stress_module *make_ai2d_module()
{
    return new ai2d_module();
}
