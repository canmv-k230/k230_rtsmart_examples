/* SPDX-License-Identifier: BSD-2-Clause */
/* CPU load: integer, floating-point and RVV memory-copy chunks, duty-cycled in
 * 100 ms windows. One unit is one chunk, so units/s tracks the achieved clock. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "hal_rvv_ops.h"
#include "stress.h"

namespace {

constexpr uint64_t window_us = 100000;
constexpr size_t mem_bytes = 4u << 20;  /* larger than the caches */
constexpr size_t mem_chunk = 64u << 10;

class cpu_module : public stress_module {
public:
    explicit cpu_module(unsigned index) : index_(index)
    {
        snprintf(name_, sizeof(name_), index ? "cpu%u" : "cpu", index);
    }
    ~cpu_module() override { teardown(); }
    const char *name() const override { return name_; }
    const char *unit() const override { return "chunk/s"; }
    /* Share time slices with KPU/AI2D feeders: a permanently runnable AI
     * feeder would otherwise starve the CPU load for the whole test. */
    int priority() const override { return 25; }

    bool setup(const stress_options &options, std::string &why, bool &skipped) override
    {
        skipped = false;
        workload_ = options.workload;
        duty.store(options.cpu_duty);
        if (workload_ == cpu_workload::mem || workload_ == cpu_workload::mix) {
            src_ = static_cast<uint8_t *>(malloc(mem_bytes));
            dst_ = static_cast<uint8_t *>(malloc(mem_bytes));
            if (!src_ || !dst_) {
                why = "out of memory for the copy buffers";
                return false;
            }
            memset(src_, 0x5a, mem_bytes);
        }
        return true;
    }

    void run(const std::atomic<bool> &stop) override
    {
        unsigned kind = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            uint64_t start = now_us();
            uint64_t busy = window_us * duty.load(std::memory_order_relaxed) / 100;
            while (now_us() - start < busy && !stop.load(std::memory_order_relaxed)) {
                uint64_t t0 = now_us();
                chunk(workload_ == cpu_workload::mix ? kind++ % 3 : static_cast<unsigned>(workload_));
                counters.add(now_us() - t0);
            }
            uint64_t spent = now_us() - start;
            if (spent < window_us)
                sleep_us(window_us - spent);
        }
    }

    void teardown() override
    {
        free(src_);
        free(dst_);
        src_ = dst_ = nullptr;
    }

private:
    void chunk(unsigned kind)
    {
        switch (kind) {
        case 0: {
            uint64_t a = acc_;
            for (unsigned i = 0; i < 20000; ++i)
                a = a * UINT64_C(6364136223846793005) + 1442695040888963407ULL;
            acc_ = a;
            break;
        }
        case 1: {
            double x0 = 1.0001, x1 = 0.9999, x2 = 1.0002, x3 = 0.9998;
            for (unsigned i = 0; i < 5000; ++i) {
                x0 = x0 * 0.999999 + 0.000001;
                x1 = x1 * 1.000001 - 0.000001;
                x2 = x2 * 0.999998 + 0.000002;
                x3 = x3 * 1.000002 - 0.000002;
            }
            fsink_ = fsink_ + x0 + x1 + x2 + x3;
            break;
        }
        default:
            hal_rvv_memcpy(dst_ + offset_, src_ + offset_, mem_chunk);
            offset_ = (offset_ + mem_chunk) % mem_bytes;
            break;
        }
    }

    unsigned index_;
    char name_[8];
    cpu_workload workload_ = cpu_workload::mix;
    uint8_t *src_ = nullptr, *dst_ = nullptr;
    size_t offset_ = 0;
    volatile uint64_t acc_ = 1;
    volatile double fsink_ = 0;
};

} // namespace

stress_module *make_cpu_module(unsigned index)
{
    return new cpu_module(index);
}
