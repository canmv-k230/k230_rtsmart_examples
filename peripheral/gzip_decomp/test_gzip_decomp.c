#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "drv_gzip_decomp.h"
#include "mpi_gsdma_api.h"
#include "mpi_sys_api.h"
#include "soft_gzip.h"

#define DMA_NODE_SIZE (128U * 1024U)
#define GUARD_SIZE 64U
#define MULTI_FIXTURE_SIZE 393339U
#define MAX_FIXTURE_SIZE (16U * 1024U * 1024U)

static const uint8_t small_expected[] = "K230 gzip decompression small fixture\n";

/* soft_gzip settings for streams generated on the board */
static const soft_gzip_opt_t gen_opt = { .window = 32768, .max_match = 258, .block_tokens = 16384 };

/* Monotonic: NTP may set the wall clock while a test runs. */
static long get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000L + ts.tv_nsec / 1000;
}

static uint8_t *read_file_buf(const char *path, long *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "Error: empty or invalid file: %s\n", path);
        fclose(fp);
        return NULL;
    }
    uint8_t *buf = malloc(sz);
    if (!buf) {
        fprintf(stderr, "Error: failed to allocate %ld bytes for %s\n", sz, path);
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, sz, fp) != (size_t)sz) {
        fprintf(stderr, "Error: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *out_size = sz;
    return buf;
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <input.gz> <original_file> [output_file]\n", prog);
    fprintf(stderr, "       %s --selftest [fixture_directory]\n", prog);
    fprintf(stderr, "       %s --sdma-contention [rounds] [gzip_KiB] [copy_KiB] [min_ms] [max_ms]\n", prog);
    fprintf(stderr, "  input.gz      - gzip compressed input file\n");
    fprintf(stderr, "  original_file - original uncompressed file for verification\n");
    fprintf(stderr, "  output_file   - optional file to write decompressed data\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "--selftest runs the fixture streams, then streams generated on the board\n");
    fprintf(stderr, "by a small dynamic-Huffman encoder (soft_gzip.c), then decompression while\n");
    fprintf(stderr, "ordinary SDMA copies keep the channels gzip borrows busy. It needs about\n");
    fprintf(stderr, "115 MB of MMZ (96 MB for three SDMA load threads) and 20 MB of heap.\n");
    fprintf(stderr, "--sdma-contention needs 8 x copy_KiB plus about 2 x gzip_KiB of MMZ.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Note: the K230 engine only supports dynamic-Huffman streams.\n");
    fprintf(stderr, "Create a marked, hardware-compatible stream on the host with:\n");
    fprintf(stderr, "  python3 tools/image_tools/k230_priv_gzip.py compress <file>\\\n");
    fprintf(stderr, "      -o <file>.k230.gz -l 9\n");
}

static int check_fill(const uint8_t *buf, size_t size, uint8_t value)
{
    size_t i;

    for (i = 0; i < size; ++i) {
        if (buf[i] != value)
            return -1;
    }
    return 0;
}

static int fixture_path(char *path, size_t path_size, const char *dir, const char *name)
{
    int len = snprintf(path, path_size, "%s/%s", dir, name);

    return (len < 0 || (size_t)len >= path_size) ? -1 : 0;
}

static int run_valid_case(drv_gzip_decomp_inst_t *inst, const char *name,
                          const char *path, uint32_t expected_size,
                          const uint8_t *expected)
{
    uint8_t *input = NULL;
    uint8_t *storage = NULL;
    uint8_t *output = NULL;
    long input_size = 0;
    int ret;
    int failed = 0;

    input = read_file_buf(path, &input_size);
    if (!input)
        return -1;

    if (expected_size > 0) {
        storage = malloc((size_t)expected_size + 2 * GUARD_SIZE);
        if (!storage) {
            free(input);
            return -1;
        }
        memset(storage, 0xa5, (size_t)expected_size + 2 * GUARD_SIZE);
        output = storage + GUARD_SIZE;
    }

    ret = drv_gzip_decomp_gunzip(inst, input, (uint32_t)input_size,
                                 output, expected_size, 5000);
    if (ret != 0) {
        printf("[FAIL] %-12s decompress ret=%d\n", name, ret);
        failed = 1;
    } else if (expected_size > 0) {
        if (expected) {
            if (memcmp(output, expected, expected_size) != 0) {
                printf("[FAIL] %-12s payload mismatch\n", name);
                failed = 1;
            }
        } else if (check_fill(output, expected_size, 0) != 0) {
            printf("[FAIL] %-12s expected all-zero payload\n", name);
            failed = 1;
        }
    }

    if (storage &&
        (check_fill(storage, GUARD_SIZE, 0xa5) != 0 ||
         check_fill(output + expected_size, GUARD_SIZE, 0xa5) != 0)) {
        printf("[FAIL] %-12s output guard overwritten\n", name);
        failed = 1;
    }

    if (!failed)
        printf("[PASS] %-12s input=%ld output=%u\n", name, input_size, expected_size);

    free(storage);
    free(input);
    return failed ? -1 : 0;
}

/* A corrupt stream may fail in the engine or never finish. Use zero when
 * there is no alternate error. */
static int run_invalid_case(drv_gzip_decomp_inst_t *inst, const char *name,
                            const char *path, int corrupt, uint32_t trim,
                            int expected, int alternate)
{
    uint8_t *input;
    uint8_t output[sizeof(small_expected) - 1];
    long input_size = 0;
    uint32_t submit_size;
    int ret;

    input = read_file_buf(path, &input_size);
    if (!input)
        return -1;

    if (corrupt && input_size > 20)
        input[10 + (input_size - 18) / 2] ^= 0x5a;

    submit_size = (uint32_t)input_size;
    if (trim >= submit_size)
        submit_size = 0;
    else
        submit_size -= trim;

    memset(output, 0xa5, sizeof(output));
    ret = drv_gzip_decomp_gunzip(inst, input, submit_size, output,
                                 sizeof(output), 1000);
    free(input);

    if (ret >= 0) {
        printf("[FAIL] %-12s invalid stream accepted\n", name);
        return -1;
    }
    if (ret != expected && ret != alternate) {
        printf("[FAIL] %-12s ret=%d, expected %d", name, ret, expected);
        if (alternate)
            printf(" or %d", alternate);
        printf("\n");
        return -1;
    }
    if (check_fill(output, sizeof(output), 0xa5) != 0) {
        printf("[FAIL] %-12s copied failed output to caller\n", name);
        return -1;
    }

    printf("[PASS] %-12s rejected ret=%d, caller output unchanged\n", name, ret);
    return 0;
}

static int run_undersized_ioctl_case(drv_gzip_decomp_inst_t *inst,
                                     const char *path)
{
    uint8_t *input = NULL;
    void *src_virt = NULL;
    void *dst_virt = NULL;
    uint64_t src_phys = 0;
    uint64_t dst_phys = 0;
    long input_size = 0;
    int ret;
    int failed = 0;

    input = read_file_buf(path, &input_size);
    if (!input)
        return -1;

    ret = kd_mpi_sys_mmz_alloc_cached(&src_phys, &src_virt, "gzip_tsrc",
                                      "anonymous", DMA_NODE_SIZE);
    if (ret != 0) {
        printf("[FAIL] undersized   source MMZ allocation ret=%d\n", ret);
        failed = 1;
        goto out;
    }

    ret = kd_mpi_sys_mmz_alloc_cached(&dst_phys, &dst_virt, "gzip_tdst",
                                      "anonymous", DMA_NODE_SIZE);
    if (ret != 0) {
        printf("[FAIL] undersized   destination MMZ allocation ret=%d\n", ret);
        failed = 1;
        goto out;
    }

    memset(src_virt, 0, DMA_NODE_SIZE);
    memcpy(src_virt, input, (size_t)input_size);
    ((uint8_t *)src_virt)[2] = 0x08;
    memset(dst_virt, 0xa5, DMA_NODE_SIZE);
    kd_mpi_sys_mmz_flush_cache(src_phys, src_virt, DMA_NODE_SIZE);
    kd_mpi_sys_mmz_flush_cache(dst_phys, dst_virt, DMA_NODE_SIZE);

    ret = drv_gzip_decomp_gunzip_phys(inst, src_phys, (uint32_t)input_size,
                                      dst_phys, sizeof(small_expected) - 2, 1000);
    kd_mpi_sys_mmz_invalidate_cache(dst_phys, dst_virt, DMA_NODE_SIZE);

    if (ret >= 0) {
        printf("[FAIL] undersized   raw ioctl accepted short destination\n");
        failed = 1;
    } else if (ret != -EINVAL) {
        printf("[FAIL] undersized   ret=%d, expected %d\n", ret, -EINVAL);
        failed = 1;
    } else if (check_fill(dst_virt, DMA_NODE_SIZE, 0xa5) != 0) {
        printf("[FAIL] undersized   raw ioctl modified destination before rejection\n");
        failed = 1;
    } else {
        printf("[PASS] undersized   raw ioctl rejected ret=%d, destination unchanged\n", ret);
    }

out:
    if (dst_virt)
        kd_mpi_sys_mmz_free(dst_phys, dst_virt);
    if (src_virt)
        kd_mpi_sys_mmz_free(src_phys, src_virt);
    free(input);
    return failed ? -1 : 0;
}

/* Payloads generated on the board and compressed with soft_gzip, so sizes and
 * content are not limited to the fixtures built on the host. */
enum { PATTERN_TEXT, PATTERN_NOISE, PATTERN_RUNS };

static void fill_pattern(uint8_t *buf, uint32_t len, uint32_t seed, int kind)
{
    static const char words[] = "k230 gzip engine dma line sram node crc huffman ";
    uint32_t x = seed * 2654435761u + 1;

    for (uint32_t i = 0; i < len; ++i) {
        x = x * 1103515245u + 12345u;
        switch (kind) {
        case PATTERN_TEXT:
            buf[i] = (x >> 24) < 8 ? (uint8_t)(x >> 16) : (uint8_t)words[(i + (x >> 28)) % (sizeof(words) - 1)];
            break;
        case PATTERN_NOISE:
            buf[i] = (uint8_t)(x >> 16);
            break;
        default:
            buf[i] = (uint8_t)((i / (1 + (seed % 7) * 97)) * 31);
            break;
        }
    }
}

static int run_soft_case(drv_gzip_decomp_inst_t *inst, const char *name, uint32_t len, int kind)
{
    uint8_t *payload = malloc(len);
    uint8_t *storage = malloc((size_t)len + 2 * GUARD_SIZE);
    uint8_t *gz = NULL;
    uint8_t *output;
    uint32_t gz_len = 0;
    int ret, failed = 0;

    if (!payload || !storage) {
        printf("[FAIL] %-12s out of memory\n", name);
        failed = 1;
        goto out;
    }
    fill_pattern(payload, len, len, kind);
    gz = soft_gzip_compress(payload, len, &gen_opt, &gz_len);
    if (!gz) {
        printf("[FAIL] %-12s soft_gzip_compress failed\n", name);
        failed = 1;
        goto out;
    }

    memset(storage, 0xa5, (size_t)len + 2 * GUARD_SIZE);
    output = storage + GUARD_SIZE;
    ret = drv_gzip_decomp_gunzip(inst, gz, gz_len, output, len, 5000);
    if (ret != 0) {
        printf("[FAIL] %-12s decompress ret=%d\n", name, ret);
        failed = 1;
    } else if (memcmp(output, payload, len) != 0) {
        printf("[FAIL] %-12s payload mismatch\n", name);
        failed = 1;
    } else if (check_fill(storage, GUARD_SIZE, 0xa5) != 0 ||
               check_fill(output + len, GUARD_SIZE, 0xa5) != 0) {
        printf("[FAIL] %-12s output guard overwritten\n", name);
        failed = 1;
    } else {
        printf("[PASS] %-12s soft gzip %u -> %u bytes, hardware output matches\n", name, len, gz_len);
    }

out:
    free(gz);
    free(storage);
    free(payload);
    return failed ? -1 : 0;
}

/* The kernel must take only buffers that are allocated while it uses them:
 * a destination that was freed is refused, not written. */
static int run_freed_dst_case(drv_gzip_decomp_inst_t *inst, const char *path)
{
    uint8_t *input = NULL;
    void *src_virt = NULL, *dst_virt = NULL;
    uint64_t src_phys = 0, dst_phys = 0;
    long input_size = 0;
    int ret, failed = 0;

    input = read_file_buf(path, &input_size);
    if (!input)
        return -1;
    if (kd_mpi_sys_mmz_alloc_cached(&src_phys, &src_virt, "gzip_fsrc", "anonymous", DMA_NODE_SIZE) != 0 ||
        kd_mpi_sys_mmz_alloc_cached(&dst_phys, &dst_virt, "gzip_fdst", "anonymous", DMA_NODE_SIZE) != 0) {
        printf("[FAIL] freed-dst    MMZ allocation\n");
        failed = 1;
        goto out;
    }
    memset(src_virt, 0, DMA_NODE_SIZE);
    memcpy(src_virt, input, (size_t)input_size);
    ((uint8_t *)src_virt)[2] = 0x08;
    kd_mpi_sys_mmz_flush_cache(src_phys, src_virt, DMA_NODE_SIZE);
    kd_mpi_sys_mmz_free(dst_phys, dst_virt);
    dst_virt = NULL;

    ret = drv_gzip_decomp_gunzip_phys(inst, src_phys, (uint32_t)input_size, dst_phys,
                                      sizeof(small_expected) - 1, 1000);
    if (ret >= 0) {
        printf("[FAIL] freed-dst    kernel decompressed into a freed buffer\n");
        failed = 1;
    } else if (ret != -EINVAL) {
        printf("[FAIL] freed-dst    ret=%d, expected %d\n", ret, -EINVAL);
        failed = 1;
    } else {
        printf("[PASS] freed-dst    rejected ret=%d\n", ret);
    }

out:
    if (dst_virt)
        kd_mpi_sys_mmz_free(dst_phys, dst_virt);
    if (src_virt)
        kd_mpi_sys_mmz_free(src_phys, src_virt);
    free(input);
    return failed ? -1 : 0;
}

/* ---- Streams generated on the board ---- */

enum gen_pattern { GEN_ZERO, GEN_RANDOM, GEN_TEXT, GEN_RUNS, GEN_MIX };

static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void gen_data(uint8_t *buf, uint32_t len, enum gen_pattern pattern, uint32_t seed)
{
    static const char *const words[] = { "kendryte ", "k230 ", "gzip ", "deflate ", "huffman ",
                                         "window ", "sram ", "sdma ", "canmv ", "\n" };
    uint32_t state = seed ? seed : 1;
    uint32_t i = 0;

    while (i < len) {
        enum gen_pattern p = pattern;
        uint32_t n;

        if (p == GEN_MIX)
            p = (enum gen_pattern)(xorshift32(&state) % GEN_MIX);
        n = 1 + xorshift32(&state) % 4096;
        if (n > len - i)
            n = len - i;
        switch (p) {
        case GEN_ZERO:
            memset(buf + i, 0, n);
            break;
        case GEN_RANDOM:
            for (uint32_t k = 0; k < n; k++)
                buf[i + k] = (uint8_t)xorshift32(&state);
            break;
        case GEN_TEXT:
            for (uint32_t k = 0; k < n;) {
                const char *w = words[xorshift32(&state) % 10];

                while (*w && k < n)
                    buf[i + k++] = (uint8_t)*w++;
            }
            break;
        case GEN_RUNS:
        default:
            /* Byte runs and copies from far back, up to the 32 KB window. */
            if (i > 0 && (xorshift32(&state) & 1)) {
                uint32_t back = 1 + xorshift32(&state) % (i < 32768 ? i : 32768);

                for (uint32_t k = 0; k < n; k++)
                    buf[i + k] = buf[i + k - back];
            } else {
                memset(buf + i, (uint8_t)xorshift32(&state), n);
            }
            break;
        }
        i += n;
    }
}

/* Decompress input and compare with expected, with guard bytes around the
 * output to catch writes past its end. */
static int check_gunzip(drv_gzip_decomp_inst_t *inst, const char *name, const uint8_t *input,
                        uint32_t input_len, const uint8_t *expected, uint32_t expected_size)
{
    uint8_t *storage = malloc((size_t)expected_size + 2 * GUARD_SIZE);
    uint8_t *output;
    int ret, failed = 0;

    if (!storage) {
        printf("[FAIL] %-12s out of memory\n", name);
        return -1;
    }
    memset(storage, 0xa5, (size_t)expected_size + 2 * GUARD_SIZE);
    output = storage + GUARD_SIZE;

    ret = drv_gzip_decomp_gunzip(inst, input, input_len, output, expected_size, 5000);
    if (ret != 0) {
        printf("[FAIL] %-12s decompress ret=%d\n", name, ret);
        failed = 1;
    } else if (memcmp(output, expected, expected_size) != 0) {
        printf("[FAIL] %-12s payload mismatch\n", name);
        failed = 1;
    } else if (check_fill(storage, GUARD_SIZE, 0xa5) != 0 ||
               check_fill(output + expected_size, GUARD_SIZE, 0xa5) != 0) {
        printf("[FAIL] %-12s output guard overwritten\n", name);
        failed = 1;
    }
    free(storage);
    return failed ? -1 : 0;
}

static uint8_t *gen_stream(uint32_t len, enum gen_pattern pattern, uint32_t seed,
                           const soft_gzip_opt_t *opt, uint8_t **plain, uint32_t *stream_len)
{
    uint8_t *data = malloc(len);
    uint8_t *stream;

    if (!data)
        return NULL;
    gen_data(data, len, pattern, seed);
    stream = soft_gzip_compress(data, len, opt, stream_len);
    if (!stream) {
        free(data);
        return NULL;
    }
    *plain = data;
    return stream;
}

/* The first deflate block (byte 10, right after soft_gzip's plain header)
 * gets the reserved block type 3. The engine drops back to IDLE within
 * microseconds, before the driver first looks, and the driver must still
 * report EBADMSG well before the timeout instead of waiting it out. The intact
 * stream runs first: it must decode, and the bad one then follows a good run
 * with no engine reset in between. */
static int run_bad_block_case(drv_gzip_decomp_inst_t *inst)
{
    uint32_t stream_len = 0, len = 64 * 1024;
    uint8_t *plain = NULL, *output = malloc(len), *stream;
    int ret, failed = 0;
    long t;

    stream = gen_stream(len, GEN_MIX, 7, &gen_opt, &plain, &stream_len);
    if (!stream || !output) {
        printf("[FAIL] %-12s setup failed\n", "bad-block");
        free(stream);
        free(plain);
        free(output);
        return -1;
    }
    ret = drv_gzip_decomp_gunzip(inst, stream, stream_len, output, len, 1000);
    if (ret != 0 || memcmp(output, plain, len) != 0) {
        printf("[FAIL] %-12s intact stream ret=%d\n", "bad-block", ret);
        failed = 1;
        goto out;
    }
    stream[10] |= 0x06; /* BTYPE 11 */

    t = get_time_us();
    ret = drv_gzip_decomp_gunzip(inst, stream, stream_len, output, len, 1000);
    t = get_time_us() - t;
    if (ret != -EBADMSG || t >= 500 * 1000) {
        printf("[FAIL] %-12s ret=%d after %ld ms, expected %d well before the 1000 ms timeout\n", "bad-block",
               ret, t / 1000, -EBADMSG);
        failed = 1;
    } else {
        printf("[PASS] %-12s rejected ret=%d in %ld ms\n", "bad-block", ret, t / 1000);
    }
out:
    free(stream);
    free(plain);
    free(output);
    return failed ? -1 : 0;
}

static int run_generated_case(drv_gzip_decomp_inst_t *inst, const char *name, uint32_t len,
                              enum gen_pattern pattern, const soft_gzip_opt_t *opt, uint32_t seed)
{
    uint8_t *plain = NULL, *stream;
    uint32_t stream_len = 0;
    int ret;

    stream = gen_stream(len, pattern, seed, opt, &plain, &stream_len);
    if (!stream) {
        printf("[FAIL] %-12s compressing %u bytes failed\n", name, len);
        return -1;
    }
    ret = check_gunzip(inst, name, stream, stream_len, plain, len);
    if (ret == 0)
        printf("[PASS] %-12s input=%u output=%u\n", name, stream_len, len);
    free(stream);
    free(plain);
    return ret;
}

static int run_generated_cases(drv_gzip_decomp_inst_t *inst, uint32_t seed)
{
    static const soft_gzip_opt_t small_window = { .window = 4096, .max_match = 258, .block_tokens = 16384 };
    static const soft_gzip_opt_t short_blocks = { .window = 32768, .max_match = 258, .block_tokens = 300 };
    static const struct {
        const char *name;
        uint32_t len;
        enum gen_pattern pattern;
        const soft_gzip_opt_t *opt;
    } cases[] = {
        { "gen-1B",      1,                     GEN_RANDOM, &gen_opt },
        { "gen-text",    4095,                  GEN_TEXT,   &gen_opt },
        { "gen-random",  65536,                 GEN_RANDOM, &gen_opt },
        { "gen-node-1",  DMA_NODE_SIZE - 1,     GEN_MIX,    &gen_opt },
        { "gen-node",    DMA_NODE_SIZE,         GEN_MIX,    &gen_opt },
        { "gen-node+1",  DMA_NODE_SIZE + 1,     GEN_MIX,    &gen_opt },
        { "gen-runs",    1024 * 1024,           GEN_RUNS,   &gen_opt },
        { "gen-zeros",   2 * 1024 * 1024,       GEN_ZERO,   &gen_opt },
        { "gen-window",  256 * 1024,            GEN_MIX,    &small_window },
        { "gen-blocks",  256 * 1024,            GEN_MIX,    &short_blocks },
        { "gen-mix-4M",  4 * 1024 * 1024,       GEN_MIX,    &gen_opt },
    };
    int failures = 0;

    printf("generated streams, seed %u\n", seed);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        failures += run_generated_case(inst, cases[i].name, cases[i].len, cases[i].pattern,
                                       cases[i].opt, seed + (uint32_t)i) != 0;
    return failures;
}

/* ---- Decompression while ordinary SDMA transfers run ---- */

#define LOAD_THREADS   3
#define LOAD_COPY_SIZE (16U * 1024U * 1024U)
#define CONTENTION_RUNS 10

struct sdma_load {
    pthread_t thread;
    atomic_int *stop;
    unsigned copies;
    unsigned failed;    /* copies that failed */
    int alloc_failed;   /* no MMZ for its buffers: never copied */
    int started;
};

/* Back-to-back SDMA copies keep the SDMA channels gzip borrows (0 and 1)
 * busy, so every decompression has to wait for them. */
static void *sdma_load_thread(void *arg)
{
    struct sdma_load *load = arg;
    k_sdma_memcpy_t cfg;
    uint64_t phys;
    void *virt;

    if (kd_mpi_sys_mmz_alloc(&phys, &virt, "gzip_load", "anonymous", LOAD_COPY_SIZE * 2) != 0) {
        load->alloc_failed = 1;
        return NULL;
    }
    while (!atomic_load(load->stop)) {
        memset(&cfg, 0, sizeof(cfg));
        cfg.src_phys_addr = phys;
        cfg.dst_phys_addr = phys + LOAD_COPY_SIZE;
        cfg.size = LOAD_COPY_SIZE;
        cfg.timeout_ms = 2000;
        if (kd_mpi_gsdma_sdma_memcpy(&cfg) == 0)
            load->copies++;
        else
            load->failed++;
    }
    kd_mpi_sys_mmz_free(phys, virt);
    return NULL;
}

static int run_contention_case(drv_gzip_decomp_inst_t *inst, uint32_t seed)
{
    struct sdma_load load[LOAD_THREADS];
    atomic_int stop = 0;
    uint8_t *plain = NULL, *stream;
    uint32_t stream_len = 0, len = 512 * 1024;
    unsigned copies = 0, load_failed = 0, not_started = 0, no_mmz = 0;
    long worst = 0;
    int failed = 0, i;

    stream = gen_stream(len, GEN_MIX, seed, &gen_opt, &plain, &stream_len);
    if (!stream || kd_mpi_gsdma_init() != 0) {
        printf("[FAIL] contention  setup failed\n");
        free(stream);
        free(plain);
        return -1;
    }

    memset(load, 0, sizeof(load));
    for (i = 0; i < LOAD_THREADS; i++) {
        load[i].stop = &stop;
        load[i].started = pthread_create(&load[i].thread, NULL, sdma_load_thread, &load[i]) == 0;
    }
    usleep(100 * 1000);

    for (i = 0; i < CONTENTION_RUNS && !failed; i++) {
        long t = get_time_us();

        failed = check_gunzip(inst, "contention", stream, stream_len, plain, len) != 0;
        t = get_time_us() - t;
        if (t > worst)
            worst = t;
    }

    atomic_store(&stop, 1);
    for (i = 0; i < LOAD_THREADS; i++) {
        if (load[i].started)
            pthread_join(load[i].thread, NULL);
        else
            not_started++;
        copies += load[i].copies;
        load_failed += load[i].failed;
        no_mmz += load[i].alloc_failed;
    }
    kd_mpi_gsdma_deinit();

    if (!failed && (not_started || no_mmz || load_failed || !copies)) {
        printf("[FAIL] contention  SDMA load: %u copies ok, %u failed; %u of %d threads not started, "
               "%u without their %u MiB of MMZ\n", copies, load_failed, not_started, LOAD_THREADS, no_mmz,
               2 * LOAD_COPY_SIZE >> 20);
        failed = 1;
    }
    if (!failed)
        printf("[PASS] contention  %d runs under %u SDMA copies, worst %ld ms\n",
               CONTENTION_RUNS, copies, worst / 1000);
    free(stream);
    free(plain);
    return failed ? -1 : 0;
}

static int run_selftest(const char *fixture_dir)
{
    drv_gzip_decomp_inst_t *inst = NULL;
    char empty_path[512];
    char small_path[512];
    char multi_path[512];
    char max_path[512];
    int failures = 0;

    if (fixture_path(empty_path, sizeof(empty_path), fixture_dir, "empty.k230.gz") ||
        fixture_path(small_path, sizeof(small_path), fixture_dir, "small.k230.gz") ||
        fixture_path(multi_path, sizeof(multi_path), fixture_dir, "multi.k230.gz") ||
        fixture_path(max_path, sizeof(max_path), fixture_dir, "max.k230.gz")) {
        fprintf(stderr, "fixture directory path is too long\n");
        return 1;
    }

    if (drv_gzip_decomp_open(&inst) != 0) {
        fprintf(stderr, "[FAIL] open gzip decompressor\n");
        return 1;
    }

    failures += run_valid_case(inst, "empty", empty_path, 0, NULL) != 0;
    failures += run_valid_case(inst, "small", small_path,
                               sizeof(small_expected) - 1, small_expected) != 0;
    failures += run_valid_case(inst, "multi-node", multi_path,
                               MULTI_FIXTURE_SIZE, NULL) != 0;
    failures += run_valid_case(inst, "max-16MiB", max_path,
                               MAX_FIXTURE_SIZE, NULL) != 0;
    failures += run_invalid_case(inst, "corrupt", small_path, 1, 0, -EBADMSG, 0) != 0;
    failures += run_bad_block_case(inst) != 0;
    failures += run_invalid_case(inst, "truncated", small_path, 0, 5, -EINVAL, 0) != 0;
    failures += run_undersized_ioctl_case(inst, small_path) != 0;
    failures += run_freed_dst_case(inst, small_path) != 0;
    failures += run_soft_case(inst, "soft-1B", 1, PATTERN_TEXT) != 0;
    failures += run_soft_case(inst, "soft-node-1", DMA_NODE_SIZE - 1, PATTERN_TEXT) != 0;
    failures += run_soft_case(inst, "soft-node", DMA_NODE_SIZE, PATTERN_RUNS) != 0;
    failures += run_soft_case(inst, "soft-node+1", DMA_NODE_SIZE + 1, PATTERN_TEXT) != 0;
    failures += run_soft_case(inst, "soft-3node", 3 * DMA_NODE_SIZE + 17, PATTERN_NOISE) != 0;
    failures += run_soft_case(inst, "soft-1MiB", 1024 * 1024, PATTERN_TEXT) != 0;
    failures += run_soft_case(inst, "soft-4MiB", 4 * 1024 * 1024, PATTERN_RUNS) != 0;
    failures += run_valid_case(inst, "recovery", small_path,
                               sizeof(small_expected) - 1, small_expected) != 0;
    failures += run_generated_cases(inst, 1);
    failures += run_contention_case(inst, 100);

    drv_gzip_decomp_close(&inst);
    printf("D-01 selftest: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

/* --sdma-contention: gzip holds SDMA channels 0 and 1 while ordinary SDMA
 * copies queue behind channels 2 and 3. When gzip hands its channels back,
 * a queued copy starts on one of them at once. It must not run in gzip's
 * handshake mode (dec_en on channel 0), where it would never finish.
 *
 * gzip reserves its channels some time into the ioctl (after its debug log),
 * so the copies start after a delay that sweeps min_ms..max_ms. In rounds
 * where they come too early, gzip may time out reserving the channels and
 * return EBUSY: counted, not an error. A PASS needs rounds where gzip ran
 * while copies waited. */
#define CONT_WORKERS 4
#define CONT_SAMPLES 512
#define CONT_COPY_TIMEOUT_MS 2000
#define CONT_GZIP_TIMEOUT_MS 5000
#define CONT_MAX_FAILURES 3
#define CONT_MAX_NODE_LEN (UINT32_MAX - (DMA_NODE_SIZE - 1))
#define CONT_MAX_ROUNDS (UINT_MAX / CONT_WORKERS)
#define CONT_MAX_DELAY_MS (UINT_MAX / 1000U)

static struct {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    unsigned gz_go, copy_go, gz_done, copies_done;
    int stop;
} cont = { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0, 0, 0 };

typedef struct {
    uint32_t size;
    uint64_t src_phys, dst_phys;
    uint32_t *src, *dst; /* uncached */
    long min_us, max_us, waited_over_us;
    unsigned ok, fail, mismatch, waited;
    int first_ret;
} cont_copy_t;

typedef struct {
    drv_gzip_decomp_inst_t *inst;
    uint64_t src_phys, dst_phys;
    void *src, *dst; /* cached */
    uint32_t gz_len, len, src_alloc, dst_alloc;
    const uint8_t *expected;
    long max_us, total_us;
    unsigned ok, fail, mismatch, busy;
    int first_ret;
} cont_gzip_t;

/* Wait until *go moves past *seen; returns nonzero when told to stop. */
static int cont_wait(const unsigned *go, unsigned *seen)
{
    int stop;

    pthread_mutex_lock(&cont.lock);
    while (*go == *seen && !cont.stop)
        pthread_cond_wait(&cont.cond, &cont.lock);
    *seen = *go;
    stop = cont.stop;
    pthread_mutex_unlock(&cont.lock);
    return stop;
}

static void cont_finish(unsigned *done)
{
    pthread_mutex_lock(&cont.lock);
    (*done)++;
    pthread_cond_broadcast(&cont.cond);
    pthread_mutex_unlock(&cont.lock);
}

static uint32_t cont_sample(uint32_t i, uint32_t words)
{
    return (uint32_t)((uint64_t)i * (words - 1) / (CONT_SAMPLES - 1));
}

/* One SDMA copy; returns its latency, or -1 if it failed. */
static long cont_copy_once(cont_copy_t *w)
{
    uint32_t words = w->size / 4;
    k_sdma_memcpy_t cfg = {
        .src_phys_addr = w->src_phys,
        .dst_phys_addr = w->dst_phys,
        .size = w->size,
        .timeout_ms = CONT_COPY_TIMEOUT_MS,
    };
    long t0, dt;
    int ret;

    for (uint32_t i = 0; i < CONT_SAMPLES; ++i)
        w->dst[cont_sample(i, words)] = ~w->src[cont_sample(i, words)];
    t0 = get_time_us();
    ret = kd_mpi_gsdma_sdma_memcpy(&cfg);
    dt = get_time_us() - t0;
    if (dt > w->max_us)
        w->max_us = dt;
    if (!w->min_us || dt < w->min_us)
        w->min_us = dt;
    if (ret != 0) {
        if (!w->fail++)
            w->first_ret = ret;
        return -1;
    }
    for (uint32_t i = 0; i < CONT_SAMPLES; ++i) {
        uint32_t at = cont_sample(i, words);
        if (w->dst[at] != w->src[at]) {
            w->mismatch++;
            return -1;
        }
    }
    w->ok++;
    return dt;
}

static void *cont_copy_worker(void *arg)
{
    cont_copy_t *w = arg;
    unsigned seen = 0;

    while (!cont_wait(&cont.copy_go, &seen)) {
        if (cont_copy_once(w) > w->waited_over_us)
            w->waited++;
        cont_finish(&cont.copies_done);
    }
    return NULL;
}

static int cont_gzip_once(cont_gzip_t *g)
{
    long t0 = get_time_us(), dt;
    int ret = drv_gzip_decomp_gunzip_phys(g->inst, g->src_phys, g->gz_len, g->dst_phys,
                                          g->len, CONT_GZIP_TIMEOUT_MS);

    dt = get_time_us() - t0;
    g->total_us += dt;
    if (dt > g->max_us)
        g->max_us = dt;
    if (ret == -EBUSY) {
        g->busy++; /* could not reserve both channels */
    } else if (ret != 0) {
        if (!g->fail++)
            g->first_ret = ret;
    } else {
        kd_mpi_sys_mmz_invalidate_cache(g->dst_phys, g->dst, g->dst_alloc);
        if (memcmp(g->dst, g->expected, g->len) != 0) {
            g->mismatch++;
            ret = -1;
        } else {
            g->ok++;
        }
    }
    /* Clear the output for the next round now, not when the round starts. */
    memset(g->dst, 0, g->dst_alloc);
    kd_mpi_sys_mmz_flush_cache(g->dst_phys, g->dst, g->dst_alloc);
    return ret;
}

static void *cont_gzip_worker(void *arg)
{
    cont_gzip_t *g = arg;
    unsigned seen = 0;

    while (!cont_wait(&cont.gz_go, &seen)) {
        cont_gzip_once(g);
        cont_finish(&cont.gz_done);
    }
    return NULL;
}

static uint32_t round_up_node(uint32_t len)
{
    return (len + DMA_NODE_SIZE - 1) / DMA_NODE_SIZE * DMA_NODE_SIZE;
}

static int run_sdma_contention(unsigned rounds, uint32_t gz_kib, uint32_t copy_kib,
                               unsigned min_ms, unsigned max_ms)
{
    cont_copy_t copies[CONT_WORKERS];
    cont_gzip_t gz;
    pthread_t copy_threads[CONT_WORKERS], gz_thread;
    uint8_t *payload = NULL, *stream = NULL;
    long base_copy_us, base_gz_us;
    unsigned copy_ok = 0, copy_fail = 0, copy_mismatch = 0, copy_waited = 0;
    unsigned done_rounds = 0, threads = 0, overlap = 0;
    long copy_min = 0, copy_max = 0;
    int failed = 1, gz_started = 0, gsdma_up = 0;

    memset(copies, 0, sizeof(copies));
    memset(&gz, 0, sizeof(gz));
    gz.len = gz_kib * 1024U;

    if (kd_mpi_gsdma_init() != 0) {
        printf("[FAIL] contention  kd_mpi_gsdma_init\n");
        return 1;
    }
    gsdma_up = 1;

    payload = malloc(gz.len);
    if (!payload) {
        printf("[FAIL] contention  out of memory\n");
        goto out;
    }
    fill_pattern(payload, gz.len, gz.len, PATTERN_TEXT);
    stream = soft_gzip_compress(payload, gz.len, &gen_opt, &gz.gz_len);
    if (!stream) {
        printf("[FAIL] contention  soft_gzip_compress failed\n");
        goto out;
    }
    if (gz.gz_len > CONT_MAX_NODE_LEN) {
        printf("[FAIL] contention  compressed stream too large\n");
        goto out;
    }
    gz.expected = payload;
    gz.src_alloc = round_up_node(gz.gz_len);
    gz.dst_alloc = round_up_node(gz.len);
    if (kd_mpi_sys_mmz_alloc_cached(&gz.src_phys, &gz.src, "gzip_csrc", "anonymous", gz.src_alloc) != 0 ||
        kd_mpi_sys_mmz_alloc_cached(&gz.dst_phys, &gz.dst, "gzip_cdst", "anonymous", gz.dst_alloc) != 0) {
        printf("[FAIL] contention  gzip MMZ allocation\n");
        goto out;
    }
    memset(gz.src, 0, gz.src_alloc);
    memcpy(gz.src, stream, gz.gz_len);
    /* As drv_gzip_decomp_gunzip() does: the engine wants DEFLATE (0x08) in
     * place of the software marker 0x09. soft_gzip writes no header extras. */
    ((uint8_t *)gz.src)[2] = 0x08;
    kd_mpi_sys_mmz_flush_cache(gz.src_phys, gz.src, gz.src_alloc);
    memset(gz.dst, 0, gz.dst_alloc);
    kd_mpi_sys_mmz_flush_cache(gz.dst_phys, gz.dst, gz.dst_alloc);

    for (int i = 0; i < CONT_WORKERS; ++i) {
        cont_copy_t *w = &copies[i];
        void *src, *dst;

        w->size = copy_kib * 1024U;
        if (kd_mpi_sys_mmz_alloc(&w->src_phys, &src, "sdma_csrc", "anonymous", w->size) != 0 ||
            kd_mpi_sys_mmz_alloc(&w->dst_phys, &dst, "sdma_cdst", "anonymous", w->size) != 0) {
            printf("[FAIL] contention  copy %d MMZ allocation\n", i);
            goto out;
        }
        w->src = src;
        w->dst = dst;
        for (uint32_t j = 0; j < w->size / 4; ++j)
            w->src[j] = ((uint32_t)i << 28) ^ (j * 2654435761u);
    }

    if (drv_gzip_decomp_open(&gz.inst) != 0) {
        printf("[FAIL] contention  open gzip decompressor\n");
        goto out;
    }

    /* Each alone first, for scale: a copy that waited for a channel takes
     * clearly longer than one copy on an idle engine. */
    base_copy_us = cont_copy_once(&copies[0]);
    if (cont_gzip_once(&gz) != 0 || base_copy_us < 0) {
        printf("[FAIL] contention  baseline gzip ret=%d copy ret=%d\n", gz.first_ret, copies[0].first_ret);
        goto out;
    }
    base_gz_us = gz.max_us;
    for (int i = 0; i < CONT_WORKERS; ++i) {
        copies[i].ok = copies[i].fail = copies[i].mismatch = copies[i].waited = 0;
        copies[i].min_us = copies[i].max_us = 0;
        copies[i].waited_over_us = base_copy_us + base_copy_us / 2;
    }
    gz.ok = gz.fail = gz.mismatch = gz.busy = 0;
    gz.max_us = gz.total_us = 0;
    printf("contention: %u rounds, gzip %u -> %u bytes (%ld us alone), "
           "%d copies of %u KiB (%ld us alone), copies %u..%u ms after gzip\n",
           rounds, gz.gz_len, gz.len, base_gz_us, CONT_WORKERS, copy_kib, base_copy_us,
           min_ms, max_ms);

    if (pthread_create(&gz_thread, NULL, cont_gzip_worker, &gz) != 0) {
        printf("[FAIL] contention  pthread_create\n");
        goto out;
    }
    gz_started = 1;
    for (; threads < CONT_WORKERS; ++threads) {
        if (pthread_create(&copy_threads[threads], NULL, cont_copy_worker, &copies[threads]) != 0) {
            printf("[FAIL] contention  pthread_create\n");
            goto stop;
        }
    }

    /* Each round: gzip takes channels 0 and 1, then the copies arrive. Two
     * run on channels 2 and 3, the others queue until a channel frees. */
    for (; done_rounds < rounds; ++done_rounds) {
        unsigned errors = 0, gz_ok = gz.ok, waited_before = 0, waited_after = 0;

        for (int i = 0; i < CONT_WORKERS; ++i)
            waited_before += copies[i].waited;
        pthread_mutex_lock(&cont.lock);
        cont.gz_go++;
        pthread_cond_broadcast(&cont.cond);
        pthread_mutex_unlock(&cont.lock);
        usleep((min_ms + done_rounds % (max_ms - min_ms + 1)) * 1000U);
        pthread_mutex_lock(&cont.lock);
        cont.copy_go++;
        pthread_cond_broadcast(&cont.cond);
        while (cont.gz_done != done_rounds + 1 ||
               cont.copies_done != (done_rounds + 1) * CONT_WORKERS)
            pthread_cond_wait(&cont.cond, &cont.lock);
        pthread_mutex_unlock(&cont.lock);

        errors = gz.fail + gz.mismatch;
        for (int i = 0; i < CONT_WORKERS; ++i) {
            errors += copies[i].fail + copies[i].mismatch;
            waited_after += copies[i].waited;
        }
        if (gz.ok != gz_ok && waited_after != waited_before)
            overlap++;
        if (errors >= CONT_MAX_FAILURES) {
            printf("contention: stopping after %u errors in round %u\n", errors, done_rounds);
            ++done_rounds;
            break;
        }
    }

stop:
    pthread_mutex_lock(&cont.lock);
    cont.stop = 1;
    pthread_cond_broadcast(&cont.cond);
    pthread_mutex_unlock(&cont.lock);
    for (unsigned i = 0; i < threads; ++i)
        pthread_join(copy_threads[i], NULL);
    if (gz_started)
        pthread_join(gz_thread, NULL);
    if (threads < CONT_WORKERS)
        goto out;

    for (int i = 0; i < CONT_WORKERS; ++i) {
        copy_ok += copies[i].ok;
        copy_fail += copies[i].fail;
        copy_mismatch += copies[i].mismatch;
        copy_waited += copies[i].waited;
        if (copies[i].max_us > copy_max)
            copy_max = copies[i].max_us;
        if (copies[i].min_us && (!copy_min || copies[i].min_us < copy_min))
            copy_min = copies[i].min_us;
        if (copies[i].fail)
            printf("  copy %d: %u failed, first ret=0x%x\n", i, copies[i].fail, copies[i].first_ret);
    }
    printf("contention: gzip ok=%u busy=%u fail=%u mismatch=%u avg=%ld us max=%ld us\n",
           gz.ok, gz.busy, gz.fail, gz.mismatch,
           done_rounds ? gz.total_us / done_rounds : 0, gz.max_us);
    if (gz.fail)
        printf("  gzip first ret=%d\n", gz.first_ret);
    printf("contention: copies ok=%u fail=%u bad_copies=%u latency %ld..%ld us, "
           "%u waited for a channel (> %ld us)\n",
           copy_ok, copy_fail, copy_mismatch, copy_min, copy_max, copy_waited,
           copies[0].waited_over_us);
    printf("contention: %u rounds where gzip ran while copies waited\n", overlap);
    failed = gz.fail || gz.mismatch || copy_fail || copy_mismatch || !overlap ||
             done_rounds != rounds || copy_ok != rounds * CONT_WORKERS;

out:
    if (gz.inst)
        drv_gzip_decomp_close(&gz.inst);
    for (int i = 0; i < CONT_WORKERS; ++i) {
        if (copies[i].dst)
            kd_mpi_sys_mmz_free(copies[i].dst_phys, copies[i].dst);
        if (copies[i].src)
            kd_mpi_sys_mmz_free(copies[i].src_phys, copies[i].src);
    }
    if (gz.dst)
        kd_mpi_sys_mmz_free(gz.dst_phys, gz.dst);
    if (gz.src)
        kd_mpi_sys_mmz_free(gz.src_phys, gz.src);
    free(stream);
    free(payload);
    if (gsdma_up)
        kd_mpi_gsdma_deinit();
    printf("D-02 sdma-contention: %s (%u rounds)\n", failed ? "FAIL" : "PASS", done_rounds);
    return failed ? 1 : 0;
}

static int parse_contention_args(int argc, char *argv[])
{
    struct {
        const char *name;
        unsigned value, min, max;
    } args[] = {
        { "rounds",   200,   1, CONT_MAX_ROUNDS },
        { "gzip_KiB", 1024,  1, CONT_MAX_NODE_LEN / 1024U },
        { "copy_KiB", 16384, 4, UINT32_MAX / 1024U },
        { "min_ms",   6,     0, CONT_MAX_DELAY_MS },
        { "max_ms",   25,    0, CONT_MAX_DELAY_MS },
    };

    if (argc > 2 + (int)(sizeof(args) / sizeof(args[0])))
        goto bad_args;
    for (int i = 0; i < argc - 2; ++i) {
        const char *arg = argv[i + 2];
        char *end;

        errno = 0;
        unsigned long value = strtoul(arg, &end, 0);
        if (*arg < '0' || *arg > '9' || end == arg || *end || errno ||
            value < args[i].min || value > args[i].max) {
            fprintf(stderr, "invalid %s '%s' (expected %u..%u)\n",
                    args[i].name, arg, args[i].min, args[i].max);
            goto bad_args;
        }
        args[i].value = (unsigned)value;
    }
    if (args[3].value > args[4].value) {
        fprintf(stderr, "min_ms must not exceed max_ms\n");
        goto bad_args;
    }
    return run_sdma_contention(args[0].value, args[1].value, args[2].value,
                               args[3].value, args[4].value);

bad_args:
    usage(argv[0]);
    return 1;
}

int main(int argc, char *argv[])
{
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0)
        return run_selftest(argc >= 3 ? argv[2] : "./gzip_decomp_data");
    if (argc >= 2 && strcmp(argv[1], "--sdma-contention") == 0)
        return parse_contention_args(argc, argv);

    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *orig_path = argv[2];
    const char *output_path = (argc > 3) ? argv[3] : NULL;

    /* Read input gzip file */
    long input_size = 0;
    uint8_t *input_buf = read_file_buf(input_path, &input_size);
    if (!input_buf)
        return 1;

    /* Read original file for verification */
    long orig_size = 0;
    uint8_t *orig_buf = read_file_buf(orig_path, &orig_size);
    if (!orig_buf) {
        free(input_buf);
        return 1;
    }

    /* Verify gzip header */
    if (input_size < 18 || input_buf[0] != 0x1f || input_buf[1] != 0x8b) {
        fprintf(stderr, "Error: input is not a gzip file (bad magic: 0x%02x 0x%02x)\n",
                input_size >= 1 ? input_buf[0] : 0xff,
                input_size >= 2 ? input_buf[1] : 0xff);
        free(orig_buf);
        free(input_buf);
        return 1;
    }

    if (input_buf[2] != 0x09) {
        if (input_buf[2] == 0x08) {
            fprintf(stderr, "Error: unmarked gzip may contain stored or fixed-Huffman blocks,\n");
            fprintf(stderr, "       which the K230 hardware cannot decompress. Use K230PrivGzip.\n");
        } else {
            fprintf(stderr, "Error: unsupported gzip method 0x%02x\n", input_buf[2]);
        }
        free(orig_buf);
        free(input_buf);
        return 1;
    }

    printf("  K230 hardware gzip stream detected\n");

    /* Allocate output buffer sized to original file */
    uint32_t output_size = (uint32_t)orig_size;
    uint8_t *output_buf = malloc(output_size);
    if (!output_buf) {
        fprintf(stderr, "Error: failed to allocate %u bytes for output\n", output_size);
        free(orig_buf);
        free(input_buf);
        return 1;
    }
    memset(output_buf, 0, output_size);

    printf("Gzip Decompress Test\n");
    printf("  Input:    %s (%ld bytes)\n", input_path, input_size);
    printf("  Original: %s (%ld bytes)\n", orig_path, orig_size);

    /* Open driver instance */
    drv_gzip_decomp_inst_t *inst = NULL;
    if (drv_gzip_decomp_open(&inst) != 0) {
        fprintf(stderr, "Error: drv_gzip_decomp_open failed\n");
        free(output_buf);
        free(orig_buf);
        free(input_buf);
        return 1;
    }

    /* Decompress */
    long t0 = get_time_us();
    int ret = drv_gzip_decomp_gunzip(inst, input_buf, (uint32_t)input_size,
                                    output_buf, output_size, 5000);
    long t1 = get_time_us();

    if (ret < 0) {
        fprintf(stderr, "Error: drv_gzip_decomp_gunzip failed (%d)\n", ret);
    } else {
        long elapsed = t1 - t0;
        printf("  Result: OK\n");
        printf("  Time:   %ld us (%.3f ms)\n", elapsed, elapsed / 1000.0);
        if (elapsed > 0) {
            double speed = (double)input_size / (elapsed / 1000000.0) / (1024 * 1024);
            printf("  Speed:  %.2f MB/s (input throughput)\n", speed);
        }

        /* Verify against original file */
        if (memcmp(output_buf, orig_buf, orig_size) == 0) {
            printf("  Verify: PASS (%ld bytes match)\n", orig_size);
        } else {
            printf("  Verify: FAIL\n");
            long first_diff = -1;
            for (long i = 0; i < orig_size; i++) {
                if (output_buf[i] != orig_buf[i]) {
                    first_diff = i;
                    break;
                }
            }
            printf("  First mismatch at offset %ld: got 0x%02x, expected 0x%02x\n",
                   first_diff, output_buf[first_diff], orig_buf[first_diff]);
            ret = -1;
        }
    }

    /* Optionally write output file */
    if (output_path && ret >= 0) {
        FILE *ofp = fopen(output_path, "wb");
        if (ofp) {
            fwrite(output_buf, 1, output_size, ofp);
            fclose(ofp);
            printf("  Output written to %s\n", output_path);
        } else {
            perror("fopen output");
        }
    }

    drv_gzip_decomp_close(&inst);
    free(output_buf);
    free(orig_buf);
    free(input_buf);

    return (ret < 0) ? 1 : 0;
}
