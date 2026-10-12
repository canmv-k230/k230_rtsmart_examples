#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "drv_gzip_decomp.h"
#include "mpi_sys_api.h"

#define DMA_NODE_SIZE (128U * 1024U)
#define GUARD_SIZE 64U
#define MULTI_FIXTURE_SIZE 393339U
#define MAX_FIXTURE_SIZE (16U * 1024U * 1024U)

static const uint8_t small_expected[] = "K230 gzip decompression small fixture\n";

static long get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000L + tv.tv_usec;
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
    fprintf(stderr, "  input.gz      - gzip compressed input file\n");
    fprintf(stderr, "  original_file - original uncompressed file for verification\n");
    fprintf(stderr, "  output_file   - optional file to write decompressed data\n");
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

static int run_invalid_case(drv_gzip_decomp_inst_t *inst, const char *name,
                            const char *path, int corrupt, uint32_t trim)
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
    failures += run_invalid_case(inst, "corrupt", small_path, 1, 0) != 0;
    failures += run_invalid_case(inst, "truncated", small_path, 0, 5) != 0;
    failures += run_undersized_ioctl_case(inst, small_path) != 0;
    failures += run_valid_case(inst, "recovery", small_path,
                               sizeof(small_expected) - 1, small_expected) != 0;

    drv_gzip_decomp_close(&inst);
    printf("D-01 selftest: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

int main(int argc, char *argv[])
{
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0)
        return run_selftest(argc >= 3 ? argv[2] : "./gzip_decomp_data");

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
