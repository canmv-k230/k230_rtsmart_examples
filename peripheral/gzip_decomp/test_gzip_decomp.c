#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "drv_gzip_decomp.h"

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
    fprintf(stderr, "  input.gz      - gzip compressed input file\n");
    fprintf(stderr, "  original_file - original uncompressed file for verification\n");
    fprintf(stderr, "  output_file   - optional file to write decompressed data\n");
}

int main(int argc, char *argv[])
{
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
    if (input_size < 10 || input_buf[0] != 0x1f || input_buf[1] != 0x8b) {
        fprintf(stderr, "Error: input is not a gzip file (bad magic: 0x%02x 0x%02x)\n",
                input_size >= 1 ? input_buf[0] : 0xff,
                input_size >= 2 ? input_buf[1] : 0xff);
        free(orig_buf);
        free(input_buf);
        return 1;
    }

    if (input_buf[2] != 0x08 && input_buf[2] != 0x09) {
        fprintf(stderr, "Error: unsupported gzip method 0x%02x (expect 0x08 standard or 0x09 K230 private)\n",
                input_buf[2]);
        free(orig_buf);
        free(input_buf);
        return 1;
    }

    if (input_buf[2] == 0x08)
        printf("  Note: standard gzip detected, HAL will auto-patch to K230 private format\n");

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
