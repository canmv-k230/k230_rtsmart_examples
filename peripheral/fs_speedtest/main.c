#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <mbedtls/sha256.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

#define TEST_TOTAL_BYTES (8ULL * 1024 * 1024)
#define MIN_BLOCK_COUNT  8U
#define FILL_VERIFY_BLOCK_SIZE (64U * 1024U)
#define SHA256_SIZE            32U
#define SOAK_DEFAULT_KIB       128U
#define SOAK_BLOCK_SIZE        (64U * 1024U)
#define SOAK_DELAY_US          500000U
#define SOAK_REPORT_US         (60ULL * 1000U * 1000U)

static volatile sig_atomic_t soak_stop;

/** cpu time *****************************************************************/
#define CPU_TICKS_PER_SECOND (27 * 1000 * 1000)

static __inline __attribute__((__always_inline__)) uint64_t utils_cpu_ticks(void)
{
    uint64_t tick;
    __asm__ __volatile__("rdtime %0" : "=r"(tick));
    return tick;
}

static __inline __attribute__((__always_inline__)) uint64_t utils_cpu_ticks_ms(void)
{
    uint64_t time;
    __asm__ __volatile__("rdtime %0" : "=r"(time));
    return (time / (CPU_TICKS_PER_SECOND / 1000));
}

static __inline __attribute__((__always_inline__)) uint64_t utils_cpu_ticks_us(void)
{
    uint64_t time;
    __asm__ __volatile__("rdtime %0" : "=r"(time));
    return (time / (CPU_TICKS_PER_SECOND / 1000000));
}

static __inline __attribute__((__always_inline__)) uint64_t utils_cpu_ticks_ns(void)
{
    uint64_t time;
    __asm__ __volatile__("rdtime %0" : "=r"(time));
    return (time * 1000000000ULL) / CPU_TICKS_PER_SECOND;
}

static const char* k_default_mount_points[] = {
    "/sdcard",
    "/data",
};

static const size_t k_block_sizes[] = {
    // 512,
    // 1024,
    // 4096,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
};

static uint32_t next_rand(uint32_t* state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static void format_size(size_t bytes, char* buffer, size_t buffer_len)
{
    if (bytes >= (1024 * 1024)) {
        snprintf(buffer, buffer_len, "%.2f MiB", (double)bytes / (1024.0 * 1024.0));
        return;
    }

    if (bytes >= 1024) {
        snprintf(buffer, buffer_len, "%.2f KiB", (double)bytes / 1024.0);
        return;
    }

    snprintf(buffer, buffer_len, "%zu B", bytes);
}

static int build_permutation(size_t* order, size_t count)
{
    uint32_t state = 0x12345678u;
    size_t   index;

    if (count == 0) {
        return -1;
    }

    for (index = 0; index < count; ++index) {
        order[index] = index;
    }

    for (index = count - 1; index > 0; --index) {
        size_t swap_index = (size_t)(next_rand(&state) % (index + 1));
        size_t tmp        = order[index];

        order[index]      = order[swap_index];
        order[swap_index] = tmp;
    }

    return 0;
}

static int write_full(int fd, const uint8_t* buffer, size_t size)
{
    size_t written = 0;

    while (written < size) {
        ssize_t ret = write(fd, buffer + written, size - written);

        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }

            return -1;
        }

        if (ret == 0) {
            errno = EIO;
            return -1;
        }

        written += (size_t)ret;
    }

    return 0;
}

static int read_full(int fd, uint8_t* buffer, size_t size)
{
    size_t received = 0;

    while (received < size) {
        ssize_t ret = read(fd, buffer + received, size - received);

        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }

            return -1;
        }

        if (ret == 0) {
            errno = EIO;
            return -1;
        }

        received += (size_t)ret;
    }

    return 0;
}

static uint64_t filesystem_free_bytes(const struct statvfs* fs)
{
    uint64_t fragment_size = fs->f_frsize ? fs->f_frsize : fs->f_bsize;

    return fragment_size * (uint64_t)fs->f_bavail;
}

static void fill_offset_pattern(uint8_t* buffer, size_t size, uint64_t offset)
{
    size_t index = 0;

    /* Mix the full word offset so that distant sectors do not repeat. */
    while (index < size) {
        uint64_t     position = offset + index;
        uint64_t     word = (position / 8U) + UINT64_C(0x9e3779b97f4a7c15);
        unsigned int byte = (unsigned int)(position % 8U);

        word = (word ^ (word >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        word = (word ^ (word >> 27)) * UINT64_C(0x94d049bb133111eb);
        word ^= word >> 31;
        for (; byte < 8U && index < size; ++byte, ++index) {
            buffer[index] = (uint8_t)(word >> (byte * 8U));
        }
    }
}

static void print_sha256(const uint8_t digest[SHA256_SIZE])
{
    size_t index;

    for (index = 0; index < SHA256_SIZE; ++index) {
        printf("%02x", digest[index]);
    }
}

static int hash_file(const char* path, uint8_t* buffer, size_t buffer_size, uint8_t digest[SHA256_SIZE])
{
    mbedtls_sha256_context context;
    int                    fd = -1;
    int                    result = -1;

    mbedtls_sha256_init(&context);
    if (mbedtls_sha256_starts(&context, 0) != 0) {
        goto cleanup;
    }

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        goto cleanup;
    }

    for (;;) {
        ssize_t received = read(fd, buffer, buffer_size);

        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            goto cleanup;
        }
        if (received == 0) {
            break;
        }
        if (mbedtls_sha256_update(&context, buffer, (size_t)received) != 0) {
            goto cleanup;
        }
    }

    if (mbedtls_sha256_finish(&context, digest) != 0) {
        goto cleanup;
    }
    result = 0;

cleanup:
    if (fd >= 0) {
        close(fd);
    }
    mbedtls_sha256_free(&context);
    return result;
}

static void soak_signal_handler(int sig)
{
    (void)sig;
    soak_stop = 1;
}

static int run_soak(const char* mount_point, unsigned int seconds, size_t file_size)
{
    uint8_t* write_buffer = NULL;
    uint8_t* read_buffer = NULL;
    uint64_t start_us = utils_cpu_ticks_us();
    uint64_t deadline_us = start_us + (uint64_t)seconds * 1000000U;
    uint64_t next_report_us = start_us + SOAK_REPORT_US;
    uint64_t total_bytes = 0;
    unsigned int cycles = 0;
    char path[256];
    int fd = -1;
    int result = 1;

    if (snprintf(path, sizeof(path), "%s/p10_sd_soak.bin", mount_point) >= (int)sizeof(path)) {
        fprintf(stderr, "Soak path is too long\n");
        return 1;
    }

    write_buffer = malloc(SOAK_BLOCK_SIZE);
    read_buffer = malloc(SOAK_BLOCK_SIZE);
    if (write_buffer == NULL || read_buffer == NULL) {
        fprintf(stderr, "Soak buffer allocation failed\n");
        goto cleanup;
    }

    soak_stop = 0;
    signal(SIGINT, soak_signal_handler);
    signal(SIGTERM, soak_signal_handler);
    unlink(path);

    printf("P10_SD_SOAK start mount=%s seconds=%u file_bytes=%zu\n",
           mount_point, seconds, file_size);

    while (!soak_stop && utils_cpu_ticks_us() < deadline_us) {
        size_t offset = 0;
        uint64_t pattern_base = (uint64_t)cycles * file_size;

        fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
        if (fd < 0) {
            fprintf(stderr, "P10_SD_SOAK open failed at cycle %u: %s\n",
                    cycles + 1, strerror(errno));
            goto cleanup;
        }

        while (offset < file_size) {
            size_t chunk_size = file_size - offset;

            if (chunk_size > SOAK_BLOCK_SIZE) {
                chunk_size = SOAK_BLOCK_SIZE;
            }
            fill_offset_pattern(write_buffer, chunk_size, pattern_base + offset);
            if (write_full(fd, write_buffer, chunk_size) != 0) {
                fprintf(stderr, "P10_SD_SOAK write failed at cycle %u offset %zu: %s\n",
                        cycles + 1, offset, strerror(errno));
                goto cleanup;
            }
            offset += chunk_size;
        }

        if (fsync(fd) != 0 || lseek(fd, 0, SEEK_SET) < 0) {
            fprintf(stderr, "P10_SD_SOAK sync/seek failed at cycle %u: %s\n",
                    cycles + 1, strerror(errno));
            goto cleanup;
        }

        offset = 0;
        while (offset < file_size) {
            size_t chunk_size = file_size - offset;

            if (chunk_size > SOAK_BLOCK_SIZE) {
                chunk_size = SOAK_BLOCK_SIZE;
            }
            if (read_full(fd, read_buffer, chunk_size) != 0) {
                fprintf(stderr, "P10_SD_SOAK read failed at cycle %u offset %zu: %s\n",
                        cycles + 1, offset, strerror(errno));
                goto cleanup;
            }
            fill_offset_pattern(write_buffer, chunk_size, pattern_base + offset);
            if (memcmp(write_buffer, read_buffer, chunk_size) != 0) {
                fprintf(stderr, "P10_SD_SOAK verify failed at cycle %u offset %zu\n",
                        cycles + 1, offset);
                goto cleanup;
            }
            offset += chunk_size;
        }

        if (close(fd) != 0) {
            fd = -1;
            fprintf(stderr, "P10_SD_SOAK close failed at cycle %u: %s\n",
                    cycles + 1, strerror(errno));
            goto cleanup;
        }
        fd = -1;
        if (unlink(path) != 0) {
            fprintf(stderr, "P10_SD_SOAK unlink failed at cycle %u: %s\n",
                    cycles + 1, strerror(errno));
            goto cleanup;
        }

        cycles++;
        total_bytes += (uint64_t)file_size * 2U;
        if (utils_cpu_ticks_us() >= next_report_us) {
            printf("P10_SD_SOAK progress cycles=%u verified_bytes=%" PRIu64 "\n",
                   cycles, total_bytes);
            next_report_us += SOAK_REPORT_US;
        }
        usleep(SOAK_DELAY_US);
    }

    if (soak_stop) {
        fprintf(stderr, "P10_SD_SOAK interrupted after %u cycles\n", cycles);
        goto cleanup;
    }

    printf("P10_SD_SOAK PASS seconds=%u cycles=%u verified_bytes=%" PRIu64 "\n",
           seconds, cycles, total_bytes);
    result = 0;

cleanup:
    if (fd >= 0) {
        close(fd);
    }
    unlink(path);
    free(read_buffer);
    free(write_buffer);
    return result;
}

static int run_fill_verify(const char* mount_point, unsigned int percent)
{
    struct statvfs         initial_fs;
    struct statvfs         written_fs;
    struct statvfs         deleted_fs;
    struct stat            file_stat;
    mbedtls_sha256_context write_context;
    uint8_t                write_digest[SHA256_SIZE];
    uint8_t                read_digest[SHA256_SIZE];
    uint8_t*               buffer = NULL;
    uint64_t               initial_free;
    uint64_t               target_bytes;
    uint64_t               written = 0;
    uint64_t               fragment_size;
    char                   path[256];
    int                    fd = -1;
    int                    result = 1;

    if ((percent == 0U) || (percent > 95U)) {
        fprintf(stderr, "Fill percentage must be between 1 and 95\n");
        return 1;
    }
    if (snprintf(path, sizeof(path), "%s/b03_fill_verify.bin", mount_point) >= (int)sizeof(path)) {
        fprintf(stderr, "Test path is too long\n");
        return 1;
    }
    if (statvfs(mount_point, &initial_fs) != 0) {
        fprintf(stderr, "statvfs(%s) failed: %s\n", mount_point, strerror(errno));
        return 1;
    }

    initial_free = filesystem_free_bytes(&initial_fs);
    target_bytes = ((initial_free / 100U) * percent) +
                   ((((initial_free % 100U) * percent) + 99U) / 100U);
    if (target_bytes == 0U) {
        fprintf(stderr, "No free space is available on %s\n", mount_point);
        return 1;
    }

    printf("B03_FILL_VERIFY mount=%s percent=%u\n", mount_point, percent);
    printf("initial_free=%" PRIu64 " target_bytes=%" PRIu64 "\n", initial_free, target_bytes);

    unlink(path);
    buffer = (uint8_t*)malloc(FILL_VERIFY_BLOCK_SIZE);
    if (buffer == NULL) {
        fprintf(stderr, "Buffer allocation failed: %s\n", strerror(errno));
        return 1;
    }

    mbedtls_sha256_init(&write_context);
    if (mbedtls_sha256_starts(&write_context, 0) != 0) {
        fprintf(stderr, "Failed to initialize SHA-256\n");
        goto cleanup;
    }

    fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        fprintf(stderr, "open(%s) failed: %s\n", path, strerror(errno));
        goto cleanup;
    }

    while (written < target_bytes) {
        size_t chunk_size = FILL_VERIFY_BLOCK_SIZE;

        if (target_bytes - written < chunk_size) {
            chunk_size = (size_t)(target_bytes - written);
        }
        fill_offset_pattern(buffer, chunk_size, written);
        if (write_full(fd, buffer, chunk_size) != 0) {
            fprintf(stderr, "Write failed at byte %" PRIu64 ": %s\n", written, strerror(errno));
            goto cleanup;
        }
        if (mbedtls_sha256_update(&write_context, buffer, chunk_size) != 0) {
            fprintf(stderr, "SHA-256 update failed\n");
            goto cleanup;
        }
        written += chunk_size;
    }

    if (fsync(fd) != 0) {
        fprintf(stderr, "fsync failed: %s\n", strerror(errno));
        goto cleanup;
    }
    if (close(fd) != 0) {
        fd = -1;
        fprintf(stderr, "close failed: %s\n", strerror(errno));
        goto cleanup;
    }
    fd = -1;

    if (mbedtls_sha256_finish(&write_context, write_digest) != 0) {
        fprintf(stderr, "Failed to finish write SHA-256\n");
        goto cleanup;
    }
    if ((stat(path, &file_stat) != 0) || ((uint64_t)file_stat.st_size != target_bytes)) {
        fprintf(stderr, "Created file size does not match target\n");
        goto cleanup;
    }
    if (statvfs(mount_point, &written_fs) != 0) {
        fprintf(stderr, "statvfs after write failed: %s\n", strerror(errno));
        goto cleanup;
    }

    printf("written_bytes=%" PRIu64 " free_after_write=%" PRIu64 " sha256=", written,
           filesystem_free_bytes(&written_fs));
    print_sha256(write_digest);
    printf("\n");

    if (hash_file(path, buffer, FILL_VERIFY_BLOCK_SIZE, read_digest) != 0) {
        fprintf(stderr, "Failed to hash the written file\n");
        goto cleanup;
    }
    printf("read_sha256=");
    print_sha256(read_digest);
    printf("\n");
    if (memcmp(write_digest, read_digest, SHA256_SIZE) != 0) {
        fprintf(stderr, "SHA-256 verification failed\n");
        goto cleanup;
    }

    if (unlink(path) != 0) {
        fprintf(stderr, "unlink(%s) failed: %s\n", path, strerror(errno));
        goto cleanup;
    }
    if (statvfs(mount_point, &deleted_fs) != 0) {
        fprintf(stderr, "statvfs after delete failed: %s\n", strerror(errno));
        goto cleanup;
    }

    fragment_size = deleted_fs.f_frsize ? deleted_fs.f_frsize : deleted_fs.f_bsize;
    printf("free_after_delete=%" PRIu64 "\n", filesystem_free_bytes(&deleted_fs));
    if (filesystem_free_bytes(&deleted_fs) + fragment_size < initial_free) {
        fprintf(stderr, "Free space was not recovered after deletion\n");
        goto cleanup;
    }

    printf("B03_FILL_VERIFY PASS\n");
    result = 0;

cleanup:
    if (fd >= 0) {
        close(fd);
    }
    if (result != 0) {
        unlink(path);
    }
    mbedtls_sha256_free(&write_context);
    free(buffer);
    return result;
}

static void print_result(const char* mode, size_t block_size, size_t total_bytes, uint64_t elapsed_us)
{
    char   block_text[32];
    char   total_text[32];
    double seconds;
    double mib_per_sec;

    format_size(block_size, block_text, sizeof(block_text));
    format_size(total_bytes, total_text, sizeof(total_text));

    seconds     = (elapsed_us > 0) ? ((double)elapsed_us / 1000000.0) : 0.0;
    mib_per_sec = (seconds > 0.0) ? ((double)total_bytes / (1024.0 * 1024.0)) / seconds : 0.0;

    printf("  %-24s block=%-10s total=%-10s time=%8.3f ms speed=%8.2f MiB/s\n", mode, block_text, total_text,
           (double)elapsed_us / 1000.0, mib_per_sec);
}

static int open_write_target(const char* path, size_t total_bytes, int preallocate, uint64_t* prealloc_elapsed_us)
{
    int fd;

    fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0) {
        return -1;
    }

    if (preallocate) {
        int      result;
        uint64_t start_us;
        uint64_t end_us;

        start_us = utils_cpu_ticks_us();
        result   = posix_fallocate(fd, 0, (off_t)total_bytes);
        end_us   = utils_cpu_ticks_us();
        if (prealloc_elapsed_us != NULL) {
            *prealloc_elapsed_us = (end_us >= start_us) ? (end_us - start_us) : 0;
        }

        if (result != 0) {
            errno = result;
            close(fd);
            return -1;
        }

        if (lseek(fd, 0, SEEK_SET) < 0) {
            close(fd);
            return -1;
        }
    } else if (prealloc_elapsed_us != NULL) {
        *prealloc_elapsed_us = 0;
    }

    return fd;
}

/* A NULL order selects sequential I/O; otherwise hash in the shuffled I/O order. */
static int benchmark_write(const char* path, uint8_t* buffer, size_t block_size, const size_t* order,
                           size_t block_count, int preallocate, uint64_t pattern_base,
                           uint64_t* prealloc_elapsed_us, uint64_t* elapsed_us, uint8_t digest[SHA256_SIZE])
{
    mbedtls_sha256_context context;
    int                    fd = -1;
    int                    result = -1;
    int                    saved_errno;
    size_t                 index;
    uint64_t               start_us;

    *elapsed_us = 0;
    mbedtls_sha256_init(&context);
    if (mbedtls_sha256_starts(&context, 0) != 0) {
        errno = EIO;
        goto cleanup;
    }

    fd = open_write_target(path, block_size * block_count, preallocate, prealloc_elapsed_us);
    if (fd < 0) {
        goto cleanup;
    }

    for (index = 0; index < block_count; ++index) {
        size_t block_index = order ? order[index] : index;
        off_t  offset = (off_t)(block_index * block_size);

        fill_offset_pattern(buffer, block_size, pattern_base + (uint64_t)offset);
        if (mbedtls_sha256_update(&context, buffer, block_size) != 0) {
            errno = EIO;
            goto cleanup;
        }

        /* Only seek/write time contributes to throughput, not pattern generation or hashing. */
        start_us = utils_cpu_ticks_us();
        if (order != NULL && lseek(fd, offset, SEEK_SET) != offset) {
            goto cleanup;
        }
        if (write_full(fd, buffer, block_size) != 0) {
            goto cleanup;
        }
        *elapsed_us += utils_cpu_ticks_us() - start_us;
    }

    start_us = utils_cpu_ticks_us();
    if (fsync(fd) != 0) {
        goto cleanup;
    }
    *elapsed_us += utils_cpu_ticks_us() - start_us;

    if (mbedtls_sha256_finish(&context, digest) != 0) {
        errno = EIO;
        goto cleanup;
    }
    result = 0;

cleanup:
    saved_errno = errno;
    if (fd >= 0 && close(fd) != 0 && result == 0) {
        saved_errno = errno;
        result = -1;
    }
    mbedtls_sha256_free(&context);
    errno = saved_errno;
    return result;
}

static int benchmark_read(const char* path, uint8_t* buffer, size_t block_size, const size_t* order,
                          size_t block_count, uint64_t* elapsed_us, uint8_t digest[SHA256_SIZE])
{
    mbedtls_sha256_context context;
    struct stat            file_stat;
    int                    fd = -1;
    int                    result = -1;
    int                    saved_errno;
    size_t                 index;
    uint64_t               start_us;

    *elapsed_us = 0;
    mbedtls_sha256_init(&context);
    if (mbedtls_sha256_starts(&context, 0) != 0) {
        errno = EIO;
        goto cleanup;
    }

    /* Reopen after fsync/close so readback goes through the filesystem again. */
    fd = open(path, O_RDONLY);
    if (fd < 0 || fstat(fd, &file_stat) != 0) {
        goto cleanup;
    }
    if ((uint64_t)file_stat.st_size != (uint64_t)block_size * block_count) {
        errno = EIO;
        goto cleanup;
    }

    for (index = 0; index < block_count; ++index) {
        size_t block_index = order ? order[index] : index;
        off_t  offset = (off_t)(block_index * block_size);

        /* Do not let a read that leaves the buffer untouched reuse written data. */
        memset(buffer, 0, block_size);
        start_us = utils_cpu_ticks_us();
        if (order != NULL && lseek(fd, offset, SEEK_SET) != offset) {
            goto cleanup;
        }
        if (read_full(fd, buffer, block_size) != 0) {
            goto cleanup;
        }
        *elapsed_us += utils_cpu_ticks_us() - start_us;

        if (mbedtls_sha256_update(&context, buffer, block_size) != 0) {
            errno = EIO;
            goto cleanup;
        }
    }

    if (mbedtls_sha256_finish(&context, digest) != 0) {
        errno = EIO;
        goto cleanup;
    }
    result = 0;

cleanup:
    saved_errno = errno;
    if (fd >= 0 && close(fd) != 0 && result == 0) {
        saved_errno = errno;
        result = -1;
    }
    mbedtls_sha256_free(&context);
    errno = saved_errno;
    return result;
}

static int run_benchmark_case(const char* path, uint8_t* buffer, size_t block_size, const size_t* order,
                             size_t block_count, int preallocate)
{
    const char* write_mode = order ? (preallocate ? "nonseq_write_prealloc" : "nonseq_write")
                                   : (preallocate ? "seq_write_prealloc" : "seq_write");
    const char* read_mode = order ? (preallocate ? "nonseq_read_prealloc" : "nonseq_read")
                                 : (preallocate ? "seq_read_prealloc" : "seq_read");
    uint8_t  write_digest[SHA256_SIZE];
    uint8_t  read_digest[SHA256_SIZE];
    uint64_t prealloc_elapsed_us;
    uint64_t elapsed_us;
    uint64_t pattern_base = utils_cpu_ticks();
    size_t   total_bytes = block_size * block_count;
    int      matches;

    if (benchmark_write(path, buffer, block_size, order, block_count, preallocate, pattern_base,
                        &prealloc_elapsed_us, &elapsed_us, write_digest) != 0) {
        fprintf(stderr, "  %s block=%zu failed: %s\n", write_mode, block_size, strerror(errno));
        return -1;
    }
    if (preallocate) {
        print_result("prealloc_time", block_size, total_bytes, prealloc_elapsed_us);
    }
    print_result(write_mode, block_size, total_bytes, elapsed_us);

    /* Verify every write case before the next case truncates the file. */
    if (benchmark_read(path, buffer, block_size, order, block_count, &elapsed_us, read_digest) != 0) {
        fprintf(stderr, "  %s block=%zu failed: %s\n", read_mode, block_size, strerror(errno));
        return -1;
    }
    print_result(read_mode, block_size, total_bytes, elapsed_us);

    matches = memcmp(write_digest, read_digest, SHA256_SIZE) == 0;
    printf("  %s block=%zu SHA-256 %s\n", write_mode, block_size, matches ? "PASS" : "FAIL");
    printf("    write_sha256=");
    print_sha256(write_digest);
    printf("\n    read_sha256=");
    print_sha256(read_digest);
    printf("\n");
    if (!matches) {
        fprintf(stderr, "  %s block=%zu: SHA-256 verification failed\n", write_mode, block_size);
        return -1;
    }
    return 0;
}

static int is_mount_available(const char* path)
{
    struct stat statbuf;

    if (stat(path, &statbuf) != 0) {
        return 0;
    }

    return S_ISDIR(statbuf.st_mode) ? 1 : 0;
}

static int create_directory_recursive(const char* path)
{
    char   tmp[256];
    char*  p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);

    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }

    return mkdir(tmp, 0755);
}

static int run_benchmark_for_mount(const char* mount_point)
{
    char   path[256];
    size_t size_index;

    if (mount_point[0] == '\0' ||
        snprintf(path, sizeof(path), "%s/helloworld_fsbench.bin", mount_point) >= (int)sizeof(path)) {
        fprintf(stderr, "Test path is empty or too long\n");
        return -1;
    }

    // Create directory if it doesn't exist
    if (!is_mount_available(mount_point)) {
        printf("[%s] directory does not exist, attempting to create...\n", mount_point);
        if (create_directory_recursive(mount_point) != 0) {
            printf("[%s] failed to create directory: %s\n", mount_point, strerror(errno));
            return -1;
        }
        printf("[%s] directory created successfully.\n", mount_point);
    }

    printf("\n[%s]\n", mount_point);

    for (size_index = 0; size_index < (sizeof(k_block_sizes) / sizeof(k_block_sizes[0])); ++size_index) {
        size_t   block_size  = k_block_sizes[size_index];
        size_t   block_count = (size_t)(TEST_TOTAL_BYTES / block_size);
        size_t*  order;
        uint8_t* buffer;

        if (block_count < MIN_BLOCK_COUNT) {
            block_count = MIN_BLOCK_COUNT;
        }

        buffer      = (uint8_t*)malloc(block_size);
        order       = (size_t*)malloc(block_count * sizeof(size_t));
        if ((buffer == NULL) || (order == NULL)) {
            printf("  allocation failed for block size %zu: %s\n", block_size, strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }

        if (build_permutation(order, block_count) != 0) {
            printf("  failed to prepare shuffled order for block size %zu\n", block_size);
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }

        if (run_benchmark_case(path, buffer, block_size, NULL, block_count, 0) != 0 ||
            run_benchmark_case(path, buffer, block_size, NULL, block_count, 1) != 0 ||
            run_benchmark_case(path, buffer, block_size, order, block_count, 0) != 0 ||
            run_benchmark_case(path, buffer, block_size, order, block_count, 1) != 0) {
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }

        free(order);
        free(buffer);
        unlink(path);
    }

    return 0;
}

static void print_usage(const char* prog_name)
{
    printf("Usage: %s [test_path ...]\n", prog_name);
    printf("       %s --fill-verify <mount_point> [percent]\n", prog_name);
    printf("       %s --soak <mount_point> <seconds> [file_kib]\n", prog_name);
    printf("  Each benchmark write is synced, closed and read back with SHA-256 verification.\n");
    printf("  Pattern generation and hashing are excluded from the reported I/O time.\n");
    printf("  test_path: Directory path where benchmark file will be created\n");
    printf("             If not provided, default paths will be used:\n");
    for (size_t i = 0; i < (sizeof(k_default_mount_points) / sizeof(k_default_mount_points[0])); ++i) {
        printf("               - %s\n", k_default_mount_points[i]);
    }
}

int main(int argc, char* argv[])
{
    size_t       mount_index;
    int          use_default = 1;
    const char** test_paths  = NULL;
    size_t       num_paths   = 0;

    if ((argc >= 2) && (strcmp(argv[1], "--fill-verify") == 0)) {
        unsigned long percent = 80;
        char*         end = NULL;

        if ((argc < 3) || (argc > 4)) {
            print_usage(argv[0]);
            return 1;
        }
        if (argc == 4) {
            percent = strtoul(argv[3], &end, 10);
            if ((end == argv[3]) || (*end != '\0') || (percent > 100U)) {
                fprintf(stderr, "Invalid fill percentage: %s\n", argv[3]);
                return 1;
            }
        }
        return run_fill_verify(argv[2], (unsigned int)percent);
    }

    if ((argc >= 2) && (strcmp(argv[1], "--soak") == 0)) {
        unsigned long seconds;
        unsigned long file_kib = SOAK_DEFAULT_KIB;
        char* end = NULL;

        if (argc < 4 || argc > 5) {
            print_usage(argv[0]);
            return 1;
        }
        seconds = strtoul(argv[3], &end, 10);
        if (end == argv[3] || *end != '\0' || seconds == 0 || seconds > UINT32_MAX) {
            fprintf(stderr, "Invalid soak duration: %s\n", argv[3]);
            return 1;
        }
        if (argc == 5) {
            end = NULL;
            file_kib = strtoul(argv[4], &end, 10);
            if (end == argv[4] || *end != '\0' || file_kib == 0 || file_kib > 1024U) {
                fprintf(stderr, "Invalid soak file size: %s\n", argv[4]);
                return 1;
            }
        }
        return run_soak(argv[2], (unsigned int)seconds,
                        (size_t)file_kib * 1024U);
    }

    printf("Filesystem read/write benchmark with SHA-256 verification\n");
    printf("Every write case is verified; pattern generation and hashing are excluded from I/O timing.\n");
    printf("Uses utils_cpu_ticks_us() for timing. Read numbers may include filesystem cache effects.\n");
    printf("Total payload per case is at least %.2f MiB.\n\n", (double)TEST_TOTAL_BYTES / (1024.0 * 1024.0));

    // Parse command line arguments
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }

        // Use the provided path(s)
        test_paths = (const char**)malloc((argc - 1) * sizeof(const char*));
        if (test_paths == NULL) {
            printf("Failed to allocate memory for test paths\n");
            return 1;
        }

        for (int i = 1; i < argc; ++i) {
            test_paths[num_paths++] = argv[i];
        }
        use_default = 0;
    }

    if (!use_default) {
        // Run benchmark on user-provided paths
        for (size_t i = 0; i < num_paths; ++i) {
            if (run_benchmark_for_mount(test_paths[i]) != 0) {
                free(test_paths);
                return 1;
            }
        }
        free(test_paths);
    } else {
        // Run benchmark on default paths
        for (mount_index = 0; mount_index < (sizeof(k_default_mount_points) / sizeof(k_default_mount_points[0]));
             ++mount_index) {
            if (run_benchmark_for_mount(k_default_mount_points[mount_index]) != 0) {
                return 1;
            }
        }
    }

    return 0;
}
