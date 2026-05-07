#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define TEST_TOTAL_BYTES (8ULL * 1024 * 1024)
#define MIN_BLOCK_COUNT  8U

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

static void fill_pattern(uint8_t* buffer, size_t size)
{
    size_t index;

    for (index = 0; index < size; ++index) {
        buffer[index] = (uint8_t)((index * 131u) + 17u);
    }
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

static int benchmark_seq_write(const char* path, const uint8_t* buffer, size_t block_size, size_t block_count, int preallocate,
                               uint64_t* prealloc_elapsed_us, uint64_t* elapsed_us)
{
    int      fd;
    size_t   index;
    uint64_t start_us;
    uint64_t end_us;
    size_t   total_bytes;

    total_bytes = block_size * block_count;
    fd          = open_write_target(path, total_bytes, preallocate, prealloc_elapsed_us);
    if (fd < 0) {
        return -1;
    }

    start_us = utils_cpu_ticks_us();
    for (index = 0; index < block_count; ++index) {
        if (write_full(fd, buffer, block_size) != 0) {
            close(fd);
            return -1;
        }
    }

    if (fsync(fd) != 0) {
        close(fd);
        return -1;
    }

    end_us = utils_cpu_ticks_us();
    close(fd);
    *elapsed_us = (end_us >= start_us) ? (end_us - start_us) : 0;
    return 0;
}

static int benchmark_seq_read(const char* path, uint8_t* buffer, size_t block_size, size_t block_count, uint64_t* elapsed_us)
{
    int      fd;
    size_t   index;
    uint64_t start_us;
    uint64_t end_us;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    start_us = utils_cpu_ticks_us();
    for (index = 0; index < block_count; ++index) {
        if (read_full(fd, buffer, block_size) != 0) {
            close(fd);
            return -1;
        }
    }

    end_us = utils_cpu_ticks_us();
    close(fd);
    *elapsed_us = (end_us >= start_us) ? (end_us - start_us) : 0;
    return 0;
}

static int benchmark_random_write(const char* path, const uint8_t* buffer, size_t block_size, const size_t* order,
                                  size_t block_count, int preallocate, uint64_t* prealloc_elapsed_us, uint64_t* elapsed_us)
{
    int      fd;
    size_t   index;
    uint64_t start_us;
    uint64_t end_us;
    size_t   total_bytes;

    total_bytes = block_size * block_count;
    fd          = open_write_target(path, total_bytes, preallocate, prealloc_elapsed_us);
    if (fd < 0) {
        return -1;
    }

    start_us = utils_cpu_ticks_us();
    for (index = 0; index < block_count; ++index) {
        off_t offset = (off_t)(order[index] * block_size);

        if (lseek(fd, offset, SEEK_SET) < 0) {
            close(fd);
            return -1;
        }

        if (write_full(fd, buffer, block_size) != 0) {
            close(fd);
            return -1;
        }
    }

    if (fsync(fd) != 0) {
        close(fd);
        return -1;
    }

    end_us = utils_cpu_ticks_us();
    close(fd);
    *elapsed_us = (end_us >= start_us) ? (end_us - start_us) : 0;
    return 0;
}

static int benchmark_random_read(const char* path, uint8_t* buffer, size_t block_size, const size_t* order, size_t block_count,
                                 uint64_t* elapsed_us)
{
    int      fd;
    size_t   index;
    uint64_t start_us;
    uint64_t end_us;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    start_us = utils_cpu_ticks_us();
    for (index = 0; index < block_count; ++index) {
        off_t offset = (off_t)(order[index] * block_size);

        if (lseek(fd, offset, SEEK_SET) < 0) {
            close(fd);
            return -1;
        }

        if (read_full(fd, buffer, block_size) != 0) {
            close(fd);
            return -1;
        }
    }

    end_us = utils_cpu_ticks_us();
    close(fd);
    *elapsed_us = (end_us >= start_us) ? (end_us - start_us) : 0;
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

    // Create directory if it doesn't exist
    if (!is_mount_available(mount_point)) {
        printf("[%s] directory does not exist, attempting to create...\n", mount_point);
        if (create_directory_recursive(mount_point) != 0) {
            printf("[%s] failed to create directory: %s\n", mount_point, strerror(errno));
            return 0;
        }
        printf("[%s] directory created successfully.\n", mount_point);
    }

    snprintf(path, sizeof(path), "%s/helloworld_fsbench.bin", mount_point);
    printf("\n[%s]\n", mount_point);

    for (size_index = 0; size_index < (sizeof(k_block_sizes) / sizeof(k_block_sizes[0])); ++size_index) {
        size_t   block_size  = k_block_sizes[size_index];
        size_t   block_count = (size_t)(TEST_TOTAL_BYTES / block_size);
        size_t   total_bytes;
        size_t*  order;
        uint8_t* buffer;
        uint64_t prealloc_elapsed_us;
        uint64_t elapsed_us;

        if (block_count < MIN_BLOCK_COUNT) {
            block_count = MIN_BLOCK_COUNT;
        }

        total_bytes = block_count * block_size;
        buffer      = (uint8_t*)malloc(block_size);
        order       = (size_t*)malloc(block_count * sizeof(size_t));
        if ((buffer == NULL) || (order == NULL)) {
            printf("  allocation failed for block size %zu: %s\n", block_size, strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }

        fill_pattern(buffer, block_size);
        if (build_permutation(order, block_count) != 0) {
            printf("  failed to prepare shuffled order for block size %zu\n", block_size);
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }

        if (benchmark_seq_write(path, buffer, block_size, block_count, 0, NULL, &elapsed_us) != 0) {
            printf("  seq_write        failed: %s\n", strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }
        print_result("seq_write", block_size, total_bytes, elapsed_us);

        if (benchmark_seq_write(path, buffer, block_size, block_count, 1, &prealloc_elapsed_us, &elapsed_us) != 0) {
            printf("  seq_write_prealloc failed: %s\n", strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }
        print_result("prealloc_time", block_size, total_bytes, prealloc_elapsed_us);
        print_result("seq_write_prealloc", block_size, total_bytes, elapsed_us);

        if (benchmark_seq_read(path, buffer, block_size, block_count, &elapsed_us) != 0) {
            printf("  seq_read         failed: %s\n", strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }
        print_result("seq_read", block_size, total_bytes, elapsed_us);

        if (benchmark_random_write(path, buffer, block_size, order, block_count, 0, NULL, &elapsed_us) != 0) {
            printf("  nonseq_write     failed: %s\n", strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }
        print_result("nonseq_write", block_size, total_bytes, elapsed_us);

        if (benchmark_random_write(path, buffer, block_size, order, block_count, 1, &prealloc_elapsed_us, &elapsed_us) != 0) {
            printf("  nonseq_write_prealloc failed: %s\n", strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }
        print_result("prealloc_time", block_size, total_bytes, prealloc_elapsed_us);
        print_result("nonseq_write_prealloc", block_size, total_bytes, elapsed_us);

        if (benchmark_random_read(path, buffer, block_size, order, block_count, &elapsed_us) != 0) {
            printf("  nonseq_read      failed: %s\n", strerror(errno));
            free(order);
            free(buffer);
            unlink(path);
            return -1;
        }
        print_result("nonseq_read", block_size, total_bytes, elapsed_us);

        free(order);
        free(buffer);
        unlink(path);
    }

    return 0;
}

static void print_usage(const char* prog_name)
{
    printf("Usage: %s [test_path]\n", prog_name);
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

    printf("Filesystem read/write benchmark\n");
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
