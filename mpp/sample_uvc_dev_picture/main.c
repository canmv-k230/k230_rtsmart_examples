#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "jpeg_data.h" // Include the JPEG data header

#include "mpi_uvc_api.h"
#include "mpi_vb_api.h"

volatile sig_atomic_t keep_running = 1;

// Signal handler for Ctrl-C
void signal_handler(int sig)
{
    keep_running = 0;
    printf("\nReceived signal %d, shutting down gracefully...\n", sig);
}

// Simple test data structure for JPEG data
typedef struct {
    const char*    filename;
    const uint8_t* data;
    uint32_t       size;
} test_data_t;

// Global test data arrays
static test_data_t test_data_array[2];
static int         current_test_index = 0;
static int         test_data_count    = 1;

// Load test data from JPEG headers
int load_test_data()
{
    // First JPEG data
    test_data_array[0].filename = "test_jpeg_data_1";
    test_data_array[0].data     = test_jpeg_data_1;
    test_data_array[0].size     = sizeof(test_jpeg_data_1);

    if (test_data_array[0].size == 0 || test_data_array[0].data == NULL) {
        printf("ERROR: Invalid JPEG data 1 from header\n");
        return -1;
    }

    printf("Loaded JPEG test data 1: %s (%u bytes)\n", test_data_array[0].filename, test_data_array[0].size);
    printf("Total test data items: %d\n", test_data_count);

    return 0;
}

// Get the next test data (alternate between the two JPEGs)
test_data_t* get_next_test_data()
{
    test_data_t* data = &test_data_array[current_test_index];

    // Move to next index for next call
    current_test_index = (current_test_index + 1) % test_data_count;

    return data;
}

// Free test data (no-op since data is static)
void free_test_data() { printf("Test data references cleared\n"); }

// Find maximum size among test data items
uint32_t get_max_test_data_size()
{
    uint32_t max_size = 0;
    for (int i = 0; i < test_data_count; i++) {
        if (test_data_array[i].size > max_size) {
            max_size = test_data_array[i].size;
        }
    }
    return max_size;
}

// Test function with continuous loop
int test_uvc_driver_continuous()
{
    int                      ret      = 0;
    int                      opened   = 0;
    uint32_t*                buffer   = NULL;
    uint32_t                 max_size = 0;
    struct uvc_device_conf_t config;
    int                      frame_count = 0;

    printf("=== UVC Driver Continuous Test Started ===\n");
    printf("Press Ctrl-C to stop\n");
    printf("Using %d embedded JPEG data items\n", test_data_count);

    // 1. Initialize UVC device
    printf("1. Initializing UVC device...\n");
    ret = uvc_device_init();
    if (ret != 0) {
        printf("ERROR: uvc_device_init failed with %d\n", ret);
        return -1;
    }

    // 2. Create buffer pool (use maximum JPEG data size)
    printf("2. Creating buffer pool...\n");
    uint32_t max_jpeg_size = get_max_test_data_size();

    ret = uvc_device_create_buffer_pool(max_jpeg_size + 1024, 5); // Add some padding
    if (ret != 0) {
        printf("ERROR: uvc_device_create_buffer_pool failed with %d\n", ret);
        uvc_device_deinit();
        return -1;
    }

    // 3. Configure device
    printf("3. Configuring UVC device...\n");
    config.frame_rate = 30; // 30 FPS
    ret               = uvc_device_conf(&config);
    if (ret != 0) {
        printf("ERROR: uvc_device_conf failed with %d\n", ret);
        uvc_device_deinit();
        return -1;
    }

    // 4. Start streaming
    printf("4. Starting video streaming...\n");
    ret = uvc_device_start();
    if (ret != 0) {
        printf("ERROR: uvc_device_start failed with %d\n", ret);
        uvc_device_deinit();
        return -1;
    }

    // 5. Continuous test loop
    printf("5. Starting continuous test loop...\n");

    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    while (keep_running) {
        if (0x00 != uvc_device_get_state(&opened)) {
            printf("ERROR: uvc_device_get_state failed\n");
            break;
        }
        if (0x00 == opened) {
            usleep(1000 * 10);
            continue;
        }

        // Get buffer
        ret = uvc_device_get_buf(&buffer, &max_size);
        if (ret != 0 || buffer == NULL) {
            continue;
        }

        // Get next test data (alternates between the two JPEGs)
        test_data_t* test_data_ptr = get_next_test_data();
        if (!test_data_ptr) {
            printf("ERROR: No test data available\n");
            break;
        }

        // Copy JPEG data to buffer
        uint32_t data_size = test_data_ptr->size;
        if (data_size > max_size) {
            printf("WARNING: Truncating data from %u to %u bytes\n", data_size, max_size);
            data_size = max_size;
        }

        memcpy(buffer, test_data_ptr->data, data_size);

        // Put buffer back
        ret = uvc_device_put_buf(buffer, data_size);
        if (ret != 0) {
            printf("ERROR: uvc_device_put_buf failed\n");
            break;
        }

        frame_count++;

        // Display progress every 100 frames
        if (frame_count % 100 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            double elapsed = (current_time.tv_sec - start_time.tv_sec) + (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            double fps     = frame_count / elapsed;

            printf("Frames: %d, Data: %s (%u bytes), FPS: %.1f\n", frame_count, test_data_ptr->filename, test_data_ptr->size,
                   fps);
        }
    }

    // 6. Stop streaming
    printf("6. Stopping video streaming...\n");
    ret = uvc_device_stop();
    if (ret != 0) {
        printf("ERROR: uvc_device_stop failed with %d\n", ret);
    }

    // 7. Deinitialize
    printf("7. Deinitializing UVC device...\n");
    uvc_device_deinit();

    // Calculate final statistics
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    double elapsed = (current_time.tv_sec - start_time.tv_sec) + (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
    double fps     = frame_count / elapsed;

    printf("=== Test Completed ===\n");
    printf("Total frames: %d\n", frame_count);
    printf("Total time: %.2f seconds\n", elapsed);
    printf("Average FPS: %.2f\n", fps);
    printf("Test data items used: %d\n", test_data_count);

    return 0;
}

int vb_init(void)
{
    k_s32       ret = 0;
    k_vb_config config;

    memset(&config, 0, sizeof(config));

    config.max_pool_cnt = 10;

    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("kd_mpi_vb_set_config fail, ret = %d\n", ret);
        goto out;
    }

    ret = kd_mpi_vb_init();
    if (ret) {
        printf("kd_mpi_vb_init fail, ret = %d\n", ret);
    }

out:
    return ret;
}

// Main test entry point
int main(int argc, char* argv[])
{
    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("UVC Driver Continuous Test Application\n");
    printf("======================================\n");
    printf("Using embedded JPEG data from header file\n\n");

    // Load test data from header
    if (load_test_data() != 0) {
        printf("ERROR: Failed to load test data from header\n");
        return 1;
    }

    printf("\nStarting continuous test...\n");
    printf("Loaded %d test JPEG data items\n", test_data_count);
    printf("Press Ctrl-C to stop the test\n\n");

    vb_init();

    int ret = test_uvc_driver_continuous();

    // Cleanup
    free_test_data();

    if (ret == 0) {
        printf("\n✅ Test completed successfully\n");
    } else {
        printf("\n❌ Test failed\n");
    }

    kd_mpi_vb_exit();

    return ret;
}
