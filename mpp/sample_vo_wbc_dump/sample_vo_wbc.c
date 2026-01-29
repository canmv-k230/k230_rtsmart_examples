#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"

#include "kd_display.h"

// ============================================================================
// Global Defines and Variables
// ============================================================================

#define ISP_WIDTH           (1920)
#define ISP_HEIGHT          (1080)
#define ISP_CH0_WIDTH       (1920)
#define ISP_CH0_HEIGHT      (1080)
#define DEFAULT_DUMP_FRAMES (10)
#define OUTPUT_FILENAME     "/data/wbc_dump_out.yuv"

// Global flag to control the main loop
static volatile bool g_app_run = true;
// Global variable to hold the CSI device ID, initialized in parse_arguments
static k_vicap_dev vicap_dev_id       = CONFIG_MPP_SENSOR_DEFAULT_CSI;
static int         g_dump_frame_count = DEFAULT_DUMP_FRAMES;

// ============================================================================
// NV12 Saving Function (Copied and used as a blueprint)
// ============================================================================

/**
 * @brief Maps the NV12 frame buffers, writes them to file, and unmaps them.
 * This logic is adapted from the user-provided function.
 */
static void save_wbc_nv12_stream(FILE* fp, const k_video_frame_info* frame_info)
{
    // The frame format is YUV420 semi-planar (NV12)
    unsigned ysize = frame_info->v_frame.width * frame_info->v_frame.height;
    // UV plane size is half of the Y size
    unsigned uvsize   = ysize / 2;
    k_u32    y_paddr  = frame_info->v_frame.phys_addr[0];
    k_u32    uv_paddr = frame_info->v_frame.phys_addr[1];

    if (y_paddr == 0 || uv_paddr == 0) {
        printf("Invalid physical address for Y/UV planes.");
        return;
    }

    // Map physical addresses to virtual addresses
    void* y_vaddr  = kd_mpi_sys_mmap_cached(y_paddr, ysize);
    void* uv_vaddr = kd_mpi_sys_mmap_cached(uv_paddr, uvsize);

    if (y_vaddr == NULL || uv_vaddr == NULL) {
        printf("Failed to map frame buffers.");
        // Unmap any successful mappings before returning
        if (y_vaddr != NULL)
            kd_mpi_sys_munmap(y_vaddr, ysize);
        if (uv_vaddr != NULL)
            kd_mpi_sys_munmap(uv_vaddr, uvsize);
        return;
    }

    // Write Y plane
    if (fwrite(y_vaddr, 1, ysize, fp) != ysize) {
        printf("Error writing Y plane.");
    }

    // Write UV plane
    if (fwrite(uv_vaddr, 1, uvsize, fp) != uvsize) {
        printf("Error writing UV plane.");
    }

    // Unmap virtual addresses
    kd_mpi_sys_munmap(y_vaddr, ysize);
    kd_mpi_sys_munmap(uv_vaddr, uvsize);
}

// ============================================================================
// Core Logic: WBC Frame Dumping
// ============================================================================

/**
 * @brief Configures and enables VO Write Back Controller (WBC) and dumps frames.
 * @param frame_count The number of frames to dump.
 */
static void sample_vo_wbc_dump(int frame_count)
{
    if (kd_display_wbc_configure(3)) {
        printf("kd_mpi_vo_set_wbc_attr error");
        return;
    }

    if (kd_display_wbc_enable()) {
        printf("kd_mpi_vo_enable_wbc error");
        return;
    }

    printf("WBC enabled. Dumping %d frames to %s...\n", frame_count, OUTPUT_FILENAME);

    FILE* fp = NULL;
    fp       = fopen(OUTPUT_FILENAME, "wb");
    if (!fp) {
        printf("Failed to open output file: %s, errno=%d", OUTPUT_FILENAME, errno);
        return;
    }

    k_video_frame_info frame_info;
    for (int i = 0; i < frame_count && g_app_run; ++i) {
        // 1. Dump frame (blocking call with timeout 50ms)
        unsigned error = kd_display_wbc_dump_frame(&frame_info, 50);
        if (error) {
            printf("kd_mpi_wbc_dump_frame error: %u. Retrying.", error);
            // This is non-fatal, might be a temporary issue. Continue to try the next frame.
            continue;
        }

        printf("Dumped frame %d (%dx%d). Writing to file...\n", i, frame_info.v_frame.width, frame_info.v_frame.height);

        // 2. Write frame data to file using NV12 save function
        save_wbc_nv12_stream(fp, &frame_info);

        // 3. Release the frame
        if (0x00 != kd_display_wbc_release_frame(&frame_info)) {
            printf("wbc dump release failed for frame %d", i);
        }
    }

    kd_display_wbc_disable();

    if (fp) {
        fclose(fp);
        printf("Successfully finished dumping %d frames to %s.\n", frame_count, OUTPUT_FILENAME);
    }
}

// ============================================================================
// Argument Parsing (Updated for completeness)
// ============================================================================

static void print_usage(const char* program_name)
{
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -c, --csi <num>    CSI device number (0-2, default: 2)\n");
    printf("  -n, --frames <num> Number of frames to dump (default: %d)\n", DEFAULT_DUMP_FRAMES);
    printf("  -h, --help         Show this help message\n");
}

static int parse_arguments(int argc, char** argv)
{
    int opt;
    int csi_num   = 2;
    int frame_num = DEFAULT_DUMP_FRAMES;

    while ((opt = getopt(argc, argv, "c:n:h")) != -1) {
        switch (opt) {
        case 'c':
            csi_num = atoi(optarg);
            if (csi_num < 0 || csi_num > 2) {
                printf("ERROR: CSI number must be between 0 and 2\n");
                return -1;
            }
            vicap_dev_id = (k_vicap_dev)csi_num;
            printf("Using CSI device: %d\n", csi_num);
            break;
        case 'n':
            frame_num = atoi(optarg);
            if (frame_num <= 0) {
                printf("ERROR: Frame count must be a positive integer\n");
                return -1;
            }
            g_dump_frame_count = frame_num;
            printf("Dumping %d frames.\n", frame_num);
            break;
        case 'h':
            print_usage(argv[0]);
            exit(0);
        default:
            print_usage(argv[0]);
            return -1;
        }
    }
    vicap_dev_id = (k_vicap_dev)csi_num;
    return 0;
}

// ============================================================================
// Module Initialization and Deinitialization (Restored and Corrected)
// ============================================================================

static k_s32 sample_vb_init(void)
{
    k_vb_config config  = { 0 };
    config.max_pool_cnt = 64;

    k_s32 ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("kd_mpi_vb_set_config failed, ret=%d", ret);
        return ret;
    }

    k_vb_supplement_config supplement_config = { 0 };
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("kd_mpi_vb_set_supplement_config failed, ret=%d", ret);
        return ret;
    }

    ret = kd_mpi_vb_init();
    if (ret) {
        printf("kd_mpi_vb_init failed, ret=%d", ret);
    }
    return ret;
}

static k_s32 sample_vicap_init(k_vicap_dev dev_chn)
{
    k_vicap_dev_attr     dev_attr;
    k_vicap_chn_attr     chn_attr;
    k_vicap_sensor_info  sensor_info;
    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_type  sensor_type;

    memset(&sensor_info, 0, sizeof(sensor_info));
    probe_cfg.csi_num = dev_chn;
    probe_cfg.width   = ISP_WIDTH;
    probe_cfg.height  = ISP_HEIGHT;
    probe_cfg.fps     = 30;

    if (0x00 != kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info)) {
        printf("ERROR: can't probe sensor on %d, output %dx%d@%d\n", probe_cfg.csi_num, probe_cfg.width, probe_cfg.height,
               probe_cfg.fps);
        return -1;
    }
    sensor_type = sensor_info.sensor_type;

    k_s32 ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
    if (ret) {
        printf("kd_mpi_vicap_get_sensor_info failed, ret=%d", ret);
        return ret;
    }

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width  = ISP_WIDTH;
    dev_attr.acq_win.height = ISP_HEIGHT;
    dev_attr.mode           = VICAP_WORK_OFFLINE_MODE;
    dev_attr.buffer_num     = 6;
    dev_attr.buffer_size    = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 1024);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret) {
        printf("kd_mpi_vicap_set_dev_attr failed, ret=%d", ret);
        return ret;
    }

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = ISP_CH0_WIDTH;
    chn_attr.out_win.height = ISP_CH0_HEIGHT;
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_FALSE;
    chn_attr.scale_enable   = K_TRUE;
    chn_attr.chn_enable     = K_TRUE;
    chn_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num     = 6;
    chn_attr.buffer_size    = VB_ALIGN_UP(ISP_CH0_WIDTH * ISP_CH0_HEIGHT * 3 / 2, 4096);
    chn_attr.alignment      = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        printf("kd_mpi_vicap_set_chn_attr failed, ret=%d", ret);
        return ret;
    }

    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    if (ret) {
        printf("kd_mpi_vicap_init failed, ret=%d", ret);
    }
    return ret;
}

/**
 * @brief Initializes the VO connector and configures/enables Layer 2.
 */
k_s32 sample_vo_init(k_connector_type type)
{
    const k_vo_layer_id target_layer  = K_VO_LAYER_VIDEO2;
    const k_u32         target_width  = ISP_CH0_WIDTH; // 1920
    const k_u32         target_height = ISP_CH0_HEIGHT; // 1080

    // 1. Initialize Connector/Device
    if (0x00 != kd_display_init(type, target_width, target_height)) {
        printf("ERROR: sample_vo_init, the connector type not supported!\n");
        return -1;
    }

    // 2. Configure and Enable VO Layer 2
    printf("Configuring VO Layer %d directly...\n", target_layer);

    // Check width constraint (integral multiple of 8)
    if (target_width == 0 || target_height == 0 || (target_width & 7) != 0) {
        printf("ERROR: width (%u) must be an integral multiple of 8 pixels and not zero.\n", target_width);
        return -1;
    }

    // Disable layer before configuration
    kd_display_layer_disable(target_layer);

    if (0x00 != kd_display_layer_configure(target_layer, PIXEL_FORMAT_YUV_SEMIPLANAR_420, target_width, target_height)) {
        printf("ERROR: sample_vo_init, configure layer failed!\n");
        return -1;
    }

    // Enable the layer
    if (0x00 != kd_display_layer_enable(target_layer)) {
        printf("ERROR: sample_vo_init, kd_display_layer_enable failed\n");
        return -1;
    }

    printf("VO Layer %d configured and enabled successfully.\n", target_layer);
    return K_SUCCESS;
}

static void sample_vicap_bind_vo()
{
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn vo_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = VICAP_CHN_ID_0;

    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = K_VO_LAYER_VIDEO2;

    kd_mpi_sys_bind(&vi_mpp_chn, &vo_mpp_chn);
}

static void sample_vicap_unbind_vo()
{
    k_mpp_chn vvi_mpp_chn;
    k_mpp_chn vvo_mpp_chn;

    vvi_mpp_chn.mod_id = K_ID_VI;
    vvi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vvi_mpp_chn.chn_id = VICAP_CHN_ID_0;

    vvo_mpp_chn.mod_id = K_ID_VO;
    vvo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vvo_mpp_chn.chn_id = K_VO_LAYER_VIDEO2;

    kd_mpi_sys_unbind(&vvi_mpp_chn, &vvo_mpp_chn);
}

// ============================================================================
// Main Function (Final)
// ============================================================================

static void handle_signal(int sig)
{
    if (sig == SIGINT) {
        printf("Caught SIGINT, signaling application to exit...\n");
        g_app_run = false;
    }
}

int main(int argc, char** argv)
{
    printf("WBC Frame Dump Demo Started\n");

    // Parse command line arguments
    if (parse_arguments(argc, argv) != 0) {
        return -1;
    }

    // Register signal handler for graceful shutdown
    signal(SIGINT, handle_signal);

    // 1. Initialize System Modules
    if (sample_vb_init() != K_SUCCESS)
        goto cleanup_none;
    if (sample_vicap_init(vicap_dev_id) != K_SUCCESS)
        goto cleanup_vb;

    // 2. Start Camera Stream
    printf("Starting VICAP stream...\n");
    if (kd_mpi_vicap_start_stream(VICAP_DEV_ID_0) != K_SUCCESS) {
        printf("ERROR: Failed to start VICAP stream\n");
        goto cleanup_vicap;
    }

    if (sample_vo_init(VIRTUAL_DISPLAY_DEVICE) != K_SUCCESS) {
        printf("ERROR: Failed to start video output\n");
        goto cleanup_vo;
    }

    // 3. Bind VICAP to VO and Run WBC Dump
    sample_vicap_bind_vo();

    sample_vo_wbc_dump(g_dump_frame_count);

    // 4. Stop and Deinitialize System Modules (in reverse order)
    printf("Stopping VICAP stream...\n");
    kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);

    sample_vicap_unbind_vo();

cleanup_vo:
    kd_display_deinit();

cleanup_vicap:
    printf("Deinitializing VICAP...\n");
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
cleanup_vb:
    printf("Deinitializing VB...\n");
    kd_mpi_vb_exit();
cleanup_none:
    printf("WBC Frame Dump Demo Finished\n");
    return 0;
}
