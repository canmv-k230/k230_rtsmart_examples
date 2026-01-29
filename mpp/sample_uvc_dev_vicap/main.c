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
#include "mpi_uvc_api.h"
#include "mpi_vb_api.h"
#include "mpi_venc_api.h"
#include "mpi_vicap_api.h"

// ============================================================================
// Global Defines and Variables
// ============================================================================

#define ISP_WIDTH  (1920)
#define ISP_HEIGHT (1080)

#define VENC_CH_ID_0 (0)

#define VENC_WIDTH  (1920)
#define VENC_HEIGHT (1080)

// Global flag to control the main loop
static volatile bool g_app_run = true;

static k_vicap_dev vicap_dev_id = VICAP_DEV_ID_2;

static k_s32 venc_pool_id = VB_INVALID_POOLID;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Handles Ctrl+C signal to allow for a graceful shutdown.
 */
static void handle_signal(int sig)
{
    if (sig == SIGINT) {
        printf("Caught SIGINT, signaling application to exit...\n");
        g_app_run = false;
    }
}

// ============================================================================
// Module Initialization and Deinitialization
// ============================================================================

/**
 * @brief Initializes the Video Buffer (VB) pool.
 * This is crucial for allocating memory for video frames.
 */
static k_s32 sample_vb_init(void)
{
    k_vb_config config  = { 0 };
    config.max_pool_cnt = 64;

    k_s32 ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("ERROR: kd_mpi_vb_set_config failed, ret=%d\n", ret);
        return ret;
    }

    k_vb_supplement_config supplement_config = { 0 };
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("ERROR: kd_mpi_vb_set_supplement_config failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vb_init();
    if (ret) {
        printf("ERROR: kd_mpi_vb_init failed, ret=%d\n", ret);
    }
    return ret;
}

/**
 * @brief Initializes the Video Capture (VICAP) module.
 * This configures the camera sensor.
 */
static k_s32 sample_vicap_init(k_vicap_dev dev_chn)
{
    k_vicap_dev_attr    dev_attr;
    k_vicap_chn_attr    chn_attr;
    k_vicap_sensor_info sensor_info;

    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_type  sensor_type;

    // 1. Get sensor info
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
        printf("ERROR: kd_mpi_vicap_get_sensor_info failed, ret=%d\n", ret);
        return ret;
    }

    // 2. Set device attributes
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
        printf("ERROR: kd_mpi_vicap_set_dev_attr failed, ret=%d\n", ret);
        return ret;
    }

    // 3. Set channel attributes
    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = VENC_WIDTH;
    chn_attr.out_win.height = VENC_HEIGHT;
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_FALSE;
    chn_attr.scale_enable   = K_TRUE;
    chn_attr.chn_enable     = K_TRUE;
    chn_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num     = 6;
    chn_attr.buffer_size    = VB_ALIGN_UP(VENC_WIDTH * VENC_HEIGHT * 3 / 2, 4096);
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;
    chn_attr.alignment      = 12;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_set_chn_attr failed, ret=%d\n", ret);
        return ret;
    }

    // 4. Initialize the device
    ret = kd_mpi_vicap_init(VICAP_DEV_ID_0);
    if (ret) {
        printf("ERROR: kd_mpi_vicap_init failed, ret=%d\n", ret);
    }
    return ret;
}

/**
 * @brief Initializes the Video Encoder (VENC) module for JPEG encoding.
 */
static k_s32 sample_venc_init(void)
{
    k_s32 poolid;

    k_venc_chn_attr attr;
    memset(&attr, 0, sizeof(attr));

    venc_pool_id = VB_INVALID_POOLID;

    poolid = kd_mpi_vb_create_pool_ex(VB_ALIGN_UP(VENC_WIDTH * VENC_HEIGHT, 4096), 4, VB_REMAP_MODE_NOCACHE);
    if (VB_INVALID_POOLID == poolid) {
        printf("ERROR: create pool failed\n");
        return -1;
    }

    if (K_SUCCESS != kd_mpi_venc_attach_vb_pool(VENC_CH_ID_0, poolid)) {
        printf("ERROR: attach venc pool failed\n");

        kd_mpi_vb_destory_pool(poolid);
        return -2;
    }
    venc_pool_id = poolid;

    attr.venc_attr.type                     = K_PT_JPEG;
    attr.venc_attr.pic_width                = VENC_WIDTH;
    attr.venc_attr.pic_height               = VENC_HEIGHT;
    attr.rc_attr.rc_mode                    = K_VENC_RC_MODE_MJPEG_FIXQP;
    attr.rc_attr.mjpeg_fixqp.src_frame_rate = 30;
    attr.rc_attr.mjpeg_fixqp.dst_frame_rate = 30;
    attr.rc_attr.mjpeg_fixqp.q_factor       = 90; // High quality

    k_s32 ret = kd_mpi_venc_create_chn(VENC_CH_ID_0, &attr);
    if (ret) {
        printf("ERROR: kd_mpi_venc_create_chn failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_venc_start_chn(VENC_CH_ID_0);
    if (ret) {
        printf("ERROR: kd_mpi_venc_start_chn failed, ret=%d\n", ret);
        kd_mpi_venc_destroy_chn(VENC_CH_ID_0);
    }
    return ret;
}

/**
 * @brief Initializes the USB Video Class (UVC) device.
 */
static k_s32 sample_uvc_init(void)
{
    k_s32 ret = uvc_device_init();
    if (ret != 0) {
        printf("ERROR: uvc_device_init failed, ret=%d\n", ret);
        return ret;
    }

    // Create a buffer pool for UVC frames
    ret = uvc_device_create_buffer_pool(VB_ALIGN_UP(VENC_WIDTH * VENC_HEIGHT, 4096), 5);
    if (ret != 0) {
        printf("ERROR: uvc_device_create_buffer_pool failed, ret=%d\n", ret);
        uvc_device_deinit();
        return ret;
    }

    // Configure the UVC device
    struct uvc_device_conf_t config;
    config.frame_rate = 30;
    ret               = uvc_device_conf(&config);
    if (ret != 0) {
        printf("ERROR: uvc_device_conf failed, ret=%d\n", ret);
        uvc_device_deinit();
        return ret;
    }

    // Start UVC streaming
    ret = uvc_device_start();
    if (ret != 0) {
        printf("ERROR: uvc_device_start failed, ret=%d\n", ret);
        uvc_device_deinit();
    }
    return ret;
}

// ============================================================================
// Main Application Logic
// ============================================================================
/**
 * @brief The main video processing loop.
 * It grabs frames from VICAP, encodes them with VENC, and sends them via UVC.
 */
static void sample_video_loop(void)
{
    k_s32              ret;
    k_video_frame_info vicap_frame;
    k_venc_chn_status  venc_status;
    k_venc_stream      venc_stream;
    uint32_t*          uvc_buffer   = NULL;
    uint32_t           uvc_max_size = 0;
    int                uvc_opened   = 0;
    unsigned long long frame_count  = 0;

    printf("Starting video pipeline loop...\n");

    while (g_app_run) {
        // 1. Get a frame from VICAP (camera)
        memset(&vicap_frame, 0, sizeof(vicap_frame));
        ret = kd_mpi_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, VICAP_DUMP_YUV, &vicap_frame, 1000);
        if (ret != K_SUCCESS) {
            printf("WARN: Failed to dump frame from VICAP, ret=%d\n", ret);
            usleep(10000);
            continue;
        }

        // 2. Send the captured frame to VENC for encoding
        ret = kd_mpi_venc_send_frame(VENC_CH_ID_0, &vicap_frame, 1000);
        // We must release the VICAP frame regardless of whether the send succeeded
        kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &vicap_frame);
        if (ret != K_SUCCESS) {
            printf("WARN: Failed to send frame to VENC, ret=%d\n", ret);
            continue;
        }

        ret = kd_mpi_venc_query_status(VENC_CH_ID_0, &venc_status);
        if (ret != K_SUCCESS) {
            printf("ERROR: kd_mpi_venc_query_status failed, ret=%d\n", ret);
            continue;
        }

        // 3. Get the encoded stream from VENC
        memset(&venc_stream, 0, sizeof(venc_stream));
        venc_stream.pack_cnt = (venc_status.cur_packs > 0) ? venc_status.cur_packs : 1;
        venc_stream.pack     = malloc(sizeof(k_venc_pack) * venc_stream.pack_cnt);
        if (!venc_stream.pack) {
            printf("ERROR: Failed to allocate memory for VENC packs\n");
            continue;
        }
        ret = kd_mpi_venc_get_stream(VENC_CH_ID_0, &venc_stream, 1000);
        if (ret != K_SUCCESS) {
            printf("WARN: Failed to get stream from VENC, ret=%d\n", ret);
            free(venc_stream.pack);
            continue;
        }

        // 4. Check if UVC is ready and get a buffer to write to
        if (uvc_device_get_state(&uvc_opened) != 0 || !uvc_opened) {
            // UVC host is not connected or ready, drop this frame
            kd_mpi_venc_release_stream(VENC_CH_ID_0, &venc_stream);
            free(venc_stream.pack);
            usleep(10000); // Wait a bit before trying again
            continue;
        }

        ret = uvc_device_get_buf(&uvc_buffer, &uvc_max_size);
        if (ret != 0 || uvc_buffer == NULL) {
            // UVC buffers might be full, drop this frame
            kd_mpi_venc_release_stream(VENC_CH_ID_0, &venc_stream);
            free(venc_stream.pack);
            continue;
        }

        // 5. Copy encoded JPEG data to the UVC buffer
        uint32_t total_len = 0;
        for (int i = 0; i < venc_stream.pack_cnt; i++) {
            if ((total_len + venc_stream.pack[i].len) > uvc_max_size) {
                printf("WARN: Encoded data too large for UVC buffer. Truncating.\n");
                break;
            }
            k_u8* src_data = (k_u8*)kd_mpi_sys_mmap(venc_stream.pack[i].phys_addr, venc_stream.pack[i].len);
            if (src_data) {
                memcpy((k_u8*)uvc_buffer + total_len, src_data, venc_stream.pack[i].len);
                total_len += venc_stream.pack[i].len;
                kd_mpi_sys_munmap(src_data, venc_stream.pack[i].len);
            }
        }

        // 6. Release the VENC stream buffer
        kd_mpi_venc_release_stream(VENC_CH_ID_0, &venc_stream);
        free(venc_stream.pack);

        // 7. Send the filled buffer to the UVC host
        ret = uvc_device_put_buf(uvc_buffer, total_len);
        if (ret != 0) {
            printf("ERROR: uvc_device_put_buf failed\n");
        }

        frame_count++;
        if (frame_count % 100 == 0) {
            printf("Sent %llu frames via UVC\n", frame_count);
        }
    }

    printf("Video pipeline loop finished.\n");
}

static void print_usage(const char* program_name)
{
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -c, --csi <num>    CSI device number (0-2, default: 2)\n");
    printf("  -h, --help         Show this help message\n");
}

static int parse_arguments(int argc, char** argv)
{
    int opt;
    int csi_num = 0;

    // Use simpler getopt instead of getopt_long for better compatibility
    while ((opt = getopt(argc, argv, "c:h")) != -1) {
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
        case 'h':
            print_usage(argv[0]);
            exit(0);
        default:
            print_usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    printf("UVC Camera Demo Started\n");

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
    if (sample_venc_init() != K_SUCCESS)
        goto cleanup_vicap;
    if (sample_uvc_init() != K_SUCCESS)
        goto cleanup_venc;

    // 2. Start Camera Stream
    printf("Starting VICAP stream...\n");
    if (kd_mpi_vicap_start_stream(VICAP_DEV_ID_0) != K_SUCCESS) {
        printf("ERROR: Failed to start VICAP stream\n");
        goto cleanup_uvc;
    }

    // 3. Run Main Processing Loop
    sample_video_loop();

    // 4. Stop and Deinitialize System Modules (in reverse order)
    printf("Stopping VICAP stream...\n");
    kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
cleanup_uvc:
    printf("Deinitializing UVC...\n");
    uvc_device_stop();
    uvc_device_deinit();
cleanup_venc:
    printf("Deinitializing VENC...\n");
    kd_mpi_venc_stop_chn(VENC_CH_ID_0);
    kd_mpi_venc_destroy_chn(VENC_CH_ID_0);
    kd_mpi_venc_detach_vb_pool(VENC_CH_ID_0);
    kd_mpi_vb_destory_pool(venc_pool_id);
cleanup_vicap:
    printf("Deinitializing VICAP...\n");
    kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
cleanup_vb:
    printf("Deinitializing VB...\n");
    kd_mpi_vb_exit();
cleanup_none:
    printf("UVC Camera Demo Finished\n");
    return 0;
}
