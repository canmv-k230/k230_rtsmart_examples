#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <getopt.h>
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
#define DEFAULT_OUTPUT_WIDTH  (1920)
#define DEFAULT_OUTPUT_HEIGHT (1080)
#define PREVIEW_CHN         VICAP_CHN_ID_1
#define DEFAULT_DUMP_FRAMES (10)
#define OUTPUT_FILENAME     "/data/wbc_dump_out.yuv"
#define WBC_DUMP_TIMEOUT_MS (50)
#define WBC_MAX_RETRIES     (100)
#define APP_INTERRUPTED     (130)

// Global flag to control the main loop
static volatile sig_atomic_t g_app_run = 1;
static k_vicap_dev vicap_dev_id = CONFIG_MPP_SENSOR_DEFAULT_CSI;
static k_vicap_mipi_lane_pref g_lane_pref = VICAP_MIPI_LANE_PREF_ANY;
static k_connector_type g_connector_type = VIRTUAL_DISPLAY_DEVICE;
static k_gdma_rotation_e g_rotation = GDMA_ROTATE_DEGREE_0;
static k_u32 g_output_width;
static k_u32 g_output_height;
static k_u32 g_output_x;
static k_u32 g_output_y;
static int g_rotation_degrees;
static int g_dump_frame_count = DEFAULT_DUMP_FRAMES;

// ============================================================================
// NV12 Saving Function
// ============================================================================

typedef struct {
    k_u8  y_min;
    k_u8  y_max;
    k_u64 y_sum;
    k_u64 hash;
} wbc_frame_stats;

static void update_hash(k_u64* hash, const k_u8* data, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        *hash ^= data[i];
        *hash *= UINT64_C(1099511628211);
    }
}

static void calculate_frame_stats(const k_u8* y, size_t ysize, const k_u8* uv, size_t uvsize,
                                  wbc_frame_stats* stats)
{
    stats->y_min = UINT8_MAX;
    stats->y_max = 0;
    stats->y_sum = 0;
    stats->hash  = UINT64_C(14695981039346656037);

    for (size_t i = 0; i < ysize; ++i) {
        if (y[i] < stats->y_min)
            stats->y_min = y[i];
        if (y[i] > stats->y_max)
            stats->y_max = y[i];
        stats->y_sum += y[i];
    }

    update_hash(&stats->hash, y, ysize);
    update_hash(&stats->hash, uv, uvsize);
}

static int save_wbc_nv12_stream(FILE* fp, const k_video_frame_info* frame_info, bool collect_stats,
                                wbc_frame_stats* stats)
{
    size_t ysize    = (size_t)frame_info->v_frame.width * frame_info->v_frame.height;
    size_t uvsize   = ysize / 2;
    k_u64  y_paddr  = frame_info->v_frame.phys_addr[0];
    k_u64  uv_paddr = frame_info->v_frame.phys_addr[1];
    int    ret      = -1;

    if (y_paddr == 0 || uv_paddr == 0) {
        printf("Invalid physical address for Y/UV planes.\n");
        return -1;
    }

    void* y_vaddr  = kd_mpi_sys_mmap_cached(y_paddr, ysize);
    void* uv_vaddr = kd_mpi_sys_mmap_cached(uv_paddr, uvsize);

    if (y_vaddr == NULL || uv_vaddr == NULL) {
        printf("Failed to map frame buffers.\n");
        goto cleanup;
    }

    if (collect_stats)
        calculate_frame_stats(y_vaddr, ysize, uv_vaddr, uvsize, stats);

    if (fwrite(y_vaddr, 1, ysize, fp) != ysize) {
        printf("Error writing Y plane: errno=%d.\n", errno);
        goto cleanup;
    }

    if (fwrite(uv_vaddr, 1, uvsize, fp) != uvsize) {
        printf("Error writing UV plane: errno=%d.\n", errno);
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (y_vaddr != NULL)
        kd_mpi_sys_munmap(y_vaddr, ysize);
    if (uv_vaddr != NULL)
        kd_mpi_sys_munmap(uv_vaddr, uvsize);
    return ret;
}

// ============================================================================
// Core Logic: WBC Frame Dumping
// ============================================================================

/**
 * @brief Configures and enables VO Write Back Controller (WBC) and dumps frames.
 * @param frame_count The number of frames to dump.
 */
static int sample_vo_wbc_dump(int frame_count)
{
    FILE* fp                    = NULL;
    int   dumped_frames         = 0;
    int   consecutive_failures  = 0;
    int   ret                   = EXIT_FAILURE;
    bool  wbc_enabled           = false;

    if (kd_display_wbc_configure(3)) {
        printf("kd_mpi_vo_set_wbc_attr error\n");
        return EXIT_FAILURE;
    }

    if (kd_display_wbc_enable()) {
        printf("kd_mpi_vo_enable_wbc error\n");
        return EXIT_FAILURE;
    }
    wbc_enabled = true;

    printf("WBC enabled. Dumping %d frames to %s...\n", frame_count, OUTPUT_FILENAME);

    fp = fopen(OUTPUT_FILENAME, "wb");
    if (!fp) {
        printf("Failed to open output file: %s, errno=%d\n", OUTPUT_FILENAME, errno);
        goto cleanup;
    }

    k_video_frame_info frame_info;
    while (dumped_frames < frame_count && g_app_run) {
        k_s32 error = kd_display_wbc_dump_frame(&frame_info, WBC_DUMP_TIMEOUT_MS);
        if (error) {
            if (!g_app_run)
                break;

            ++consecutive_failures;
            printf("kd_mpi_wbc_dump_frame error: %d. Retry %d/%d.\n", error, consecutive_failures,
                   WBC_MAX_RETRIES);
            if (consecutive_failures >= WBC_MAX_RETRIES)
                goto cleanup;
            continue;
        }
        consecutive_failures = 0;

        printf("Dumped frame %d (%dx%d). Writing to file...\n", dumped_frames, frame_info.v_frame.width,
               frame_info.v_frame.height);

        bool            collect_stats = dumped_frames == 0 || dumped_frames == frame_count / 2 ||
                             dumped_frames == frame_count - 1;
        wbc_frame_stats stats;
        int             write_ret = save_wbc_nv12_stream(fp, &frame_info, collect_stats, &stats);
        k_s32           release_ret = kd_display_wbc_release_frame(&frame_info);

        if (release_ret != K_SUCCESS) {
            printf("wbc dump release failed for frame %d, ret=%d\n", dumped_frames, release_ret);
            goto cleanup;
        }

        if (write_ret != 0)
            goto cleanup;

        if (collect_stats) {
            k_u64 y_pixels = (k_u64)frame_info.v_frame.width * frame_info.v_frame.height;
            printf("Frame %d content: y_min=%u y_max=%u y_mean=%llu fnv1a64=%016llx\n", dumped_frames,
                   stats.y_min, stats.y_max, (unsigned long long)(stats.y_sum / y_pixels),
                   (unsigned long long)stats.hash);
        }

        ++dumped_frames;
    }

    ret = g_app_run ? EXIT_SUCCESS : APP_INTERRUPTED;

cleanup:
    if (fp != NULL) {
        if (fclose(fp) != 0) {
            printf("Failed to close output file: errno=%d\n", errno);
            ret = EXIT_FAILURE;
        }
        fp = NULL;
    }

    if (wbc_enabled && kd_display_wbc_disable() != K_SUCCESS) {
        printf("kd_mpi_vo_disable_wbc error\n");
        ret = EXIT_FAILURE;
    }

    if (ret == EXIT_SUCCESS) {
        printf("Successfully finished dumping %d frames to %s.\n", frame_count, OUTPUT_FILENAME);
    } else if (ret == APP_INTERRUPTED) {
        printf("Interrupted after dumping %d of %d frames to %s.\n", dumped_frames, frame_count, OUTPUT_FILENAME);
    } else {
        printf("Failed after dumping %d of %d frames to %s.\n", dumped_frames, frame_count, OUTPUT_FILENAME);
    }

    return ret;
}

// ============================================================================
// Argument Parsing (Updated for completeness)
// ============================================================================

static void print_usage(const char* program_name)
{
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -c, --connector <type> Connector type from list_connector (default: virtual)\n");
    printf("  -s, --csi <num>        CSI device number (0-2, default: %d)\n",
           CONFIG_MPP_SENSOR_DEFAULT_CSI);
    printf("  -w, --width <pixels>   Output width (default: panel width)\n");
    printf("  -h, --height <pixels>  Output height (default: panel height)\n");
    printf("  -r, --rotation <deg>   Display rotation: 0, 90, 180, or 270 (default: 0)\n");
    printf("  -n, --frames <num>     Number of frames to dump (default: %d)\n", DEFAULT_DUMP_FRAMES);
    printf("  -L, --lane <2|4>       MIPI lane preference (default: ANY)\n");
    printf("  -H, --help             Show this help message\n");
    printf("\nExample for ST7701 480x800 in landscape:\n");
    printf("  %s -c %u -s 0 -r 90 -n 10\n", program_name, ST7701_V1_MIPI_2LAN_480X800_30FPS);
}

static int parse_u32(const char* text, k_u32* value)
{
    char*         end;
    unsigned long parsed;

    errno  = 0;
    parsed = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX)
        return -1;

    *value = (k_u32)parsed;
    return 0;
}

static k_gdma_rotation_e rotation_from_degrees(int degrees)
{
    switch (degrees) {
    case 90:
        return GDMA_ROTATE_DEGREE_90;
    case 180:
        return GDMA_ROTATE_DEGREE_180;
    case 270:
        return GDMA_ROTATE_DEGREE_270;
    default:
        return GDMA_ROTATE_DEGREE_0;
    }
}

static bool rotation_swaps_axes(int degrees)
{
    return degrees == 90 || degrees == 270;
}

static int parse_arguments(int argc, char** argv)
{
    static const struct option long_options[] = {
        { "connector", required_argument, NULL, 'c' },
        { "csi", required_argument, NULL, 's' },
        { "width", required_argument, NULL, 'w' },
        { "height", required_argument, NULL, 'h' },
        { "rotation", required_argument, NULL, 'r' },
        { "frames", required_argument, NULL, 'n' },
        { "lane", required_argument, NULL, 'L' },
        { "help", no_argument, NULL, 'H' },
        { NULL, 0, NULL, 0 },
    };
    int opt;
    int csi_num = CONFIG_MPP_SENSOR_DEFAULT_CSI;

    while ((opt = getopt_long(argc, argv, "c:s:w:h:r:n:L:H", long_options, NULL)) != -1) {
        switch (opt) {
        case 'c':
            if (parse_u32(optarg, &g_connector_type) != 0) {
                printf("ERROR: invalid connector type: %s\n", optarg);
                return -1;
            }
            break;
        case 's':
            csi_num = atoi(optarg);
            if (csi_num < 0 || csi_num > 2) {
                printf("ERROR: CSI number must be between 0 and 2\n");
                return -1;
            }
            break;
        case 'w':
            if (parse_u32(optarg, &g_output_width) != 0 || g_output_width == 0) {
                printf("ERROR: output width must be a positive integer\n");
                return -1;
            }
            break;
        case 'h':
            if (parse_u32(optarg, &g_output_height) != 0 || g_output_height == 0) {
                printf("ERROR: output height must be a positive integer\n");
                return -1;
            }
            break;
        case 'r':
            g_rotation_degrees = atoi(optarg);
            if (g_rotation_degrees != 0 && g_rotation_degrees != 90 &&
                g_rotation_degrees != 180 && g_rotation_degrees != 270) {
                printf("ERROR: rotation must be 0, 90, 180, or 270\n");
                return -1;
            }
            g_rotation = rotation_from_degrees(g_rotation_degrees);
            break;
        case 'n':
            g_dump_frame_count = atoi(optarg);
            if (g_dump_frame_count <= 0) {
                printf("ERROR: Frame count must be a positive integer\n");
                return -1;
            }
            break;
        case 'L': {
            int lane = atoi(optarg);
            if (lane == 2)
                g_lane_pref = VICAP_MIPI_LANE_PREF_2LANE;
            else if (lane == 4)
                g_lane_pref = VICAP_MIPI_LANE_PREF_4LANE;
            else {
                printf("ERROR: -L must be 2 or 4\n");
                return -1;
            }
            break;
        }
        case 'H':
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

static int resolve_output_configuration(void)
{
    k_connector_info info;
    k_u32 panel_width = 0;
    k_u32 panel_height = 0;
    k_u32 logical_width;
    k_u32 logical_height;

    if ((g_output_width == 0) != (g_output_height == 0)) {
        printf("ERROR: output width and height must be specified together\n");
        return -1;
    }

    if (g_connector_type == VIRTUAL_DISPLAY_DEVICE) {
        if (g_rotation_degrees != 0) {
            printf("ERROR: rotation is not supported by the virtual display\n");
            return -1;
        }
        if (g_output_width == 0) {
            g_output_width = DEFAULT_OUTPUT_WIDTH;
            g_output_height = DEFAULT_OUTPUT_HEIGHT;
        }
        panel_width = g_output_width;
        panel_height = g_output_height;
    } else {
        memset(&info, 0, sizeof(info));
        if (kd_mpi_get_connector_info(g_connector_type, &info) == K_SUCCESS) {
            panel_width = info.resolution.hactive;
            panel_height = info.resolution.vactive;
        }
        if (panel_width == 0)
            panel_width = K_CONN_WIDTH(g_connector_type);
        if (panel_height == 0)
            panel_height = K_CONN_HEIGHT(g_connector_type);
        if (panel_width == 0 || panel_height == 0) {
            printf("ERROR: cannot resolve connector %u dimensions\n", g_connector_type);
            return -1;
        }
    }

    logical_width = panel_width;
    logical_height = panel_height;
    if (rotation_swaps_axes(g_rotation_degrees)) {
        logical_width = panel_height;
        logical_height = panel_width;
    }

    if (g_output_width == 0) {
        g_output_width = logical_width;
        g_output_height = logical_height;
    }
    if ((g_output_width & 7U) != 0 || (g_output_height & 1U) != 0) {
        printf("ERROR: output width must be a multiple of 8 and height must be even\n");
        return -1;
    }
    if (g_output_width > logical_width || g_output_height > logical_height) {
        printf("ERROR: output %ux%u exceeds logical display %ux%u\n", g_output_width,
               g_output_height, logical_width, logical_height);
        return -1;
    }

    g_output_x = (logical_width - g_output_width) / 2;
    g_output_y = (logical_height - g_output_height) / 2;
    printf("Configuration: connector=%u panel=%ux%u logical=%ux%u output=%ux%u at (%u,%u), "
           "rotation=%d, CSI=%d, frames=%d\n",
           g_connector_type, panel_width, panel_height, logical_width, logical_height,
           g_output_width, g_output_height, g_output_x, g_output_y, g_rotation_degrees,
           vicap_dev_id, g_dump_frame_count);
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
    memset(&probe_cfg, 0, sizeof(probe_cfg));
    probe_cfg.csi_num = dev_chn;
    probe_cfg.width   = ISP_WIDTH;
    probe_cfg.height  = ISP_HEIGHT;
    probe_cfg.fps     = 30;

    if (0x00 != kd_mpi_sensor_adapt_get_ex(&probe_cfg, &sensor_info, g_lane_pref)) {
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
    dev_attr.mode           = VICAP_WORK_ONLINE_MODE;
    dev_attr.buffer_num     = 6;
    dev_attr.buffer_size    = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 2, 1024);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    dev_attr.pipe_ctrl.data = UINT32_MAX;
    dev_attr.pipe_ctrl.bits.ae_enable   = K_TRUE;
    dev_attr.pipe_ctrl.bits.awb_enable  = K_TRUE;
    dev_attr.pipe_ctrl.bits.ahdr_enable = K_FALSE;
    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(VICAP_DEV_ID_0, dev_attr);
    if (ret) {
        printf("kd_mpi_vicap_set_dev_attr failed, ret=%d", ret);
        return ret;
    }

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = dev_attr.acq_win.width;
    chn_attr.out_win.height = dev_attr.acq_win.height;
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_FALSE;
    chn_attr.scale_enable   = K_FALSE;
    chn_attr.chn_enable     = K_TRUE;
    chn_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num     = 6;
    chn_attr.buffer_size    = VB_ALIGN_UP(ISP_WIDTH * ISP_HEIGHT * 3 / 2, 4096);
    chn_attr.alignment      = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, VICAP_CHN_ID_0, chn_attr);
    if (ret) {
        printf("kd_mpi_vicap_set_chn_attr CHN0 failed, ret=%d\n", ret);
        return ret;
    }

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width  = g_output_width;
    chn_attr.out_win.height = g_output_height;
    chn_attr.crop_win       = dev_attr.acq_win;
    chn_attr.scale_win      = chn_attr.out_win;
    chn_attr.crop_enable    = K_FALSE;
    chn_attr.scale_enable   = g_output_width != ISP_WIDTH || g_output_height != ISP_HEIGHT;
    chn_attr.chn_enable     = K_TRUE;
    chn_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num     = 6;
    chn_attr.buffer_size    = VB_ALIGN_UP(g_output_width * g_output_height * 3 / 2, 4096);
    chn_attr.alignment      = 12;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(VICAP_DEV_ID_0, PREVIEW_CHN, chn_attr);
    if (ret) {
        printf("kd_mpi_vicap_set_chn_attr CHN1 failed, ret=%d\n", ret);
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
static k_s32 sample_vo_init(k_connector_type type)
{
    const k_vo_layer_id target_layer  = K_VO_LAYER_VIDEO2;
    const k_u32         target_width  = g_output_width;
    const k_u32         target_height = g_output_height;
    k_u32               display_width;
    k_u32               display_height;

    if (0x00 != kd_display_init(type, target_width, target_height, g_rotation)) {
        printf("ERROR: sample_vo_init, the connector type not supported!\n");
        return -1;
    }

    if (kd_display_get_resolution(&display_width, &display_height) != K_SUCCESS) {
        printf("ERROR: failed to query display resolution\n");
        return -1;
    }
    if (g_output_x + target_width > display_width || g_output_y + target_height > display_height) {
        printf("ERROR: output %ux%u at (%u,%u) exceeds display %ux%u\n", target_width,
               target_height, g_output_x, g_output_y, display_width, display_height);
        return -1;
    }

    printf("Configuring VO Layer %d at (%u,%u), display=%ux%u...\n", target_layer,
           g_output_x, g_output_y, display_width, display_height);

    // Check width constraint (integral multiple of 8)
    if (target_width == 0 || target_height == 0 || (target_width & 7) != 0) {
        printf("ERROR: width (%u) must be an integral multiple of 8 pixels and not zero.\n", target_width);
        return -1;
    }

    kd_display_layer_disable(target_layer);

    if (0x00 != kd_display_layer_configure(target_layer, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                           target_width, target_height, g_output_x, g_output_y)) {
        printf("ERROR: sample_vo_init, configure layer failed!\n");
        return -1;
    }

    if (0x00 != kd_display_layer_enable(target_layer)) {
        printf("ERROR: sample_vo_init, kd_display_layer_enable failed\n");
        return -1;
    }

    printf("VO Layer %d configured and enabled successfully.\n", target_layer);
    return K_SUCCESS;
}

static k_s32 sample_wait_vicap_frame(void)
{
    k_video_frame_info frame;
    k_s32 ret;

    memset(&frame, 0, sizeof(frame));
    ret = kd_mpi_vicap_dump_frame(VICAP_DEV_ID_0, VICAP_CHN_ID_0, VICAP_DUMP_YUV,
                                  &frame, 3000);
    if (ret != K_SUCCESS) {
        printf("ERROR: VICAP did not produce a frame within 3000 ms, ret=%d\n", ret);
        return ret;
    }

    printf("VICAP ready: %ux%u format=%d phys=0x%llx\n", frame.v_frame.width,
           frame.v_frame.height, frame.v_frame.pixel_format,
           (unsigned long long)frame.v_frame.phys_addr[0]);
    ret = kd_mpi_vicap_dump_release(VICAP_DEV_ID_0, VICAP_CHN_ID_0, &frame);
    if (ret != K_SUCCESS)
        printf("ERROR: failed to release VICAP readiness frame, ret=%d\n", ret);
    return ret;
}

static k_s32 sample_vicap_bind_vo(void)
{
    k_mpp_chn vi_mpp_chn;
    k_mpp_chn vo_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vi_mpp_chn.chn_id = PREVIEW_CHN;

    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = K_VO_LAYER_VIDEO2;

    return kd_mpi_sys_bind(&vi_mpp_chn, &vo_mpp_chn);
}

static k_s32 sample_vicap_unbind_vo(void)
{
    k_mpp_chn vvi_mpp_chn;
    k_mpp_chn vvo_mpp_chn;

    vvi_mpp_chn.mod_id = K_ID_VI;
    vvi_mpp_chn.dev_id = VICAP_DEV_ID_0;
    vvi_mpp_chn.chn_id = PREVIEW_CHN;

    vvo_mpp_chn.mod_id = K_ID_VO;
    vvo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vvo_mpp_chn.chn_id = K_VO_LAYER_VIDEO2;

    return kd_mpi_sys_unbind(&vvi_mpp_chn, &vvo_mpp_chn);
}

// ============================================================================
// Main Function (Final)
// ============================================================================

static void handle_signal(int sig)
{
    if (sig == SIGINT)
        g_app_run = 0;
}

int main(int argc, char** argv)
{
    int  exit_status       = EXIT_FAILURE;
    bool vb_initialized    = false;
    bool vicap_initialized = false;
    bool stream_started    = false;
    bool display_attempted = false;
    bool vicap_bound       = false;

    printf("WBC Frame Dump Demo Started\n");

    if (parse_arguments(argc, argv) != 0) {
        return -1;
    }
    if (resolve_output_configuration() != 0)
        return -1;

    signal(SIGINT, handle_signal);

    if (sample_vb_init() != K_SUCCESS)
        goto cleanup;
    vb_initialized = true;

    if (sample_vicap_init(vicap_dev_id) != K_SUCCESS)
        goto cleanup;
    vicap_initialized = true;

    display_attempted = true;
    if (sample_vo_init(g_connector_type) != K_SUCCESS) {
        printf("ERROR: Failed to start video output\n");
        goto cleanup;
    }

    if (sample_vicap_bind_vo() != K_SUCCESS) {
        printf("ERROR: Failed to bind VICAP to VO\n");
        goto cleanup;
    }
    vicap_bound = true;

    printf("Starting VICAP stream...\n");
    if (kd_mpi_vicap_start_stream(VICAP_DEV_ID_0) != K_SUCCESS) {
        printf("ERROR: Failed to start VICAP stream\n");
        goto cleanup;
    }
    stream_started = true;

    if (sample_wait_vicap_frame() != K_SUCCESS)
        goto cleanup;

    exit_status = sample_vo_wbc_dump(g_dump_frame_count);

cleanup:
    if (stream_started) {
        printf("Stopping VICAP stream...\n");
        if (kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0) != K_SUCCESS)
            exit_status = EXIT_FAILURE;
    }

    if (vicap_bound && sample_vicap_unbind_vo() != K_SUCCESS) {
        printf("ERROR: Failed to unbind VICAP from VO\n");
        exit_status = EXIT_FAILURE;
    }

    if (display_attempted && kd_display_deinit() != K_SUCCESS)
        exit_status = EXIT_FAILURE;

    if (vicap_initialized) {
        printf("Deinitializing VICAP...\n");
        if (kd_mpi_vicap_deinit(VICAP_DEV_ID_0) != K_SUCCESS)
            exit_status = EXIT_FAILURE;
    }

    if (vb_initialized) {
        printf("Deinitializing VB...\n");
        if (kd_mpi_vb_exit() != K_SUCCESS)
            exit_status = EXIT_FAILURE;
    }

    printf("WBC Frame Dump Demo Finished\n");
    return exit_status;
}
