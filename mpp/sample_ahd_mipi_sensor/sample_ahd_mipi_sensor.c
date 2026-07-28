/* Preview one MIPI sensor and one XS9950 MCM sensor in center-cropped full-height slots. */

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k_connector_comm.h"
#include "k_dewarp_comm.h"
#include "k_module.h"
#include "k_nonai_2d_comm.h"
#include "k_sys_comm.h"
#include "k_vb_comm.h"
#include "k_vicap_comm.h"
#include "k_video_comm.h"
#include "k_vo_comm.h"
#include "kd_display.h"
#include "mpi_dewarp_api.h"
#include "mpi_nonai_2d_api.h"
#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"

#define MIPI_CAPTURE_WIDTH  1920
#define MIPI_CAPTURE_HEIGHT 1080
#define MCM_CAPTURE_WIDTH   1280
#define MCM_CAPTURE_HEIGHT  720
#define VICAP_BUFFER_COUNT  6
#define MODULE_BUFFER_COUNT 6
#define NONAI_2D_CHANNEL    0
#define DEWARP_DEVICE       0
#define MIPI_LAYER           K_VO_LAYER_VIDEO1
#define MCM_LAYER            K_VO_LAYER_VIDEO2
#define DEFAULT_MIPI_CSI    VICAP_DEV_ID_2
#define DEFAULT_MCM_CSI     VICAP_DEV_ID_0
#define SENSOR_FPS          30
#define MCM_SENSOR_PREFIX   "xs9950"
#define CROP_ALIGN_WIDTH    16
#define CROP_ALIGN_HEIGHT   8

typedef struct {
    k_connector_type connector;
    k_vicap_dev mipi_csi;
    k_vicap_dev mcm_csi;
    k_vicap_sensor_info mipi_sensor_info;
    k_vicap_sensor_info mcm_sensor_info;
    k_u32 preview_width;
    k_u32 preview_height;
    k_u32 csc_pool;
    k_u32 dewarp_pool;
    k_bool vb_initialized;
    k_bool display_initialized;
    k_bool csc_pool_attached;
    k_bool csc_channel_created;
    k_bool csc_channel_started;
    k_bool dewarp_initialized;
    k_bool mipi_initialized;
    k_bool mcm_initialized;
    k_bool mipi_bound;
    k_bool mcm_vi_bound;
    k_bool mcm_csc_bound;
    k_bool mcm_dw_bound;
    k_bool mipi_stream_started;
    k_bool mcm_stream_started;
} app_context;

static volatile sig_atomic_t g_exit_requested;

static void signal_handler(int signo)
{
    (void)signo;
    g_exit_requested = 1;
}

static void print_usage(const char *program)
{
    printf("Usage: %s -c connector_type [-m mipi_csi] [-a mcm_csi]\n", program);
    printf("  -c connector_type Display connector type from list_connector (required)\n");
    printf("  -m mipi_csi       Normal MIPI sensor CSI id (0-%d, default: %d)\n",
           VICAP_DEV_ID_MAX - 1, DEFAULT_MIPI_CSI);
    printf("  -a mcm_csi        XS9950 CSI id (0-%d, default: %d)\n",
           VICAP_DEV_ID_MAX - 1, DEFAULT_MCM_CSI);
    printf("  -s csi_id         Alias for -m (MIPI CSI id)\n");
    printf("  -h, --help        Show this help message\n");
    printf("\nThe two CSI ids must be different. Each preview fills its half of the display\n");
    printf("using a centered crop followed by hardware scaling. Press Ctrl+C to stop.\n");
}

static k_s32 parse_id(const char *text, k_vicap_dev *id)
{
    char *end;
    long value;

    if (text == NULL || *text == '\0')
        return K_FAILED;
    value = strtol(text, &end, 10);
    if (*end != '\0' || value < 0 || value >= VICAP_DEV_ID_MAX)
        return K_FAILED;
    *id = (k_vicap_dev)value;
    return K_SUCCESS;
}

static k_s32 parse_connector(const char *text, k_connector_type *connector)
{
    char *end;
    long value;

    if (text == NULL || *text == '\0')
        return K_FAILED;
    value = strtol(text, &end, 10);
    if (*end != '\0' || value < 0)
        return K_FAILED;
    *connector = (k_connector_type)value;
    return K_SUCCESS;
}

static k_s32 parse_options(int argc, char *argv[], app_context *ctx)
{
    k_bool connector_set = K_FALSE;
    int i;

    ctx->mipi_csi = DEFAULT_MIPI_CSI;
    ctx->mcm_csi = DEFAULT_MCM_CSI;
    for (i = 1; i < argc; ++i) {
        const char *option = argv[i];
        k_vicap_dev *csi = NULL;

        if (strcmp(option, "-h") == 0 || strcmp(option, "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (strcmp(option, "-c") == 0) {
            if (++i >= argc || parse_connector(argv[i], &ctx->connector) != K_SUCCESS) {
                printf("ERROR: invalid connector type\n");
                return K_FAILED;
            }
            connector_set = K_TRUE;
            continue;
        }
        if (strcmp(option, "-m") == 0 || strcmp(option, "-s") == 0 ||
            strcmp(option, "--mipi-csi") == 0 || strcmp(option, "-mipi_csi") == 0) {
            csi = &ctx->mipi_csi;
        } else if (strcmp(option, "-a") == 0 || strcmp(option, "--mcm-csi") == 0 ||
                   strcmp(option, "-mcm_csi") == 0) {
            csi = &ctx->mcm_csi;
        } else {
            printf("ERROR: unknown option '%s'\n", option);
            print_usage(argv[0]);
            return K_FAILED;
        }
        if (++i >= argc || parse_id(argv[i], csi) != K_SUCCESS) {
            printf("ERROR: CSI id must be in the range 0-%d\n", VICAP_DEV_ID_MAX - 1);
            return K_FAILED;
        }
    }
    if (!connector_set) {
        printf("ERROR: connector type is required\n");
        print_usage(argv[0]);
        return K_FAILED;
    }
    if (ctx->mipi_csi == ctx->mcm_csi) {
        printf("ERROR: MIPI and MCM sensors cannot use the same CSI id (%d)\n",
               ctx->mipi_csi);
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_bool sensor_name_is_mcm(const char *sensor_name)
{
    return sensor_name != NULL &&
           strncmp(sensor_name, MCM_SENSOR_PREFIX, sizeof(MCM_SENSOR_PREFIX) - 1) == 0;
}

static k_s32 probe_sensor(k_vicap_dev csi, k_u32 width, k_u32 height, k_u32 fps,
                          k_vicap_sensor_info *sensor_info)
{
    k_vicap_probe_config probe_config;
    k_s32 ret;

    memset(&probe_config, 0, sizeof(probe_config));
    probe_config.csi_num = csi;
    probe_config.width = width;
    probe_config.height = height;
    probe_config.fps = fps;

    ret = kd_mpi_sensor_adapt_get(&probe_config, sensor_info);
    if (ret != K_SUCCESS) {
        printf("ERROR: cannot probe a sensor on CSI %d for %ux%u@%u\n",
               csi, width, height, fps);
        return ret;
    }
    if (sensor_info->sensor_name == NULL || sensor_info->width == 0 || sensor_info->height == 0) {
        printf("ERROR: sensor probe returned incomplete information on CSI %d\n", csi);
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 create_pool(k_u32 block_count, k_u64 block_size, k_u32 *pool_id)
{
    k_vb_pool_config config;

    memset(&config, 0, sizeof(config));
    config.blk_cnt = block_count;
    config.blk_size = VICAP_ALIGN_UP(block_size, 0x1000);
    config.mode = VB_REMAP_MODE_NOCACHE;
    *pool_id = kd_mpi_vb_create_pool(&config);
    if (*pool_id == VB_INVALID_POOLID) {
        printf("ERROR: failed to create VB pool (%u blocks, %llu bytes)\n",
               block_count, (unsigned long long)config.blk_size);
        return K_FAILED;
    }
    return K_SUCCESS;
}

static void destroy_pool(k_u32 *pool_id)
{
    if (*pool_id != VB_INVALID_POOLID) {
        kd_mpi_vb_destory_pool(*pool_id);
        *pool_id = VB_INVALID_POOLID;
    }
}

static k_s32 vb_init(app_context *ctx)
{
    k_vb_config config;
    k_vb_supplement_config supplement;
    k_s32 ret;

    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;
    ret = kd_mpi_vb_set_config(&config);
    if (ret != K_SUCCESS)
        return ret;

    memset(&supplement, 0, sizeof(supplement));
    supplement.supplement_config = VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement);
    if (ret != K_SUCCESS)
        return ret;

    ret = kd_mpi_vb_init();
    if (ret == K_SUCCESS)
        ctx->vb_initialized = K_TRUE;
    return ret;
}

static k_gdma_rotation_e display_rotation(k_connector_type connector)
{
    if (K_CONN_WIDTH(connector) != 0 && K_CONN_WIDTH(connector) < K_CONN_HEIGHT(connector))
        return GDMA_ROTATE_DEGREE_90;
    return GDMA_ROTATE_NONE;
}

static k_s32 choose_preview_size(app_context *ctx, k_u32 screen_width, k_u32 screen_height)
{
    /* Use the complete display height; each source is cropped to this slot's aspect ratio. */
    k_u32 width = screen_width / 2;
    k_u32 height = screen_height;

    width &= ~(CROP_ALIGN_WIDTH - 1);
    height &= ~(CROP_ALIGN_HEIGHT - 1);
    if (width == 0 || height == 0)
        return K_FAILED;
    ctx->preview_width = width;
    ctx->preview_height = height;
    return K_SUCCESS;
}

static k_u32 align_down(k_u32 value, k_u32 alignment)
{
    if (alignment <= 1)
        return value;
    return value - value % alignment;
}

static k_u32 centered_crop_start(k_u32 source_start, k_u32 source_size,
                                 k_u32 crop_size, k_u32 alignment)
{
    k_u32 start = source_start + (source_size - crop_size) / 2;
    k_u32 max_start = source_start + source_size - crop_size;

    start = align_down(start, alignment);
    if (start < source_start)
        start = source_start;
    if (start > max_start)
        start = max_start;
    return start;
}

static k_s32 choose_center_crop(const k_vicap_window *source,
                                k_u32 output_width, k_u32 output_height,
                                k_u32 width_alignment, k_u32 height_alignment,
                                k_vicap_window *crop)
{
    k_u32 crop_width;
    k_u32 crop_height;

    if (source == NULL || crop == NULL || source->width == 0 || source->height == 0 ||
        output_width == 0 || output_height == 0)
        return K_FAILED;

    *crop = *source;
    if ((k_u64)source->width * output_height > (k_u64)source->height * output_width) {
        crop_width = (k_u32)((k_u64)source->height * output_width / output_height);
        crop_width = align_down(crop_width, width_alignment);
        if (crop_width == 0 || crop_width > source->width)
            return K_FAILED;
        crop->h_start = (k_u16)centered_crop_start(source->h_start, source->width,
                                                    crop_width, width_alignment);
        crop->width = (k_u16)crop_width;
    } else {
        crop_height = (k_u32)((k_u64)source->width * output_height / output_width);
        crop_height = align_down(crop_height, height_alignment);
        if (crop_height == 0 || crop_height > source->height)
            return K_FAILED;
        crop->v_start = (k_u16)centered_crop_start(source->v_start, source->height,
                                                    crop_height, height_alignment);
        crop->height = (k_u16)crop_height;
    }
    return K_SUCCESS;
}

static k_s32 display_init(app_context *ctx)
{
    k_u32 screen_width;
    k_u32 screen_height;
    k_u32 y_offset;
    k_s32 ret;

    ret = kd_display_init(ctx->connector, 0, 0, display_rotation(ctx->connector));
    if (ret != K_SUCCESS)
        return ret;
    ctx->display_initialized = K_TRUE;

    ret = kd_display_get_resolution(&screen_width, &screen_height);
    if (ret != K_SUCCESS || screen_width == 0 || screen_height == 0) {
        printf("ERROR: cannot get display resolution for connector %d\n", ctx->connector);
        return ret == K_SUCCESS ? K_FAILED : ret;
    }
    ret = choose_preview_size(ctx, screen_width, screen_height);
    if (ret != K_SUCCESS) {
        printf("ERROR: connector resolution is too small for two previews (%ux%u)\n",
               screen_width, screen_height);
        return ret;
    }
    y_offset = (screen_height - ctx->preview_height) / 2;

    ret = kd_display_layer_configure(MIPI_LAYER, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                      ctx->preview_width, ctx->preview_height, 0, y_offset);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_display_layer_enable(MIPI_LAYER);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_display_layer_configure(MCM_LAYER, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                      ctx->preview_width, ctx->preview_height,
                                      ctx->preview_width, y_offset);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_display_layer_enable(MCM_LAYER);
    if (ret == K_SUCCESS) {
        printf("Display: connector=%d, each preview=%ux%u, MIPI x=0, MCM x=%u\n",
               ctx->connector, ctx->preview_width, ctx->preview_height,
               ctx->preview_width);
    }
    return ret;
}

static k_s32 csc_init(app_context *ctx)
{
    k_nonai_2d_chn_attr attr;
    k_s32 ret;

    ret = create_pool(MODULE_BUFFER_COUNT,
                      (k_u64)MCM_CAPTURE_WIDTH * MCM_CAPTURE_HEIGHT * 3,
                      &ctx->csc_pool);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_mpi_nonai_2d_attach_vb_pool(NONAI_2D_CHANNEL, ctx->csc_pool);
    if (ret != K_SUCCESS)
        return ret;
    ctx->csc_pool_attached = K_TRUE;

    memset(&attr, 0, sizeof(attr));
    attr.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr.dst_fmt = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ret = kd_mpi_nonai_2d_create_chn(NONAI_2D_CHANNEL, &attr);
    if (ret != K_SUCCESS)
        return ret;
    ctx->csc_channel_created = K_TRUE;
    ret = kd_mpi_nonai_2d_start_chn(NONAI_2D_CHANNEL);
    if (ret != K_SUCCESS)
        return ret;
    ctx->csc_channel_started = K_TRUE;
    return K_SUCCESS;
}

static void set_common_pipe_control(k_vicap_dev_attr *dev_attr)
{
    dev_attr->pipe_ctrl.data = 0xffffffff;
    dev_attr->pipe_ctrl.bits.af_enable = 0;
    dev_attr->pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr->pipe_ctrl.bits.dnr3_enable = 0;
}

static k_s32 vicap_init_mipi(app_context *ctx)
{
    k_vicap_sensor_info sensor_info;
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_s32 ret;

    memset(&sensor_info, 0, sizeof(sensor_info));
    ret = probe_sensor(ctx->mipi_csi, MIPI_CAPTURE_WIDTH, MIPI_CAPTURE_HEIGHT,
                       SENSOR_FPS, &sensor_info);
    if (ret != K_SUCCESS)
        return ret;
    if (sensor_name_is_mcm(sensor_info.sensor_name)) {
        printf("ERROR: sensor '%s' on CSI %d is an MCM sensor, expected a normal MIPI sensor\n",
               sensor_info.sensor_name, ctx->mipi_csi);
        return K_FAILED;
    }
    ctx->mipi_sensor_info = sensor_info;
    printf("Detected normal MIPI sensor '%s' on CSI %d (%ux%u@%u)\n",
           sensor_info.sensor_name, ctx->mipi_csi, sensor_info.width,
           sensor_info.height, sensor_info.fps);

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width = sensor_info.width;
    dev_attr.acq_win.height = sensor_info.height;
    dev_attr.input_type = VICAP_INPUT_TYPE_SENSOR;
    dev_attr.mode = VICAP_WORK_ONLINE_MODE;
    dev_attr.buffer_num = VICAP_BUFFER_COUNT;
    dev_attr.buffer_size = VICAP_ALIGN_UP(sensor_info.width * sensor_info.height * 2,
                                          VICAP_ALIGN_1K);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    set_common_pipe_control(&dev_attr);
    dev_attr.pipe_ctrl.bits.ae_enable = 1;
    dev_attr.pipe_ctrl.bits.awb_enable = 1;
    memcpy(&dev_attr.sensor_info, &ctx->mipi_sensor_info, sizeof(ctx->mipi_sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(ctx->mipi_csi, dev_attr);
    if (ret != K_SUCCESS)
        return ret;

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width = ctx->preview_width;
    chn_attr.out_win.height = ctx->preview_height;
    ret = choose_center_crop(&dev_attr.acq_win, ctx->preview_width, ctx->preview_height,
                             CROP_ALIGN_WIDTH, CROP_ALIGN_HEIGHT, &chn_attr.crop_win);
    if (ret != K_SUCCESS)
        return ret;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = (chn_attr.crop_win.h_start != dev_attr.acq_win.h_start ||
                            chn_attr.crop_win.v_start != dev_attr.acq_win.v_start ||
                            chn_attr.crop_win.width != dev_attr.acq_win.width ||
                            chn_attr.crop_win.height != dev_attr.acq_win.height);
    chn_attr.scale_enable = (chn_attr.scale_win.width != chn_attr.crop_win.width ||
                             chn_attr.scale_win.height != chn_attr.crop_win.height);
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num = VICAP_BUFFER_COUNT;
    chn_attr.buffer_size = VICAP_ALIGN_UP(ctx->preview_width * ctx->preview_height * 3 / 2,
                                          VICAP_ALIGN_1K);
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;
    ret = kd_mpi_vicap_set_chn_attr(ctx->mipi_csi, VICAP_CHN_ID_0, chn_attr);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_mpi_vicap_set_database_parse_mode(ctx->mipi_csi, VICAP_DATABASE_PARSE_XML_JSON);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_mpi_vicap_init(ctx->mipi_csi);
    if (ret == K_SUCCESS)
        ctx->mipi_initialized = K_TRUE;
    return ret;
}

static k_s32 vicap_init_mcm(app_context *ctx)
{
    k_vicap_sensor_info sensor_info;
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_s32 ret;

    memset(&sensor_info, 0, sizeof(sensor_info));
    ret = probe_sensor(ctx->mcm_csi, MCM_CAPTURE_WIDTH, MCM_CAPTURE_HEIGHT,
                       SENSOR_FPS, &sensor_info);
    if (ret != K_SUCCESS)
        return ret;
    if (!sensor_name_is_mcm(sensor_info.sensor_name)) {
        printf("ERROR: sensor '%s' on CSI %d is not an XS9950 MCM sensor\n",
               sensor_info.sensor_name, ctx->mcm_csi);
        return K_FAILED;
    }
    ctx->mcm_sensor_info = sensor_info;
    printf("Detected MCM sensor '%s' on CSI %d (%ux%u@%u)\n",
           sensor_info.sensor_name, ctx->mcm_csi, sensor_info.width,
           sensor_info.height, sensor_info.fps);

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width = sensor_info.width;
    dev_attr.acq_win.height = sensor_info.height;
    dev_attr.input_type = VICAP_INPUT_TYPE_SENSOR;
    dev_attr.mode = VICAP_WORK_ONLY_MCM_MODE;
    dev_attr.buffer_num = VICAP_BUFFER_COUNT;
    dev_attr.buffer_size = VICAP_ALIGN_UP(sensor_info.width * sensor_info.height * 3,
                                          VICAP_ALIGN_1K);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    set_common_pipe_control(&dev_attr);
    memcpy(&dev_attr.sensor_info, &ctx->mcm_sensor_info, sizeof(ctx->mcm_sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(ctx->mcm_csi, dev_attr);
    if (ret != K_SUCCESS)
        return ret;

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width = MCM_CAPTURE_WIDTH;
    chn_attr.out_win.height = MCM_CAPTURE_HEIGHT;
    chn_attr.crop_win = dev_attr.acq_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_444;
    chn_attr.buffer_num = VICAP_BUFFER_COUNT;
    chn_attr.buffer_size = VICAP_ALIGN_UP(MCM_CAPTURE_WIDTH * MCM_CAPTURE_HEIGHT * 3,
                                          VICAP_ALIGN_1K);
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;
    ret = kd_mpi_vicap_set_chn_attr(ctx->mcm_csi, VICAP_CHN_ID_0, chn_attr);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_mpi_vicap_init(ctx->mcm_csi);
    if (ret == K_SUCCESS)
        ctx->mcm_initialized = K_TRUE;
    return ret;
}

static k_s32 dewarp_init(app_context *ctx)
{
    struct k_dw_settings settings;
    k_s32 ret;

    ret = create_pool(MODULE_BUFFER_COUNT,
                      (k_u64)ctx->preview_width * ctx->preview_height * 3 / 2,
                      &ctx->dewarp_pool);
    if (ret != K_SUCCESS)
        return ret;
    printf("MCM dewarp: input=%ux%u, output=%ux%u\n",
           MCM_CAPTURE_WIDTH, MCM_CAPTURE_HEIGHT,
           ctx->preview_width, ctx->preview_height);
    memset(&settings, 0, sizeof(settings));
    settings.vdev_id = DEWARP_DEVICE;
    settings.input.width = MCM_CAPTURE_WIDTH;
    settings.input.height = MCM_CAPTURE_HEIGHT;
    settings.input.format = K_DW_PIX_YUV420SP;
    settings.output_enable_mask = 1;
    settings.output[0].width = ctx->preview_width;
    settings.output[0].height = ctx->preview_height;
    settings.output[0].format = K_DW_PIX_YUV420SP;
    settings.attach_pool_id = ctx->dewarp_pool;
    ret = kd_mpi_dw_init(&settings);
    if (ret == K_SUCCESS)
        ctx->dewarp_initialized = K_TRUE;
    return ret;
}

static k_mpp_chn mpp_channel(k_mod_id module, k_s32 device, k_s32 channel)
{
    k_mpp_chn result;

    result.mod_id = module;
    result.dev_id = device;
    result.chn_id = channel;
    return result;
}

static k_s32 bind_pipeline(app_context *ctx)
{
    k_mpp_chn mipi = mpp_channel(K_ID_VI, ctx->mipi_csi, VICAP_CHN_ID_0);
    k_mpp_chn mipi_vo = mpp_channel(K_ID_VO, K_VO_DISPLAY_DEV_ID, MIPI_LAYER);
    k_mpp_chn mcm = mpp_channel(K_ID_VI, ctx->mcm_csi, VICAP_CHN_ID_0);
    k_mpp_chn csc = mpp_channel(K_ID_NONAI_2D, 0, NONAI_2D_CHANNEL);
    k_mpp_chn dw = mpp_channel(K_ID_DW200, DEWARP_DEVICE, 0);
    k_mpp_chn mcm_vo = mpp_channel(K_ID_VO, K_VO_DISPLAY_DEV_ID, MCM_LAYER);
    k_s32 ret;

    ret = kd_mpi_sys_bind(&mipi, &mipi_vo);
    if (ret != K_SUCCESS)
        return ret;
    ctx->mipi_bound = K_TRUE;

    ret = kd_mpi_sys_bind(&mcm, &csc);
    if (ret != K_SUCCESS)
        goto fail;
    ctx->mcm_vi_bound = K_TRUE;
    ret = kd_mpi_sys_bind(&csc, &dw);
    if (ret != K_SUCCESS)
        goto fail;
    ctx->mcm_csc_bound = K_TRUE;
    ret = kd_mpi_sys_bind(&dw, &mcm_vo);
    if (ret != K_SUCCESS)
        goto fail;
    ctx->mcm_dw_bound = K_TRUE;
    return K_SUCCESS;

fail:
    if (ctx->mcm_dw_bound)
        kd_mpi_sys_unbind(&dw, &mcm_vo);
    if (ctx->mcm_csc_bound)
        kd_mpi_sys_unbind(&csc, &dw);
    if (ctx->mcm_vi_bound)
        kd_mpi_sys_unbind(&mcm, &csc);
    if (ctx->mipi_bound)
        kd_mpi_sys_unbind(&mipi, &mipi_vo);
    ctx->mcm_dw_bound = K_FALSE;
    ctx->mcm_csc_bound = K_FALSE;
    ctx->mcm_vi_bound = K_FALSE;
    ctx->mipi_bound = K_FALSE;
    return ret;
}

static void unbind_pipeline(app_context *ctx)
{
    k_mpp_chn mipi = mpp_channel(K_ID_VI, ctx->mipi_csi, VICAP_CHN_ID_0);
    k_mpp_chn mipi_vo = mpp_channel(K_ID_VO, K_VO_DISPLAY_DEV_ID, MIPI_LAYER);
    k_mpp_chn mcm = mpp_channel(K_ID_VI, ctx->mcm_csi, VICAP_CHN_ID_0);
    k_mpp_chn csc = mpp_channel(K_ID_NONAI_2D, 0, NONAI_2D_CHANNEL);
    k_mpp_chn dw = mpp_channel(K_ID_DW200, DEWARP_DEVICE, 0);
    k_mpp_chn mcm_vo = mpp_channel(K_ID_VO, K_VO_DISPLAY_DEV_ID, MCM_LAYER);

    if (ctx->mcm_dw_bound) {
        kd_mpi_sys_unbind(&dw, &mcm_vo);
        ctx->mcm_dw_bound = K_FALSE;
    }
    if (ctx->mcm_csc_bound) {
        kd_mpi_sys_unbind(&csc, &dw);
        ctx->mcm_csc_bound = K_FALSE;
    }
    if (ctx->mcm_vi_bound) {
        kd_mpi_sys_unbind(&mcm, &csc);
        ctx->mcm_vi_bound = K_FALSE;
    }
    if (ctx->mipi_bound) {
        kd_mpi_sys_unbind(&mipi, &mipi_vo);
        ctx->mipi_bound = K_FALSE;
    }
}

static void cleanup(app_context *ctx)
{
    if (ctx->mcm_stream_started) {
        kd_mpi_vicap_stop_stream(ctx->mcm_csi);
        ctx->mcm_stream_started = K_FALSE;
    }
    if (ctx->mipi_stream_started) {
        kd_mpi_vicap_stop_stream(ctx->mipi_csi);
        ctx->mipi_stream_started = K_FALSE;
    }
    if (ctx->mipi_bound || ctx->mcm_vi_bound || ctx->mcm_csc_bound || ctx->mcm_dw_bound)
        unbind_pipeline(ctx);
    if (ctx->dewarp_initialized) {
        kd_mpi_dw_exit(DEWARP_DEVICE);
        ctx->dewarp_initialized = K_FALSE;
    }
    if (ctx->mcm_initialized) {
        kd_mpi_vicap_deinit(ctx->mcm_csi);
        ctx->mcm_initialized = K_FALSE;
    }
    if (ctx->mipi_initialized) {
        kd_mpi_vicap_deinit(ctx->mipi_csi);
        ctx->mipi_initialized = K_FALSE;
    }
    if (ctx->csc_channel_started) {
        kd_mpi_nonai_2d_stop_chn(NONAI_2D_CHANNEL);
        ctx->csc_channel_started = K_FALSE;
    }
    if (ctx->csc_channel_created) {
        kd_mpi_nonai_2d_destroy_chn(NONAI_2D_CHANNEL);
        ctx->csc_channel_created = K_FALSE;
    }
    if (ctx->csc_pool_attached) {
        kd_mpi_nonai_2d_detach_vb_pool(NONAI_2D_CHANNEL);
        ctx->csc_pool_attached = K_FALSE;
    }
    destroy_pool(&ctx->dewarp_pool);
    destroy_pool(&ctx->csc_pool);
    kd_mpi_nonai_2d_close();
    if (ctx->display_initialized) {
        kd_display_layer_disable(MIPI_LAYER);
        kd_display_layer_disable(MCM_LAYER);
        kd_display_deinit();
        ctx->display_initialized = K_FALSE;
    }
    if (ctx->vb_initialized) {
        kd_mpi_vb_exit();
        ctx->vb_initialized = K_FALSE;
    }
}

static void context_init(app_context *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->csc_pool = VB_INVALID_POOLID;
    ctx->dewarp_pool = VB_INVALID_POOLID;
}

int main(int argc, char *argv[])
{
    app_context ctx;
    k_s32 ret;

    context_init(&ctx);
    ret = parse_options(argc, argv, &ctx);
    if (ret != K_SUCCESS)
        return ret > 0 ? 0 : 1;
    printf("Using MIPI CSI %d and MCM CSI %d\n", ctx.mipi_csi, ctx.mcm_csi);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    ret = vb_init(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = display_init(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = csc_init(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = vicap_init_mipi(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = vicap_init_mcm(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    /* VICAP initialization resets DW200, so create DW after both devices. */
    ret = dewarp_init(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = bind_pipeline(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = kd_mpi_vicap_start_stream(ctx.mipi_csi);
    if (ret != K_SUCCESS)
        goto done;
    ctx.mipi_stream_started = K_TRUE;
    ret = kd_mpi_vicap_start_stream(ctx.mcm_csi);
    if (ret != K_SUCCESS)
        goto done;
    ctx.mcm_stream_started = K_TRUE;

    printf("MIPI + MCM preview running; press Ctrl+C to stop.\n");
    while (!g_exit_requested)
        usleep(100 * 1000);
    ret = K_SUCCESS;

done:
    if (ret != K_SUCCESS)
        printf("ERROR: MIPI + MCM sample failed, ret=%d\n", ret);
    cleanup(&ctx);
    return ret == K_SUCCESS ? 0 : 1;
}
