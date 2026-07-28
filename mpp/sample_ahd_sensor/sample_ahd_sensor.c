/* One-channel XS9950 AHD capture and display sample. */

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

#define CAPTURE_WIDTH       1280
#define CAPTURE_HEIGHT      720
#define VICAP_BUFFER_COUNT  10
#define MODULE_BUFFER_COUNT 4
#define NONAI_2D_CHANNEL    0
#define DEWARP_DEVICE       0
#define VO_LAYER             K_VO_LAYER_VIDEO1
#define SENSOR_FPS          30
#define MCM_SENSOR_PREFIX   "xs9950"

typedef struct {
    k_connector_type connector;
    k_vicap_dev csi;
    k_vicap_sensor_info sensor_info;
    k_u32 screen_width;
    k_u32 screen_height;
    k_u32 csc_pool;
    k_u32 dewarp_pool;
    k_bool vb_initialized;
    k_bool display_initialized;
    k_bool csc_pool_attached;
    k_bool csc_channel_created;
    k_bool csc_channel_started;
    k_bool dewarp_initialized;
    k_bool vicap_initialized;
    k_bool pipeline_bound;
    k_bool stream_started;
} app_context;

static volatile sig_atomic_t g_exit_requested;

static void signal_handler(int signo)
{
    (void)signo;
    g_exit_requested = 1;
}

static void print_usage(const char *program)
{
    printf("Usage: %s -c connector_type [-s csi_id]\n", program);
    printf("  -c connector_type Display connector type from list_connector (required)\n");
    printf("  -s csi_id         XS9950 CSI id (0-%d, default: 0)\n", VICAP_DEV_ID_MAX - 1);
    printf("  -h                 Show this help message\n");
    printf("\nPreview one XS9950 1280x720 input. Press Ctrl+C to stop.\n");
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

static k_s32 parse_options(int argc, char *argv[], app_context *ctx)
{
    k_bool connector_set = K_FALSE;
    int option;

    ctx->csi = VICAP_DEV_ID_0;
    optind = 1;
    while ((option = getopt(argc, argv, "c:hs:")) != -1) {
        switch (option) {
        case 'c': {
            char *end;
            long value = strtol(optarg, &end, 10);

            if (*optarg == '\0' || *end != '\0' || value < 0) {
                printf("ERROR: invalid connector type '%s'\n", optarg);
                return K_FAILED;
            }
            ctx->connector = (k_connector_type)value;
            connector_set = K_TRUE;
            break;
        }
        case 's':
            if (parse_id(optarg, &ctx->csi) != K_SUCCESS) {
                printf("ERROR: CSI id must be in the range 0-%d\n", VICAP_DEV_ID_MAX - 1);
                return K_FAILED;
            }
            break;
        case 'h':
            print_usage(argv[0]);
            return 1;
        default:
            print_usage(argv[0]);
            return K_FAILED;
        }
    }
    if (optind != argc) {
        print_usage(argv[0]);
        return K_FAILED;
    }
    if (!connector_set) {
        printf("ERROR: connector type is required\n");
        print_usage(argv[0]);
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

static k_s32 display_init(app_context *ctx)
{
    k_u32 screen_width;
    k_u32 screen_height;
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
    ctx->screen_width = screen_width;
    ctx->screen_height = screen_height;

    ret = kd_display_layer_configure(VO_LAYER, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                     ctx->screen_width, ctx->screen_height, 0, 0);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_display_layer_enable(VO_LAYER);
    if (ret == K_SUCCESS)
        printf("Display: connector=%d, layer=%d, size=%ux%u\n", ctx->connector,
               VO_LAYER, ctx->screen_width, ctx->screen_height);
    return ret;
}

static k_s32 csc_init(app_context *ctx)
{
    k_nonai_2d_chn_attr attr;
    k_s32 ret;

    ret = create_pool(MODULE_BUFFER_COUNT,
                      (k_u64)CAPTURE_WIDTH * CAPTURE_HEIGHT * 3,
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

static k_s32 vicap_init(app_context *ctx)
{
    k_vicap_sensor_info sensor_info;
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_s32 ret;

    memset(&sensor_info, 0, sizeof(sensor_info));
    ret = probe_sensor(ctx->csi, CAPTURE_WIDTH, CAPTURE_HEIGHT, SENSOR_FPS, &sensor_info);
    if (ret != K_SUCCESS) {
        return ret;
    }
    if (!sensor_name_is_mcm(sensor_info.sensor_name)) {
        printf("ERROR: sensor '%s' on CSI %d is not an XS9950 MCM sensor\n",
               sensor_info.sensor_name, ctx->csi);
        return K_FAILED;
    }
    ctx->sensor_info = sensor_info;
    printf("Detected MCM sensor '%s' on CSI %d (%ux%u@%u)\n",
           sensor_info.sensor_name, ctx->csi, sensor_info.width,
           sensor_info.height, sensor_info.fps);

    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.width = sensor_info.width;
    dev_attr.acq_win.height = sensor_info.height;
    dev_attr.input_type = VICAP_INPUT_TYPE_SENSOR;
    dev_attr.mode = VICAP_WORK_ONLY_MCM_MODE;
    dev_attr.buffer_num = VICAP_BUFFER_COUNT;
    dev_attr.buffer_size = VICAP_ALIGN_UP(CAPTURE_WIDTH * CAPTURE_HEIGHT * 3,
                                          VICAP_ALIGN_1K);
    dev_attr.buffer_pool_id = VB_INVALID_POOLID;
    dev_attr.pipe_ctrl.data = 0xffffffff;
    dev_attr.pipe_ctrl.bits.af_enable = 0;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.dw_enable = K_FALSE;
    memcpy(&dev_attr.sensor_info, &ctx->sensor_info, sizeof(ctx->sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(ctx->csi, dev_attr);
    if (ret != K_SUCCESS)
        return ret;

    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.width = CAPTURE_WIDTH;
    chn_attr.out_win.height = CAPTURE_HEIGHT;
    chn_attr.crop_win = dev_attr.acq_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_444;
    chn_attr.buffer_num = VICAP_BUFFER_COUNT;
    chn_attr.buffer_size = VICAP_ALIGN_UP(CAPTURE_WIDTH * CAPTURE_HEIGHT * 3,
                                          VICAP_ALIGN_1K);
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_chn_attr(ctx->csi, VICAP_CHN_ID_0, chn_attr);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_mpi_vicap_init(ctx->csi);
    if (ret == K_SUCCESS)
        ctx->vicap_initialized = K_TRUE;
    return ret;
}

static k_s32 dewarp_init(app_context *ctx)
{
    struct k_dw_settings settings;
    k_s32 ret;

    ret = create_pool(MODULE_BUFFER_COUNT,
                      (k_u64)ctx->screen_width * ctx->screen_height * 3 / 2,
                      &ctx->dewarp_pool);
    if (ret != K_SUCCESS)
        return ret;

    memset(&settings, 0, sizeof(settings));
    settings.vdev_id = DEWARP_DEVICE;
    settings.input.width = CAPTURE_WIDTH;
    settings.input.height = CAPTURE_HEIGHT;
    settings.input.format = K_DW_PIX_YUV420SP;
    settings.output_enable_mask = 1;
    settings.output[0].width = ctx->screen_width;
    settings.output[0].height = ctx->screen_height;
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
    k_mpp_chn vi = mpp_channel(K_ID_VI, ctx->csi, VICAP_CHN_ID_0);
    k_mpp_chn csc = mpp_channel(K_ID_NONAI_2D, 0, NONAI_2D_CHANNEL);
    k_mpp_chn dw = mpp_channel(K_ID_DW200, DEWARP_DEVICE, 0);
    k_mpp_chn vo = mpp_channel(K_ID_VO, K_VO_DISPLAY_DEV_ID, VO_LAYER);
    k_s32 ret;

    ret = kd_mpi_sys_bind(&vi, &csc);
    if (ret != K_SUCCESS)
        return ret;
    ret = kd_mpi_sys_bind(&csc, &dw);
    if (ret != K_SUCCESS) {
        kd_mpi_sys_unbind(&vi, &csc);
        return ret;
    }
    ret = kd_mpi_sys_bind(&dw, &vo);
    if (ret != K_SUCCESS) {
        kd_mpi_sys_unbind(&csc, &dw);
        kd_mpi_sys_unbind(&vi, &csc);
        return ret;
    }
    ctx->pipeline_bound = K_TRUE;
    return K_SUCCESS;
}

static void unbind_pipeline(app_context *ctx)
{
    k_mpp_chn vi = mpp_channel(K_ID_VI, ctx->csi, VICAP_CHN_ID_0);
    k_mpp_chn csc = mpp_channel(K_ID_NONAI_2D, 0, NONAI_2D_CHANNEL);
    k_mpp_chn dw = mpp_channel(K_ID_DW200, DEWARP_DEVICE, 0);
    k_mpp_chn vo = mpp_channel(K_ID_VO, K_VO_DISPLAY_DEV_ID, VO_LAYER);

    kd_mpi_sys_unbind(&dw, &vo);
    kd_mpi_sys_unbind(&csc, &dw);
    kd_mpi_sys_unbind(&vi, &csc);
    ctx->pipeline_bound = K_FALSE;
}

static void cleanup(app_context *ctx)
{
    if (ctx->stream_started) {
        kd_mpi_vicap_stop_stream(ctx->csi);
        ctx->stream_started = K_FALSE;
    }
    if (ctx->pipeline_bound)
        unbind_pipeline(ctx);
    if (ctx->dewarp_initialized) {
        kd_mpi_dw_exit(DEWARP_DEVICE);
        ctx->dewarp_initialized = K_FALSE;
    }
    if (ctx->vicap_initialized) {
        kd_mpi_vicap_deinit(ctx->csi);
        ctx->vicap_initialized = K_FALSE;
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
        kd_display_layer_disable(VO_LAYER);
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
    printf("Using CSI %d\n", ctx.csi);
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
    ret = vicap_init(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = dewarp_init(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = bind_pipeline(&ctx);
    if (ret != K_SUCCESS)
        goto done;
    ret = kd_mpi_vicap_start_stream(ctx.csi);
    if (ret != K_SUCCESS)
        goto done;
    ctx.stream_started = K_TRUE;

    printf("AHD preview running on CSI %d; press Ctrl+C to stop.\n", ctx.csi);
    while (!g_exit_requested)
        usleep(100 * 1000);
    ret = K_SUCCESS;

done:
    if (ret != K_SUCCESS)
        printf("ERROR: AHD sample failed, ret=%d\n", ret);
    cleanup(&ctx);
    return ret == K_SUCCESS ? 0 : 1;
}
