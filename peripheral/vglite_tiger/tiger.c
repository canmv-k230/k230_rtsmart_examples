#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <stdarg.h>
#include <unistd.h>
#include "vg_lite.h"
#include "tiger_paths.h"

#include "mpi_connector_api.h"
#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "kd_display.h"
#include "hal_utils.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"

static const char *error_type[] =
{
    "VG_LITE_SUCCESS",
    "VG_LITE_INVALID_ARGUMENT",
    "VG_LITE_OUT_OF_MEMORY",
    "VG_LITE_NO_CONTEXT",      
    "VG_LITE_TIMEOUT",
    "VG_LITE_OUT_OF_RESOURCES",
    "VG_LITE_GENERIC_IO",
    "VG_LITE_NOT_SUPPORT",
};
#define ARRAY_SIZE(array)        (sizeof(array) / sizeof((array)[0]))
#define ALIGN_UP(x, a)           (((x) + ((a) - 1)) & ~((a) - 1))
#define OSD_BPP                  4
#define DISPLAY_BUFFER_DEFAULT   4
#define DISPLAY_BUFFER_PIPELINE  8
#define DISPLAY_BUFFER_MAX       DISPLAY_BUFFER_PIPELINE
#define IS_ERROR(status)         ((status) > 0)
#define CHECK_ERROR(Function) \
    error = Function; \
    if (IS_ERROR(error)) \
    { \
        printf("[%s:%d] %s failed: %s\n", __func__, __LINE__, #Function, \
               error < ARRAY_SIZE(error_type) ? error_type[error] : "UNKNOWN"); \
        goto ErrorHandler; \
    }
static int fb_width = 640, fb_height = 480;
static vg_lite_buffer_t buffer;     //offscreen framebuffer object for rendering.
static vg_lite_buffer_t * fb;
typedef struct {
    vg_lite_buffer_t buffer;
    k_vb_blk_handle blk;
    k_u64 physical;
    void *memory;
    k_u32 size;
} display_buffer_t;

static int vglite_inited;
static int display_inited;
static int layer_enabled;
static int vb_inited;
static k_vo_layer_id display_layer = K_VO_LAYER_OSD0;
static k_s32 display_pool_id = -1;
static k_u32 display_size;
static int display_buffer_count = 1;
static display_buffer_t display_buffers[DISPLAY_BUFFER_MAX] = {
    { .blk = VB_INVALID_HANDLE },
    { .blk = VB_INVALID_HANDLE },
    { .blk = VB_INVALID_HANDLE },
    { .blk = VB_INVALID_HANDLE },
    { .blk = VB_INVALID_HANDLE },
    { .blk = VB_INVALID_HANDLE },
    { .blk = VB_INVALID_HANDLE },
    { .blk = VB_INVALID_HANDLE },
};
static int trace_enabled;
static int trace_paths;
static int trace_every = 30;
static volatile sig_atomic_t exit_requested;

typedef struct {
    uint64_t clear_us;
    uint64_t draw_us;
    uint64_t submit_us;
    uint64_t finish_us;
    uint64_t present_us;
} profile_stats_t;

static void sig_handler(int sig)
{
    (void)sig;
    exit_requested = 1;
}

static uint64_t get_time_us(void)
{
    return utils_cpu_ticks_us();
}

static uint64_t get_frame_period_us(int fps)
{
    return fps > 0 ? (1000000ULL + (uint64_t)fps / 2ULL) / (uint64_t)fps : 0;
}

static uint64_t sleep_until_next_frame(uint64_t next_frame_time, uint64_t frame_period_us)
{
    uint64_t now;

    if (frame_period_us == 0 || exit_requested) {
        return next_frame_time;
    }

    next_frame_time += frame_period_us;
    now = get_time_us();
    if (now < next_frame_time) {
        usleep((useconds_t)(next_frame_time - now));
    } else if (now - next_frame_time > frame_period_us) {
        next_frame_time = now;
    }

    return next_frame_time;
}

static void trace_log(const char *fmt, ...)
{
    va_list ap;

    if (!trace_enabled) {
        return;
    }

    printf("[tiger] ");
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static int should_trace_frame(int frame)
{
    return trace_enabled && (frame < 3 || trace_every <= 1 || (frame % trace_every) == 0);
}

static const char *skip_arg_prefix(const char *arg, const char *prefix)
{
    size_t len = strlen(prefix);

    return strncmp(arg, prefix, len) == 0 ? arg + len : NULL;
}

static int parse_positive_int(const char *arg, int *value)
{
    char *end = NULL;
    long parsed;

    parsed = strtol(arg, &end, 10);
    if (arg == end || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        return 0;
    }

    *value = (int)parsed;
    return 1;
}

static int parse_int_arg(const char *arg, int *value)
{
    char *end = NULL;
    long parsed;

    parsed = strtol(arg, &end, 10);
    if (arg == end || *end != '\0' || parsed < INT32_MIN || parsed > INT32_MAX) {
        return 0;
    }

    *value = (int)parsed;
    return 1;
}

static int parse_quality_arg(const char *arg, vg_lite_quality_t *quality)
{
    if (strcmp(arg, "high") == 0) {
        *quality = VG_LITE_HIGH;
    } else if (strcmp(arg, "upper") == 0) {
        *quality = VG_LITE_UPPER;
    } else if (strcmp(arg, "medium") == 0) {
        *quality = VG_LITE_MEDIUM;
    } else if (strcmp(arg, "low") == 0) {
        *quality = VG_LITE_LOW;
    } else {
        return 0;
    }

    return 1;
}

static const char *quality_name(vg_lite_quality_t quality)
{
    switch (quality) {
    case VG_LITE_HIGH:
        return "high";
    case VG_LITE_UPPER:
        return "upper";
    case VG_LITE_MEDIUM:
        return "medium";
    case VG_LITE_LOW:
        return "low";
    default:
        return "unknown";
    }
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

static int rotation_swaps_axes(int degrees)
{
    return degrees == 90 || degrees == 270;
}

static k_u32 get_timing_fps(const k_vo_timing *timing)
{
    uint64_t htotal;
    uint64_t vtotal;
    uint64_t pixels_per_frame;

    htotal = (uint64_t)timing->hactive + timing->hsync_len +
             timing->hback_porch + timing->hfront_porch;
    vtotal = (uint64_t)timing->vactive + timing->vsync_len +
             timing->vback_porch + timing->vfront_porch;
    pixels_per_frame = htotal * vtotal;
    if (timing->pclk_khz == 0 || pixels_per_frame == 0) {
        return 0;
    }

    return (k_u32)(((uint64_t)timing->pclk_khz * 1000ULL +
                    pixels_per_frame / 2ULL) / pixels_per_frame);
}

static void get_connector_size(k_connector_type connector_type, k_u32 *width,
                               k_u32 *height, k_u32 *fps)
{
    k_connector_info info;
    k_u32 conn_w;
    k_u32 conn_h;

    *fps = 0;

    memset(&info, 0, sizeof(info));
    if (kd_mpi_get_connector_info(connector_type, &info) == K_SUCCESS) {
        conn_w = info.resolution.hactive;
        conn_h = info.resolution.vactive;
        if (conn_w != 0 && conn_h != 0) {
            *width = conn_w;
            *height = conn_h;
            *fps = get_timing_fps(&info.resolution);
            return;
        }
    }

    conn_w = K_CONN_WIDTH(connector_type);
    conn_h = K_CONN_HEIGHT(connector_type);
    if (conn_w != 0 && conn_h != 0) {
        *width = conn_w;
        *height = conn_h;
        return;
    }

    *width = (k_u32)fb_width;
    *height = (k_u32)fb_height;
}

static void setup_tiger_matrix(vg_lite_matrix_t *matrix)
{
    vg_lite_identity(matrix);
    vg_lite_translate(fb_width / 2 - 20 * fb_width / 640.0f,
                      fb_height / 2 - 100 * fb_height / 480.0f, matrix);
    vg_lite_scale(4, 4, matrix);
    vg_lite_scale(fb_width / 640.0f, fb_height / 480.0f, matrix);
}

static void setup_tiger_matrix_frame(vg_lite_matrix_t *matrix, int frame)
{
    int phase_x = frame % 120;
    int wave_x = phase_x < 60 ? phase_x : 120 - phase_x;
    int phase_y = frame % 80;
    int wave_y = phase_y < 40 ? phase_y : 80 - phase_y;
    vg_lite_float_t sx = fb_width / 640.0f;
    vg_lite_float_t sy = fb_height / 480.0f;
    int dx = (wave_x - 30) * 2;
    int dy = wave_y - 20;

    /* Keep geometry and coverage stable while exercising dynamic rendering. */
    vg_lite_identity(matrix);
    vg_lite_translate(fb_width / 2 - 20.0f * sx + (vg_lite_float_t)dx,
                      fb_height / 2 - 100.0f * sy + (vg_lite_float_t)dy, matrix);
    vg_lite_scale(4.0f, 4.0f, matrix);
    vg_lite_scale(sx, sy, matrix);
}

static vg_lite_error_t enqueue_tiger(vg_lite_buffer_t *target, vg_lite_matrix_t *matrix,
                                    int frame, int trace_frame, profile_stats_t *profile)
{
    int i;
    vg_lite_error_t error;
    uint64_t stage_start = 0;

    if (trace_frame) {
        trace_log("frame %d clear begin target=0x%08x", frame, target->address);
    }
    if (profile != NULL) {
        stage_start = get_time_us();
    }
    error = vg_lite_clear(target, NULL, 0xFFFF0000);
    if (profile != NULL) {
        profile->clear_us += get_time_us() - stage_start;
    }
    if (IS_ERROR(error)) {
        return error;
    }

    if (trace_frame) {
        trace_log("frame %d draw begin", frame);
    }
    if (profile != NULL) {
        stage_start = get_time_us();
    }
    for (i = 0; i < pathCount; i++) {
        if (trace_frame && trace_paths) {
            trace_log("frame %d draw path %d/%d", frame, i + 1, pathCount);
        }
        error = vg_lite_draw(target, &path[i], VG_LITE_FILL_EVEN_ODD, matrix,
                             VG_LITE_BLEND_NONE, color_data[i]);
        if (IS_ERROR(error)) {
            return error;
        }
    }
    if (profile != NULL) {
        profile->draw_us += get_time_us() - stage_start;
    }

    return VG_LITE_SUCCESS;
}

static vg_lite_error_t upload_tiger_paths(void)
{
    int i;
    vg_lite_error_t error;

    for (i = 0; i < pathCount; i++) {
        if (trace_enabled && (i < 3 || (i % 50) == 0 || i == pathCount - 1)) {
            trace_log("upload path %d/%d", i + 1, pathCount);
        }
        error = vg_lite_upload_path(&path[i]);
        if (IS_ERROR(error)) {
            return error;
        }
    }

    return VG_LITE_SUCCESS;
}

static void set_tiger_quality(vg_lite_quality_t quality)
{
    int i;

    for (i = 0; i < pathCount; i++) {
        path[i].quality = quality;
    }
}

static int init_display_buffers(k_u32 width, k_u32 height, int buffer_count)
{
    k_vb_config vb_cfg;
    k_vb_pool_config pool_cfg;
    int i;

    memset(&vb_cfg, 0, sizeof(vb_cfg));
    vb_cfg.max_pool_cnt = 10;
    trace_log("vb init begin buffers=%d size=%ux%u", buffer_count, width, height);
    if (kd_mpi_vb_set_config(&vb_cfg) != K_SUCCESS || kd_mpi_vb_init() != K_SUCCESS) {
        printf("VB init failed\n");
        return -1;
    }
    vb_inited = 1;

    memset(&pool_cfg, 0, sizeof(pool_cfg));
    pool_cfg.blk_cnt = (k_u32)buffer_count;
    pool_cfg.blk_size = ALIGN_UP(width * height * OSD_BPP + 4096, 0x1000);
    pool_cfg.mode = VB_REMAP_MODE_NOCACHE;

    trace_log("vb create pool begin blk_size=%u blk_cnt=%u", pool_cfg.blk_size, pool_cfg.blk_cnt);
    display_pool_id = kd_mpi_vb_create_pool(&pool_cfg);
    if (display_pool_id < 0) {
        printf("vb_create_pool failed\n");
        return -1;
    }
    trace_log("vb create pool done pool=%d", display_pool_id);

    display_size = pool_cfg.blk_size;
    display_buffer_count = buffer_count;

    for (i = 0; i < display_buffer_count; i++) {
        trace_log("vb get block %d begin", i);
        display_buffers[i].size = display_size;
        display_buffers[i].blk = kd_mpi_vb_get_block(display_pool_id, display_size, NULL);
        if (display_buffers[i].blk == VB_INVALID_HANDLE) {
            printf("vb_get_block %d failed\n", i);
            return -1;
        }

        display_buffers[i].physical = kd_mpi_vb_handle_to_phyaddr(display_buffers[i].blk);
        if (display_buffers[i].physical == 0) {
            printf("vb_handle_to_phyaddr %d failed\n", i);
            return -1;
        }

        trace_log("vb mmap block %d phys=0x%llx size=%u begin", i,
                  (unsigned long long)display_buffers[i].physical, display_size);
        display_buffers[i].memory = kd_mpi_sys_mmap(display_buffers[i].physical, display_size);
        if (display_buffers[i].memory == NULL) {
            printf("sys_mmap %d failed\n", i);
            return -1;
        }
        trace_log("vb block %d ready virt=%p", i, display_buffers[i].memory);
    }

    return 0;
}

static int present_display_frame(k_vo_layer_id layer, k_u32 width, k_u32 height, k_u64 physical)
{
    k_video_frame_info vf;

    memset(&vf, 0, sizeof(vf));
    vf.mod_id = K_ID_VO;
    vf.pool_id = display_pool_id;
    vf.v_frame.width = width;
    vf.v_frame.height = height;
    vf.v_frame.pixel_format = PIXEL_FORMAT_RGBA_8888;
    vf.v_frame.stride[0] = width * OSD_BPP;
    vf.v_frame.phys_addr[0] = physical;

    if (kd_display_layer_push_frame(layer, &vf) != K_SUCCESS) {
        printf("display push frame failed\n");
        return -1;
    }

    return 0;
}

static void wait_display_hold(int hold_seconds)
{
    if (hold_seconds < 0) {
        printf("Showing tiger. Press Ctrl+C to exit.\n");
        fflush(stdout);
        while (!exit_requested) {
            usleep(100000);
        }
        return;
    }

    if (hold_seconds > 0) {
        sleep((unsigned int)hold_seconds);
    }
}

void cleanup(void)
{
    int32_t i;

    if (layer_enabled) {
        kd_display_layer_disable(display_layer);
        layer_enabled = 0;
    }

    if (display_inited) {
        kd_display_deinit();
        display_inited = 0;
    }

    for (i = 0; i < display_buffer_count; i++) {
        if (display_buffers[i].buffer.handle != NULL) {
            vg_lite_unmap(&display_buffers[i].buffer);
        }
    }

    if (buffer.handle != NULL) {
        vg_lite_free(&buffer);
    }

    for (i = 0; i < pathCount; i++)
    {
        vg_lite_clear_path(&path[i]);
    }

    if (vglite_inited) {
        vg_lite_close();
        vglite_inited = 0;
    }

    for (i = 0; i < display_buffer_count; i++) {
        if (display_buffers[i].memory != NULL) {
            kd_mpi_sys_munmap(display_buffers[i].memory, display_buffers[i].size);
            display_buffers[i].memory = NULL;
        }

        if (display_buffers[i].blk != VB_INVALID_HANDLE) {
            kd_mpi_vb_release_block(display_buffers[i].blk);
            display_buffers[i].blk = VB_INVALID_HANDLE;
        }
    }

    if (display_pool_id >= 0) {
        kd_mpi_vb_destory_pool(display_pool_id);
        display_pool_id = -1;
    }

    if (vb_inited) {
        kd_mpi_vb_exit();
        vb_inited = 0;
    }
}

int main(int argc, const char * argv[])
{
    int frame;
    int frames = 30;
    int frames_set = 0;
    int animate = 0;
    int display = 1;
    int upload_paths = 1;
    int profile = 0;
    int pipeline = 0;
    int requested_display_buffers = 0;
    int sync_fps = -1;
    int cmd_buffer_kb = 512;
    int tess_width = 0;
    int tess_height = 0;
    int connector_set = 1;
    int hold_seconds = -1;
    int rotation_degrees = 0;
    int rotation_set = 0;
    int width_set = 0;
    int height_set = 0;
    int x_set = 0;
    int y_set = 0;
    int offset_x = 0;
    int offset_y = 0;
    k_connector_type connector_type = ST7701_480_800_DSI_V1;
    k_u32 panel_width = 0;
    k_u32 panel_height = 0;
    k_u32 panel_fps = 0;
    k_u32 visible_width = 0;
    k_u32 visible_height = 0;
    uint64_t init_start;
    uint64_t init_us;
    uint64_t upload_start;
    uint64_t upload_us = 0;
    uint64_t alloc_start;
    uint64_t alloc_us;
    uint64_t render_start;
    uint64_t render_us;
    uint64_t work_us = 0;
    uint64_t sync_sleep_us = 0;
    uint64_t frame_period_us = 0;
    uint64_t next_frame_time = 0;
    k_u64 pending_physical = 0;
    int pending_frame = -1;
    vg_lite_quality_t render_quality = VG_LITE_HIGH;
    vg_lite_matrix_t matrix;
    profile_stats_t profile_stats = { 0 };

    /* Initialize vglite. */
    vg_lite_error_t error = VG_LITE_SUCCESS;

    for (int i = 1; i < argc; i++) {
        const char *value;

        if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--connector") == 0) && i + 1 < argc) {
            int parsed;
            if (!parse_int_arg(argv[++i], &parsed)) {
                printf("invalid connector: %s\n", argv[i]);
                return 1;
            }
            connector_type = (k_connector_type)parsed;
            connector_set = 1;
            display = 1;
        } else if ((strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--layer") == 0) && i + 1 < argc) {
            int parsed;
            if (!parse_int_arg(argv[++i], &parsed)) {
                printf("invalid layer: %s\n", argv[i]);
                return 1;
            }
            display_layer = (k_vo_layer_id)parsed;
        } else if (strcmp(argv[i], "--display") == 0) {
            display = 1;
        } else if (strcmp(argv[i], "--no-display") == 0) {
            display = 0;
        } else if (strcmp(argv[i], "--animate") == 0 || strcmp(argv[i], "--dynamic") == 0) {
            animate = 1;
        } else if (strcmp(argv[i], "--no-upload") == 0) {
            upload_paths = 0;
        } else if (strcmp(argv[i], "--trace") == 0) {
            trace_enabled = 1;
        } else if (strcmp(argv[i], "--trace-paths") == 0) {
            trace_enabled = 1;
            trace_paths = 1;
        } else if (strcmp(argv[i], "--profile") == 0) {
            profile = 1;
        } else if (strcmp(argv[i], "--pipeline") == 0) {
            pipeline = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--trace-every=")) != NULL) {
            if (!parse_positive_int(value, &trace_every)) {
                printf("invalid trace every: %s\n", value);
                return 1;
            }
            trace_enabled = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--fps=")) != NULL) {
            if (!parse_positive_int(value, &sync_fps)) {
                printf("invalid fps: %s\n", value);
                return 1;
            }
        } else if (strcmp(argv[i], "--no-sync") == 0) {
            sync_fps = 0;
        } else if ((value = skip_arg_prefix(argv[i], "--cmd-buf-kb=")) != NULL) {
            if (!parse_positive_int(value, &cmd_buffer_kb)) {
                printf("invalid command buffer size: %s\n", value);
                return 1;
            }
        } else if ((value = skip_arg_prefix(argv[i], "--quality=")) != NULL) {
            if (!parse_quality_arg(value, &render_quality)) {
                printf("invalid quality: %s\n", value);
                return 1;
            }
        } else if ((value = skip_arg_prefix(argv[i], "--tess-width=")) != NULL) {
            if (!parse_positive_int(value, &tess_width)) {
                printf("invalid tessellation width: %s\n", value);
                return 1;
            }
        } else if ((value = skip_arg_prefix(argv[i], "--tess-height=")) != NULL) {
            if (!parse_positive_int(value, &tess_height)) {
                printf("invalid tessellation height: %s\n", value);
                return 1;
            }
        } else if ((value = skip_arg_prefix(argv[i], "--buffers=")) != NULL) {
            if (!parse_positive_int(value, &requested_display_buffers) ||
                requested_display_buffers > DISPLAY_BUFFER_MAX) {
                printf("invalid buffers: %s\n", value);
                return 1;
            }
        } else if ((value = skip_arg_prefix(argv[i], "--connector=")) != NULL) {
            int parsed;
            if (!parse_int_arg(value, &parsed)) {
                printf("invalid connector: %s\n", value);
                return 1;
            }
            connector_type = (k_connector_type)parsed;
            connector_set = 1;
            display = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--frames=")) != NULL) {
            if (!parse_positive_int(value, &frames)) {
                printf("invalid frames: %s\n", value);
                return 1;
            }
            frames_set = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--width=")) != NULL) {
            if (!parse_positive_int(value, &fb_width)) {
                printf("invalid width: %s\n", value);
                return 1;
            }
            width_set = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--height=")) != NULL) {
            if (!parse_positive_int(value, &fb_height)) {
                printf("invalid height: %s\n", value);
                return 1;
            }
            height_set = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--layer=")) != NULL) {
            int parsed;
            if (!parse_int_arg(value, &parsed)) {
                printf("invalid layer: %s\n", value);
                return 1;
            }
            display_layer = (k_vo_layer_id)parsed;
        } else if ((value = skip_arg_prefix(argv[i], "--hold=")) != NULL) {
            if (!parse_int_arg(value, &hold_seconds)) {
                printf("invalid hold: %s\n", value);
                return 1;
            }
        } else if ((value = skip_arg_prefix(argv[i], "--rotate=")) != NULL) {
            if (!parse_int_arg(value, &rotation_degrees) ||
                (rotation_degrees != 0 && rotation_degrees != 90 &&
                 rotation_degrees != 180 && rotation_degrees != 270)) {
                printf("invalid rotation: %s\n", value);
                return 1;
            }
            rotation_set = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--x=")) != NULL) {
            if (!parse_int_arg(value, &offset_x) || offset_x < 0) {
                printf("invalid x: %s\n", value);
                return 1;
            }
            x_set = 1;
        } else if ((value = skip_arg_prefix(argv[i], "--y=")) != NULL) {
            if (!parse_int_arg(value, &offset_y) || offset_y < 0) {
                printf("invalid y: %s\n", value);
                return 1;
            }
            y_set = 1;
        } else {
            if (!parse_positive_int(argv[i], &frames)) {
                printf("usage: %s [frames] [-c connector] [--animate] [--fps=N] [--no-sync] [--pipeline] [--quality=high|upper|medium|low] [--cmd-buf-kb=N] [--tess-width=N] [--tess-height=N] [--buffers=1..8] [--profile] [--trace] [--trace-paths] [--trace-every=N] [--no-upload] [--no-display] [--width=N] [--height=N] [--layer=N] [--x=N] [--y=N] [--hold=N|-1]\n", argv[0]);
                return 1;
            }
            frames_set = 1;
        }
    }

    if (animate && !frames_set) {
        frames = 600;
    }

    if (display && !connector_set) {
        printf("display needs -c <connector_type>\n");
        return 1;
    }
    if (pipeline && (!display || !animate)) {
        printf("--pipeline requires --animate with display enabled\n");
        return 1;
    }

    if (display) {
        display_buffer_count = requested_display_buffers > 0 ? requested_display_buffers :
                               (animate ? (pipeline ? DISPLAY_BUFFER_PIPELINE : DISPLAY_BUFFER_DEFAULT) : 1);
    } else {
        display_buffer_count = 0;
    }

    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    if (display) {
        trace_log("query connector size type=%u", connector_type);
        get_connector_size(connector_type, &panel_width, &panel_height, &panel_fps);
        if (!rotation_set && ((panel_height > panel_width) != (fb_height > fb_width))) {
            rotation_degrees = 90;
        }

        visible_width = rotation_swaps_axes(rotation_degrees) ? panel_height : panel_width;
        visible_height = rotation_swaps_axes(rotation_degrees) ? panel_width : panel_height;

        if (!width_set) {
            fb_width = (int)visible_width;
        }
        if (!height_set) {
            fb_height = (int)visible_height;
        }

        if (!x_set) {
            offset_x = (visible_width > (k_u32)fb_width) ? (int)((visible_width - (k_u32)fb_width) / 2) : 0;
        }
        if (!y_set) {
            offset_y = (visible_height > (k_u32)fb_height) ? (int)((visible_height - (k_u32)fb_height) / 2) : 0;
        }

        if (animate && sync_fps < 0) {
            sync_fps = panel_fps > 0 ? (int)panel_fps : 60;
        }

        trace_log("display init begin connector=%u rotate=%d", connector_type, rotation_degrees);
        if (kd_display_init(connector_type, 0, 0, rotation_from_degrees(rotation_degrees)) != K_SUCCESS) {
            printf("display init failed\n");
            return 1;
        }
        display_inited = 1;
        trace_log("display init done");

        if (init_display_buffers((k_u32)fb_width, (k_u32)fb_height, display_buffer_count) != 0) {
            goto ErrorHandler;
        }
    } else if (sync_fps < 0) {
        sync_fps = 0;
    }

    if (tess_width == 0) {
        tess_width = fb_width;
    }
    if (tess_height == 0) {
        tess_height = fb_height;
    }

    init_start = get_time_us();
    CHECK_ERROR(vg_lite_set_command_buffer_size((vg_lite_uint32_t)cmd_buffer_kb * 1024U));
    trace_log("vg_lite_init begin tess=%dx%d", tess_width, tess_height);
    CHECK_ERROR(vg_lite_init(tess_width, tess_height));
    vglite_inited = 1;
    init_us = get_time_us() - init_start;
    trace_log("vg_lite_init done %llu us", (unsigned long long)init_us);

    set_tiger_quality(render_quality);
    if (upload_paths) {
        upload_start = get_time_us();
        trace_log("upload paths begin");
        CHECK_ERROR(upload_tiger_paths());
        upload_us = get_time_us() - upload_start;
        trace_log("upload paths done %llu us", (unsigned long long)upload_us);
    }

    printf("Framebuffer size: %d x %d\n", fb_width, fb_height);
    if (display) {
        printf("Panel size: %u x %u, visible: %u x %u, rotate: %d, offset: %d,%d, buffers: %d, sync: ",
               panel_width, panel_height, visible_width, visible_height,
               rotation_degrees, offset_x, offset_y, display_buffer_count);
        if (sync_fps > 0) {
            printf("%d fps\n", sync_fps);
        } else {
            printf("off\n");
        }
    }
    printf("Tiger paths: %d, frames: %d, quality: %s, tess: %dx%d, cmd buffer: %d KB, pipeline: %s\n",
           pathCount, frames, quality_name(render_quality), tess_width, tess_height,
           cmd_buffer_kb, pipeline ? "on" : "off");
    if (display && animate && sync_fps > 0) {
        frame_period_us = get_frame_period_us(sync_fps);
    }

    /* Allocate the off-screen buffer. */
    alloc_start = get_time_us();
    buffer.width  = fb_width;
    buffer.height = fb_height;
    buffer.format = VG_LITE_RGBA8888;
    if (display) {
        for (int i = 0; i < display_buffer_count; i++) {
            trace_log("vg_lite_map display buffer %d phys=0x%llx begin", i,
                      (unsigned long long)display_buffers[i].physical);
            display_buffers[i].buffer.width = fb_width;
            display_buffers[i].buffer.height = fb_height;
            display_buffers[i].buffer.format = VG_LITE_RGBA8888;
            display_buffers[i].buffer.stride = fb_width * OSD_BPP;
            display_buffers[i].buffer.memory = display_buffers[i].memory;
            display_buffers[i].buffer.address = (vg_lite_uint32_t)display_buffers[i].physical;
            CHECK_ERROR(vg_lite_map(&display_buffers[i].buffer, VG_LITE_MAP_USER_MEMORY, -1));
            trace_log("vg_lite_map display buffer %d done handle=%p", i,
                      display_buffers[i].buffer.handle);
        }
        fb = &display_buffers[0].buffer;
    } else {
        trace_log("vg_lite_allocate begin");
        CHECK_ERROR(vg_lite_allocate(&buffer));
        trace_log("vg_lite_allocate done phys=0x%08x", buffer.address);
        fb = &buffer;
    }
    alloc_us = get_time_us() - alloc_start;

    if (display) {
        trace_log("layer configure begin layer=%d size=%dx%d offset=%d,%d",
                  display_layer, fb_width, fb_height, offset_x, offset_y);
        if (kd_display_layer_configure(display_layer, PIXEL_FORMAT_RGBA_8888,
                                       (k_u32)fb_width, (k_u32)fb_height,
                                       (k_u32)offset_x, (k_u32)offset_y) != K_SUCCESS) {
            printf("display layer configure failed\n");
            goto ErrorHandler;
        }
        trace_log("layer configure done");
        trace_log("layer enable begin layer=%d", display_layer);
        if (kd_display_layer_enable(display_layer) != K_SUCCESS) {
            printf("display layer enable failed\n");
            goto ErrorHandler;
        }
        layer_enabled = 1;
        trace_log("layer enable done");
    }

    setup_tiger_matrix(&matrix);

    render_start = get_time_us();
    next_frame_time = render_start;
    for (frame = 0; frame < frames && !exit_requested; frame++) {
        vg_lite_buffer_t *target = fb;
        k_u64 target_physical = 0;
        int trace_frame = should_trace_frame(frame);
        uint64_t frame_start = get_time_us();
        uint64_t render_done;
        uint64_t frame_done;
        uint64_t sync_start;
        uint64_t stage_start = 0;

        if (display) {
            int index = animate ? frame % display_buffer_count : 0;
            target = &display_buffers[index].buffer;
            target_physical = display_buffers[index].physical;
            fb = target;
            if (trace_frame) {
                trace_log("frame %d begin buffer=%d phys=0x%llx", frame, index,
                          (unsigned long long)target_physical);
            }
        } else if (trace_frame) {
            trace_log("frame %d begin offscreen phys=0x%08x", frame, target->address);
        }

        if (animate) {
            setup_tiger_matrix_frame(&matrix, frame);
        }
        CHECK_ERROR(enqueue_tiger(target, &matrix, frame, trace_frame,
                                  profile ? &profile_stats : NULL));

        if (pipeline) {
            if (trace_frame) {
                trace_log("frame %d submit begin", frame);
            }
            if (profile) {
                stage_start = get_time_us();
            }
            CHECK_ERROR(vg_lite_flush());
            if (profile) {
                profile_stats.submit_us += get_time_us() - stage_start;
            }
            if (trace_frame) {
                trace_log("frame %d submit end", frame);
            }

            /* A second flush waits for the previous command buffer before it
               submits the current one, so the pending frame is now complete. */
            if (pending_frame >= 0) {
                if (trace_frame) {
                    trace_log("frame %d present begin", pending_frame);
                }
                stage_start = get_time_us();
                if (present_display_frame(display_layer, (k_u32)fb_width, (k_u32)fb_height,
                                          pending_physical) != 0) {
                    goto ErrorHandler;
                }
                if (profile) {
                    profile_stats.present_us += get_time_us() - stage_start;
                }
                if (trace_frame) {
                    trace_log("frame %d present end", pending_frame);
                }
            }

            pending_physical = target_physical;
            pending_frame = frame;
        } else {
            if (trace_frame) {
                trace_log("frame %d finish begin", frame);
            }
            if (profile) {
                stage_start = get_time_us();
            }
            CHECK_ERROR(vg_lite_finish());
            if (profile) {
                profile_stats.finish_us += get_time_us() - stage_start;
            }
            render_done = get_time_us();
            if (trace_frame) {
                trace_log("frame %d finish end", frame);
                trace_log("frame %d render done %llu us", frame,
                          (unsigned long long)(render_done - frame_start));
            }

            if (display) {
                if (trace_frame) {
                    trace_log("frame %d present begin", frame);
                }
                stage_start = get_time_us();
                if (present_display_frame(display_layer, (k_u32)fb_width, (k_u32)fb_height,
                                          target_physical) != 0) {
                    goto ErrorHandler;
                }
                if (profile) {
                    profile_stats.present_us += get_time_us() - stage_start;
                }
                if (trace_frame) {
                    trace_log("frame %d present done %llu us", frame,
                              (unsigned long long)(get_time_us() - render_done));
                }
            }
        }

        frame_done = get_time_us();
        work_us += frame_done - frame_start;
        if (display && animate && frame_period_us > 0) {
            sync_start = get_time_us();
            next_frame_time = sleep_until_next_frame(next_frame_time, frame_period_us);
            sync_sleep_us += get_time_us() - sync_start;
        }
    }

    if (pipeline && pending_frame >= 0) {
        uint64_t drain_start = get_time_us();
        uint64_t present_start;

        if (profile) {
            present_start = get_time_us();
        }
        CHECK_ERROR(vg_lite_finish());
        if (profile) {
            profile_stats.finish_us += get_time_us() - present_start;
        }

        present_start = get_time_us();
        if (present_display_frame(display_layer, (k_u32)fb_width, (k_u32)fb_height,
                                  pending_physical) != 0) {
            goto ErrorHandler;
        }
        if (profile) {
            profile_stats.present_us += get_time_us() - present_start;
        }
        work_us += get_time_us() - drain_start;
    }
    render_us = get_time_us() - render_start;

    if (display) {
        printf("Displayed on connector %u layer %d\n", connector_type, display_layer);
    }

    printf("Init: %llu us\n", (unsigned long long)init_us);
    if (upload_paths) {
        printf("Upload paths: %llu us\n", (unsigned long long)upload_us);
    }
    printf("Allocate: %llu us\n", (unsigned long long)alloc_us);
    printf("%s: %llu us total, %llu us/frame, %.2f fps\n",
           display ? (animate ? "Dynamic render+present" : "Render+present") : (animate ? "Dynamic render" : "Render"),
           (unsigned long long)render_us,
           (unsigned long long)(frame ? render_us / (uint64_t)frame : 0),
           render_us ? (double)frame * 1000000.0 / (double)render_us : 0.0);
    if (sync_sleep_us > 0) {
        printf("Work only: %llu us total, %llu us/frame, %.2f fps\n",
               (unsigned long long)work_us,
               (unsigned long long)(frame ? work_us / (uint64_t)frame : 0),
               work_us ? (double)frame * 1000000.0 / (double)work_us : 0.0);
        printf("Display sync sleep: %llu us\n", (unsigned long long)sync_sleep_us);
    }
    if (profile && frame > 0) {
        uint64_t render_profile_us = profile_stats.clear_us + profile_stats.draw_us +
                                     profile_stats.submit_us + profile_stats.finish_us;
        uint64_t profiled_us = render_profile_us + profile_stats.present_us;
        uint64_t other_us = work_us > profiled_us ? work_us - profiled_us : 0;

        printf("Profile (%d frames), total / average us:\n", frame);
        printf("  clear enqueue: %llu / %llu\n",
               (unsigned long long)profile_stats.clear_us,
               (unsigned long long)(profile_stats.clear_us / (uint64_t)frame));
        printf("  draw enqueue:  %llu / %llu\n",
               (unsigned long long)profile_stats.draw_us,
               (unsigned long long)(profile_stats.draw_us / (uint64_t)frame));
        if (pipeline) {
            printf("  submit+prewait: %llu / %llu\n",
                   (unsigned long long)profile_stats.submit_us,
                   (unsigned long long)(profile_stats.submit_us / (uint64_t)frame));
            printf("  final drain:    %llu / %llu\n",
                   (unsigned long long)profile_stats.finish_us,
                   (unsigned long long)(profile_stats.finish_us / (uint64_t)frame));
        } else {
            printf("  finish+wait:    %llu / %llu\n",
                   (unsigned long long)profile_stats.finish_us,
                   (unsigned long long)(profile_stats.finish_us / (uint64_t)frame));
        }
        if (!pipeline) {
            printf("  render total:  %llu / %llu (%.2f fps)\n",
                   (unsigned long long)render_profile_us,
                   (unsigned long long)(render_profile_us / (uint64_t)frame),
                   render_profile_us ? (double)frame * 1000000.0 / (double)render_profile_us : 0.0);
        }
        if (display) {
            printf("  display push:  %llu / %llu\n",
                   (unsigned long long)profile_stats.present_us,
                   (unsigned long long)(profile_stats.present_us / (uint64_t)frame));
        }
        printf("  other:         %llu / %llu\n",
               (unsigned long long)other_us,
               (unsigned long long)(other_us / (uint64_t)frame));
    }
    if (display) {
        wait_display_hold(hold_seconds);
    }

ErrorHandler:
    // Cleanup.
    cleanup();
    return IS_ERROR(error) ? 1 : 0;
}
