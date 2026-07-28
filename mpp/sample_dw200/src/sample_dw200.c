/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cJSON.h>

#include "dewarp_map.h"
#include "dw200_config.h"
#include "k_dewarp_ioctl.h"
#include "mpi_dewarp_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"

#define COLOR_NONE "\033[0m"
#define RED "\033[1;31;40m"
#define GREEN "\033[1;32;40m"
#define YELLOW "\033[1;33;40m"

#define LOG_LEVEL 3
#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#define ALIGN_UP(value, alignment) (((value) + ((alignment) - 1)) & ~((alignment) - 1))

#define pr_info(fmt, ...)                                              \
    do {                                                               \
        if (LOG_LEVEL >= 2) {                                          \
            fprintf(stderr, GREEN fmt "\n" COLOR_NONE, ##__VA_ARGS__); \
        }                                                              \
    } while (0)
#define pr_warn(fmt, ...)                                               \
    do {                                                                \
        if (LOG_LEVEL >= 1) {                                           \
            fprintf(stderr, YELLOW fmt "\n" COLOR_NONE, ##__VA_ARGS__); \
        }                                                               \
    } while (0)
#define pr_err(fmt, ...)                                             \
    do {                                                             \
        if (LOG_LEVEL >= 0) {                                        \
            fprintf(stderr, RED fmt "\n" COLOR_NONE, ##__VA_ARGS__); \
        }                                                            \
    } while (0)

#define MMZ_NAME "anonymous"
#define MMB_NAME "vivdw200_mmz"
#define DEWARP_BUFFER_ALIGNMENT 16U
#define DEWARP_BLOCK_SHIFT 4U
#define DEWARP_WIDTH_ALIGNMENT 16U
#define DEWARP_HEIGHT_ALIGNMENT 8U
#define DEWARP_SCALE_SHIFT 12U
#define DEWARP_SCALE_ONE (1U << DEWARP_SCALE_SHIFT)
#define MAX_IMAGE_WIDTH 4096U
#define MAP_BITS 16
#define MAP_FRACTIONAL_BITS 4
#define IRQ_POLL_LIMIT 10U
#define PATH_BUFFER_SIZE 256U
#define OUTPUT_FILENAME_SIZE 512U
#define DWE_IRQ_CHANNEL 1
#define VSE_IRQ_CHANNEL 0
#define VSE_INPUT_DWE 4U
#define VSE_INPUT_DMA 5U
#define VSE_OUTPUT_IRQ_MASK 0x7U
#define VSE_ERROR_IRQ_FLAG 0x80000000U
#define VSE_IRQ_ENABLE_MASK 0x7007U
#define VB_MAX_POOL_COUNT 64U
#define DW200_DEVICE_PATH "/dev/" DW_DEV_NAME
#define DW200_COMMAND_NOP 0x18000000ULL
#define DW200_COMMAND_END 0x1000011aULL
#define CHECK_ERROR(expression)                                          \
    do {                                                                 \
        pr_info("enter %s", #expression);                                \
        int error = (expression);                                        \
        if (error != 0) {                                                \
            pr_err("error %d at %s:%d", error, __FILE_NAME__, __LINE__); \
            goto cleanup;                                                \
        }                                                                \
    } while (0)
#define RETURN_IF_ERROR(expression)                                      \
    do {                                                                 \
        pr_info("enter %s", #expression);                                \
        int error = (expression);                                        \
        if (error != 0) {                                                \
            pr_err("error %d at %s:%d", error, __FILE_NAME__, __LINE__); \
            return -1;                                                   \
        }                                                                \
    } while (0)

/* Values are shared by the DW200 and VSE format enums. */
static const char* const format_names[] = { "YUV422SP", "YUV422I", "YUV420SP", "YUV444", "RGB888", "RGB888P", "RAW8", "RAW12" };

struct video_buffer {
    k_u64 phy_addr;
    void* virt_addr;
    uint32_t size;
};

enum driver_api {
    DRIVER_API_LOW_LEVEL,
    DRIVER_API_LEGACY,
    DRIVER_API_VDEV,
    DRIVER_API_ALL,
};

struct command_line_options {
    enum driver_api api;
    unsigned int frame_count;
    const char* config_filename;
};

struct managed_output_map {
    unsigned int sample_output;
    unsigned int driver_output;
};

enum {
    DEWARP_MODEL_LENS_DISTORTION_CORRECTION = 1 << 0,
    DEWARP_MODEL_FISHEYE_EXPAND = 1 << 1,
    DEWARP_MODEL_SPLIT_SCREEN = 1 << 2,
    DEWARP_MODEL_FISHEYE_DEWARP = 1 << 3,
    DEWARP_MODEL_PERSPECTIVE = 1 << 4,
};

static const char* const dewarp_mode_names[] = {
    "LENS_CORRECTION",
    "FISHEYE_EXPAND",
    "SPLIT_SCREEN",
    "FISHEYE_DEWARP",
    "PERSPECTIVE",
};

static const char* driver_api_name(enum driver_api api)
{
    static const char* const names[] = {
        [DRIVER_API_LOW_LEVEL] = "low-level",
        [DRIVER_API_LEGACY] = "legacy",
        [DRIVER_API_VDEV] = "vdev",
        [DRIVER_API_ALL] = "all",
    };

    return (unsigned int)api < ARRAY_SIZE(names) ? names[api] : "unknown";
}

static int parse_driver_api(const char* value, enum driver_api* api)
{
    for (unsigned int i = 0; i <= DRIVER_API_ALL; i++) {
        if (strcmp(value, driver_api_name(i)) == 0) {
            *api = i;
            return 0;
        }
    }
    return -1;
}

static void print_usage(const char* program)
{
    fprintf(stderr, "Usage: %s [--api low-level|legacy|vdev|all] [--frames count] <config JSON>\n", program);
}

static int parse_command_line(int argc, char* argv[], struct command_line_options* options)
{
    *options = (struct command_line_options) {
        .api = DRIVER_API_LOW_LEVEL,
        .frame_count = 1,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--api") == 0) {
            if (++i >= argc || parse_driver_api(argv[i], &options->api) < 0) {
                return -1;
            }
            continue;
        }
        if (strcmp(argv[i], "--frames") == 0) {
            if (++i >= argc) {
                return -1;
            }
            errno = 0;
            char* end = NULL;
            unsigned long frame_count = strtoul(argv[i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || frame_count == 0 || frame_count > UINT32_MAX) {
                return -1;
            }
            options->frame_count = (unsigned int)frame_count;
            continue;
        }
        if (argv[i][0] == '-' || options->config_filename != NULL) {
            return -1;
        }
        options->config_filename = argv[i];
    }

    return options->config_filename != NULL ? 0 : -1;
}

static int allocate_video_buffer(struct video_buffer* buffer, uint32_t size)
{
    buffer->size = size;
    /* Keep CPU and DW200 coherent without cache maintenance in this sample. */
    RETURN_IF_ERROR(kd_mpi_sys_mmz_alloc(&buffer->phy_addr, &buffer->virt_addr, MMB_NAME, MMZ_NAME, buffer->size));
    return 0;
}

static void free_video_buffer(struct video_buffer* buffer)
{
    if (buffer->virt_addr == NULL) {
        return;
    }

    kd_mpi_sys_mmz_free(buffer->phy_addr, buffer->virt_addr);
    memset(buffer, 0, sizeof(*buffer));
}

static int load_input_image(const char* filename, unsigned int input_id, struct video_buffer* buffer)
{
    if (filename[0] == '\0') {
        return 0;
    }

    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("open image file");
        return -1;
    }

    size_t bytes_read = fread(buffer->virt_addr, 1, buffer->size, file);
    fclose(file);
    if (bytes_read != buffer->size) {
        pr_err("input image is smaller than the configured frame size");
        return -1;
    }

    pr_info("loaded input %u from %s (%u bytes)", input_id, filename, buffer->size);
    return 0;
}

static enum format_t get_format(const char* format)
{
    if (format == NULL) {
        return MEDIA_PIX_FMT_YUV420SP;
    }

    for (unsigned i = 0; i < ARRAY_SIZE(format_names); i++) {
        if (strcmp(format, format_names[i]) == 0) {
            return i;
        }
    }
    return MEDIA_PIX_FMT_YUV420SP;
}

static uint32_t get_dewarp_model(const cJSON* mode)
{
    if (cJSON_IsString(mode) && mode->valuestring != NULL) {
        for (unsigned i = 0; i < ARRAY_SIZE(dewarp_mode_names); i++) {
            if (strcmp(mode->valuestring, dewarp_mode_names[i]) == 0) {
                return 1U << i;
            }
        }
    }
    return DEWARP_MODEL_FISHEYE_EXPAND;
}

static void copy_json_string(char* destination, size_t destination_size, const cJSON* item)
{
    if (destination_size == 0) {
        return;
    }

    if (!cJSON_IsString(item) || item->valuestring == NULL) {
        destination[0] = '\0';
        return;
    }

    snprintf(destination, destination_size, "%s", item->valuestring);
}

static void resolve_path_from_config(const char* config_path, char* path, size_t path_size)
{
    if (path_size == 0 || path[0] == '\0' || path[0] == '/') {
        return;
    }

    const char* slash = strrchr(config_path, '/');
    if (slash == NULL) {
        return;
    }

    size_t directory_size = (size_t)(slash - config_path) + 1;
    if (directory_size >= path_size) {
        path[0] = '\0';
        return;
    }

    char relative_path[PATH_BUFFER_SIZE];
    size_t relative_size = strnlen(path, path_size);
    if (relative_size >= sizeof(relative_path)) {
        path[0] = '\0';
        return;
    }
    memcpy(relative_path, path, relative_size + 1);
    if (directory_size + relative_size >= path_size) {
        path[0] = '\0';
        return;
    }
    memcpy(path, config_path, directory_size);
    memcpy(path + directory_size, relative_path, relative_size + 1);
}

static void get_config_directory(const char* config_path, char* directory, size_t directory_size)
{
    if (directory_size == 0) {
        return;
    }

    const char* slash = strrchr(config_path, '/');
    if (slash == NULL) {
        snprintf(directory, directory_size, ".");
        return;
    }

    size_t length = (size_t)(slash - config_path);
    if (length == 0) {
        length = 1;
    }
    if (length >= directory_size) {
        directory[0] = '\0';
        return;
    }

    memcpy(directory, config_path, length);
    directory[length] = '\0';
}

static int json_int(const cJSON* object, const char* key, int fallback)
{
    const cJSON* item;

    if (object == NULL) {
        return fallback;
    }
    item = cJSON_GetObjectItem(object, key);
    return item != NULL ? item->valueint : fallback;
}

static double json_number(const cJSON* object, const char* key, double fallback)
{
    const cJSON* item;

    if (object == NULL) {
        return fallback;
    }
    item = cJSON_GetObjectItem(object, key);
    return item != NULL ? item->valuedouble : fallback;
}

static void parse_output(const cJSON* node, unsigned int id, struct dw200_parameters* params)
{
    if (!cJSON_IsObject(node) || id >= DW200_OUTPUT_COUNT) {
        return;
    }
    params->output_res[id].enable = json_int(node, "enabled", params->output_res[id].enable);
    params->output_res[id].width = json_int(node, "width", 0);
    params->output_res[id].height = json_int(node, "height", 0);
    params->output_res[id].format = get_format(cJSON_GetStringValue(cJSON_GetObjectItem(node, "format")));
    params->output_res[id].yuvbit = json_int(node, "yuvbit", 0);
    if (!params->output_res[id].enable) {
        return;
    }

    if (id > 0) {
        params->mi_settings[id - 1].width = params->output_res[id].width;
        params->mi_settings[id - 1].height = params->output_res[id].height;
        params->mi_settings[id - 1].out_format = params->output_res[id].format;
        params->mi_settings[id - 1].yuvbit = params->output_res[id].yuvbit;
        params->mi_settings[id - 1].enable = 1;
        params->vse_format_conv[id - 1].out_format = params->output_res[id].format;

        const cJSON* crop = cJSON_GetObjectItem(node, "crop");
        if (cJSON_IsArray(crop) && cJSON_GetArraySize(crop) >= 4) {
            params->vse_crop_size[id - 1].left = cJSON_GetArrayItem(crop, 0)->valueint;
            params->vse_crop_size[id - 1].right = cJSON_GetArrayItem(crop, 1)->valueint;
            params->vse_crop_size[id - 1].top = cJSON_GetArrayItem(crop, 2)->valueint;
            params->vse_crop_size[id - 1].bottom = cJSON_GetArrayItem(crop, 3)->valueint;
        }
    }
}

static int load_user_map(const char* filename, uint32_t* map_buffer, size_t capacity)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("open user map");
        return -1;
    }

    char line[1024];
    uint32_t value;
    size_t count = 0;
    while (count < capacity && fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "%08x", &value) == 1) {
            map_buffer[count++] = value;
        }
    }
    fclose(file);
    return (int)count;
}

static void parse_input(const cJSON* json, const char* key, char* filename, size_t filename_size,
    struct dw200_resolution* resolution, uint32_t* vse_input_select)
{
    const cJSON* input = cJSON_GetObjectItem(json, key);

    if (!cJSON_IsObject(input)) {
        filename[0] = '\0';
        return;
    }

    copy_json_string(filename, filename_size, cJSON_GetObjectItem(input, "file"));
    resolution->format = get_format(cJSON_GetStringValue(cJSON_GetObjectItem(input, "format")));
    resolution->width = json_int(input, "width", 0);
    resolution->height = json_int(input, "height", 0);
    resolution->enable = json_int(input, "enabled", 0);
    resolution->yuvbit = json_int(input, "yuvbit", 0);
    if (vse_input_select != NULL) {
        *vse_input_select = json_int(input, "channel", 0);
    }
}

static void parse_number_array(const cJSON* json, const char* key, double* values, size_t value_count)
{
    const cJSON* array = cJSON_GetObjectItem(json, key);

    if (!cJSON_IsArray(array)) {
        return;
    }

    int array_size = cJSON_GetArraySize(array);
    for (size_t i = 0; i < value_count && i < (size_t)array_size; i++) {
        values[i] = cJSON_GetArrayItem(array, (int)i)->valuedouble;
    }
}

static void parse_fov(const cJSON* json, struct fov_parameter* fov)
{
    const cJSON* node = cJSON_GetObjectItem(json, "fov");

    fov->off_angle_ul = json_number(node, "offAngleUL", fov->off_angle_ul);
    fov->off_angle_ur = json_number(node, "offAngleUR", fov->off_angle_ur);
    fov->off_angle_dl = json_number(node, "offAngleDL", fov->off_angle_dl);
    fov->off_angle_dr = json_number(node, "offAngleDR", fov->off_angle_dr);
    fov->fov_ul = json_number(node, "fovUL", fov->fov_ul);
    fov->fov_ur = json_number(node, "fovUR", fov->fov_ur);
    fov->fov_dl = json_number(node, "fovDL", fov->fov_dl);
    fov->fov_dr = json_number(node, "fovDR", fov->fov_dr);
    fov->pano_at_win = json_int(node, "panoAtWin", fov->pano_at_win);
    fov->center_offset_ratio_ul = json_number(node, "centerOffsetRatioUL", fov->center_offset_ratio_ul);
    fov->center_offset_ratio_ur = json_number(node, "centerOffsetRatioUR", fov->center_offset_ratio_ur);
    fov->center_offset_ratio_dl = json_number(node, "centerOffsetRatioDL", fov->center_offset_ratio_dl);
    fov->center_offset_ratio_dr = json_number(node, "centerOffsetRatioDR", fov->center_offset_ratio_dr);
    fov->circle_offset_ratio_ul = json_number(node, "circleOffsetRatioUL", fov->circle_offset_ratio_ul);
    fov->circle_offset_ratio_ur = json_number(node, "circleOffsetRatioUR", fov->circle_offset_ratio_ur);
    fov->circle_offset_ratio_dl = json_number(node, "circleOffsetRatioDL", fov->circle_offset_ratio_dl);
    fov->circle_offset_ratio_dr = json_number(node, "circleOffsetRatioDR", fov->circle_offset_ratio_dr);
}

static int parse_config(const char* config, char* input0_filename, char* input1_filename, char* user_map_filename,
    size_t path_size, struct dewarp_distortion_map* distortion_map, struct dw200_parameters* params)
{
    FILE* file = fopen(config, "r");
    if (file == NULL) {
        perror("open config");
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    size_t size = (size_t)file_size;
    char* buffer = malloc(size + 1);
    if (buffer == NULL || fread(buffer, 1, size, file) != size) {
        fclose(file);
        free(buffer);
        return -1;
    }
    fclose(file);
    buffer[size] = '\0';

    pr_info("config file size: %zu", size);

    cJSON* json = cJSON_Parse(buffer);
    if (!json) {
        pr_err("parse error");
        free(buffer);
        return -1;
    }
    parse_input(json, "input 0", input0_filename, path_size, &params->input_res[0], NULL);
    parse_input(json, "input 1", input1_filename, path_size, &params->input_res[1], &params->vse_input_select);
    copy_json_string(user_map_filename, path_size, cJSON_GetObjectItem(json, "userMap"));

    for (unsigned i = 0; i < DW200_OUTPUT_COUNT; i++) {
        char output_key[16];
        snprintf(output_key, sizeof(output_key), "output %u", i);
        parse_output(cJSON_GetObjectItem(json, output_key), i, params);
    }

    params->dewarp_type = get_dewarp_model(cJSON_GetObjectItem(json, "dewarpMode"));
    params->scale_factor = DEWARP_SCALE_ONE;
    const cJSON* scale = cJSON_GetObjectItem(json, "scale");
    if (cJSON_IsObject(scale)) {
        params->roi_start.width = json_int(scale, "roix", params->roi_start.width);
        params->roi_start.height = json_int(scale, "roiy", params->roi_start.height);
        params->scale_factor = json_number(scale, "factor", params->scale_factor / (double)DEWARP_SCALE_ONE) * DEWARP_SCALE_ONE;
    }

    const cJSON* split = cJSON_GetObjectItem(json, "split");
    if (cJSON_IsObject(split)) {
        params->split_horizon_line = json_int(split, "horizon_line", params->split_horizon_line);
        params->split_vertical_line_up = json_int(split, "vertical_line_up", params->split_vertical_line_up);
        params->split_vertical_line_down = json_int(split, "vertical_line_down", params->split_vertical_line_down);
    }

    params->hflip = json_int(json, "hflip", params->hflip);
    params->vflip = json_int(json, "vflip", params->vflip);
    params->rotation = json_int(json, "rotation", params->rotation);
    params->bypass = json_int(json, "bypass", params->bypass);
    parse_number_array(json, "camera_matrix", distortion_map->camera_matrix, ARRAY_SIZE(distortion_map->camera_matrix));
    parse_number_array(json, "distortion_coeff", distortion_map->distortion_coeff,
        ARRAY_SIZE(distortion_map->distortion_coeff));
    parse_number_array(json, "perspective", distortion_map->perspective_matrix, ARRAY_SIZE(distortion_map->perspective_matrix));
    parse_fov(json, &params->fov);

    params->boundary_pixel.y = 0;
    params->boundary_pixel.u = 128;
    params->boundary_pixel.v = 128;

    cJSON_Delete(json);
    free(buffer);
    return 0;
}

static uint32_t resolution_size(const struct dw200_resolution* res)
{
    uint32_t stride = ALIGN_UP(res->width * (res->yuvbit + 1U), DEWARP_BUFFER_ALIGNMENT);
    uint32_t pixels = stride * res->height;

    switch (res->format) {
    case MEDIA_PIX_FMT_YUV422SP:
    case MEDIA_PIX_FMT_YUV422I:
        return pixels * 2U;
    case MEDIA_PIX_FMT_YUV420SP:
        return pixels + ALIGN_UP(pixels / 2U, DEWARP_BUFFER_ALIGNMENT);
    case MEDIA_PIX_FMT_YUV444:
    case MEDIA_PIX_FMT_RGB888:
    case MEDIA_PIX_FMT_RGB888P:
        return pixels * 3U;
    case MEDIA_PIX_FMT_RAW8:
        return res->width * res->height;
    case MEDIA_PIX_FMT_RAW12:
        return res->width * res->height * 2U;
    }
    return 0;
}

static bool needs_vse_output0(const struct dw200_parameters* params)
{
    /* The DW200 native output is YUV422SP.  Use VSE for YUV420/RGB output. */
    return params->output_res[0].enable && params->output_res[0].format != MEDIA_PIX_FMT_YUV422SP;
}

static bool has_vse_output(const struct dw200_parameters* params, bool use_vse_output0)
{
    if (use_vse_output0) {
        return true;
    }
    for (unsigned i = 1; i < DW200_OUTPUT_COUNT; i++) {
        if (params->output_res[i].enable) {
            return true;
        }
    }
    return false;
}

static void set_params(const struct dw200_parameters* params, bool use_vse_output0, struct k_dwe_hw_info* dwe_info,
    struct k_vse_params* vse_info)
{
    dwe_info->src_w = params->input_res[0].width;
    dwe_info->src_h = params->input_res[0].height;
    dwe_info->in_yuvbit = params->input_res[0].yuvbit;
    dwe_info->out_yuvbit = use_vse_output0 ? params->input_res[0].yuvbit : params->output_res[0].yuvbit;
    dwe_info->roi_x = params->roi_start.width;
    dwe_info->roi_y = params->roi_start.height;
    dwe_info->map_w = (ALIGN_UP(dwe_info->src_w, BLOCK_SIZE) >> DEWARP_BLOCK_SHIFT) + 1;
    dwe_info->map_h = (ALIGN_UP(dwe_info->src_h, BLOCK_SIZE) >> DEWARP_BLOCK_SHIFT) + 1;

    if (params->dewarp_type == DEWARP_MODEL_SPLIT_SCREEN) {
        if (params->split_horizon_line > dwe_info->src_h && params->split_vertical_line_up > dwe_info->src_w
            && params->split_vertical_line_down > dwe_info->src_w) {
            dwe_info->dst_w = MIN(MAX_IMAGE_WIDTH, ALIGN_UP((uint32_t)(dwe_info->src_h * VS_PI), DEWARP_WIDTH_ALIGNMENT));
            dwe_info->dst_h = ALIGN_UP(dwe_info->src_h / 2, DEWARP_HEIGHT_ALIGNMENT);
        } else if ((params->split_vertical_line_up > dwe_info->src_w || params->split_vertical_line_down > dwe_info->src_w)
            && params->split_horizon_line > BLOCK_SIZE && params->split_horizon_line < dwe_info->src_h) {
            dwe_info->dst_w = MIN(MAX_IMAGE_WIDTH, ALIGN_UP((uint32_t)(dwe_info->src_h / 2 * VS_PI), DEWARP_WIDTH_ALIGNMENT));
            dwe_info->dst_h = dwe_info->src_h;
        } else if (params->split_vertical_line_up > BLOCK_SIZE && params->split_vertical_line_up < dwe_info->src_w
            && params->split_vertical_line_down > BLOCK_SIZE && params->split_vertical_line_down < dwe_info->src_w
            && params->split_horizon_line > BLOCK_SIZE && params->split_horizon_line < dwe_info->src_h) {
            dwe_info->dst_w = dwe_info->src_w;
            dwe_info->dst_h = dwe_info->src_h;
        } else if (params->split_vertical_line_up > BLOCK_SIZE && params->split_vertical_line_up < dwe_info->src_w
            && params->split_vertical_line_down > BLOCK_SIZE && params->split_vertical_line_down < dwe_info->src_w
            && params->split_vertical_line_up == params->split_vertical_line_down
            && params->split_horizon_line > dwe_info->src_h) {
            dwe_info->dst_w = MIN(MAX_IMAGE_WIDTH, ALIGN_UP((uint32_t)(dwe_info->src_h * VS_PI), DEWARP_WIDTH_ALIGNMENT));
            dwe_info->dst_h = ALIGN_UP(dwe_info->src_h / 2, DEWARP_HEIGHT_ALIGNMENT);
        }

        dwe_info->map_w = (ALIGN_UP(dwe_info->dst_w, BLOCK_SIZE) >> DEWARP_BLOCK_SHIFT) + 1;
        dwe_info->map_h = (ALIGN_UP(dwe_info->dst_h, BLOCK_SIZE) >> DEWARP_BLOCK_SHIFT) + 1;
        dwe_info->map_w++;
        dwe_info->map_h++;
    } else {
        dwe_info->roi_x = (dwe_info->roi_x >> DEWARP_BLOCK_SHIFT) << DEWARP_BLOCK_SHIFT;
        dwe_info->roi_y = (dwe_info->roi_y >> DEWARP_BLOCK_SHIFT) << DEWARP_BLOCK_SHIFT;

        dwe_info->dst_w = ALIGN_UP(((dwe_info->src_w - dwe_info->roi_x) * params->scale_factor) >> DEWARP_SCALE_SHIFT,
            DEWARP_WIDTH_ALIGNMENT);
        dwe_info->dst_h = ALIGN_UP(((dwe_info->src_h - dwe_info->roi_y) * params->scale_factor) >> DEWARP_SCALE_SHIFT,
            DEWARP_HEIGHT_ALIGNMENT);
    }

    if (!use_vse_output0 && params->output_res[0].width > 0) {
        dwe_info->dst_w = ALIGN_UP(params->output_res[0].width, DEWARP_WIDTH_ALIGNMENT);
        dwe_info->dst_h = ALIGN_UP(params->output_res[0].height, DEWARP_HEIGHT_ALIGNMENT);
    }

    dwe_info->src_stride = ALIGN_UP(dwe_info->src_w * (dwe_info->in_yuvbit + 1), DEWARP_BUFFER_ALIGNMENT);
    dwe_info->dst_stride = ALIGN_UP(dwe_info->dst_w * (dwe_info->out_yuvbit + 1), DEWARP_BUFFER_ALIGNMENT);

    if (params->input_res[0].format == MEDIA_PIX_FMT_YUV422I) {
        dwe_info->src_stride *= 2;
    }
    if (!use_vse_output0 && params->output_res[0].format == MEDIA_PIX_FMT_YUV422I) {
        dwe_info->dst_stride *= 2;
    }

    dwe_info->split_line = params->rotation || params->dewarp_type == DEWARP_MODEL_SPLIT_SCREEN;
    dwe_info->scale_factor = (uint32_t)(512.0f * 1024.0f / params->scale_factor) & 0xffff;
    dwe_info->in_format = params->input_res[0].format;
    dwe_info->out_format = use_vse_output0 ? MEDIA_PIX_FMT_YUV422SP : params->output_res[0].format;

    dwe_info->hand_shake = 0;
    dwe_info->boundary_y = params->boundary_pixel.y;
    dwe_info->boundary_u = params->boundary_pixel.u;
    dwe_info->boundary_v = params->boundary_pixel.v;
    dwe_info->src_auto_shadow = 0;
    dwe_info->dst_auto_shadow = 0;
    dwe_info->split_h = params->split_horizon_line;
    dwe_info->split_v1 = params->split_vertical_line_up;
    dwe_info->split_v2 = params->split_vertical_line_down;
    if (dwe_info->out_format == MEDIA_PIX_FMT_YUV420SP) {
        dwe_info->dst_size_uv = ALIGN_UP(dwe_info->dst_stride * dwe_info->dst_h / 2, DEWARP_BUFFER_ALIGNMENT);
    } else {
        dwe_info->dst_size_uv = ALIGN_UP(dwe_info->dst_stride * dwe_info->dst_h, DEWARP_BUFFER_ALIGNMENT);
    }

    memcpy(vse_info->crop_size, params->vse_crop_size, sizeof(vse_info->crop_size));
    memcpy(vse_info->format_conv, params->vse_format_conv, sizeof(vse_info->format_conv));
    memcpy(vse_info->mi_settings, params->mi_settings, sizeof(vse_info->mi_settings));
    if (use_vse_output0) {
        vse_info->out_size[0].width = params->output_res[0].width;
        vse_info->out_size[0].height = params->output_res[0].height;
        vse_info->resize_enable[0] = 1;
        vse_info->mi_settings[0].enable = 1;
        vse_info->mi_settings[0].width = params->output_res[0].width;
        vse_info->mi_settings[0].height = params->output_res[0].height;
        vse_info->mi_settings[0].out_format = params->output_res[0].format;
        vse_info->mi_settings[0].yuvbit = params->output_res[0].yuvbit;
        vse_info->format_conv[0].out_format = params->output_res[0].format;
    }

    for (unsigned i = 1; i < DW200_OUTPUT_COUNT; i++) {
        if (!params->output_res[i].enable) {
            continue;
        }
        vse_info->out_size[i - 1].width = params->output_res[i].width;
        vse_info->out_size[i - 1].height = params->output_res[i].height;
        vse_info->resize_enable[i - 1] = params->output_res[i].enable;
        vse_info->mi_settings[i - 1].enable = vse_info->resize_enable[i - 1];
    }

    if (params->input_res[1].enable) {
        vse_info->src_w = params->input_res[1].width;
        vse_info->src_h = params->input_res[1].height;
        vse_info->in_format = params->input_res[1].format;
        vse_info->in_yuvbit = params->input_res[1].yuvbit;
        vse_info->input_select = params->vse_input_select;
    } else if (has_vse_output(params, use_vse_output0)) {
        vse_info->src_w = dwe_info->dst_w;
        vse_info->src_h = dwe_info->dst_h;
        vse_info->in_format = MEDIA_PIX_FMT_YUV422SP;
        vse_info->in_yuvbit = dwe_info->out_yuvbit;
        vse_info->input_select = VSE_INPUT_DWE;
    }
}

static void create_bypass_map(unsigned int* map, int map_width, int map_height, int image_width, int image_height)
{
    int y = 0;
    for (int row = 0; row < map_height; row++, y += BLOCK_SIZE) {
        if (row == map_height - 1) {
            y = image_height - 1;
        }

        int x = 0;
        for (int column = 0; column < map_width; column++, x += BLOCK_SIZE) {
            if (column == map_width - 1) {
                x = image_width - 1;
            }
            int dx = (x * 16) & 0xffff;
            int dy = (y * 16) & 0xffff;
            map[row * map_width + column] = (dy << 16) | dx;
        }
    }
}

static int build_dewarp_map(const struct dewarp_distortion_map* distortion_map, const struct dw200_parameters* params,
    struct k_dwe_hw_info* dwe_info, const char* user_map_filename, unsigned int* map_buffer)
{
    if (user_map_filename[0] != '\0') {
        pr_info("use user map");
        if (load_user_map(user_map_filename, map_buffer, MAX_MAP_SIZE / sizeof(*map_buffer)) < 0) {
            return -1;
        }
        return 0;
    }

    const float perspective_scale_x = 1.0f;
    const float perspective_scale_y = 1.0f;
    const int perspective_offset_x = 0;
    const int perspective_offset_y = 0;

    if (params->bypass) {
        pr_info("use bypass map");
        create_bypass_map(map_buffer, dwe_info->map_w, dwe_info->map_h, dwe_info->src_w, dwe_info->src_h);
    } else if (params->dewarp_type == DEWARP_MODEL_SPLIT_SCREEN) {
        pr_info("use polar map");
        CreateUpdateWarpPolarMap(
            map_buffer, dwe_info->map_w, dwe_info->map_h, MAP_BITS, MAP_FRACTIONAL_BITS, dwe_info->src_w, dwe_info->src_h,
            dwe_info->dst_w, dwe_info->dst_h, dwe_info->src_w / 2, dwe_info->src_h / 2, dwe_info->src_h / 2,
            params->split_horizon_line, params->split_vertical_line_up, params->split_vertical_line_down, BLOCK_SIZE,
            BLOCK_SIZE, 0x20, params->fov.off_angle_ul, params->fov.off_angle_ur, params->fov.off_angle_dl,
            params->fov.off_angle_dr, params->fov.fov_ul, params->fov.fov_ur, params->fov.fov_dl, params->fov.fov_dr,
            params->fov.pano_at_win, params->fov.center_offset_ratio_ul, params->fov.center_offset_ratio_ur,
            params->fov.center_offset_ratio_dl, params->fov.center_offset_ratio_dr, params->fov.circle_offset_ratio_ul,
            params->fov.circle_offset_ratio_ur, params->fov.circle_offset_ratio_dl, params->fov.circle_offset_ratio_dr);
    } else {
        switch (params->dewarp_type) {
        case DEWARP_MODEL_LENS_DISTORTION_CORRECTION:
            pr_info("use dewarp map");
            CreateUpdateDewarpMap(map_buffer, dwe_info->map_w, dwe_info->map_h, MAP_BITS, MAP_FRACTIONAL_BITS,
                distortion_map->camera_matrix, distortion_map->distortion_coeff, dwe_info->src_w,
                dwe_info->src_h, 1.0f, BLOCK_SIZE, BLOCK_SIZE);
            break;
        case DEWARP_MODEL_FISHEYE_EXPAND:
            pr_info("use fisheye expand map");
            CreateUpdateFisheyeExpandMap(map_buffer, dwe_info->map_w, dwe_info->map_h, MAP_BITS, MAP_FRACTIONAL_BITS,
                dwe_info->src_w, dwe_info->src_h, dwe_info->src_w, dwe_info->src_h,
                dwe_info->src_w / 2, dwe_info->src_h / 2, dwe_info->src_h / 2, BLOCK_SIZE, BLOCK_SIZE);
            break;
        case DEWARP_MODEL_FISHEYE_DEWARP:
            pr_info("use fisheye dewarp map");
            CreateUpdateFisheyeDewarpMap(map_buffer, dwe_info->map_w, dwe_info->map_h, MAP_BITS, MAP_FRACTIONAL_BITS,
                distortion_map->camera_matrix, distortion_map->distortion_coeff, dwe_info->src_w,
                dwe_info->src_h, 1.0f, BLOCK_SIZE, BLOCK_SIZE);
            break;
        case DEWARP_MODEL_PERSPECTIVE:
            pr_info("use perspective map");
            CreateUpdatePerspectiveMap(map_buffer, dwe_info->map_w, dwe_info->map_h, MAP_BITS, MAP_FRACTIONAL_BITS,
                distortion_map->perspective_matrix, dwe_info->src_w, dwe_info->src_h, BLOCK_SIZE,
                BLOCK_SIZE, perspective_scale_x, perspective_scale_y, perspective_offset_x,
                perspective_offset_y);
            break;
        }
    }

    if (params->rotation) {
        unsigned tmp = dwe_info->map_w;
        dwe_info->map_w = dwe_info->map_h;
        dwe_info->map_h = tmp;
    }
    return 0;
}

static int consume_dwe_irq(bool* frame_done)
{
    uint32_t irq_status = kd_mpi_dewarp_read_irq(DWE_IRQ_CHANNEL);
    if (irq_status == 0) {
        return 0;
    }

    pr_info("polled DWE IRQ 0x%08x", irq_status);
    if (irq_status & INT_ERR_STATUS_MASK) {
        pr_err("dewarp error: %u", (irq_status & INT_ERR_STATUS_MASK) >> INT_ERR_STATUS_SHIFT);
        return -1;
    }
    if (irq_status & INT_FRAME_BUSY) {
        pr_warn("dewarp frame busy");
    }
    if (irq_status & INT_FRAME_DONE) {
        *frame_done = true;
    }
    return 0;
}

static int consume_vse_irq(uint32_t* received_output_mask)
{
    uint32_t irq_status = kd_mpi_dewarp_read_irq(VSE_IRQ_CHANNEL);
    if (irq_status == 0) {
        return 0;
    }

    pr_info("polled VSE IRQ 0x%08x", irq_status);
    if (irq_status & VSE_ERROR_IRQ_FLAG) {
        pr_err("VSE reported an error interrupt");
        return -1;
    }

    *received_output_mask |= irq_status & VSE_OUTPUT_IRQ_MASK;
    return 0;
}

static int wait_for_frame(bool expect_dwe, uint32_t enabled_vse_output_mask)
{
    bool dwe_frame_done = !expect_dwe;
    uint32_t received_vse_output_mask = 0;
    unsigned int timeout_count = 0;

    while (!dwe_frame_done || (received_vse_output_mask & enabled_vse_output_mask) != enabled_vse_output_mask) {
        int poll_result = kd_mpi_dewarp_poll_irq();
        if (poll_result < 0) {
            perror("poll dewarp irq");
            return -1;
        }
        if (poll_result == 0) {
            if (++timeout_count > IRQ_POLL_LIMIT) {
                pr_err("poll timeout, exit");
                return -1;
            }
            pr_warn("poll timeout");
            continue;
        }

        timeout_count = 0;
        uint32_t previous_vse_output_mask = received_vse_output_mask;
        if (consume_dwe_irq(&dwe_frame_done) < 0 || consume_vse_irq(&received_vse_output_mask) < 0) {
            return -1;
        }

        if (received_vse_output_mask != previous_vse_output_mask) {
            uint32_t remaining_mask = enabled_vse_output_mask & ~received_vse_output_mask;
            RETURN_IF_ERROR(kd_mpi_dewarp_mask_irq(remaining_mask == 0 ? 0 : VSE_IRQ_ENABLE_MASK & ~received_vse_output_mask));
        }
    }
    return 0;
}

static const char* const format_suffixes[]
    = { "yuv422sp", "yuv422i", "yuv420sp", "yuv444", "rgb888", "rgb888p", "raw8", "raw12" };

static int save_frame(const char* output_directory, unsigned int output_id, const struct dw200_resolution* resolution,
    const char* api_suffix, const void* data, size_t size)
{
    if (resolution->format >= ARRAY_SIZE(format_suffixes)) {
        pr_err("unsupported output format %u", resolution->format);
        return -1;
    }

    char filename[OUTPUT_FILENAME_SIZE];
    snprintf(filename, sizeof(filename), "%s/channel%u_%ux%u_%s%s%s.bin", output_directory, output_id, resolution->width,
        resolution->height, format_suffixes[resolution->format], api_suffix == NULL ? "" : "_",
        api_suffix == NULL ? "" : api_suffix);

    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        perror("open output file");
        return -1;
    }
    size_t bytes_written = fwrite(data, 1, size, file);
    int close_result = fclose(file);
    if (bytes_written != size || close_result != 0) {
        perror("write output file");
        return -1;
    }
    pr_info("wrote channel %u to %s (%zu bytes)", output_id, filename, bytes_written);
    return 0;
}

static int save_low_level_outputs(const struct dw200_parameters* params, const struct video_buffer* output_buffers,
    const char* output_directory)
{
    for (unsigned int i = 0; i < DW200_OUTPUT_COUNT; i++) {
        if (!params->output_res[i].enable || output_buffers[i].virt_addr == NULL) {
            continue;
        }
        if (save_frame(output_directory, i, &params->output_res[i], NULL, output_buffers[i].virt_addr, output_buffers[i].size)
            < 0) {
            return -1;
        }
    }
    return 0;
}

static int run_low_level_api(const struct dw200_parameters* params, struct k_dwe_hw_info* dwe_info,
    struct k_vse_params* vse_info, const struct video_buffer* input_buffers,
    const struct video_buffer* lut_map, const char* output_directory, unsigned int frame_count)
{
    struct video_buffer output_buffers[DW200_OUTPUT_COUNT] = { 0 };
    struct video_buffer dwe_output_buffer = { 0 };
    bool use_vse_output0 = needs_vse_output0(params);
    bool dwe_enabled = params->input_res[0].enable;
    bool vse_enabled = has_vse_output(params, use_vse_output0);
    bool vse_dma_input = params->input_res[1].enable;
    bool dewarp_initialized = false;
    bool dwe_bus_enabled = false;
    int result = -1;

    if (!dwe_enabled && !vse_enabled) {
        pr_err("configuration enables neither DWE nor VSE");
        return -1;
    }
    if (vse_dma_input && params->vse_input_select != VSE_INPUT_DMA) {
        pr_err("input 1 requires VSE channel %u", VSE_INPUT_DMA);
        return -1;
    }
    if (!vse_dma_input && vse_enabled && !dwe_enabled) {
        pr_err("VSE requires either DWE input or input 1 DMA");
        return -1;
    }

    for (unsigned int i = 0; i < DW200_OUTPUT_COUNT; i++) {
        if (!params->output_res[i].enable) {
            continue;
        }
        uint32_t size = resolution_size(&params->output_res[i]);
        if (size == 0 || allocate_video_buffer(&output_buffers[i], size) < 0) {
            goto cleanup;
        }
        pr_info("output %u buffer: %08x,%p,%u", i, (uint32_t)output_buffers[i].phy_addr, output_buffers[i].virt_addr,
            output_buffers[i].size);
    }

    if (dwe_enabled && (use_vse_output0 || !params->output_res[0].enable)) {
        const struct dw200_resolution intermediate_resolution = {
            .yuvbit = dwe_info->out_yuvbit,
            .width = dwe_info->dst_w,
            .height = dwe_info->dst_h,
            .format = MEDIA_PIX_FMT_YUV422SP,
            .enable = 1,
        };
        if (allocate_video_buffer(&dwe_output_buffer, resolution_size(&intermediate_resolution)) < 0) {
            goto cleanup;
        }
        pr_info("DWE intermediate buffer: %08x,%p,%u", (uint32_t)dwe_output_buffer.phy_addr, dwe_output_buffer.virt_addr,
            dwe_output_buffer.size);
    }

    uint32_t vse_enabled_mask = 0;
    uint32_t vse_output_addresses[DW200_VSE_OUTPUT_COUNT] = { 0 };
    if (vse_enabled) {
        for (unsigned int i = 0; i < DW200_VSE_OUTPUT_COUNT; i++) {
            if (vse_info->resize_enable[i]) {
                vse_enabled_mask |= 1U << i;
            }
        }
        if (use_vse_output0) {
            vse_output_addresses[0] = (uint32_t)output_buffers[0].phy_addr;
        } else {
            for (unsigned int i = 0; i < DW200_VSE_OUTPUT_COUNT; i++) {
                vse_output_addresses[i] = (uint32_t)output_buffers[i + 1].phy_addr;
            }
        }
    }

    if (kd_mpi_dewarp_init() < 0) {
        perror("open dewarp device");
        goto cleanup;
    }
    dewarp_initialized = true;
    if (kd_mpi_dewarp_reset() != 0) {
        pr_err("failed to reset DW200");
        goto cleanup;
    }

    if (dwe_enabled) {
        CHECK_ERROR(kd_mpi_dewarp_dwe_disable_irq());
        CHECK_ERROR(kd_mpi_dewarp_set_map_lut_addr((uint32_t)lut_map->phy_addr));
        CHECK_ERROR(kd_mpi_dewarp_dwe_s_params(dwe_info));
        pr_info("DWE %ux%u, map %ux%u, strides %u/%u", dwe_info->src_w, dwe_info->src_h, dwe_info->map_w, dwe_info->map_h,
            dwe_info->src_stride, dwe_info->dst_stride);
    }
    if (vse_enabled) {
        CHECK_ERROR(kd_mpi_dewarp_vse_s_params(vse_info));
        pr_info("VSE %ux%u format %u, output mask 0x%x", vse_info->src_w, vse_info->src_h, vse_info->in_format,
            vse_enabled_mask);
    }

    for (unsigned int frame = 0; frame < frame_count; frame++) {
        pr_info("low-level frame %u/%u", frame + 1, frame_count);
        if (vse_enabled) {
            CHECK_ERROR(kd_mpi_dewarp_clear_irq(VSE_IRQ_CHANNEL));
            CHECK_ERROR(kd_mpi_dewarp_update_buffer(vse_output_addresses));
            CHECK_ERROR(kd_mpi_dewarp_set_mi_info());
            CHECK_ERROR(kd_mpi_dewarp_mask_irq(VSE_IRQ_ENABLE_MASK));
        }

        if (dwe_enabled) {
            uint32_t destination = dwe_output_buffer.virt_addr != NULL ? (uint32_t)dwe_output_buffer.phy_addr
                                                                       : (uint32_t)output_buffers[0].phy_addr;
            CHECK_ERROR(kd_mpi_dewarp_set_dst_buffer_addr(destination));
            CHECK_ERROR(kd_mpi_dewarp_set_map_lut_addr((uint32_t)lut_map->phy_addr));
            CHECK_ERROR(kd_mpi_dewarp_start_dma_read((uint32_t)input_buffers[0].phy_addr));
            if (frame == 0) {
                CHECK_ERROR(kd_mpi_dewarp_enable_bus());
                dwe_bus_enabled = true;
                CHECK_ERROR(kd_mpi_dewarp_start_dwe());
            } else {
                /* CLEAR_IRQ acknowledges the previous frame and restarts DWE. */
                CHECK_ERROR(kd_mpi_dewarp_clear_irq(DWE_IRQ_CHANNEL));
            }
        }
        if (vse_dma_input) {
            CHECK_ERROR(kd_mpi_dewarp_set_dma_buffer_info((uint32_t)input_buffers[1].phy_addr));
        }
        if (wait_for_frame(dwe_enabled, vse_enabled_mask) < 0) {
            goto cleanup;
        }
    }

    if (vse_enabled) {
        CHECK_ERROR(kd_mpi_dewarp_mask_irq(0));
        CHECK_ERROR(kd_mpi_dewarp_clear_irq(VSE_IRQ_CHANNEL));
    }
    if (dwe_enabled) {
        CHECK_ERROR(kd_mpi_dewarp_disable_irq());
        CHECK_ERROR(kd_mpi_dewarp_disable_bus());
        dwe_bus_enabled = false;
    }
    if (save_low_level_outputs(params, output_buffers, output_directory) < 0) {
        goto cleanup;
    }
    result = 0;

cleanup:
    if (dewarp_initialized) {
        if (dwe_bus_enabled) {
            kd_mpi_dewarp_disable_irq();
            kd_mpi_dewarp_disable_bus();
        }
        if (vse_enabled) {
            kd_mpi_dewarp_mask_irq(0);
        }
        kd_mpi_dewarp_exit();
    }
    free_video_buffer(&dwe_output_buffer);
    for (unsigned int i = 0; i < DW200_OUTPUT_COUNT; i++) {
        free_video_buffer(&output_buffers[i]);
    }
    return result;
}

static uint32_t managed_output_size(const struct k_dw_frame_info* frame)
{
    const struct dw200_resolution resolution = {
        .yuvbit = frame->bit10,
        .width = frame->width,
        .height = frame->height,
        .format = frame->format,
        .enable = 1,
    };
    return resolution_size(&resolution);
}

static int build_managed_settings(const struct dw200_parameters* params, const struct k_dwe_hw_info* dwe_info,
    const struct video_buffer* lut_map, struct k_dw_settings* settings,
    struct managed_output_map* output_map, unsigned int* output_count, unsigned int* input_id)
{
    if (params->input_res[0].enable) {
        *input_id = 0;
    } else if (params->input_res[1].enable) {
        *input_id = 1;
    } else {
        pr_err("managed API requires one input");
        return -1;
    }

    const struct dw200_resolution* input = &params->input_res[*input_id];
    if (input->format > K_DW_PIX_RGB888P) {
        pr_err("managed API does not support input format %u", input->format);
        return -1;
    }

    *settings = (struct k_dw_settings) {
        .input = {
            .width = input->width,
            .height = input->height,
            .format = input->format,
            .bit10 = input->yuvbit != 0,
            .alignment = 4,
        },
        .split_enable = params->rotation || params->dewarp_type == DEWARP_MODEL_SPLIT_SCREEN,
        .split_horizon_line = params->split_horizon_line,
        .split_vertical_line_up = params->split_vertical_line_up,
        .split_vertical_line_down = params->split_vertical_line_down,
        .vdev_id = UINT8_MAX,
    };

    if (*input_id == 0) {
        settings->lut_phy_addr = (uint32_t)lut_map->phy_addr;
        settings->lut_user_virt_addr = lut_map->virt_addr;
        settings->lut_width = dwe_info->map_w;
        settings->lut_height = dwe_info->map_h;
        settings->lut_size = dwe_info->map_w * dwe_info->map_h * sizeof(uint32_t);
    }

    *output_count = 0;
    for (unsigned int sample_output = 0; sample_output < DW200_OUTPUT_COUNT; sample_output++) {
        const struct dw200_resolution* output = &params->output_res[sample_output];
        if (!output->enable) {
            continue;
        }
        if (*output_count >= DW200_VSE_OUTPUT_COUNT || output->format > K_DW_PIX_RGB888P) {
            pr_err("managed API cannot map output %u", sample_output);
            return -1;
        }

        unsigned int driver_output = *output_count;
        settings->output[driver_output] = (struct k_dw_frame_info) {
            .width = output->width,
            .height = output->height,
            .format = output->format,
            .bit10 = output->yuvbit != 0,
            .alignment = 4,
        };
        if (sample_output > 0) {
            settings->crop[driver_output] = params->vse_crop_size[sample_output - 1];
        }
        settings->output_enable_mask |= 1U << driver_output;
        output_map[*output_count] = (struct managed_output_map) {
            .sample_output = sample_output,
            .driver_output = driver_output,
        };
        (*output_count)++;
    }

    if (*output_count == 0) {
        pr_err("managed API requires at least one output");
        return -1;
    }
    return 0;
}

static int initialize_managed_vb(const struct k_dw_settings* settings)
{
    uint32_t maximum_size = 0;
    unsigned int output_count = 0;
    for (unsigned int i = 0; i < DW200_VSE_OUTPUT_COUNT; i++) {
        if (!(settings->output_enable_mask & (1U << i))) {
            continue;
        }
        uint32_t size = managed_output_size(&settings->output[i]);
        if (size > maximum_size) {
            maximum_size = size;
        }
        output_count++;
    }

    k_vb_config config = {
        .max_pool_cnt = VB_MAX_POOL_COUNT,
        .comm_pool[0] = {
            .blk_size = maximum_size,
            .blk_cnt = output_count + 1,
            .mode = VB_REMAP_MODE_NOCACHE,
        },
    };
    int error = kd_mpi_vb_set_config(&config);
    if (error != 0) {
        pr_err("kd_mpi_vb_set_config failed: %d", error);
        return -1;
    }
    error = kd_mpi_vb_init();
    if (error != 0) {
        pr_err("kd_mpi_vb_init failed: %d", error);
        return -1;
    }
    return 0;
}

static k_pixel_format driver_pixel_format(uint32_t format)
{
    static const k_pixel_format formats[] = {
        PIXEL_FORMAT_YUV_SEMIPLANAR_422,
        PIXEL_FORMAT_YUYV_PACKAGE_422,
        PIXEL_FORMAT_YUV_SEMIPLANAR_420,
        PIXEL_FORMAT_YVU_PLANAR_444,
        PIXEL_FORMAT_RGB_888,
        PIXEL_FORMAT_BGR_888_PLANAR,
        PIXEL_FORMAT_RGB_BAYER_8BPP,
        PIXEL_FORMAT_RGB_BAYER_12BPP,
    };

    return format < ARRAY_SIZE(formats) ? formats[format] : PIXEL_FORMAT_BUTT;
}

static void prepare_load_request(const struct dw200_parameters* params, const struct video_buffer* input_buffers,
    unsigned int input_id, struct k_dw_load_request* request)
{
    const struct dw200_resolution* input = &params->input_res[input_id];
    uint32_t stride = ALIGN_UP(input->width * (input->yuvbit + 1U), DEWARP_BUFFER_ALIGNMENT);
    uint32_t plane_size = stride * input->height;

    memset(request, 0, sizeof(*request));
    request->vb_info.phys_addr[0] = input_buffers[input_id].phy_addr;
    if (input->format == MEDIA_PIX_FMT_YUV422SP || input->format == MEDIA_PIX_FMT_YUV420SP
        || input->format == MEDIA_PIX_FMT_YUV444 || input->format == MEDIA_PIX_FMT_RGB888P) {
        request->vb_info.phys_addr[1] = input_buffers[input_id].phy_addr + plane_size;
    }
    if (input->format == MEDIA_PIX_FMT_YUV444 || input->format == MEDIA_PIX_FMT_RGB888P) {
        request->vb_info.phys_addr[2] = input_buffers[input_id].phy_addr + plane_size * 2U;
    }
    request->vb_info.dev_num = 0;
    request->vb_info.width = input->width;
    request->vb_info.height = input->height;
    request->vb_info.format = driver_pixel_format(input->format);
    request->vb_info.alignment = 4;
}

static void release_managed_outputs(struct k_dw_load_request* request)
{
    for (unsigned int i = 0; i < DW200_VSE_OUTPUT_COUNT; i++) {
        if (request->output_vb_info[i].phys_addr[0] == 0) {
            continue;
        }
        k_vb_blk_handle handle = kd_mpi_vb_phyaddr_to_handle(request->output_vb_info[i].phys_addr[0]);
        if (handle != VB_INVALID_HANDLE) {
            int error = kd_mpi_vb_release_block(handle);
            if (error != 0) {
                pr_warn("release output %u failed: %d", i, error);
            }
        }
        memset(&request->output_vb_info[i], 0, sizeof(request->output_vb_info[i]));
    }
}

static int save_managed_outputs(const struct dw200_parameters* params, const struct k_dw_settings* settings,
    const struct managed_output_map* output_map, unsigned int output_count,
    const struct k_dw_load_request* request, const char* output_directory, const char* api_suffix)
{
    for (unsigned int i = 0; i < output_count; i++) {
        unsigned int sample_output = output_map[i].sample_output;
        unsigned int driver_output = output_map[i].driver_output;
        k_u64 physical_address = request->output_vb_info[driver_output].phys_addr[0];
        uint32_t size = managed_output_size(&settings->output[driver_output]);
        if (physical_address == 0 || size == 0) {
            pr_err("driver did not return output %u", driver_output);
            return -1;
        }

        void* data = kd_mpi_sys_mmap(physical_address, size);
        if (data == NULL) {
            pr_err("map output %u failed", driver_output);
            return -1;
        }
        int result = save_frame(output_directory, sample_output, &params->output_res[sample_output], api_suffix, data, size);
        kd_mpi_sys_munmap(data, size);
        if (result < 0) {
            return -1;
        }
    }
    return 0;
}

static int call_driver_ioctl(int fd, int command, void* argument, const char* name)
{
    pr_info("enter ioctl(%s)", name);
    int result = ioctl(fd, command, argument);
    if (result != 0) {
        pr_err("ioctl(%s) failed: %d", name, result);
        return -1;
    }
    return 0;
}

static int print_driver_version(int fd)
{
    char version[64];
    ssize_t size = read(fd, version, sizeof(version) - 1);
    if (size < 0) {
        perror("read dewarp version");
        return -1;
    }
    version[size] = '\0';
    if (size > 0 && version[size - 1] == '\n') {
        version[size - 1] = '\0';
    }
    pr_info("driver %s", version);
    return 0;
}

static int dump_vdev_commands(int fd, const char* output_directory)
{
    union k_dewarp_command* commands = calloc(DW_COMMAND_BUFFER_LEN, sizeof(*commands));
    if (commands == NULL) {
        return -1;
    }
    if (call_driver_ioctl(fd, K_DWVIOC_DUMP_PARAMS, commands, "K_DWVIOC_DUMP_PARAMS") < 0) {
        free(commands);
        return -1;
    }

    char filename[OUTPUT_FILENAME_SIZE];
    snprintf(filename, sizeof(filename), "%s/dw200_vdev_registers.txt", output_directory);
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("open register dump");
        free(commands);
        return -1;
    }

    bool found_end = false;
    for (unsigned int i = 0; i < DW_COMMAND_BUFFER_LEN; i++) {
        if (commands[i].value == DW200_COMMAND_END) {
            found_end = true;
            break;
        }
        if (commands[i].value == DW200_COMMAND_NOP) {
            continue;
        }
        fprintf(file, "0x%08x, 0x%08x\n", commands[i].bytes.addr, commands[i].bytes.value);
    }
    fclose(file);
    free(commands);
    if (!found_end) {
        pr_err("vdev command buffer has no end marker");
        return -1;
    }
    pr_info("wrote vdev register commands to %s", filename);
    return 0;
}

static int run_managed_api(enum driver_api api, const struct dw200_parameters* params, const struct k_dwe_hw_info* dwe_info,
    const struct video_buffer* input_buffers, const struct video_buffer* lut_map,
    const char* output_directory, unsigned int frame_count)
{
    struct k_dw_settings settings;
    struct managed_output_map output_map[DW200_VSE_OUTPUT_COUNT];
    unsigned int output_count;
    unsigned int input_id;
    struct k_dw_load_request request = { 0 };
    bool vb_initialized = false;
    int fd = -1;
    int result = -1;

    if (build_managed_settings(params, dwe_info, lut_map, &settings, output_map, &output_count, &input_id) < 0) {
        return -1;
    }
    if (api == DRIVER_API_LEGACY && input_id != 0) {
        pr_err("legacy API only supports the DWE input path");
        return -1;
    }
    if (initialize_managed_vb(&settings) < 0) {
        return -1;
    }
    vb_initialized = true;

    fd = open(DW200_DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("open " DW200_DEVICE_PATH);
        goto cleanup;
    }
    if (print_driver_version(fd) < 0) {
        goto cleanup;
    }

    int setup_command = api == DRIVER_API_LEGACY ? K_DWIOC_SETUP : K_DWVIOC_SETUP;
    const char* setup_name = api == DRIVER_API_LEGACY ? "K_DWIOC_SETUP" : "K_DWVIOC_SETUP";
    if (call_driver_ioctl(fd, setup_command, &settings, setup_name) < 0) {
        goto cleanup;
    }
    if (api == DRIVER_API_VDEV && dump_vdev_commands(fd, output_directory) < 0) {
        goto cleanup;
    }

    int load_command = api == DRIVER_API_LEGACY ? K_DWIOC_LOAD : K_DWVIOC_LOAD;
    const char* load_name = api == DRIVER_API_LEGACY ? "K_DWIOC_LOAD" : "K_DWVIOC_LOAD";
    for (unsigned int frame = 0; frame < frame_count; frame++) {
        pr_info("%s frame %u/%u", driver_api_name(api), frame + 1, frame_count);
        prepare_load_request(params, input_buffers, input_id, &request);
        if (call_driver_ioctl(fd, load_command, &request, load_name) < 0
            || save_managed_outputs(params, &settings, output_map, output_count, &request, output_directory,
                   driver_api_name(api))
                < 0) {
            goto cleanup;
        }
        release_managed_outputs(&request);
    }
    result = 0;

cleanup:
    release_managed_outputs(&request);
    if (fd >= 0) {
        close(fd);
    }
    if (vb_initialized) {
        kd_mpi_vb_exit();
    }
    return result;
}

int main(int argc, char* argv[])
{
    pr_info("version: %s %s", __DATE__, __TIME__);
    struct command_line_options options;
    if (parse_command_line(argc, argv, &options) < 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    struct dw200_parameters params = { 0 };
    struct dewarp_distortion_map distortion_map = { 0 };
    struct k_dwe_hw_info dwe_info = { 0 };
    struct k_vse_params vse_info = { 0 };
    struct video_buffer input_buffers[DW200_INPUT_COUNT] = { 0 };
    struct video_buffer lut_map = { 0 };
    char user_map_filename[PATH_BUFFER_SIZE] = { 0 };
    char input_filenames[DW200_INPUT_COUNT][PATH_BUFFER_SIZE] = { 0 };
    char output_directory[PATH_BUFFER_SIZE] = { 0 };
    int result = EXIT_FAILURE;

    get_config_directory(options.config_filename, output_directory, sizeof(output_directory));
    if (output_directory[0] == '\0') {
        pr_err("configuration path is too long");
        goto cleanup;
    }
    if (parse_config(options.config_filename, input_filenames[0], input_filenames[1], user_map_filename,
            sizeof(user_map_filename), &distortion_map, &params)
        < 0) {
        goto cleanup;
    }

    bool use_vse_output0 = needs_vse_output0(&params);
    if (use_vse_output0
        && (params.output_res[1].enable || params.output_res[2].enable || params.output_res[3].enable
            || params.input_res[1].enable)) {
        pr_err("output 0 conversion cannot be combined with output 1-3 or input 1");
        goto cleanup;
    }
    if (options.api == DRIVER_API_ALL && !params.input_res[0].enable) {
        pr_err("--api all requires input 0 because the legacy API has no VSE DMA mode");
        goto cleanup;
    }

    for (unsigned int i = 0; i < DW200_INPUT_COUNT; i++) {
        resolve_path_from_config(options.config_filename, input_filenames[i], sizeof(input_filenames[i]));
    }
    pr_info("API: %s, frames: %u", driver_api_name(options.api), options.frame_count);
    pr_info("input 0: %s, input 1: %s", input_filenames[0], input_filenames[1]);

    set_params(&params, use_vse_output0, &dwe_info, &vse_info);
    if (params.input_res[0].enable) {
        if (allocate_video_buffer(&lut_map, MAX_MAP_SIZE) < 0) {
            goto cleanup;
        }
        if (build_dewarp_map(&distortion_map, &params, &dwe_info, user_map_filename, lut_map.virt_addr) < 0) {
            goto cleanup;
        }
        pr_info("LUT map buffer: %08x,%p", (uint32_t)lut_map.phy_addr, lut_map.virt_addr);
    }

    for (unsigned int i = 0; i < DW200_INPUT_COUNT; i++) {
        if (!params.input_res[i].enable) {
            continue;
        }
        uint32_t size = resolution_size(&params.input_res[i]);
        if (size == 0 || allocate_video_buffer(&input_buffers[i], size) < 0) {
            goto cleanup;
        }
        if (load_input_image(input_filenames[i], i, &input_buffers[i]) < 0) {
            goto cleanup;
        }
        pr_info("input %u buffer: %08x,%p,%u", i, (uint32_t)input_buffers[i].phy_addr, input_buffers[i].virt_addr,
            input_buffers[i].size);
    }

    if ((options.api == DRIVER_API_LOW_LEVEL || options.api == DRIVER_API_ALL)
        && run_low_level_api(&params, &dwe_info, &vse_info, input_buffers, &lut_map, output_directory, options.frame_count)
            < 0) {
        goto cleanup;
    }
    if ((options.api == DRIVER_API_LEGACY || options.api == DRIVER_API_ALL)
        && run_managed_api(DRIVER_API_LEGACY, &params, &dwe_info, input_buffers, &lut_map, output_directory,
               options.frame_count)
            < 0) {
        goto cleanup;
    }
    if ((options.api == DRIVER_API_VDEV || options.api == DRIVER_API_ALL)
        && run_managed_api(DRIVER_API_VDEV, &params, &dwe_info, input_buffers, &lut_map, output_directory, options.frame_count)
            < 0) {
        goto cleanup;
    }
    result = EXIT_SUCCESS;

cleanup:
    free_video_buffer(&lut_map);
    for (unsigned int i = 0; i < DW200_INPUT_COUNT; i++) {
        free_video_buffer(&input_buffers[i]);
    }
    pr_info("test done");
    return result;
}
