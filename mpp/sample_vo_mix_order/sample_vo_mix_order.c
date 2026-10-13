#include "kd_display.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vo_api.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OSD0_BLUE K_VO_LAYER_OSD0
#define OSD1_RED  K_VO_LAYER_OSD1

#define FRAME_WIDTH  300U
#define FRAME_HEIGHT 300U
#define FRAME_BPP    4U
#define FRAME_SIZE   (FRAME_WIDTH * FRAME_HEIGHT * FRAME_BPP)

static void print_usage(const char* program_name)
{
    printf("Usage: %s [-c connector_type]\n", program_name);
    printf("       %s [connector_type]\n", program_name);
    printf("Default connector: %u (ST7701 480x800)\n", ST7701_V1_MIPI_2LAN_480X800_30FPS);
}

static int parse_connector_type(const char* text, k_connector_type* connector_type)
{
    char*         end;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX)
        return -1;

    *connector_type = (k_connector_type)value;
    return 0;
}

static int parse_arguments(int argc, char** argv, k_connector_type* connector_type)
{
    int opt;

    *connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;

    /* Keep the original positional form while supporting the common -c form. */
    if (argc == 2 && argv[1][0] != '-')
        return parse_connector_type(argv[1], connector_type);

    while ((opt = getopt(argc, argv, "c:h")) != -1) {
        switch (opt) {
        case 'c':
            if (parse_connector_type(optarg, connector_type) != 0) {
                printf("ERROR: invalid connector type: %s\n", optarg);
                return -1;
            }
            break;
        case 'h':
            print_usage(argv[0]);
            return 1;
        default:
            return -1;
        }
    }

    if (optind != argc) {
        printf("ERROR: unexpected argument: %s\n", argv[optind]);
        return -1;
    }

    return 0;
}

static k_s32 fill_frame(k_video_frame_info* frame, k_u32 color)
{
    size_t size = (size_t)frame->v_frame.height * frame->v_frame.stride[0];
    k_u8*  data = kd_mpi_sys_mmap(frame->v_frame.phys_addr[0], size);

    if (data == NULL) {
        printf("ERROR: failed to map frame buffer\n");
        return K_FAILED;
    }

    for (k_u32 y = 0; y < frame->v_frame.height; ++y) {
        k_u32* row = (k_u32*)(data + (size_t)y * frame->v_frame.stride[0]);
        for (k_u32 x = 0; x < frame->v_frame.width; ++x)
            row[x] = color;
    }

    kd_mpi_sys_munmap(data, size);
    return K_SUCCESS;
}

static k_s32 set_z_order(k_vo_layer_id bottom_id, k_vo_layer_id top_id)
{
    struct vo_disp_layer_mix_priority_t mix_prio;

    mix_prio.reg = K_VO_DEFAULT_MIX_ORDER;
    mix_prio.bits.layer4_sel = bottom_id;
    mix_prio.bits.layer5_sel = top_id;

    printf("[DEBUG] Writing Priority Reg: 0x%016lx\n", mix_prio.reg);
    return kd_display_set_layer_mix_order(mix_prio.reg);
}

int main(int argc, char** argv)
{
    k_connector_type   connector_type;
    k_vb_blk_handle    blue_handle = VB_INVALID_HANDLE;
    k_vb_blk_handle    red_handle  = VB_INVALID_HANDLE;
    k_video_frame_info blue_frame;
    k_video_frame_info red_frame;
    k_s32              pool_id             = (k_s32)VB_INVALID_POOLID;
    int                display_initialized = 0;
    int                vb_initialized      = 0;
    int                blue_enabled        = 0;
    int                red_enabled         = 0;
    int                result              = EXIT_FAILURE;
    int                parse_result;

    parse_result = parse_arguments(argc, argv, &connector_type);
    if (parse_result != 0) {
        if (parse_result < 0)
            print_usage(argv[0]);
        return parse_result < 0 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    if (kd_display_init(connector_type) != K_SUCCESS) {
        printf("ERROR: failed to initialize connector %u\n", connector_type);
        goto cleanup;
    }
    display_initialized = 1;

    if (kd_mpi_vb_set_config(&(k_vb_config) { .max_pool_cnt = 64 }) != K_SUCCESS) {
        printf("ERROR: failed to configure VB\n");
        goto cleanup;
    }
    if (kd_mpi_vb_init() != K_SUCCESS) {
        printf("ERROR: failed to initialize VB\n");
        goto cleanup;
    }
    vb_initialized = 1;

    pool_id = kd_mpi_vb_create_pool_ex(FRAME_SIZE, 2, VB_REMAP_MODE_NOCACHE);
    if (pool_id == (k_s32)VB_INVALID_POOLID) {
        printf("ERROR: failed to create VB pool\n");
        goto cleanup;
    }

    memset(&blue_frame, 0, sizeof(blue_frame));
    blue_handle = kd_mpi_vb_get_block(pool_id, FRAME_SIZE, NULL);
    if (blue_handle == VB_INVALID_HANDLE) {
        printf("ERROR: failed to allocate blue frame\n");
        goto cleanup;
    }
    blue_frame.pool_id              = pool_id;
    blue_frame.v_frame.phys_addr[0] = kd_mpi_vb_handle_to_phyaddr(blue_handle);
    blue_frame.v_frame.width        = FRAME_WIDTH;
    blue_frame.v_frame.height       = FRAME_HEIGHT;
    blue_frame.v_frame.stride[0]    = FRAME_WIDTH * FRAME_BPP;
    blue_frame.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    if (fill_frame(&blue_frame, 0xFF0000FF) != K_SUCCESS)
        goto cleanup;

    memset(&red_frame, 0, sizeof(red_frame));
    red_handle = kd_mpi_vb_get_block(pool_id, FRAME_SIZE, NULL);
    if (red_handle == VB_INVALID_HANDLE) {
        printf("ERROR: failed to allocate red frame\n");
        goto cleanup;
    }
    red_frame.pool_id              = pool_id;
    red_frame.v_frame.phys_addr[0] = kd_mpi_vb_handle_to_phyaddr(red_handle);
    red_frame.v_frame.width        = FRAME_WIDTH;
    red_frame.v_frame.height       = FRAME_HEIGHT;
    red_frame.v_frame.stride[0]    = FRAME_WIDTH * FRAME_BPP;
    red_frame.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    if (fill_frame(&red_frame, 0xFFFF0000) != K_SUCCESS)
        goto cleanup;

    printf("\n--- Step 1: Initial Mix Order (Red on Top) ---\n");
    if (set_z_order(OSD0_BLUE, OSD1_RED) != K_SUCCESS
        || kd_display_layer_configure(OSD0_BLUE, PIXEL_FORMAT_ARGB_8888, FRAME_WIDTH, FRAME_HEIGHT, 50, 50) != K_SUCCESS
        || kd_display_layer_configure(OSD1_RED, PIXEL_FORMAT_ARGB_8888, FRAME_WIDTH, FRAME_HEIGHT, 150, 150) != K_SUCCESS) {
        printf("ERROR: failed to configure display layers\n");
        goto cleanup;
    }

    if (kd_display_layer_enable(OSD0_BLUE) != K_SUCCESS) {
        printf("ERROR: failed to enable OSD0\n");
        goto cleanup;
    }
    blue_enabled = 1;
    if (kd_display_layer_enable(OSD1_RED) != K_SUCCESS) {
        printf("ERROR: failed to enable OSD1\n");
        goto cleanup;
    }
    red_enabled = 1;

    if (kd_display_layer_push_frame(OSD0_BLUE, &blue_frame) != K_SUCCESS
        || kd_display_layer_push_frame(OSD1_RED, &red_frame) != K_SUCCESS) {
        printf("ERROR: failed to push initial frames\n");
        goto cleanup;
    }
    sleep(3);

    printf("\n--- Step 2: Swap Mix Order (Blue on Top) ---\n");
    if (kd_display_layer_disable(OSD0_BLUE) != K_SUCCESS
        || kd_display_layer_disable(OSD1_RED) != K_SUCCESS) {
        printf("ERROR: failed to disable display layers\n");
        goto cleanup;
    }
    blue_enabled = 0;
    red_enabled  = 0;

    if (set_z_order(OSD1_RED, OSD0_BLUE) != K_SUCCESS) {
        printf("ERROR: failed to update mix order\n");
        goto cleanup;
    }

    if (kd_display_layer_enable(OSD0_BLUE) != K_SUCCESS) {
        printf("ERROR: failed to re-enable OSD0\n");
        goto cleanup;
    }
    blue_enabled = 1;
    if (kd_display_layer_enable(OSD1_RED) != K_SUCCESS) {
        printf("ERROR: failed to re-enable OSD1\n");
        goto cleanup;
    }
    red_enabled = 1;

    if (kd_display_layer_push_frame(OSD0_BLUE, &blue_frame) != K_SUCCESS
        || kd_display_layer_push_frame(OSD1_RED, &red_frame) != K_SUCCESS) {
        printf("ERROR: failed to push reordered frames\n");
        goto cleanup;
    }
    sleep(3);
    result = EXIT_SUCCESS;

cleanup:
    if (blue_enabled)
        kd_display_layer_disable(OSD0_BLUE);
    if (red_enabled)
        kd_display_layer_disable(OSD1_RED);
    if (display_initialized)
        kd_display_deinit();
    if (blue_handle != VB_INVALID_HANDLE)
        kd_mpi_vb_release_block(blue_handle);
    if (red_handle != VB_INVALID_HANDLE)
        kd_mpi_vb_release_block(red_handle);
    if (pool_id != (k_s32)VB_INVALID_POOLID)
        kd_mpi_vb_destory_pool((k_u32)pool_id);
    if (vb_initialized)
        kd_mpi_vb_exit();

    return result;
}
