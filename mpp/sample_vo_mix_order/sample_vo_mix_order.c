#include "kd_display.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vo_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OSD0_BLUE K_VO_LAYER_OSD0 // ID 4
#define OSD1_RED  K_VO_LAYER_OSD1 // ID 5

static void fill_frame(k_video_frame_info* vf, uint32_t color)
{
    uint32_t* vaddr = (uint32_t*)kd_mpi_sys_mmap(vf->v_frame.phys_addr[0], vf->v_frame.height * vf->v_frame.stride[0]);
    for (int i = 0; i < vf->v_frame.width * vf->v_frame.height; i++)
        vaddr[i] = color;
    kd_mpi_sys_munmap(vaddr, vf->v_frame.height * vf->v_frame.stride[0]);
}

/**
 * Updates the Z-order using the new struct.
 * Note: Per requirement, this must be called before layers are enabled.
 */
void set_z_order(k_vo_layer_id bottom_id, k_vo_layer_id top_id)
{
    struct vo_disp_layer_mix_priority_t mix_prio;

    // Start with default: Slot 0=ID 0, Slot 1=ID 1 ... Slot 7=ID 7
    mix_prio.reg = K_VO_DEFAULT_MIX_ORDER;

    // Reassign Slot 4 and Slot 5
    mix_prio.bits.layer4_sel = bottom_id;
    mix_prio.bits.layer5_sel = top_id;

    printf("[DEBUG] Writing Priority Reg: 0x%016lx\n", mix_prio.reg);
    kd_display_set_layer_mix_order(mix_prio.reg);
}

int main(int argc, char** argv)
{
    k_connector_type   conn = (argc > 1) ? (k_connector_type)atoi(argv[1]) : 20;
    k_vb_blk_handle    h0, h1;
    k_video_frame_info vf_blue, vf_red;
    k_s32              pool_id;

    kd_display_init(conn);
    kd_mpi_vb_set_config(&(k_vb_config) { .max_pool_cnt = 64 });
    kd_mpi_vb_init();

    pool_id
        = kd_mpi_vb_create_pool(&(k_vb_pool_config) { .blk_cnt = 6, .blk_size = 300 * 300 * 4, .mode = VB_REMAP_MODE_NONE });

    /* 1. Prepare Buffers */
    h0                           = kd_mpi_vb_get_block(pool_id, 300 * 300 * 4, NULL);
    vf_blue.pool_id              = pool_id;
    vf_blue.v_frame.phys_addr[0] = kd_mpi_vb_handle_to_phyaddr(h0);
    vf_blue.v_frame.width        = 300;
    vf_blue.v_frame.height       = 300;
    vf_blue.v_frame.stride[0]    = 300 * 4;
    vf_blue.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    fill_frame(&vf_blue, 0xFF0000FF); // Blue

    h1                          = kd_mpi_vb_get_block(pool_id, 300 * 300 * 4, NULL);
    vf_red.pool_id              = pool_id;
    vf_red.v_frame.phys_addr[0] = kd_mpi_vb_handle_to_phyaddr(h1);
    vf_red.v_frame.width        = 300;
    vf_red.v_frame.height       = 300;
    vf_red.v_frame.stride[0]    = 300 * 4;
    vf_red.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    fill_frame(&vf_red, 0xFFFF0000); // Red

    /* --- STEP 1: Initial Setup (Red Slot 5 over Blue Slot 4) --- */
    printf("\n--- Step 1: Initial Mix Order (Red on Top) ---\n");
    set_z_order(OSD0_BLUE, OSD1_RED); // Slot 4=Blue, Slot 5=Red

    kd_display_layer_configure(OSD0_BLUE, PIXEL_FORMAT_ARGB_8888, 300, 300, 50, 50);
    kd_display_layer_configure(OSD1_RED, PIXEL_FORMAT_ARGB_8888, 300, 300, 150, 150);

    kd_display_layer_enable(OSD0_BLUE);
    kd_display_layer_enable(OSD1_RED);

    kd_display_layer_push_frame(OSD0_BLUE, &vf_blue);
    kd_display_layer_push_frame(OSD1_RED, &vf_red);
    sleep(3);

    /* --- STEP 2: Priority Swap (Requires Disable -> Config Mix -> Enable) --- */
    printf("\n--- Step 2: Swap Mix Order (Blue on Top) ---\n");
    kd_display_layer_disable(OSD0_BLUE);
    kd_display_layer_disable(OSD1_RED);

    // Swap: Slot 4 gets Red (ID 5), Slot 5 gets Green (ID 4)
    set_z_order(OSD1_RED, OSD0_BLUE);

    kd_display_layer_enable(OSD0_BLUE);
    kd_display_layer_enable(OSD1_RED);

    // Push Green to OSD0 to visually confirm the content update and priority latch
    kd_display_layer_push_frame(OSD0_BLUE, &vf_blue);
    kd_display_layer_push_frame(OSD1_RED, &vf_red);
    sleep(3);

    /* --- Cleanup --- */
    kd_display_layer_disable(OSD0_BLUE);
    kd_display_layer_disable(OSD1_RED);
    kd_display_deinit();

    kd_mpi_vb_release_block(h0);
    kd_mpi_vb_release_block(h1);

    kd_mpi_vb_exit();
    return 0;
}
