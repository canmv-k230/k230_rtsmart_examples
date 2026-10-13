/* Copyright (c) 2026 Canaan Bright Sight Co., Ltd */

#ifndef MPP_LIFETIME_IOCTL_H
#define MPP_LIFETIME_IOCTL_H

/*
 * The public MPP repository does not export the kernel ioctl headers. This
 * test exercises the device ABI directly, so keep the subset it needs local.
 *
 * ABI snapshot source:
 *   src/rtsmart/mpp @ f2ec742b7cd3492ccd46bbdf4ac3d95ff6c5b948
 *   include/ioctl/k_{sys,vb,vicap,vdec,venc,vpu_test}_ioctl.h
 */
#include "k_ioctl.h"
#include "k_sys_comm.h"
#include "k_vb_comm.h"
#include "k_vdec_comm.h"
#include "k_venc_comm.h"
#include "k_vicap_comm.h"
#include "k_video_comm.h"

typedef struct {
    k_mpp_chn src_chn;
    k_mpp_chn dest_chn;
} k_sys_bind_args;

#define K_IOC_SYS_BIND_CTRL \
    _IOW(K_IOC_TYPE_SYS, 0, k_sys_bind_args)
#define K_IOC_SYS_UNBIND_CTRL \
    _IOW(K_IOC_TYPE_SYS, 1, k_sys_bind_args)
#define K_IOC_SYS_GETBINDBYDEST \
    _IOWR(K_IOC_TYPE_SYS, 2, k_sys_bind_args)

typedef struct {
    k_u32 pool_id;
    k_u32 blk_id;
    k_u64 blk_phys_addr;
    k_u32 blk_size;
    k_vb_blk_handle blk_handle;
    char mmz_name[MAX_MMZ_NAME_LEN];
} k_vb_blk_info;

typedef struct {
    k_vb_blk_handle handle;
    k_video_supplement supplement;
} k_vb_supplement_info;

#define K_VB_SetConfig \
    _IOW(K_IOC_TYPE_VB, 0, k_vb_config)
#define K_VB_GetConfig \
    _IOWR(K_IOC_TYPE_VB, 1, k_vb_config)
#define K_VB_Init \
    _IO(K_IOC_TYPE_VB, 2)
#define K_VB_Exit \
    _IO(K_IOC_TYPE_VB, 3)
#define K_VB_CreatePool \
    _IOWR(K_IOC_TYPE_VB, 4, k_vb_pool_info)
/* "Destory" is the spelling used by the source UAPI. */
#define K_VB_DestoryPool \
    _IOW(K_IOC_TYPE_VB, 5, k_u32)
#define K_VB_GetBlock \
    _IOWR(K_IOC_TYPE_VB, 6, k_vb_blk_info)
#define K_VB_ReleaseBlock \
    _IOW(K_IOC_TYPE_VB, 7, k_vb_blk_handle)
#define K_VB_PhysAddr2Handle \
    _IOW(K_IOC_TYPE_VB, 8, k_u64)
#define K_VB_Handle2PhysAddr \
    _IOW(K_IOC_TYPE_VB, 9, k_u64)
#define K_VB_Handle2PoolId \
    _IOW(K_IOC_TYPE_VB, 10, k_vb_blk_handle)
#define K_VB_InquireUserCnt \
    _IOW(K_IOC_TYPE_VB, 18, k_vb_blk_handle)
#define K_VB_GetSupplementAttr \
    _IOW(K_IOC_TYPE_VB, 19, k_vb_supplement_info)
#define K_VB_SetSupplementConfig \
    _IOW(K_IOC_TYPE_VB, 20, k_vb_supplement_config)
#define K_VB_GetSupplementConfig \
    _IOW(K_IOC_TYPE_VB, 21, k_vb_supplement_config)
#define K_VB_IncBlockRefCnt \
    _IOW(K_IOC_TYPE_VB, 22, k_vb_blk_handle)
#define K_VB_DecBlockRefCnt \
    _IOW(K_IOC_TYPE_VB, 23, k_vb_blk_handle)
#define K_VB_GetBlockRefCnt \
    _IOW(K_IOC_TYPE_VB, 24, k_vb_blk_handle)

#define K_IOC_VICAP_DevReset \
    _IOW(K_IOC_TYPE_VICAP, 0, k_u8)
#define K_IOC_VICAP_SetMclk \
    _IOW(K_IOC_TYPE_VICAP, 13, k_vicap_mclk)

typedef struct {
    k_u32 chn_num;
    k_vdec_chn_attr attr;
} vdec_chn_info;

typedef struct {
    k_u32 chn_num;
    k_u32 pool_id;
} vdec_chn_pool_info;

#define K_IOC_VDEC_RequestChn \
    _IOW(K_IOC_TYPE_VDEC, 0, int)
#define K_IOC_VDEC_ReleaseChn \
    _IOW(K_IOC_TYPE_VDEC, 1, int)
#define K_IOC_VDEC_CreateChn \
    _IOW(K_IOC_TYPE_VDEC, 2, vdec_chn_info)
#define K_IOC_VDEC_StartChn \
    _IOW(K_IOC_TYPE_VDEC, 4, k_u32)
#define K_IOC_VDEC_AttachVbPool \
    _IOW(K_IOC_TYPE_VDEC, 12, vdec_chn_pool_info)

typedef struct {
    k_u32 chn_num;
    k_venc_chn_attr attr;
} venc_chn_info;

typedef struct {
    k_u32 chn_num;
    k_venc_pack *pack;
    k_u32 pack_cnt;
    k_s32 milli_sec;
} venc_stream;

typedef struct {
    k_u32 chn_num;
    k_video_frame_info frame;
    k_s32 milli_sec;
} venc_frame;

typedef struct {
    k_u32 chn_num;
    k_venc_chn_status status;
} venc_chn_status;

typedef struct {
    k_u8 index;
    union {
        k_venc_2d_osd_attr osd_attr;
        k_venc_2d_border_attr border_attr;
    };
} k_venc_2d_index_attr;

typedef union {
    k_venc_2d_calc_mode mode;
    k_venc_2d_csc_attr csc_attr;
    k_venc_2d_index_attr index_attr;
    k_venc_2d_coef_attr coef_attr;
    k_venc_2d_color_gamut color_gamut;
} k_venc_2d_attr;

typedef struct {
    k_u32 chn_num;
    k_venc_2d_attr attr;
} venc_2d_chn_info;

typedef struct {
    k_u32 chn_num;
    k_u32 pool_id;
    k_s32 buff_num;
} venc_chn_pool_info;

#if defined(__riscv_xlen) && (__riscv_xlen == 64)
_Static_assert(sizeof(k_sys_bind_args) == 24,
               "k_sys_bind_args ABI snapshot changed");
_Static_assert(sizeof(k_vb_blk_info) == 40,
               "k_vb_blk_info ABI snapshot changed");
_Static_assert(sizeof(vdec_chn_info) == 24,
               "vdec_chn_info ABI snapshot changed");
_Static_assert(sizeof(venc_stream) == 24,
               "venc_stream ABI snapshot changed");
#endif

#define K_IOC_VENC_RequestChn \
    _IOW(K_IOC_TYPE_VENC, 0, int)
#define K_IOC_VENC_ReleaseChn \
    _IOW(K_IOC_TYPE_VENC, 1, int)
#define K_IOC_VENC_CreateChn \
    _IOW(K_IOC_TYPE_VENC, 2, venc_chn_info)
#define K_IOC_VENC_DestroyChn \
    _IOW(K_IOC_TYPE_VENC, 3, k_u32)
#define K_IOC_VENC_StartChn \
    _IOW(K_IOC_TYPE_VENC, 4, k_u32)
#define K_IOC_VENC_StopChn \
    _IOW(K_IOC_TYPE_VENC, 5, k_u32)
#define K_IOC_VENC_GetStream \
    _IOW(K_IOC_TYPE_VENC, 6, venc_stream)
#define K_IOC_VENC_RelStream \
    _IOW(K_IOC_TYPE_VENC, 7, venc_stream)
#define K_IOC_VENC_QueryStatus \
    _IOW(K_IOC_TYPE_VENC, 8, venc_chn_status)
#define K_IOC_VENC_SendFrame \
    _IOW(K_IOC_TYPE_VENC, 9, venc_frame)
#define K_IOC_VENC_2D_Attach \
    _IOW(K_IOC_TYPE_VENC, 39, venc_2d_chn_info)
#define K_IOC_VENC_2D_Detach \
    _IOW(K_IOC_TYPE_VENC, 40, venc_2d_chn_info)
#define K_IOC_VENC_AttachVbPool \
    _IOW(K_IOC_TYPE_VENC, 45, venc_chn_pool_info)
#define K_IOC_VENC_DetachVbPool \
    _IOW(K_IOC_TYPE_VENC, 46, k_u32)

/* Coverage-only controls. Production kernels do not handle these commands. */
#define K_IOC_VENC_TEST_CLEANUP_BUSY_ONCE \
    _IOW(K_IOC_TYPE_VENC, 0x7f, k_u32)
#define K_IOC_VDEC_TEST_CLEANUP_BUSY_ONCE \
    _IOW(K_IOC_TYPE_VDEC, 0x7f, k_u32)

#endif
