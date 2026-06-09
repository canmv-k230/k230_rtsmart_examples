/* Copyright (c) 2025, Canaan Bright Sight Co., Ltd
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

/*
 * Simple VO‑OSD format sweep test.
 *
 * 参考 src/rtsmart/mpp/userapps/sample/sample_vo/vo_test.c 里的 OSD 相关代码，
 * 把 OSD 图层测试单独拆出来，方便在指定分辨率上快速把所有支持的 OSD 格式
 * 各显示一遍彩条。
 *
 * 命令行：
 *   vo_test_osd -c <connector_type> -l <layer_id> [-w <width>] [-h <height>] ...
 *
 * 示例（先用 list_connector 查看 connector type，再测试 OSD0）：
 *   vo_test_osd -c <connector_type> -l 4
 *
 * 程序行为：
 *   - 初始化 connector、电源、VO；
 *   - 依次枚举所有 OSD 支持的像素格式：
 *       RGB565/BGR565/RGB565_LE/BGR565_LE
 *       RGB888/BGR888
 *       ARGB8888/ABGR8888/BGRA8888
 *       ARGB4444/ABGR4444
 *       ARGB1555/ABGR1555
 *       MONO8
 *   - 每种格式在指定 OSD layer 上显示一张彩条图（带橙色边框），停留约 2 秒；
 *   - 按 Ctrl+C 可提前结束。
 *
 * 实现尽量保持“直接可读”，不做多余抽象。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>
#include <sys/mman.h>

#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"

#include "mpi_vb_api.h"
#include "mpi_connector_api.h"
#include "mpi_sys_api.h"

#include "kd_display.h"

#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align)-1))

static volatile bool g_exit = false;

static int resolve_connector_size(k_connector_type *connector_type, k_u32 *width, k_u32 *height)
{
    k_connector_info info;

    memset(&info, 0, sizeof(info));
    if (kd_mpi_get_connector_info(*connector_type, &info) == K_SUCCESS) {
        if (*width == 0)
            *width = info.resolution.hactive;
        if (*height == 0)
            *height = info.resolution.vactive;
    }

    if (*width == 0)
        *width = K_CONN_WIDTH(*connector_type);
    if (*height == 0)
        *height = K_CONN_HEIGHT(*connector_type);

    if (*width == 0 || *height == 0) {
        printf("ERROR: unable to resolve display size for connector type %u, please pass -w/-h\n", *connector_type);
        return -1;
    }

    return 0;
}

static void sig_handler(int sig_no)
{
    (void)sig_no;
    g_exit = true;
    printf("Caught signal, will exit after current format...\n");
}

typedef struct {
    k_pixel_format format;
    k_vo_position  offset;
    k_vo_size      act_size;
    k_u32          size;         /* total bytes of image */
    k_u32          stride;       /* stride in 64‑bit words (for VO attr) */
    k_u8           global_alpha;
} osd_info;

typedef struct {
    const char    *name;
    k_pixel_format fmt;
} osd_fmt_entry;

/* All OSD formats supported by drv_vo.c::vo_osd_set_stride_and_format() */
static const osd_fmt_entry g_osd_formats[] = {
    { "rgb565",      PIXEL_FORMAT_RGB_565 },
    { "bgr565",      PIXEL_FORMAT_BGR_565 },
    { "rgb565_le",   PIXEL_FORMAT_RGB_565_LE },
    { "bgr565_le",   PIXEL_FORMAT_BGR_565_LE },
    { "rgb888",      PIXEL_FORMAT_RGB_888 },
    { "bgr888",      PIXEL_FORMAT_BGR_888 },
    { "argb8888",    PIXEL_FORMAT_ARGB_8888 },
    { "abgr8888",    PIXEL_FORMAT_ABGR_8888 },
    { "bgra8888",    PIXEL_FORMAT_BGRA_8888 },
    { "argb4444",    PIXEL_FORMAT_ARGB_4444 },
    { "abgr4444",    PIXEL_FORMAT_ABGR_4444 },
    { "argb1555",    PIXEL_FORMAT_ARGB_1555 },
    { "abgr1555",    PIXEL_FORMAT_ABGR_1555 },
    { "mono8",       PIXEL_FORMAT_RGB_MONOCHROME_8BPP },
};

static const int g_osd_formats_count =
    (int)(sizeof(g_osd_formats) / sizeof(g_osd_formats[0]));

static k_s32 g_osd_pool_id = VB_INVALID_POOLID;

/* ------------------------------------------------------------------------- */
/* basic helpers shared by pool/attr/fill logic                             */
/* ------------------------------------------------------------------------- */

static inline k_u16 rgb888_to_rgb565(k_u8 r, k_u8 g, k_u8 b)
{
    return (k_u16)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

static inline k_u32 osd_bytes_per_pixel(k_pixel_format fmt)
{
    switch (fmt) {
    case PIXEL_FORMAT_ARGB_8888:
    case PIXEL_FORMAT_ABGR_8888:
    case PIXEL_FORMAT_BGRA_8888:
        return 4;
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888:
        return 3;
    case PIXEL_FORMAT_RGB_565:
    case PIXEL_FORMAT_BGR_565:
    case PIXEL_FORMAT_RGB_565_LE:
    case PIXEL_FORMAT_BGR_565_LE:
    case PIXEL_FORMAT_ARGB_4444:
    case PIXEL_FORMAT_ABGR_4444:
    case PIXEL_FORMAT_ARGB_1555:
    case PIXEL_FORMAT_ABGR_1555:
        return 2;
    case PIXEL_FORMAT_RGB_MONOCHROME_8BPP:
        return 1;
    default:
        /* fall back to 32bpp for unknown formats */
        return 4;
    }
}

static inline k_u16 pack_argb4444(k_u8 a, k_u8 r, k_u8 g, k_u8 b)
{
    /* ARGB4444: [15:12]A [11:8]R [7:4]G [3:0]B */
    return (k_u16)(((a & 0xF0) << 8) |
                   ((r & 0xF0) << 4) |
                   ((g & 0xF0) << 0) |
                   ((b & 0xF0) >> 4));
}

static inline k_u16 pack_abgr4444(k_u8 a, k_u8 r, k_u8 g, k_u8 b)
{
    /* ABGR4444: [15:12]A [11:8]B [7:4]G [3:0]R */
    return (k_u16)(((a & 0xF0) << 8) |
                   ((b & 0xF0) << 4) |
                   ((g & 0xF0) << 0) |
                   ((r & 0xF0) >> 4));
}

static inline k_u16 pack_argb1555(k_u8 a, k_u8 r, k_u8 g, k_u8 b)
{
    /* ARGB1555: [15]A [14:10]R [9:5]G [4:0]B */
    k_u16 aa = (a >= 128) ? 1 : 0;
    k_u16 rr = (k_u16)(r >> 3) & 0x1F;
    k_u16 gg = (k_u16)(g >> 3) & 0x1F;
    k_u16 bb = (k_u16)(b >> 3) & 0x1F;
    return (k_u16)((aa << 15) | (rr << 10) | (gg << 5) | bb);
}

static inline k_u16 pack_abgr1555(k_u8 a, k_u8 r, k_u8 g, k_u8 b)
{
    /* ABGR1555: [15]A [14:10]B [9:5]G [4:0]R */
    k_u16 aa = (a >= 128) ? 1 : 0;
    k_u16 rr = (k_u16)(r >> 3) & 0x1F;
    k_u16 gg = (k_u16)(g >> 3) & 0x1F;
    k_u16 bb = (k_u16)(b >> 3) & 0x1F;
    return (k_u16)((aa << 15) | (bb << 10) | (gg << 5) | rr);
}

/* ------------------------------------------------------------------------- */
/* VB and connector helpers                                                  */
/* ------------------------------------------------------------------------- */

/* VB init/exit：仅给本 sample 用，配置非常保守 */
static k_s32 osd_vb_init(void)
{
    k_vb_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_pool_cnt = 10;

    k_s32 ret = kd_mpi_vb_set_config(&cfg);
    if (ret) {
        printf("kd_mpi_vb_set_config failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vb_init();
    if (ret) {
        printf("kd_mpi_vb_init failed, ret=%d\n", ret);
        return ret;
    }
    return K_SUCCESS;
}

static void osd_vb_exit(void)
{
    kd_mpi_vb_exit();
}

/* 从指定 VB pool 申请一块 buffer，并填充到 k_video_frame_info 里 */
static k_vb_blk_handle vo_alloc_frame_to_vf(k_s32 pool_id,
                                            k_video_frame_info *vf_info,
                                            void **vaddr)
{
    k_u32 size;
    k_vb_blk_handle handle;
    k_u64 phys;
    void *virt;

    if (!vf_info || !vaddr) {
        return VB_INVALID_HANDLE;
    }

    {
        k_u32 bpp = osd_bytes_per_pixel(vf_info->v_frame.pixel_format);
        size = vf_info->v_frame.height * vf_info->v_frame.width * bpp;
    }

    size = ALIGN_UP(size + 4096, 0x1000);

    handle = kd_mpi_vb_get_block(pool_id, size, NULL);
    if (handle == VB_INVALID_HANDLE) {
        printf("vb_get_block failed\n");
        return VB_INVALID_HANDLE;
    }
    phys = kd_mpi_vb_handle_to_phyaddr(handle);
    if (!phys) {
        printf("vb_handle_to_phyaddr failed\n");
        kd_mpi_vb_release_block(handle);
        return VB_INVALID_HANDLE;
    }

    virt = kd_mpi_sys_mmap(phys, size);
    if (!virt) {
        printf("sys_mmap failed\n");
        kd_mpi_vb_release_block(handle);
        return VB_INVALID_HANDLE;
    }

    vf_info->mod_id = K_ID_VO;
    vf_info->pool_id = pool_id;
    vf_info->v_frame.phys_addr[0] = phys;
    *vaddr = virt;
    return handle;
}

/* ------------------------------------------------------------------------- */
/* 生成 OSD 彩条图案（和 vo_test.c 中保持一致）                         */
/* ------------------------------------------------------------------------- */

static void osd_fill_color_bars(const osd_info *osd, void *vaddr)
{
    const int w = osd->act_size.width;
    const int h = osd->act_size.height;
    const int bars = 8;
    /* 小画面也留一点边框 */
    const int border = (w >= 32 && h >= 32) ? 4 : 2;
    const int inner_w = (w > 2 * border) ? (w - 2 * border) : w;

    if (osd->format == PIXEL_FORMAT_ARGB_8888) {

        static const k_u32 c[8] = {
            0xFFFFFFFF, 0xFFFFFF00, 0xFF00FFFF, 0xFF00FF00,
            0xFFFF00FF, 0xFFFF0000, 0xFF0000FF, 0xFF000000
        };

        /* 亮橙色边框，避免与 8 色条重复 */
        const k_u32 border_color = 0xFFFF8000;

        for (int y = 0; y < h; ++y) {
            k_u32 *row = (k_u32 *)vaddr + y * w;
            for (int x = 0; x < w; ++x) {
                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    row[x] = border_color;
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    if (idx >= bars) idx = bars - 1;
                    row[x] = c[idx];
                }
            }
        }
        return;
    }

    if (osd->format == PIXEL_FORMAT_ABGR_8888) {
        static const k_u32 c[8] = {
            0xFFFFFFFF, 0xFF00FFFF, 0xFFFFFF00, 0xFF00FF00,
            0xFFFF00FF, 0xFF0000FF, 0xFFFF0000, 0xFF000000
        };
        const k_u32 border_color = 0xFF0080FF; /* 亮橙色，ABGR */

        for (int y = 0; y < h; ++y) {
            k_u32 *row = (k_u32 *)vaddr + y * w;
            for (int x = 0; x < w; ++x) {
                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    row[x] = border_color;
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    if (idx >= bars) idx = bars - 1;
                    row[x] = c[idx];
                }
            }
        }
        return;
    }

    if (osd->format == PIXEL_FORMAT_BGRA_8888) {
        const k_u32 c[8] = {
            0xFFFFFFFF, /* white  (B=255,G=255,R=255,A=255) */
            0x00FFFFff, /* yellow (B=0,  G=255,R=255,A=255) */
            0xFFFF00ff, /* cyan   (B=255,G=255,R=0,  A=255) */
            0x00FF00ff, /* green  (B=0,  G=255,R=0,  A=255) */
            0xFF00FFff, /* magenta(B=255,G=0,  R=255,A=255) */
            0x0000FFff, /* red    (B=0,  G=0,  R=255,A=255) */
            0xFF0000ff, /* blue   (B=255,G=0,  R=0,  A=255) */
            0x000000ff, /* black  (B=0,  G=0,  R=0,  A=255) */
        };
        const k_u32 border_color = 0x0080FFff; /* 橙色边框 (B=0,G=128,R=255,A=255) */

        for (int y = 0; y < h; ++y) {
            k_u32 *row = (k_u32 *)vaddr + y * w;
            for (int x = 0; x < w; ++x) {
                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    row[x] = border_color;
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    if (idx >= bars) idx = bars - 1;
                    row[x] = c[idx];
                }
            }
        }
        return;
    }

    if (osd->format == PIXEL_FORMAT_RGB_565 || osd->format == PIXEL_FORMAT_BGR_565 ||
        osd->format == PIXEL_FORMAT_RGB_565_LE || osd->format == PIXEL_FORMAT_BGR_565_LE) {
        /* 通用约定（不依赖 CPU 大小端，直接按字节构造） */
        const int is_bgr =
            (osd->format == PIXEL_FORMAT_BGR_565) ||
            (osd->format == PIXEL_FORMAT_BGR_565_LE);
        const int is_le =
            (osd->format == PIXEL_FORMAT_RGB_565_LE) ||
            (osd->format == PIXEL_FORMAT_BGR_565_LE);

        for (int y = 0; y < h; ++y) {
            k_u8 *row = (k_u8 *)vaddr + y * w * 2;
            for (int x = 0; x < w; ++x) {
                k_u8 r = 0, g = 0, b = 0;

                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    /* 橙色边框：R=255,G=128,B=0 */
                    r = 255; g = 128; b = 0;
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    if (idx >= bars) idx = bars - 1;
                    switch (idx) {
                        case 0: r=255; g=255; b=255; break; // white
                        case 1: r=255; g=255; b=0;   break; // yellow
                        case 2: r=0;   g=255; b=255; break; // cyan
                        case 3: r=0;   g=255; b=0;   break; // green
                        case 4: r=255; g=0;   b=255; break; // magenta
                        case 5: r=255; g=0;   b=0;   break; // red
                        case 6: r=0;   g=0;   b=255; break; // blue
                        default: r=0;  g=0;   b=0;   break; // black
                    }
                }

                k_u16 v;
                if (!is_bgr) {
                    /* RGB565: R5G6B5 */
                    v = rgb888_to_rgb565(r, g, b);
                } else {
                    /* BGR565: B5G6R5：把 R/B 对调后按同一 packing 算法得到 16bit 值 */
                    v = rgb888_to_rgb565(b, g, r);
                }

                k_u8 hi = (k_u8)(v >> 8);
                k_u8 lo = (k_u8)(v & 0xFF);
                int pos = x * 2;

                if (!is_le) {
                    /* 非 LE：大端 */
                    row[pos + 0] = hi;
                    row[pos + 1] = lo;
                } else {
                    /* LE：小端 */
                    row[pos + 0] = lo;
                    row[pos + 1] = hi;
                }
            }
        }
        return;
    }

    if (osd->format == PIXEL_FORMAT_RGB_888 || osd->format == PIXEL_FORMAT_BGR_888) {
        const int bpp = 3;
        const k_u8 br = 255, bg = 128, bb = 0; /* 橙色边框 */

        for (int y = 0; y < h; ++y) {
            k_u8 *row = (k_u8 *)vaddr + y * w * bpp;
            for (int x = 0; x < w; ++x) {
                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    /* 边框像素 */
                    if (osd->format == PIXEL_FORMAT_RGB_888) {
                        row[x*3+0] = br; row[x*3+1] = bg; row[x*3+2] = bb;
                    } else { /* BGR888 */
                        row[x*3+0] = bb; row[x*3+1] = bg; row[x*3+2] = br;
                    }
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    k_u8 r=0,g=0,b=0;
                    if (idx >= bars) idx = bars - 1;
                    switch (idx) {
                        case 0: r=255; g=255; b=255; break; // white
                        case 1: r=255; g=255; b=0;   break; // yellow
                        case 2: r=0;   g=255; b=255; break; // cyan
                        case 3: r=0;   g=255; b=0;   break; // green
                        case 4: r=255; g=0;   b=255; break; // magenta
                        case 5: r=255; g=0;   b=0;   break; // red
                        case 6: r=0;   g=0;   b=255; break; // blue
                        default: r=0;  g=0;   b=0;   break; // black
                    }
                    if (osd->format == PIXEL_FORMAT_RGB_888) {
                        row[x*3+0] = r; row[x*3+1] = g; row[x*3+2] = b;
                    } else { // BGR888
                        row[x*3+0] = b; row[x*3+1] = g; row[x*3+2] = r;
                    }
                }
            }
        }
        return;
    }

    if (osd->format == PIXEL_FORMAT_ARGB_4444 || osd->format == PIXEL_FORMAT_ABGR_4444) {
        const k_u16 border_color =
            (osd->format == PIXEL_FORMAT_ARGB_4444) ?
                pack_argb4444(0xF0, 255, 128, 0) :   /* 橙色边框 */
                pack_abgr4444(0xF0, 255, 128, 0);

        for (int y = 0; y < h; ++y) {
            k_u16 *row = (k_u16 *)vaddr + y * w;
            for (int x = 0; x < w; ++x) {
                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    row[x] = border_color;
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    k_u8 r = 0, g = 0, b = 0;
                    if (idx >= bars) idx = bars - 1;
                    switch (idx) {
                        case 0: r=255; g=255; b=255; break; // white
                        case 1: r=255; g=255; b=0;   break; // yellow
                        case 2: r=0;   g=255; b=255; break; // cyan
                        case 3: r=0;   g=255; b=0;   break; // green
                        case 4: r=255; g=0;   b=255; break; // magenta
                        case 5: r=255; g=0;   b=0;   break; // red
                        case 6: r=0;   g=0;   b=255; break; // blue
                        default: r=0;  g=0;   b=0;   break; // black
                    }
                    if (osd->format == PIXEL_FORMAT_ARGB_4444)
                        row[x] = pack_argb4444(0xF0, r, g, b);
                    else
                        row[x] = pack_abgr4444(0xF0, r, g, b);
                }
            }
        }
        return;
    }

    if (osd->format == PIXEL_FORMAT_ARGB_1555 || osd->format == PIXEL_FORMAT_ABGR_1555) {
        const k_u16 border_color =
            (osd->format == PIXEL_FORMAT_ARGB_1555) ?
                pack_argb1555(0xFF, 255, 128, 0) :   /* 橙色边框 */
                pack_abgr1555(0xFF, 255, 128, 0);

        for (int y = 0; y < h; ++y) {
            k_u16 *row = (k_u16 *)vaddr + y * w;
            for (int x = 0; x < w; ++x) {
                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    row[x] = border_color;
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    k_u8 r = 0, g = 0, b = 0;
                    if (idx >= bars) idx = bars - 1;
                    switch (idx) {
                        case 0: r=255; g=255; b=255; break; // white
                        case 1: r=255; g=255; b=0;   break; // yellow
                        case 2: r=0;   g=255; b=255; break; // cyan
                        case 3: r=0;   g=255; b=0;   break; // green
                        case 4: r=255; g=0;   b=255; break; // magenta
                        case 5: r=255; g=0;   b=0;   break; // red
                        case 6: r=0;   g=0;   b=255; break; // blue
                        default: r=0;  g=0;   b=0;   break; // black
                    }
                    if (osd->format == PIXEL_FORMAT_ARGB_1555)
                        row[x] = pack_argb1555(0xFF, r, g, b);
                    else
                        row[x] = pack_abgr1555(0xFF, r, g, b);
                }
            }
        }
        return;
    }

    if (osd->format == PIXEL_FORMAT_RGB_MONOCHROME_8BPP) {
        const k_u8 border_level = 220;
        const k_u8 level[8] = {255, 220, 190, 160, 130, 100, 70, 40};

        for (int y = 0; y < h; ++y) {
            k_u8 *row = (k_u8 *)vaddr + y * w;
            for (int x = 0; x < w; ++x) {
                if (x < border || x >= (w - border) ||
                    y < border || y >= (h - border)) {
                    row[x] = border_level;
                } else {
                    int rel_x = x - border;
                    int idx = (rel_x * bars) / inner_w;
                    if (idx >= bars) idx = bars - 1;
                    row[x] = level[idx];
                }
            }
        }
        return;
    }

    /* default: solid black, 作为兜底 */
    memset(vaddr, 0x00, osd->size);
}

/* ------------------------------------------------------------------------- */
/* main: 依次跑完所有 OSD format                                            */
/* ------------------------------------------------------------------------- */

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -c <type>   Connector type. Use list_connector to view supported types\n");
    printf("  -w <width>  OSD image width [default: panel width]\n");
    printf("  -h <height> OSD image height [default: panel height]\n");
    printf("  -l <layer>  VO Layer ID (4-7 for OSD)\n");
    printf("  -r <degree> Rotation (0, 90, 180, 270) [default: 0]\n");
    printf("  -x <offset> X offset [default: 0]\n");
    printf("  -y <offset> Y offset [default: 0]\n");
    printf("\nExample:\n");
    printf("  list_connector\n");
    printf("  %s -c <connector_type> -l 4 -r 90 -x 10 -y 10\n", prog);
}

int main(int argc, char **argv)
{
    k_connector_type connector_type = 0;
    k_u32 width = 0, height = 0;
    k_vo_layer_id layer_id = 4;
    int rot_val = 0;
    k_s32 offset_x = 0, offset_y = 0;

    int opt;
    bool c_set = false;

    while ((opt = getopt(argc, argv, "c:w:h:l:r:x:y:")) != -1) {
        switch (opt) {
            case 'c': connector_type = (k_connector_type)atoi(optarg); c_set = true; break;
            case 'w': width = (k_u32)atoi(optarg); break;
            case 'h': height = (k_u32)atoi(optarg); break;
            case 'l': layer_id = (k_vo_layer_id)atoi(optarg); break;
            case 'r': rot_val = atoi(optarg); break;
            case 'x': offset_x = atoi(optarg); break;
            case 'y': offset_y = atoi(optarg); break;
            default: print_usage(argv[0]); return -1;
        }
    }

    if (!c_set) {
        printf("ERROR: Missing required arguments (-c)\n");
        print_usage(argv[0]);
        return -1;
    }

    if (resolve_connector_size(&connector_type, &width, &height) != 0) {
        print_usage(argv[0]);
        return -1;
    }

    k_gdma_rotation_e rotate = GDMA_ROTATE_DEGREE_0;
    if (rot_val == 90) rotate = GDMA_ROTATE_DEGREE_90;
    else if (rot_val == 180) rotate = GDMA_ROTATE_DEGREE_180;
    else if (rot_val == 270) rotate = GDMA_ROTATE_DEGREE_270;

    printf("vo_test_osd: connector=%u, size=%ux%u, layer=%d, rotate=%d, offset=(%d,%d)\n",
           connector_type, width, height, layer_id, rot_val, offset_x, offset_y);

    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* 1) VB System Init */
    if (osd_vb_init() != K_SUCCESS) return -1;

    /* 2) Initialize Display */
    if(kd_display_init(connector_type, 0, 0, rotate) != 0) {
        printf("ERROR: kd_display_init failed\n");
        osd_vb_exit();
        return -1;
    }

    /* Create VB Pool */
    k_vb_pool_config pool_cfg;
    memset(&pool_cfg, 0, sizeof(pool_cfg));
    pool_cfg.blk_cnt = 1;
    pool_cfg.blk_size = ALIGN_UP(width * height * 4 + 4096, 0x1000);
    g_osd_pool_id = kd_mpi_vb_create_pool(&pool_cfg);

    /* 3) Format Loop */
    for (int i = 0; i < g_osd_formats_count && !g_exit; ++i) {
        const osd_fmt_entry *fmt = &g_osd_formats[i];
        printf("\n=== [%2d/%2d] Testing %s ===\n", i + 1, g_osd_formats_count, fmt->name);

        /* Configure Layer with offsets */
        if(kd_display_layer_configure(layer_id, fmt->fmt, width, height, offset_x, offset_y) != 0) continue;

        // Note: Check your kd_display.h if it supports setting offsets. 
        // If not, you may need to call kd_mpi_vo_set_video_layer_attr directly.
        kd_display_layer_enable(layer_id);

        /* Alloc and Fill Frame */
        k_video_frame_info vf;
        void *osd_vaddr = NULL;
        memset(&vf, 0, sizeof(vf));
        vf.v_frame.width = width;
        vf.v_frame.height = height;
        vf.v_frame.pixel_format = fmt->fmt;
        vf.v_frame.stride[0] = width * osd_bytes_per_pixel(fmt->fmt);

        k_vb_blk_handle blk = vo_alloc_frame_to_vf(g_osd_pool_id, &vf, &osd_vaddr);
        if (blk != VB_INVALID_HANDLE) {
            osd_info info = { .format = fmt->fmt, .act_size = {width, height}, .size = vf.v_frame.stride[0] * height };
            osd_fill_color_bars(&info, osd_vaddr);

            kd_display_layer_push_frame(layer_id, &vf);

            for (int t = 0; t < 20 && !g_exit; ++t) usleep(100000);

            kd_display_layer_disable(layer_id);
            kd_mpi_sys_munmap(osd_vaddr, pool_cfg.blk_size);
            kd_mpi_vb_release_block(blk);
        }
    }

    kd_display_deinit();
    osd_vb_exit();
    return 0;
}
