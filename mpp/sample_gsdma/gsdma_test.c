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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>

#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "k_gsdma_comm.h"
#include "mpi_vb_api.h"
#include "mpi_gsdma_api.h"
#include "mpi_sys_api.h"
#include "hal_utils.h"

#define GDMA_TEST_WIDTH         640
#define GDMA_TEST_HEIGHT        480
#define GDMA_TEST_TIMEOUT_MS    1000

#define SDMA_TEST_TIMEOUT_MS    1000

/* Test configuration */
typedef struct {
    k_pixel_format pixel_format;
    k_gdma_rotation_e rotation;
    k_bool x_mirror;
    k_bool y_mirror;
    const char *desc;
} gdma_test_case_t;

/* SDMA memcpy test configuration */
typedef struct {
    k_u64 size;
    k_u32 src_alignment;
    k_u32 dst_alignment;
    const char *desc;
} sdma_memcpy_test_case_t;

/* SDMA memset test configuration */
typedef struct {
    k_u64 size;
    k_u32 alignment;
    k_u32 data;
    k_sdma_data_size_e data_size;
    const char *desc;
} sdma_memset_test_case_t;

/* Test statistics */
typedef struct {
    k_u32 total_tests;
    k_u32 passed_tests;
    k_u32 failed_tests;
} gdma_test_stats_t;

static gdma_test_stats_t g_test_stats = {0};

/* ==================== SDMA 测试相关定义 ==================== */

typedef struct {
    k_u32 line_size;        /* 单行有效数据长度（字节） */
    k_u32 line_num;         /* 行数（2D 模式） */
    k_u32 stride;           /* 源缓冲 stride（字节）: stride >= line_size */
    k_u8  dimension;        /* 0:1D, 1:2D */
    const char *desc;
} sdma_test_case_t;

/* 1D + 2D（gap=0 和 gap>0）测试用例 */
static sdma_test_case_t g_sdma_test_cases[] = {
    /* 1D 模式：总长度即 line_size */
    {4096,  1, 4096, 0, "SDMA 1D Transfer 4KB"},
    {8192,  1, 8192, 0, "SDMA 1D Transfer 8KB"},
    {16384, 1, 16384,0, "SDMA 1D Transfer 16KB"},
    {32768, 1, 32768,0, "SDMA 1D Transfer 32KB"},

    /* 2D 模式，gap=0：stride == line_size */
    {1024, 64, 1024, 1, "SDMA 2D 1024x64 contiguous (gap=0)"},

    /* 2D 模式，真实 gap：stride > line_size */
    {512,  128, 1024,1, "SDMA 2D 512x128 with gap (stride=1024, gap=512)"},
};

#define SDMA_TEST_CASE_COUNT (sizeof(g_sdma_test_cases) / sizeof(g_sdma_test_cases[0]))

/* SDMA memcpy test cases - including unaligned sizes */
static sdma_memcpy_test_case_t g_sdma_memcpy_test_cases[] = {
    /* Small unaligned sizes */
    {1,     1, 1, "SDMA Memcpy 1 byte (unaligned)"},
    {3,     1, 1, "SDMA Memcpy 3 bytes (unaligned)"},
    {7,     1, 1, "SDMA Memcpy 7 bytes (unaligned)"},
    {15,    1, 1, "SDMA Memcpy 15 bytes (unaligned)"},
    {31,    1, 1, "SDMA Memcpy 31 bytes (unaligned)"},
    {63,    1, 1, "SDMA Memcpy 63 bytes (unaligned)"},
    {127,   1, 1, "SDMA Memcpy 127 bytes (unaligned)"},
    {255,   1, 1, "SDMA Memcpy 255 bytes (unaligned)"},

    /* Mixed alignments with unaligned sizes */
    {1023,  1, 1, "SDMA Memcpy 1023 bytes (unaligned)"},
    {1025,  1, 1, "SDMA Memcpy 1025 bytes (unaligned)"},
    {4095,  1, 1, "SDMA Memcpy 4095 bytes (unaligned)"},
    {4097,  1, 1, "SDMA Memcpy 4097 bytes (unaligned)"},

    /* Various sizes with different alignments */
    {1024,  1, 1, "SDMA Memcpy 1KB (unaligned)"},
    {1024,  2, 2, "SDMA Memcpy 1KB (2-byte aligned)"},
    {1024,  4, 4, "SDMA Memcpy 1KB (4-byte aligned)"},
    {1024,  1, 4, "SDMA Memcpy 1KB (src unaligned, dst aligned)"},
    {1024,  4, 1, "SDMA Memcpy 1KB (src aligned, dst unaligned)"},

    {2048,  1, 1, "SDMA Memcpy 2KB (unaligned)"},
    {2048,  2, 2, "SDMA Memcpy 2KB (2-byte aligned)"},
    {2048,  4, 4, "SDMA Memcpy 2KB (4-byte aligned)"},
    {2048,  1, 2, "SDMA Memcpy 2KB (src unaligned, dst 2-byte aligned)"},

    {4096,  1, 1, "SDMA Memcpy 4KB (unaligned)"},
    {4096,  2, 2, "SDMA Memcpy 4KB (2-byte aligned)"},
    {4096,  4, 4, "SDMA Memcpy 4KB (4-byte aligned)"},
    {4096,  1, 4, "SDMA Memcpy 4KB (src unaligned, dst aligned)"},
    {4096,  4, 1, "SDMA Memcpy 4KB (src aligned, dst unaligned)"},

    {8192,  1, 1, "SDMA Memcpy 8KB (unaligned)"},
    {8192,  4, 4, "SDMA Memcpy 8KB (4-byte aligned)"},

    {16384, 1, 1, "SDMA Memcpy 16KB (unaligned)"},
    {16384, 2, 2, "SDMA Memcpy 16KB (2-byte aligned)"},

    {32768, 1, 1, "SDMA Memcpy 32KB (unaligned)"},
    {32768, 4, 4, "SDMA Memcpy 32KB (4-byte aligned)"},

    {65536, 1, 1, "SDMA Memcpy 64KB (unaligned)"},
    {65536, 2, 2, "SDMA Memcpy 64KB (2-byte aligned)"},

    /* Large unaligned sizes */
    {65535, 1, 1, "SDMA Memcpy 65535 bytes (unaligned)"},
    {131071,1, 1, "SDMA Memcpy 131071 bytes (unaligned)"},
};

#define SDMA_MEMCPY_TEST_CASE_COUNT (sizeof(g_sdma_memcpy_test_cases) / sizeof(g_sdma_memcpy_test_cases[0]))

/* SDMA memset test cases - including unaligned sizes */
static sdma_memset_test_case_t g_sdma_memset_test_cases[] = {
    /* 1-byte patterns with unaligned sizes */
    { 1, 1, 0x00, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 1 byte 0x00 (1-byte)" },
    { 3, 1, 0xFF, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 3 bytes 0xFF (1-byte)" },
    { 7, 1, 0xAA, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 7 bytes 0xAA (1-byte)" },
    { 15, 1, 0x55, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 15 bytes 0x55 (1-byte)" },
    { 31, 1, 0x12, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 31 bytes 0x12 (1-byte)" },
    { 63, 1, 0x34, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 63 bytes 0x34 (1-byte)" },
    { 127, 1, 0x56, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 127 bytes 0x56 (1-byte)" },
    { 255, 1, 0x78, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 255 bytes 0x78 (1-byte)" },

    /* 1-byte patterns with various sizes */
    { 1023, 1, 0x00, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 1023 bytes 0x00 (1-byte)" },
    { 1025, 1, 0xFF, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 1025 bytes 0xFF (1-byte)" },
    { 1024, 1, 0xAA, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 1KB 0xAA (1-byte)" },
    { 1024, 1, 0x55, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 1KB 0x55 (1-byte)" },

    /* 2-byte patterns with unaligned sizes and alignments */
    { 2, 2, 0x0000, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 2 bytes 0x0000 (2-byte aligned)" },
    { 3, 2, 0xFFFF, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 3 bytes 0xFFFF (2-byte aligned)" },
    { 4, 2, 0xAAAA, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 4 bytes 0xAAAA (2-byte aligned)" },
    { 5, 2, 0x5555, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 5 bytes 0x5555 (2-byte aligned)" },
    { 6, 2, 0x1234, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 6 bytes 0x1234 (2-byte aligned)" },
    { 7, 2, 0x5678, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 7 bytes 0x5678 (2-byte aligned)" },

    { 1023, 2, 0x1234, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 1023 bytes 0x1234 (2-byte aligned)" },
    { 1025, 2, 0x5678, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 1025 bytes 0x5678 (2-byte aligned)" },
    { 1022, 2, 0x9ABC, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 1022 bytes 0x9ABC (2-byte aligned)" },
    { 1026, 2, 0xDEF0, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 1026 bytes 0xDEF0 (2-byte aligned)" },

    { 2048, 2, 0x0000, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 2KB 0x0000 (2-byte aligned)" },
    { 2048, 2, 0xFFFF, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 2KB 0xFFFF (2-byte aligned)" },
    { 2048, 1, 0xAAAA, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 2KB 0xAAAA (2-byte unaligned)" },
    { 2048, 1, 0x5555, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 2KB 0x5555 (2-byte unaligned)" },

    /* 4-byte patterns with unaligned sizes and alignments */
    { 4, 4, 0x00000000, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 4 bytes 0x00000000 (4-byte aligned)" },
    { 5, 4, 0xFFFFFFFF, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 5 bytes 0xFFFFFFFF (4-byte aligned)" },
    { 6, 4, 0xDEADBEEF, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 6 bytes 0xDEADBEEF (4-byte aligned)" },
    { 7, 4, 0x12345678, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 7 bytes 0x12345678 (4-byte aligned)" },
    { 8, 4, 0x87654321, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 8 bytes 0x87654321 (4-byte aligned)" },
    { 9, 4, 0x11223344, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 9 bytes 0x11223344 (4-byte aligned)" },

    { 1023, 4, 0x12345678, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 1023 bytes 0x12345678 (4-byte aligned)" },
    { 1025, 4, 0x87654321, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 1025 bytes 0x87654321 (4-byte aligned)" },
    { 1020, 4, 0xDEADBEEF, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 1020 bytes 0xDEADBEEF (4-byte aligned)" },
    { 1028, 4, 0xCAFEBABE, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 1028 bytes 0xCAFEBABE (4-byte aligned)" },

    { 4096, 4, 0x00000000, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 4KB 0x00000000 (4-byte aligned)" },
    { 4096, 4, 0xFFFFFFFF, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 4KB 0xFFFFFFFF (4-byte aligned)" },
    { 4096, 1, 0xDEADBEEF, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 4KB 0xDEADBEEF (4-byte unaligned)" },
    { 4096, 1, 0x12345678, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 4KB 0x12345678 (4-byte unaligned)" },

    /* Large unaligned sizes */
    { 65535, 1, 0xAA, SDMA_DATA_SIZE_1_BYTE, "SDMA Memset 65535 bytes 0xAA (1-byte unaligned)" },
    { 65535, 2, 0x1234, SDMA_DATA_SIZE_2_BYTE, "SDMA Memset 65535 bytes 0x1234 (2-byte aligned)" },
    { 65535, 4, 0xDEADBEEF, SDMA_DATA_SIZE_4_BYTE, "SDMA Memset 65535 bytes 0xDEADBEEF (4-byte aligned)" },
};

#define SDMA_MEMSET_TEST_CASE_COUNT (sizeof(g_sdma_memset_test_cases) / sizeof(g_sdma_memset_test_cases[0]))

/* Test cases covering all formats and rotation angles */
static gdma_test_case_t g_test_cases[] = {
    /* ========== YUV 400 format (单平面) ========== */
    {PIXEL_FORMAT_YUV_400, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "YUV400 0deg"},
    {PIXEL_FORMAT_YUV_400, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "YUV400 90deg"},
    {PIXEL_FORMAT_YUV_400, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "YUV400 180deg"},
    {PIXEL_FORMAT_YUV_400, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "YUV400 270deg"},

    /* ========== YUV 420 Semiplanar (双平面) ========== */
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "YUV420SP 0deg"},
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "YUV420SP 90deg"},
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "YUV420SP 180deg"},
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "YUV420SP 270deg"},

    /* ========== YVU 420 Semiplanar (双平面) ========== */
    {PIXEL_FORMAT_YVU_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "YVU420SP 0deg"},
    {PIXEL_FORMAT_YVU_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "YVU420SP 90deg"},
    {PIXEL_FORMAT_YVU_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "YVU420SP 180deg"},
    {PIXEL_FORMAT_YVU_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "YVU420SP 270deg"},

    /* ========== YVU 420 Planar (三平面) ========== */
    {PIXEL_FORMAT_YVU_PLANAR_420, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "YVU420P 0deg"},
    {PIXEL_FORMAT_YVU_PLANAR_420, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "YVU420P 90deg"},
    {PIXEL_FORMAT_YVU_PLANAR_420, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "YVU420P 180deg"},
    {PIXEL_FORMAT_YVU_PLANAR_420, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "YVU420P 270deg"},

    /* ========== YVU 444 Planar (三平面，全分辨率) ========== */
    {PIXEL_FORMAT_YVU_PLANAR_444, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "YVU444P 0deg"},
    {PIXEL_FORMAT_YVU_PLANAR_444, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "YVU444P 90deg"},
    {PIXEL_FORMAT_YVU_PLANAR_444, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "YVU444P 180deg"},
    {PIXEL_FORMAT_YVU_PLANAR_444, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "YVU444P 270deg"},

    /* ========== RGB 16-bit formats ========== */
    {PIXEL_FORMAT_RGB_555, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "RGB555 0deg"},
    {PIXEL_FORMAT_RGB_555, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "RGB555 90deg"},
    {PIXEL_FORMAT_RGB_555, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "RGB555 180deg"},
    {PIXEL_FORMAT_RGB_555, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "RGB555 270deg"},

    {PIXEL_FORMAT_RGB_565, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "RGB565 0deg"},
    {PIXEL_FORMAT_RGB_565, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "RGB565 90deg"},
    {PIXEL_FORMAT_RGB_565, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "RGB565 180deg"},
    {PIXEL_FORMAT_RGB_565, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "RGB565 270deg"},

    /* ========== BGR 16-bit formats ========== */
    {PIXEL_FORMAT_BGR_555, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "BGR555 0deg"},
    {PIXEL_FORMAT_BGR_555, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "BGR555 90deg"},
    {PIXEL_FORMAT_BGR_555, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "BGR555 180deg"},
    {PIXEL_FORMAT_BGR_555, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "BGR555 270deg"},

    {PIXEL_FORMAT_BGR_565, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "BGR565 0deg"},
    {PIXEL_FORMAT_BGR_565, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "BGR565 90deg"},
    {PIXEL_FORMAT_BGR_565, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "BGR565 180deg"},
    {PIXEL_FORMAT_BGR_565, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "BGR565 270deg"},

    /* ========== RGB 24-bit formats ========== */
    {PIXEL_FORMAT_RGB_888, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "RGB888 0deg"},
    {PIXEL_FORMAT_RGB_888, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "RGB888 90deg"},
    {PIXEL_FORMAT_RGB_888, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "RGB888 180deg"},
    {PIXEL_FORMAT_RGB_888, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "RGB888 270deg"},

    {PIXEL_FORMAT_BGR_888, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "BGR888 0deg"},
    {PIXEL_FORMAT_BGR_888, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "BGR888 90deg"},
    {PIXEL_FORMAT_BGR_888, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "BGR888 180deg"},
    {PIXEL_FORMAT_BGR_888, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "BGR888 270deg"},

    /* ========== ARGB 16-bit formats ========== */
    {PIXEL_FORMAT_ARGB_1555, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ARGB1555 0deg"},
    {PIXEL_FORMAT_ARGB_1555, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ARGB1555 90deg"},
    {PIXEL_FORMAT_ARGB_1555, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ARGB1555 180deg"},
    {PIXEL_FORMAT_ARGB_1555, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ARGB1555 270deg"},

    {PIXEL_FORMAT_ARGB_4444, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ARGB4444 0deg"},
    {PIXEL_FORMAT_ARGB_4444, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ARGB4444 90deg"},
    {PIXEL_FORMAT_ARGB_4444, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ARGB4444 180deg"},
    {PIXEL_FORMAT_ARGB_4444, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ARGB4444 270deg"},

    /* ========== ARGB 24-bit formats ========== */
    {PIXEL_FORMAT_ARGB_8565, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ARGB8565 0deg"},
    {PIXEL_FORMAT_ARGB_8565, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ARGB8565 90deg"},
    {PIXEL_FORMAT_ARGB_8565, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ARGB8565 180deg"},
    {PIXEL_FORMAT_ARGB_8565, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ARGB8565 270deg"},

    /* ========== ARGB 32-bit formats ========== */
    {PIXEL_FORMAT_ARGB_8888, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ARGB8888 0deg"},
    {PIXEL_FORMAT_ARGB_8888, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ARGB8888 90deg"},
    {PIXEL_FORMAT_ARGB_8888, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ARGB8888 180deg"},
    {PIXEL_FORMAT_ARGB_8888, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ARGB8888 270deg"},

    /* ========== ABGR 16-bit formats ========== */
    {PIXEL_FORMAT_ABGR_1555, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ABGR1555 0deg"},
    {PIXEL_FORMAT_ABGR_1555, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ABGR1555 90deg"},
    {PIXEL_FORMAT_ABGR_1555, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ABGR1555 180deg"},
    {PIXEL_FORMAT_ABGR_1555, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ABGR1555 270deg"},

    {PIXEL_FORMAT_ABGR_4444, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ABGR4444 0deg"},
    {PIXEL_FORMAT_ABGR_4444, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ABGR4444 90deg"},
    {PIXEL_FORMAT_ABGR_4444, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ABGR4444 180deg"},
    {PIXEL_FORMAT_ABGR_4444, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ABGR4444 270deg"},

    /* ========== ABGR 24-bit formats ========== */
    {PIXEL_FORMAT_ABGR_8565, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ABGR8565 0deg"},
    {PIXEL_FORMAT_ABGR_8565, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ABGR8565 90deg"},
    {PIXEL_FORMAT_ABGR_8565, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ABGR8565 180deg"},
    {PIXEL_FORMAT_ABGR_8565, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ABGR8565 270deg"},

    /* ========== ABGR 32-bit formats ========== */
    {PIXEL_FORMAT_ABGR_8888, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "ABGR8888 0deg"},
    {PIXEL_FORMAT_ABGR_8888, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "ABGR8888 90deg"},
    {PIXEL_FORMAT_ABGR_8888, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "ABGR8888 180deg"},
    {PIXEL_FORMAT_ABGR_8888, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "ABGR8888 270deg"},

    /* ========== BGR 888 Planar (三平面) ========== */
    {PIXEL_FORMAT_BGR_888_PLANAR, GDMA_ROTATE_DEGREE_0,   K_FALSE, K_FALSE, "BGR888P 0deg"},
    {PIXEL_FORMAT_BGR_888_PLANAR, GDMA_ROTATE_DEGREE_90,  K_FALSE, K_FALSE, "BGR888P 90deg"},
    {PIXEL_FORMAT_BGR_888_PLANAR, GDMA_ROTATE_DEGREE_180, K_FALSE, K_FALSE, "BGR888P 180deg"},
    {PIXEL_FORMAT_BGR_888_PLANAR, GDMA_ROTATE_DEGREE_270, K_FALSE, K_FALSE, "BGR888P 270deg"},

    /* ========== Mirror tests (使用YUV420SP作为代表) ========== */
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_0, K_TRUE,  K_FALSE, "YUV420SP X-Mirror"},
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_0, K_FALSE, K_TRUE,  "YUV420SP Y-Mirror"},
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, GDMA_ROTATE_DEGREE_0, K_TRUE,  K_TRUE,  "YUV420SP XY-Mirror"},
};

#define TEST_CASE_COUNT (sizeof(g_test_cases) / sizeof(g_test_cases[0]))

/* Calculate frame size based on pixel format */
static k_u32 gdma_calc_frame_size(k_pixel_format pixel_format, k_u16 width, k_u16 height)
{
    k_u32 size = 0;

    switch (pixel_format) {
    /* 单平面 8-bit */
    case PIXEL_FORMAT_YUV_400:
        size = width * height;
        break;

    /* 双平面/三平面 YUV 420 */
    case PIXEL_FORMAT_YUV_SEMIPLANAR_420:
    case PIXEL_FORMAT_YVU_SEMIPLANAR_420:
    case PIXEL_FORMAT_YVU_PLANAR_420:
        size = width * height * 3 / 2;
        break;

    /* 三平面 YUV/BGR 444 (全分辨率) */
    case PIXEL_FORMAT_YVU_PLANAR_444:
    case PIXEL_FORMAT_BGR_888_PLANAR:
        size = width * height * 3;
        break;

    /* 单平面 16-bit */
    case PIXEL_FORMAT_RGB_555:
    case PIXEL_FORMAT_RGB_565:
    case PIXEL_FORMAT_BGR_555:
    case PIXEL_FORMAT_BGR_565:
    case PIXEL_FORMAT_ARGB_1555:
    case PIXEL_FORMAT_ARGB_4444:
    case PIXEL_FORMAT_ABGR_1555:
    case PIXEL_FORMAT_ABGR_4444:
        size = width * height * 2;
        break;

    /* 单平面 24-bit */
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888:
    case PIXEL_FORMAT_ARGB_8565:
    case PIXEL_FORMAT_ABGR_8565:
        size = width * height * 3;
        break;

    /* 单平面 32-bit */
    case PIXEL_FORMAT_ARGB_8888:
    case PIXEL_FORMAT_ABGR_8888:
        size = width * height * 4;
        break;

    default:
        printf("Unsupported pixel format: %d\n", pixel_format);
        size = width * height * 3 / 2;
        break;
    }

    return size;
}

/* Setup frame addresses and strides based on pixel format */
static void gdma_setup_frame_info(k_video_frame_info *frame, k_pixel_format pixel_format,
                                   k_u64 phys_addr, k_u64 virt_addr, k_u16 width, k_u16 height)
{
    frame->v_frame.pixel_format = pixel_format;
    frame->v_frame.width = width;
    frame->v_frame.height = height;

    switch (pixel_format) {
    /* 单平面 8-bit (YUV400) */
    case PIXEL_FORMAT_YUV_400:
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width;
        break;

    /* 双平面 YUV420 Semiplanar */
    case PIXEL_FORMAT_YUV_SEMIPLANAR_420:
    case PIXEL_FORMAT_YVU_SEMIPLANAR_420:
        /* Y plane */
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width;
        /* UV plane (交错) */
        frame->v_frame.phys_addr[1] = phys_addr + width * height;
        frame->v_frame.virt_addr[1] = virt_addr + width * height;
        frame->v_frame.stride[1] = width;  /* UV交错，stride与Y相同 */
        break;

    /* 三平面 YUV420 Planar */
    case PIXEL_FORMAT_YVU_PLANAR_420:
        /* Y plane */
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width;
        /* U plane */
        frame->v_frame.phys_addr[1] = phys_addr + width * height;
        frame->v_frame.virt_addr[1] = virt_addr + width * height;
        frame->v_frame.stride[1] = width / 2;  /* U/V独立，宽度减半 */
        /* V plane */
        frame->v_frame.phys_addr[2] = phys_addr + width * height + width * height / 4;
        frame->v_frame.virt_addr[2] = virt_addr + width * height + width * height / 4;
        frame->v_frame.stride[2] = width / 2;
        break;

    /* 三平面 YUV444 Planar (全分辨率) */
    case PIXEL_FORMAT_YVU_PLANAR_444:
        /* Y plane */
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width;
        /* U plane */
        frame->v_frame.phys_addr[1] = phys_addr + width * height;
        frame->v_frame.virt_addr[1] = virt_addr + width * height;
        frame->v_frame.stride[1] = width;  /* 444格式，U/V平面与Y相同尺寸 */
        /* V plane */
        frame->v_frame.phys_addr[2] = phys_addr + width * height * 2;
        frame->v_frame.virt_addr[2] = virt_addr + width * height * 2;
        frame->v_frame.stride[2] = width;
        break;

    /* 三平面 BGR888 Planar */
    case PIXEL_FORMAT_BGR_888_PLANAR:
        /* B plane */
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width;
        /* G plane */
        frame->v_frame.phys_addr[1] = phys_addr + width * height;
        frame->v_frame.virt_addr[1] = virt_addr + width * height;
        frame->v_frame.stride[1] = width;
        /* R plane */
        frame->v_frame.phys_addr[2] = phys_addr + width * height * 2;
        frame->v_frame.virt_addr[2] = virt_addr + width * height * 2;
        frame->v_frame.stride[2] = width;
        break;

    /* 单平面 16-bit RGB/ARGB/ABGR */
    case PIXEL_FORMAT_RGB_555:
    case PIXEL_FORMAT_RGB_565:
    case PIXEL_FORMAT_BGR_555:
    case PIXEL_FORMAT_BGR_565:
    case PIXEL_FORMAT_ARGB_1555:
    case PIXEL_FORMAT_ARGB_4444:
    case PIXEL_FORMAT_ABGR_1555:
    case PIXEL_FORMAT_ABGR_4444:
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width * 2;  /* 16bit per pixel */
        break;

    /* 单平面 24-bit RGB/ARGB/ABGR */
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888:
    case PIXEL_FORMAT_ARGB_8565:
    case PIXEL_FORMAT_ABGR_8565:
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width * 3;  /* 24bit per pixel */
        break;

    /* 单平面 32-bit ARGB/ABGR */
    case PIXEL_FORMAT_ARGB_8888:
    case PIXEL_FORMAT_ABGR_8888:
        frame->v_frame.phys_addr[0] = phys_addr;
        frame->v_frame.virt_addr[0] = virt_addr;
        frame->v_frame.stride[0] = width * 4;  /* 32bit per pixel */
        break;

    default:
        break;
    }
}

/* Initialize test pattern data */
static void gdma_init_test_pattern(k_u8 *virt_addr, k_pixel_format pixel_format, k_u16 width, k_u16 height)
{
    k_u32 i, j;
    k_u8 *y_plane, *u_plane, *v_plane;
    k_u16 *rgb16_data;
    k_u32 *rgb32_data;

    switch (pixel_format) {
    /* 单平面 8-bit (YUV400) */
    case PIXEL_FORMAT_YUV_400:
        /* Simple gradient: value = row index */
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                virt_addr[i * width + j] = i % 256;
            }
        }
        break;

    /* 双平面 YUV420 Semiplanar */
    case PIXEL_FORMAT_YUV_SEMIPLANAR_420:
    case PIXEL_FORMAT_YVU_SEMIPLANAR_420:
        /* Y plane: value = row index */
        y_plane = virt_addr;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                y_plane[i * width + j] = i % 256;
            }
        }
        /* UV plane: value = column index */
        u_plane = virt_addr + width * height;
        for (i = 0; i < height / 2; i++) {
            for (j = 0; j < width; j++) {  /* UV交错，每行width个字节 */
                u_plane[i * width + j] = j % 256;
            }
        }
        break;

    /* 三平面 YUV420 Planar */
    case PIXEL_FORMAT_YVU_PLANAR_420:
        /* Y plane: value = row index */
        y_plane = virt_addr;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                y_plane[i * width + j] = i % 256;
            }
        }
        /* U plane */
        u_plane = virt_addr + width * height;
        for (i = 0; i < height / 2; i++) {
            for (j = 0; j < width / 2; j++) {
                u_plane[i * width / 2 + j] = (j % 2 == 0) ? 1 : 2;
            }
        }
        /* V plane */
        v_plane = virt_addr + width * height + width * height / 4;
        for (i = 0; i < height / 2; i++) {
            for (j = 0; j < width / 2; j++) {
                v_plane[i * width / 2 + j] = (j % 2 == 0) ? 2 : 1;
            }
        }
        break;

    /* 三平面 YUV444 Planar / BGR888 Planar */
    case PIXEL_FORMAT_YVU_PLANAR_444:
    case PIXEL_FORMAT_BGR_888_PLANAR:
        /* Plane 0: value = row index */
        y_plane = virt_addr;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                y_plane[i * width + j] = i % 256;
            }
        }
        /* Plane 1: value = column index */
        u_plane = virt_addr + width * height;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                u_plane[i * width + j] = j % 256;
            }
        }
        /* Plane 2: value = (row + column) / 2 */
        v_plane = virt_addr + width * height * 2;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                v_plane[i * width + j] = ((i + j) / 2) % 256;
            }
        }
        break;

    /* 单平面 16-bit RGB/BGR/ARGB/ABGR */
    case PIXEL_FORMAT_RGB_555:
    case PIXEL_FORMAT_RGB_565:
    case PIXEL_FORMAT_BGR_555:
    case PIXEL_FORMAT_BGR_565:
    case PIXEL_FORMAT_ARGB_1555:
    case PIXEL_FORMAT_ARGB_4444:
    case PIXEL_FORMAT_ABGR_1555:
    case PIXEL_FORMAT_ABGR_4444:
        rgb16_data = (k_u16 *)virt_addr;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                /* 简单模式：高字节是行号，低字节是列号 */
                rgb16_data[i * width + j] = ((i & 0xFF) << 8) | (j & 0xFF);
            }
        }
        break;

    /* 单平面 24-bit RGB/BGR/ARGB/ABGR */
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888:
    case PIXEL_FORMAT_ARGB_8565:
    case PIXEL_FORMAT_ABGR_8565:
        /* R/G/B gradient pattern */
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                k_u32 offset = (i * width + j) * 3;
                virt_addr[offset + 0] = i % 256;  /* Component 0 */
                virt_addr[offset + 1] = j % 256;  /* Component 1 */
                virt_addr[offset + 2] = ((i + j) / 2) % 256;  /* Component 2 */
            }
        }
        break;

    /* 单平面 32-bit ARGB/ABGR */
    case PIXEL_FORMAT_ARGB_8888:
    case PIXEL_FORMAT_ABGR_8888:
        rgb32_data = (k_u32 *)virt_addr;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                /* Pattern: Alpha=0xFF, R=row, G=col, B=(row+col)/2 */
                rgb32_data[i * width + j] =
                    (0xFF << 24) | ((i & 0xFF) << 16) | ((j & 0xFF) << 8) | (((i + j) / 2) & 0xFF);
            }
        }
        break;

    default:
        memset(virt_addr, 0x80, gdma_calc_frame_size(pixel_format, width, height));
        break;
    }
}

/* Helper function to verify a single plane (rotation + optional mirror) */
static k_bool gdma_verify_plane(k_u8 *src_addr, k_u8 *dst_addr,
                                k_u16 src_width, k_u16 src_height,
                                k_u16 src_stride, k_u16 dst_stride,
                                k_gdma_rotation_e rotation,
                                k_bool x_mirror, k_bool y_mirror,
                                k_u8 bytes_per_pixel,
                                k_u32 *error_count,
                                k_u32 max_errors_to_print,
                                const char *plane_name)
{
    k_u32 i, j;
    k_u32 src_index, dst_index;
    k_u32 plane_errors = 0;
    /* destination dimensions in pixels after rotation (before mirror) */
    k_u16 dst_width, dst_height;

    if (rotation == GDMA_ROTATE_DEGREE_90 || rotation == GDMA_ROTATE_DEGREE_270) {
        dst_width = src_height;
        dst_height = src_width;
    } else {
        dst_width = src_width;
        dst_height = src_height;
    }

    if (bytes_per_pixel == 1) {
        /* 8-bit per pixel */
        for (i = 0; i < src_height && plane_errors < max_errors_to_print; i++) {
            for (j = 0; j < src_width && plane_errors < max_errors_to_print; j++) {
                k_u16 dst_i, dst_j;

                src_index = i * src_stride + j;

                /* 根据旋转角度计算目标位置（先旋转，再镜像） */
                if (rotation == GDMA_ROTATE_DEGREE_0) {
                    dst_i = i;
                    dst_j = j;
                } else if (rotation == GDMA_ROTATE_DEGREE_90) {
                    dst_i = j;
                    dst_j = src_height - i - 1;
                } else if (rotation == GDMA_ROTATE_DEGREE_180) {
                    dst_i = src_height - i - 1;
                    dst_j = src_width - j - 1;
                } else { /* GDMA_ROTATE_DEGREE_270 */
                    dst_i = src_width - j - 1;
                    dst_j = i;
                }

                /* 镜像操作在旋转之后进行，坐标基于目标图像 */
                if (x_mirror) {
                    dst_j = dst_width - dst_j - 1;
                }
                if (y_mirror) {
                    dst_i = dst_height - dst_i - 1;
                }

                dst_index = dst_i * dst_stride + dst_j;

                if (src_addr[src_index] != dst_addr[dst_index]) {
                    if (plane_errors < max_errors_to_print) {
                        printf("  %s: Mismatch at [%u,%u] src=0x%02x dst=0x%02x\n",
                               plane_name, i, j, src_addr[src_index], dst_addr[dst_index]);
                    }
                    plane_errors++;
                }
            }
        }
    } else if (bytes_per_pixel == 2) {
        /* 16-bit per pixel */
        k_u16 *src_16 = (k_u16 *)src_addr;
        k_u16 *dst_16 = (k_u16 *)dst_addr;
        k_u16 src_stride_16 = src_stride / 2;
        k_u16 dst_stride_16 = dst_stride / 2;

        for (i = 0; i < src_height && plane_errors < max_errors_to_print; i++) {
            for (j = 0; j < src_width && plane_errors < max_errors_to_print; j++) {
                k_u16 dst_i, dst_j;

                src_index = i * src_stride_16 + j;

                if (rotation == GDMA_ROTATE_DEGREE_0) {
                    dst_i = i;
                    dst_j = j;
                } else if (rotation == GDMA_ROTATE_DEGREE_90) {
                    dst_i = j;
                    dst_j = src_height - i - 1;
                } else if (rotation == GDMA_ROTATE_DEGREE_180) {
                    dst_i = src_height - i - 1;
                    dst_j = src_width - j - 1;
                } else { /* GDMA_ROTATE_DEGREE_270 */
                    dst_i = src_width - j - 1;
                    dst_j = i;
                }

                if (x_mirror) {
                    dst_j = dst_width - dst_j - 1;
                }
                if (y_mirror) {
                    dst_i = dst_height - dst_i - 1;
                }

                dst_index = dst_i * dst_stride_16 + dst_j;

                if (src_16[src_index] != dst_16[dst_index]) {
                    if (plane_errors < max_errors_to_print) {
                        printf("  %s: Mismatch at [%u,%u] src=0x%04x dst=0x%04x\n",
                               plane_name, i, j, src_16[src_index], dst_16[dst_index]);
                    }
                    plane_errors++;
                }
            }
        }
    }

    *error_count += plane_errors;
    return (plane_errors == 0);
}

/* Verify rotation result - check pixel-by-pixel transformation for all planes */
static k_bool gdma_verify_rotation(k_video_frame_info *src_frame, k_video_frame_info *dst_frame,
                                   k_gdma_rotation_e rotation,
                                   k_bool x_mirror, k_bool y_mirror,
                                   k_pixel_format pixel_format)
{
    k_u16 src_width = src_frame->v_frame.width;
    k_u16 src_height = src_frame->v_frame.height;
    k_u32 error_count = 0;
    k_u32 max_errors_to_print = 10;
    k_u8 plane_count = 0;
    k_u8 *src_addr[3] = {NULL, NULL, NULL};
    k_u8 *dst_addr[3] = {NULL, NULL, NULL};
    k_u32 dst_size[3] = {0};
    k_bool result = K_TRUE;
    k_u8 i;

    /* 确定平面数量和每个平面的参数 */
    switch (pixel_format) {
    /* 单平面格式 */
    case PIXEL_FORMAT_YUV_400:
    case PIXEL_FORMAT_RGB_555:
    case PIXEL_FORMAT_RGB_565:
    case PIXEL_FORMAT_BGR_555:
    case PIXEL_FORMAT_BGR_565:
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888:
    case PIXEL_FORMAT_ARGB_1555:
    case PIXEL_FORMAT_ARGB_4444:
    case PIXEL_FORMAT_ARGB_8565:
    case PIXEL_FORMAT_ARGB_8888:
    case PIXEL_FORMAT_ABGR_1555:
    case PIXEL_FORMAT_ABGR_4444:
    case PIXEL_FORMAT_ABGR_8565:
    case PIXEL_FORMAT_ABGR_8888:
        plane_count = 1;
        break;

    /* 双平面格式 */
    case PIXEL_FORMAT_YUV_SEMIPLANAR_420:
    case PIXEL_FORMAT_YVU_SEMIPLANAR_420:
        plane_count = 2;
        break;

    /* 三平面格式 */
    case PIXEL_FORMAT_YVU_PLANAR_420:
    case PIXEL_FORMAT_YVU_PLANAR_444:
    case PIXEL_FORMAT_BGR_888_PLANAR:
        plane_count = 3;
        break;

    default:
        plane_count = 1;
        break;
    }

    /* 映射所有平面 */
    for (i = 0; i < plane_count; i++) {
        src_addr[i] = (k_u8 *)src_frame->v_frame.virt_addr[i];

        /* 计算目标平面大小 */
        if (i == 0) {
            dst_size[i] = dst_frame->v_frame.stride[i] * dst_frame->v_frame.height;
        } else {
            /* UV平面高度是Y平面的一半（对于420格式） */
            if (pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_420 ||
                pixel_format == PIXEL_FORMAT_YVU_SEMIPLANAR_420 ||
                pixel_format == PIXEL_FORMAT_YVU_PLANAR_420) {
                dst_size[i] = dst_frame->v_frame.stride[i] * (dst_frame->v_frame.height / 2);
            } else {
                dst_size[i] = dst_frame->v_frame.stride[i] * dst_frame->v_frame.height;
            }
        }

        dst_addr[i] = (k_u8 *)(intptr_t)kd_mpi_sys_mmap(dst_frame->v_frame.phys_addr[i], dst_size[i]);
        if (!dst_addr[i]) {
            printf("Failed to map dst frame plane %d\n", i);
            /* 清理已映射的平面 */
            for (k_u8 j = 0; j < i; j++) {
                kd_mpi_sys_munmap(dst_addr[j], dst_size[j]);
            }
            return K_FALSE;
        }
    }

    /* 验证每个平面 */
    for (i = 0; i < plane_count; i++) {
        k_u16 plane_src_width, plane_src_height;
        k_u8 bytes_per_pixel;
        char plane_name[32];

        /* 确定平面名称和参数 */
        if (i == 0) {
            /* 第一平面 (Y/B/RGB) */
            if (pixel_format == PIXEL_FORMAT_BGR_888_PLANAR) {
                snprintf(plane_name, sizeof(plane_name), "Plane0(B)");
            } else {
                snprintf(plane_name, sizeof(plane_name), "Plane0(Y)");
            }
            plane_src_width = src_width;
            plane_src_height = src_height;

            /* 确定字节/像素 */
            if (pixel_format == PIXEL_FORMAT_RGB_555 || pixel_format == PIXEL_FORMAT_RGB_565 ||
                pixel_format == PIXEL_FORMAT_BGR_555 || pixel_format == PIXEL_FORMAT_BGR_565 ||
                pixel_format == PIXEL_FORMAT_ARGB_1555 || pixel_format == PIXEL_FORMAT_ARGB_4444 ||
                pixel_format == PIXEL_FORMAT_ABGR_1555 || pixel_format == PIXEL_FORMAT_ABGR_4444) {
                bytes_per_pixel = 2;
            } else if (pixel_format == PIXEL_FORMAT_RGB_888 || pixel_format == PIXEL_FORMAT_BGR_888 ||
                       pixel_format == PIXEL_FORMAT_ARGB_8565 || pixel_format == PIXEL_FORMAT_ABGR_8565) {
                bytes_per_pixel = 3;
            } else if (pixel_format == PIXEL_FORMAT_ARGB_8888 || pixel_format == PIXEL_FORMAT_ABGR_8888) {
                bytes_per_pixel = 4;
            } else {
                bytes_per_pixel = 1;
            }
        } else if (i == 1) {
            /* 第二平面 */
            if (pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_420 ||
                pixel_format == PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
                /* YUV420 Semiplanar UV plane */
                snprintf(plane_name, sizeof(plane_name), "Plane1(UV)");
                plane_src_width = src_width / 2;  /* UV交错，宽度减半 */
                plane_src_height = src_height / 2;
                bytes_per_pixel = 2;  /* UV交错，每个位置2字节 */
            } else if (pixel_format == PIXEL_FORMAT_YVU_PLANAR_420) {
                /* YUV420 Planar U plane */
                snprintf(plane_name, sizeof(plane_name), "Plane1(U)");
                plane_src_width = src_width / 2;
                plane_src_height = src_height / 2;
                bytes_per_pixel = 1;
            } else if (pixel_format == PIXEL_FORMAT_YVU_PLANAR_444) {
                /* YUV444 Planar U plane */
                snprintf(plane_name, sizeof(plane_name), "Plane1(U)");
                plane_src_width = src_width;  /* 444格式，全分辨率 */
                plane_src_height = src_height;
                bytes_per_pixel = 1;
            } else if (pixel_format == PIXEL_FORMAT_BGR_888_PLANAR) {
                /* BGR Planar G plane */
                snprintf(plane_name, sizeof(plane_name), "Plane1(G)");
                plane_src_width = src_width;
                plane_src_height = src_height;
                bytes_per_pixel = 1;
            } else {
                snprintf(plane_name, sizeof(plane_name), "Plane1");
                plane_src_width = src_width / 2;
                plane_src_height = src_height / 2;
                bytes_per_pixel = 1;
            }
        } else {
            /* 第三平面 */
            if (pixel_format == PIXEL_FORMAT_YVU_PLANAR_420) {
                /* YUV420 Planar V plane */
                snprintf(plane_name, sizeof(plane_name), "Plane2(V)");
                plane_src_width = src_width / 2;
                plane_src_height = src_height / 2;
                bytes_per_pixel = 1;
            } else if (pixel_format == PIXEL_FORMAT_YVU_PLANAR_444) {
                /* YUV444 Planar V plane */
                snprintf(plane_name, sizeof(plane_name), "Plane2(V)");
                plane_src_width = src_width;  /* 444格式，全分辨率 */
                plane_src_height = src_height;
                bytes_per_pixel = 1;
            } else if (pixel_format == PIXEL_FORMAT_BGR_888_PLANAR) {
                /* BGR Planar R plane */
                snprintf(plane_name, sizeof(plane_name), "Plane2(R)");
                plane_src_width = src_width;
                plane_src_height = src_height;
                bytes_per_pixel = 1;
            } else {
                snprintf(plane_name, sizeof(plane_name), "Plane2");
                plane_src_width = src_width / 2;
                plane_src_height = src_height / 2;
                bytes_per_pixel = 1;
            }
        }

        if (!gdma_verify_plane(src_addr[i], dst_addr[i],
                               plane_src_width, plane_src_height,
                               src_frame->v_frame.stride[i],
                               dst_frame->v_frame.stride[i],
                               rotation, x_mirror, y_mirror,
                               bytes_per_pixel,
                               &error_count, max_errors_to_print,
                               plane_name)) {
            result = K_FALSE;
        }
    }

    /* 清理映射 */
    for (i = 0; i < plane_count; i++) {
        kd_mpi_sys_munmap(dst_addr[i], dst_size[i]);
    }

    if (error_count > 0) {
        printf("Total mismatches across all planes: %u\n", error_count);
        return K_FALSE;
    }

    return result;
}

/* Run a single test case */
static k_s32 gdma_run_test_case(gdma_test_case_t *test_case, k_u32 pool_id)
{
    k_s32 ret;
    k_vb_blk_handle src_handle, dst_handle;
    k_u64 src_phys_addr, dst_phys_addr;
    k_u8 *src_virt_addr, *dst_virt_addr;
    k_u32 src_size, dst_size;
    k_video_frame_info src_frame, dst_frame;
    k_gdma_chn_cfg_t gdma_cfg;
    k_u16 dst_width, dst_height;

    g_test_stats.total_tests++;

    printf("Running test: %-30s ... ", test_case->desc);
    fflush(stdout);

    /* Calculate source and destination sizes */
    src_size = gdma_calc_frame_size(test_case->pixel_format, GDMA_TEST_WIDTH, GDMA_TEST_HEIGHT);

    /* Determine destination dimensions based on rotation */
    if (test_case->rotation == GDMA_ROTATE_DEGREE_90 || test_case->rotation == GDMA_ROTATE_DEGREE_270) {
        dst_width = GDMA_TEST_HEIGHT;
        dst_height = GDMA_TEST_WIDTH;
    } else {
        dst_width = GDMA_TEST_WIDTH;
        dst_height = GDMA_TEST_HEIGHT;
    }
    dst_size = gdma_calc_frame_size(test_case->pixel_format, dst_width, dst_height);

    /* Allocate source buffer */
    src_handle = kd_mpi_vb_get_block(pool_id, src_size, NULL);
    if (src_handle == VB_INVALID_HANDLE) {
        printf("FAILED (no source buffer)\n");
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    /* Allocate destination buffer */
    dst_handle = kd_mpi_vb_get_block(pool_id, dst_size, NULL);
    if (dst_handle == VB_INVALID_HANDLE) {
        printf("FAILED (no dest buffer)\n");
        kd_mpi_vb_release_block(src_handle);
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    /* Get physical addresses */
    src_phys_addr = kd_mpi_vb_handle_to_phyaddr(src_handle);
    dst_phys_addr = kd_mpi_vb_handle_to_phyaddr(dst_handle);

    /* Map to virtual addresses */
    src_virt_addr = (k_u8 *)kd_mpi_sys_mmap(src_phys_addr, src_size);
    dst_virt_addr = (k_u8 *)kd_mpi_sys_mmap(dst_phys_addr, dst_size);

    if (!src_virt_addr || !dst_virt_addr) {
        printf("FAILED (mmap failed)\n");
        if (src_virt_addr) kd_mpi_sys_munmap(src_virt_addr, src_size);
        if (dst_virt_addr) kd_mpi_sys_munmap(dst_virt_addr, dst_size);
        kd_mpi_vb_release_block(dst_handle);
        kd_mpi_vb_release_block(src_handle);
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    /* Initialize test pattern */
    gdma_init_test_pattern(src_virt_addr, test_case->pixel_format, GDMA_TEST_WIDTH, GDMA_TEST_HEIGHT);

    /* Prepare source frame info */
    memset(&src_frame, 0, sizeof(src_frame));
    gdma_setup_frame_info(&src_frame, test_case->pixel_format, src_phys_addr, (k_u64)(intptr_t)src_virt_addr,
                          GDMA_TEST_WIDTH, GDMA_TEST_HEIGHT);
    src_frame.pool_id = kd_mpi_vb_handle_to_pool_id(src_handle);

    /* Configure GDMA */
    memset(&gdma_cfg, 0, sizeof(gdma_cfg));
    gdma_cfg.rotation = test_case->rotation;

    if(test_case->x_mirror) {
        gdma_cfg.rotation |= GDMA_ROTATE_MIRROR_HOR;
    }

    if(test_case->y_mirror) {
        gdma_cfg.rotation |= GDMA_ROTATE_MIRROR_VER;
    }

    /* Perform GDMA operation with timing */
    uint64_t start_us = utils_cpu_ticks_us();
    ret = kd_mpi_gsdma_send_frame(&gdma_cfg, &src_frame, &dst_frame, dst_handle, GDMA_TEST_TIMEOUT_MS);
    uint64_t end_us = utils_cpu_ticks_us();
    double xfer_ms = (end_us >= start_us) ? (end_us - start_us) / 1000.0 : 0.0;

    /* Verify result (also measure verification time) */
    if (ret == K_SUCCESS) {
        uint64_t verify_start_us = utils_cpu_ticks_us();

        /* 检查维度 */
        if (dst_frame.v_frame.width != dst_width || dst_frame.v_frame.height != dst_height) {
            uint64_t verify_end_us = utils_cpu_ticks_us();
            double verify_ms = (verify_end_us >= verify_start_us) ? (verify_end_us - verify_start_us) / 1000.0 : 0.0;
            printf("FAILED (dimension mismatch: expected %dx%d, got %dx%d, xfer=%.2f ms, verify=%.2f ms)\n",
                   dst_width, dst_height, dst_frame.v_frame.width, dst_frame.v_frame.height,
                   xfer_ms, verify_ms);
            g_test_stats.failed_tests++;
            ret = K_FAILED;
        } else {
            /* 所有配置（包括镜像）都进行像素级验证 */
            k_bool verify_ok = gdma_verify_rotation(&src_frame, &dst_frame,
                                                    test_case->rotation,
                                                    test_case->x_mirror,
                                                    test_case->y_mirror,
                                                    test_case->pixel_format);
            uint64_t verify_end_us = utils_cpu_ticks_us();
            double verify_ms = (verify_end_us >= verify_start_us) ? (verify_end_us - verify_start_us) / 1000.0 : 0.0;

            if (verify_ok) {
                printf("PASSED (pixel verify, xfer=%.2f ms, verify=%.2f ms)\n", xfer_ms, verify_ms);
                g_test_stats.passed_tests++;
            } else {
                printf("FAILED (pixel verification failed, xfer=%.2f ms, verify=%.2f ms)\n", xfer_ms, verify_ms);
                g_test_stats.failed_tests++;
                ret = K_FAILED;
            }
        }
    } else {
        printf("FAILED (GDMA operation failed, ret=%d, xfer=%.2f ms, verify=0.00 ms)\n", ret, xfer_ms);
        g_test_stats.failed_tests++;
    }

    /* Cleanup */
    kd_mpi_sys_munmap(src_virt_addr, src_size);
    kd_mpi_sys_munmap(dst_virt_addr, dst_size);
    kd_mpi_vb_release_block(dst_handle);
    kd_mpi_vb_release_block(src_handle);

    return ret;
}

/* 简单 1D 模式图案填充：线性递增 */
static void sdma_init_pattern_1d(k_u8 *virt_addr, k_u32 size)
{
    for (k_u32 i = 0; i < size; i++) {
        virt_addr[i] = (k_u8)((0xA5 + i) & 0xFF);
    }
}

/* 2D 模式图案填充：按 (row, col) 生成，支持 stride >= line_size */
static void sdma_init_pattern_2d(k_u8 *virt_addr,
                                 k_u32 line_size,
                                 k_u32 line_num,
                                 k_u32 stride)
{
    for (k_u32 row = 0; row < line_num; row++) {
        k_u32 row_off = row * stride;
        for (k_u32 col = 0; col < line_size; col++) {
            /* 简单模式：value = (row + col) & 0xFF */
            virt_addr[row_off + col] = (k_u8)((row + col) & 0xFF);
        }
    }
}

/* 1D 结果校验 */
static k_bool sdma_verify_1d(k_u8 *src, k_u8 *dst, k_u32 size)
{
    k_u32 err_cnt = 0;
    const k_u32 max_print = 10;

    for (k_u32 i = 0; i < size; i++) {
        if (src[i] != dst[i]) {
            if (err_cnt < max_print) {
                printf("  [1D] Mismatch at 0x%x: src=0x%02x dst=0x%02x\n",
                       i, src[i], dst[i]);
            }
            err_cnt++;
        }
    }

    if (err_cnt) {
        printf("  [1D] Total mismatches: %u bytes\n", err_cnt);
        return K_FALSE;
    }
    return K_TRUE;
}

/* 2D 结果校验：源是带 gap 的二维数据，目标是连续存储 */
static k_bool sdma_verify_2d(k_u8 *src,
                             k_u8 *dst,
                             k_u32 line_size,
                             k_u32 line_num,
                             k_u32 stride)
{
    k_u32 err_cnt = 0;
    const k_u32 max_print = 10;

    for (k_u32 row = 0; row < line_num; row++) {
        k_u32 src_row_off = row * stride;
        k_u32 dst_row_off = row * line_size;
        for (k_u32 col = 0; col < line_size; col++) {
            k_u8 s = src[src_row_off + col];
            k_u8 d = dst[dst_row_off + col];
            if (s != d) {
                if (err_cnt < max_print) {
                    printf("  [2D] Mismatch at row=%u col=%u: src=0x%02x dst=0x%02x\n",
                           row, col, s, d);
                }
                err_cnt++;
            }
        }
    }

    if (err_cnt) {
        printf("  [2D] Total mismatches: %u bytes\n", err_cnt);
        return K_FALSE;
    }
    return K_TRUE;
}

/* Verify memcpy result */
static k_bool sdma_verify_memcpy(k_u8 *src, k_u8 *dst, k_u32 size)
{
    k_u32 err_cnt = 0;
    const k_u32 max_print = 10;

    for (k_u32 i = 0; i < size; i++) {
        if (src[i] != dst[i]) {
            if (err_cnt < max_print) {
                printf("  [Memcpy] Mismatch at offset 0x%x: src=0x%02x dst=0x%02x\n",
                       i, src[i], dst[i]);
            }
            err_cnt++;
        }
    }

    if (err_cnt) {
        printf("  [Memcpy] Total mismatches: %u bytes\n", err_cnt);
        return K_FALSE;
    }
    return K_TRUE;
}

/* Verify memset result based on data size */
static k_bool sdma_verify_memset(k_u8 *dst, k_u32 size, k_u32 data, 
                                k_sdma_data_size_e data_size)
{
    k_u32 err_cnt = 0;
    const k_u32 max_print = 10;
    k_u32 i;

    switch (data_size) {
    case SDMA_DATA_SIZE_1_BYTE: {
        k_u8 expected = data & 0xFF;
        for (i = 0; i < size; i++) {
            if (dst[i] != expected) {
                if (err_cnt < max_print) {
                    printf("  [Memset] Mismatch at offset 0x%x: expected=0x%02x actual=0x%02x\n",
                           i, expected, dst[i]);
                }
                err_cnt++;
            }
        }
        break;
    }
    case SDMA_DATA_SIZE_2_BYTE: {
        k_u16 expected = data & 0xFFFF;
        
        /* For 2-byte mode, verify as 16-bit values */
        k_u16 *dst_16 = (k_u16 *)dst;
        k_u32 size_16 = size / 2;
        
        for (i = 0; i < size_16; i++) {
            if (dst_16[i] != expected) {
                if (err_cnt < max_print) {
                    printf("  [Memset] Mismatch at offset 0x%x: expected=0x%04x actual=0x%04x\n",
                           i * 2, expected, dst_16[i]);
                }
                err_cnt++;
            }
        }
        break;
    }
    case SDMA_DATA_SIZE_4_BYTE: {
        k_u32 expected = data;

        /* For 4-byte mode, verify as 32-bit values */
        k_u32 *dst_32 = (k_u32 *)dst;
        k_u32 size_32 = size / 4;
        
        for (i = 0; i < size_32; i++) {
            if (dst_32[i] != expected) {
                if (err_cnt < max_print) {
                    printf("  [Memset] Mismatch at offset 0x%x: expected=0x%08x actual=0x%08x\n",
                           i * 4, expected, dst_32[i]);
                }
                err_cnt++;
            }
        }
        break;
    }
    default:
        printf("  [Memset] Unsupported data size: %d\n", data_size);
        return K_FALSE;
    }

    if (err_cnt) {
        printf("  [Memset] Total mismatches: %u bytes\n", err_cnt);
        return K_FALSE;
    }
    return K_TRUE;
}

/* Run SDMA memcpy test case */
static k_s32 sdma_run_memcpy_test_case(sdma_memcpy_test_case_t *test_case, k_u32 pool_id)
{
    k_s32 ret;
    k_vb_blk_handle src_handle, dst_handle;
    k_u64 src_phys_addr, dst_phys_addr;
    k_u8 *src_virt_addr, *dst_virt_addr;
    k_u32 src_size, dst_size;
    k_u64 aligned_src_phys, aligned_dst_phys;
    k_u8 *aligned_src_virt, *aligned_dst_virt;

    g_test_stats.total_tests++;

    printf("Running SDMA memcpy test: %-50s ... ", test_case->desc);
    fflush(stdout);

    /* Allocate source and destination buffers with extra space for alignment */
    src_size = test_case->size + test_case->src_alignment;
    dst_size = test_case->size + test_case->dst_alignment;

    src_handle = kd_mpi_vb_get_block(pool_id, src_size, NULL);
    if (src_handle == VB_INVALID_HANDLE) {
        printf("FAILED (no source buffer)\n");
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    dst_handle = kd_mpi_vb_get_block(pool_id, dst_size, NULL);
    if (dst_handle == VB_INVALID_HANDLE) {
        printf("FAILED (no dest buffer)\n");
        kd_mpi_vb_release_block(src_handle);
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    /* Get physical addresses */
    src_phys_addr = kd_mpi_vb_handle_to_phyaddr(src_handle);
    dst_phys_addr = kd_mpi_vb_handle_to_phyaddr(dst_handle);

    /* Map to virtual addresses */
    src_virt_addr = (k_u8 *)kd_mpi_sys_mmap(src_phys_addr, src_size);
    dst_virt_addr = (k_u8 *)kd_mpi_sys_mmap(dst_phys_addr, dst_size);

    if (!src_virt_addr || !dst_virt_addr) {
        printf("FAILED (mmap failed)\n");
        if (src_virt_addr) kd_mpi_sys_munmap(src_virt_addr, src_size);
        if (dst_virt_addr) kd_mpi_sys_munmap(dst_virt_addr, dst_size);
        kd_mpi_vb_release_block(dst_handle);
        kd_mpi_vb_release_block(src_handle);
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    /* Calculate aligned addresses */
    aligned_src_phys = (src_phys_addr + test_case->src_alignment - 1) & ~(test_case->src_alignment - 1);
    aligned_dst_phys = (dst_phys_addr + test_case->dst_alignment - 1) & ~(test_case->dst_alignment - 1);
    aligned_src_virt = src_virt_addr + (aligned_src_phys - src_phys_addr);
    aligned_dst_virt = dst_virt_addr + (aligned_dst_phys - dst_phys_addr);

    /* Initialize source pattern */
    sdma_init_pattern_1d(aligned_src_virt, test_case->size);
    memset(aligned_dst_virt, 0, test_case->size);

    /* Perform SDMA memcpy with timing */
    uint64_t start_us = utils_cpu_ticks_us();
    
    k_sdma_memcpy_t memcpy_cfg;
    memcpy_cfg.dst_phys_addr = aligned_dst_phys;
    memcpy_cfg.src_phys_addr = aligned_src_phys;
    memcpy_cfg.size = test_case->size;
    memcpy_cfg.timeout_ms = SDMA_TEST_TIMEOUT_MS;
    ret = kd_mpi_gsdma_sdma_memcpy(&memcpy_cfg);

    uint64_t end_us = utils_cpu_ticks_us();
    double xfer_ms = (end_us >= start_us) ? (end_us - start_us) / 1000.0 : 0.0;

    /* Verify result */
    if (ret == K_SUCCESS) {
        uint64_t verify_start_us = utils_cpu_ticks_us();
        k_bool verify_ok = sdma_verify_memcpy(aligned_src_virt, aligned_dst_virt, test_case->size);
        uint64_t verify_end_us = utils_cpu_ticks_us();
        double verify_ms = (verify_end_us >= verify_start_us) ? (verify_end_us - verify_start_us) / 1000.0 : 0.0;

        if (verify_ok) {
            printf("PASSED (xfer=%.2f ms, verify=%.2f ms)\n", xfer_ms, verify_ms);
            g_test_stats.passed_tests++;
        } else {
            printf("FAILED (data verification failed, xfer=%.2f ms, verify=%.2f ms)\n", xfer_ms, verify_ms);
            g_test_stats.failed_tests++;
            ret = K_FAILED;
        }
    } else {
        printf("FAILED (SDMA memcpy failed, ret=%d, xfer=%.2f ms)\n", ret, xfer_ms);
        g_test_stats.failed_tests++;
    }

    /* Cleanup */
    kd_mpi_sys_munmap(src_virt_addr, src_size);
    kd_mpi_sys_munmap(dst_virt_addr, dst_size);
    kd_mpi_vb_release_block(dst_handle);
    kd_mpi_vb_release_block(src_handle);

    return ret;
}

/* Run SDMA memset test case */
static k_s32 sdma_run_memset_test_case(sdma_memset_test_case_t *test_case, k_u32 pool_id)
{
    k_s32 ret;
    k_vb_blk_handle blk_handle;
    k_u64 phys_addr;
    k_u8 *virt_addr;
    k_u32 size;
    k_u64 aligned_phys;
    k_u8 *aligned_virt;

    g_test_stats.total_tests++;

    printf("Running SDMA memset test: %-50s ... ", test_case->desc);
    fflush(stdout);

    /* Allocate buffer with extra space for alignment */
    size = test_case->size + test_case->alignment;

    blk_handle = kd_mpi_vb_get_block(pool_id, size, NULL);
    if (blk_handle == VB_INVALID_HANDLE) {
        printf("FAILED (no buffer)\n");
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    /* Get physical address */
    phys_addr = kd_mpi_vb_handle_to_phyaddr(blk_handle);

    /* Map to virtual address */
    virt_addr = (k_u8 *)kd_mpi_sys_mmap(phys_addr, size);
    if (!virt_addr) {
        printf("FAILED (mmap failed)\n");
        kd_mpi_vb_release_block(blk_handle);
        g_test_stats.failed_tests++;
        return K_FAILED;
    }

    /* Calculate aligned address */
    aligned_phys = (phys_addr + test_case->alignment - 1) & ~(test_case->alignment - 1);
    aligned_virt = virt_addr + (aligned_phys - phys_addr);

    /* Initialize buffer with different pattern to verify memset works */
    memset(aligned_virt, 0xCC, test_case->size);

    /* Perform SDMA memset with timing */
    uint64_t start_us = utils_cpu_ticks_us();

    k_sdma_memset_t memset_cfg;

    memset_cfg.phys_addr = aligned_phys;
    memset_cfg.size = test_case->size;
    memset_cfg.data = test_case->data;
    memset_cfg.data_size = test_case->data_size;
    memset_cfg.timeout_ms = SDMA_TEST_TIMEOUT_MS;

    ret = kd_mpi_gsdma_sdma_memset(&memset_cfg);
    uint64_t end_us = utils_cpu_ticks_us();
    double xfer_ms = (end_us >= start_us) ? (end_us - start_us) / 1000.0 : 0.0;

    /* Verify result */
    if (ret == K_SUCCESS) {
        uint64_t verify_start_us = utils_cpu_ticks_us();
        k_bool verify_ok = sdma_verify_memset(aligned_virt, test_case->size, test_case->data,
                                             test_case->data_size);
        uint64_t verify_end_us = utils_cpu_ticks_us();
        double verify_ms = (verify_end_us >= verify_start_us) ? (verify_end_us - verify_start_us) / 1000.0 : 0.0;

        if (verify_ok) {
            printf("PASSED (xfer=%.2f ms, verify=%.2f ms)\n", xfer_ms, verify_ms);
            g_test_stats.passed_tests++;
        } else {
            printf("FAILED (data verification failed, xfer=%.2f ms, verify=%.2f ms)\n", xfer_ms, verify_ms);
            g_test_stats.failed_tests++;
            ret = K_FAILED;
        }
    } else {
        printf("FAILED (SDMA memset failed, ret=%d, xfer=%.2f ms)\n", ret, xfer_ms);
        g_test_stats.failed_tests++;
    }

    /* Cleanup */
    kd_mpi_sys_munmap(virt_addr, size);
    kd_mpi_vb_release_block(blk_handle);

    return ret;
}

int main(void)
{
    k_s32 ret;
    k_u32 i;
    k_vb_config vb_config;
    k_vb_pool_config pool_config;
    k_u32 pool_id;
    k_gsdma_dev_attr_t dev_attr;

    printf("========================================\n");
    printf("GSDMA Comprehensive Test Suite\n");
    printf("========================================\n");
    printf("GDMA test dimensions: %dx%d\n", GDMA_TEST_WIDTH, GDMA_TEST_HEIGHT);
    printf("GDMA test cases: %lu\n", TEST_CASE_COUNT);
    printf("SDMA transfer test cases: %lu\n", (unsigned long)SDMA_TEST_CASE_COUNT);
    printf("SDMA memcpy test cases: %lu\n", (unsigned long)SDMA_MEMCPY_TEST_CASE_COUNT);
    printf("SDMA memset test cases: %lu\n", (unsigned long)SDMA_MEMSET_TEST_CASE_COUNT);
    printf("========================================\n\n");

    /* Initialize VB */
    memset(&vb_config, 0, sizeof(vb_config));
    vb_config.max_pool_cnt = 64;
    ret = kd_mpi_vb_set_config(&vb_config);
    if (ret != K_SUCCESS) {
        printf("Failed to set VB config: %d\n", ret);
        return -1;
    }

    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret != K_SUCCESS) {
        printf("Failed to set VB supplement config: %d\n", ret);
    }

    ret = kd_mpi_vb_init();
    if (ret != K_SUCCESS) {
        printf("Failed to initialize VB: %d\n", ret);
        return -1;
    }

    /* Create VB pool (large enough for all formats) */
    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = 10;   /* Increased for additional test cases */
    pool_config.blk_size = GDMA_TEST_WIDTH * GDMA_TEST_HEIGHT * 4;  /* 128KB blocks for large transfers */
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    pool_id = kd_mpi_vb_create_pool(&pool_config);
    if (pool_id == VB_INVALID_POOLID) {
        printf("Failed to create VB pool\n");
        kd_mpi_vb_exit();
        return -1;
    }

    printf("Created VB pool %u (blk_cnt=%u, blk_size=%lu)\n\n",
           pool_id, pool_config.blk_cnt, pool_config.blk_size);

    /* Initialize GDMA */
    ret = kd_mpi_gsdma_init();
    if (ret != K_SUCCESS) {
        printf("Failed to initialize GDMA: %d\n", ret);
        kd_mpi_vb_destory_pool(pool_id);
        kd_mpi_vb_exit();
        return -1;
    }

    /* Set GDMA device attributes */
    memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.burst_len = 0;
    dev_attr.outstanding = 7;
    dev_attr.ckg_bypass = 0xFF;
    dev_attr.arbitration_weight = 0x444210;

    ret = kd_mpi_gsdma_set_dev_attr(&dev_attr);
    if (ret != K_SUCCESS) {
        printf("Failed to set GDMA device attributes: %d\n", ret);
        kd_mpi_gsdma_deinit();
        kd_mpi_vb_destory_pool(pool_id);
        kd_mpi_vb_exit();
        return -1;
    }

    printf("GDMA initialized successfully\n\n");

    /* -------- GDMA 图像测试 -------- */
    printf("Starting GDMA test execution...\n");
    printf("========================================\n");

    for (i = 0; i < TEST_CASE_COUNT; i++) {
        gdma_run_test_case(&g_test_cases[i], pool_id);
        usleep(10000);  /* Small delay between tests */
    }

    printf("========================================\n");
    printf("GDMA test execution completed\n\n");

    /* -------- SDMA 内存搬运测试 -------- */
    printf("Starting SDMA transfer test execution...\n");
    printf("========================================\n");

    for (i = 0; i < SDMA_TEST_CASE_COUNT; i++) {
        sdma_test_case_t *tc = &g_sdma_test_cases[i];
        k_s32 ret_case;
        k_vb_blk_handle src_handle, dst_handle;
        k_u64 src_phys, dst_phys;
        k_u8 *src_virt = NULL;
        k_u8 *dst_virt = NULL;
        k_u32 src_size, dst_size;
        k_sdma_transfer_cfg_t cfg;

        g_test_stats.total_tests++;

        printf("Running SDMA test: %-40s ... ", tc->desc);
        fflush(stdout);

        if (tc->dimension == DIMENSION1) {
            /* 1D：总长度即 line_size */
            src_size = tc->line_size;
            dst_size = tc->line_size;
        } else {
            /* 2D：源缓冲大小 stride * line_num，目标缓冲大小 line_size * line_num */
            src_size = tc->stride * tc->line_num;
            dst_size = tc->line_size * tc->line_num;
        }

        src_handle = kd_mpi_vb_get_block(pool_id, src_size, NULL);
        if (src_handle == VB_INVALID_HANDLE) {
            printf("FAILED (no SDMA source buffer)\n");
            g_test_stats.failed_tests++;
            continue;
        }
        dst_handle = kd_mpi_vb_get_block(pool_id, dst_size, NULL);
        if (dst_handle == VB_INVALID_HANDLE) {
            printf("FAILED (no SDMA dest buffer)\n");
            kd_mpi_vb_release_block(src_handle);
            g_test_stats.failed_tests++;
            continue;
        }

        src_phys = kd_mpi_vb_handle_to_phyaddr(src_handle);
        dst_phys = kd_mpi_vb_handle_to_phyaddr(dst_handle);

        src_virt = (k_u8 *)kd_mpi_sys_mmap(src_phys, src_size);
        dst_virt = (k_u8 *)kd_mpi_sys_mmap(dst_phys, dst_size);
        if (!src_virt || !dst_virt) {
            printf("FAILED (SDMA mmap failed)\n");
            if (src_virt) kd_mpi_sys_munmap(src_virt, src_size);
            if (dst_virt) kd_mpi_sys_munmap(dst_virt, dst_size);
            kd_mpi_vb_release_block(dst_handle);
            kd_mpi_vb_release_block(src_handle);
            g_test_stats.failed_tests++;
            continue;
        }

        /* 初始化源数据，目标清零 */
        if (tc->dimension == DIMENSION1) {
            sdma_init_pattern_1d(src_virt, tc->line_size);
        } else {
            sdma_init_pattern_2d(src_virt, tc->line_size, tc->line_num, tc->stride);
        }
        memset(dst_virt, 0, dst_size);

        memset(&cfg, 0, sizeof(cfg));
        cfg.src_addr   = (void *)(intptr_t)src_phys;
        cfg.dst_addr   = (void *)(intptr_t)dst_phys;
        cfg.dimension  = tc->dimension;
        cfg.line_size  = tc->line_size;
        cfg.line_num   = (tc->dimension == DIMENSION1) ? 1 : tc->line_num;
        cfg.line_space = (tc->dimension == DIMENSION1) ? 0 : (tc->stride - tc->line_size); /* 硬件定义：line_space = stride - line_size */
        cfg.timeout_ms = SDMA_TEST_TIMEOUT_MS;
        cfg.user_data  = 0;
        cfg.ch_cfg.value = 0; /* 默认配置：8bit, normal mode */

        uint64_t start_us = utils_cpu_ticks_us();
        ret_case = kd_mpi_gsdma_mem_transfer(&cfg);
        uint64_t end_us = utils_cpu_ticks_us();
        double xfer_ms = (end_us >= start_us) ? (end_us - start_us) / 1000.0 : 0.0;

        if (ret_case == K_SUCCESS) {
            uint64_t verify_start_us = utils_cpu_ticks_us();
            k_bool ok;
            if (tc->dimension == DIMENSION1) {
                ok = sdma_verify_1d(src_virt, dst_virt, tc->line_size);
            } else {
                ok = sdma_verify_2d(src_virt, dst_virt, tc->line_size,
                                    tc->line_num, tc->stride);
            }
            uint64_t verify_end_us = utils_cpu_ticks_us();
            double verify_ms = (verify_end_us >= verify_start_us) ? (verify_end_us - verify_start_us) / 1000.0 : 0.0;

            if (ok) {
                printf("PASSED (xfer=%.2f ms, verify=%.2f ms)\n", xfer_ms, verify_ms);
                g_test_stats.passed_tests++;
            } else {
                printf("FAILED (data verification failed, xfer=%.2f ms, verify=%.2f ms)\n",
                       xfer_ms, verify_ms);
                g_test_stats.failed_tests++;
            }
        } else {
            printf("FAILED (SDMA operation failed, ret=%d, xfer=%.2f ms, verify=0.00 ms)\n",
                   ret_case, xfer_ms);
            g_test_stats.failed_tests++;
        }

        kd_mpi_sys_munmap(src_virt, src_size);
        kd_mpi_sys_munmap(dst_virt, dst_size);
        kd_mpi_vb_release_block(dst_handle);
        kd_mpi_vb_release_block(src_handle);
    }

    printf("========================================\n");
    printf("SDMA transfer test execution completed\n\n");

    /* -------- SDMA Memcpy 测试 -------- */
    printf("Starting SDMA memcpy test execution...\n");
    printf("========================================\n");

    for (i = 0; i < SDMA_MEMCPY_TEST_CASE_COUNT; i++) {
        sdma_run_memcpy_test_case(&g_sdma_memcpy_test_cases[i], pool_id);
        usleep(5000);  /* Small delay between tests */
    }

    printf("========================================\n");
    printf("SDMA memcpy test execution completed\n\n");

    /* -------- SDMA Memset 测试 -------- */
    printf("Starting SDMA memset test execution...\n");
    printf("========================================\n");

    for (i = 0; i < SDMA_MEMSET_TEST_CASE_COUNT; i++) {
        sdma_run_memset_test_case(&g_sdma_memset_test_cases[i], pool_id);
        usleep(5000);  /* Small delay between tests */
    }

    printf("========================================\n");
    printf("SDMA memset test execution completed\n\n");

    /* Print summary */
    printf("========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Total tests:  %u\n", g_test_stats.total_tests);
    printf("Passed:       %u (%.1f%%)\n", g_test_stats.passed_tests,
           (float)g_test_stats.passed_tests * 100.0f / g_test_stats.total_tests);
    printf("Failed:       %u (%.1f%%)\n", g_test_stats.failed_tests,
           (float)g_test_stats.failed_tests * 100.0f / g_test_stats.total_tests);
    printf("========================================\n");

    if (g_test_stats.failed_tests == 0) {
        printf("\n✓ All tests PASSED!\n\n");
    } else {
        printf("\n✗ Some tests FAILED!\n\n");
    }

    /* Cleanup */
    kd_mpi_gsdma_deinit();
    kd_mpi_vb_destory_pool(pool_id);
    kd_mpi_vb_exit();

    return (g_test_stats.failed_tests == 0) ? 0 : -1;
}