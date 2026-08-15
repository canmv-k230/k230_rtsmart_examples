/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "lv_k230_image_convert.h"
#include "lv_k230_vglite.h"
#include "lvgl.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"

#define RVV_TEST_WIDTH 37u
#define RVV_TEST_HEIGHT 5u
#define CSC_TEST_WIDTH 64u
#define CSC_TEST_HEIGHT 64u
#define CSC_BLOCK_SIZE VB_ALIGN_UP(CSC_TEST_WIDTH * CSC_TEST_HEIGHT * 4u, 4096u)
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
    uint8_t alpha;
} ref_pixel_t;

typedef struct {
    lv_color_format_t format;
    const char * name;
} lv_format_case_t;

typedef struct {
    k_pixel_format mpp_format;
    lv_color_format_t lv_format;
    const char * name;
    uint8_t bytes_per_pixel;
    uint8_t rgb_tolerance;
    uint8_t alpha_tolerance;
    bool has_alpha;
    bool yuv;
} csc_case_t;

typedef struct {
    unsigned total;
    unsigned passed;
    unsigned rvv;
    unsigned scalar;
    unsigned details;
    bool verbose;
} test_stats_t;

static const lv_format_case_t s_lv_formats[] = {
    {LV_COLOR_FORMAT_L8, "L8"},
    {LV_COLOR_FORMAT_A8, "A8"},
    {LV_COLOR_FORMAT_AL88, "AL88"},
    {LV_COLOR_FORMAT_RGB565, "RGB565"},
    {LV_COLOR_FORMAT_RGB565_SWAPPED, "RGB565_SWAPPED"},
    {LV_COLOR_FORMAT_RGB888, "RGB888"},
    {LV_COLOR_FORMAT_XRGB8888, "XRGB8888"},
    {LV_COLOR_FORMAT_ARGB8888, "ARGB8888"},
    {LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED, "ARGB8888_PREMULT"},
    {LV_COLOR_FORMAT_ARGB1555, "ARGB1555"},
    {LV_COLOR_FORMAT_ARGB4444, "ARGB4444"},
    {LV_COLOR_FORMAT_ARGB2222, "ARGB2222"},
    {LV_COLOR_FORMAT_ARGB8565, "ARGB8565"},
};

static const csc_case_t s_csc_formats[] = {
    {PIXEL_FORMAT_RGB_565, LV_COLOR_FORMAT_RGB565, "RGB565", 2, 9, 0, false, false},
    {PIXEL_FORMAT_BGR_565, LV_COLOR_FORMAT_RGB565, "BGR565", 2, 9, 0, false, false},
    {PIXEL_FORMAT_RGB_565_LE, LV_COLOR_FORMAT_RGB565, "RGB565_LE", 2, 9, 0, false, false},
    {PIXEL_FORMAT_BGR_565_LE, LV_COLOR_FORMAT_RGB565, "BGR565_LE", 2, 9, 0, false, false},
    {PIXEL_FORMAT_RGB_888, LV_COLOR_FORMAT_RGB888, "RGB888", 3, 2, 0, false, false},
    {PIXEL_FORMAT_BGR_888, LV_COLOR_FORMAT_RGB888, "BGR888", 3, 2, 0, false, false},
    {PIXEL_FORMAT_ARGB_1555, LV_COLOR_FORMAT_ARGB1555, "ARGB1555", 2, 9, 1, true, false},
    {PIXEL_FORMAT_ARGB_4444, LV_COLOR_FORMAT_ARGB4444, "ARGB4444", 2, 18, 18, true, false},
    {PIXEL_FORMAT_ARGB_8888, LV_COLOR_FORMAT_ARGB8888, "ARGB8888", 4, 2, 2, true, false},
    {PIXEL_FORMAT_BGRA_8888, LV_COLOR_FORMAT_ARGB8888, "BGRA8888", 4, 2, 2, true, false},
    {PIXEL_FORMAT_YUV_SEMIPLANAR_420, LV_COLOR_FORMAT_NV12, "NV12", 0, 4, 0, false, true},
    {PIXEL_FORMAT_YVU_SEMIPLANAR_420, LV_COLOR_FORMAT_NV21, "NV21", 0, 4, 0, false, true},
    {PIXEL_FORMAT_YVU_PLANAR_420, LV_COLOR_FORMAT_I420, "I420", 0, 4, 0, false, true},
};

static const ref_pixel_t s_pattern[] = {
    {0, 0, 0, 0},
    {255, 255, 255, 255},
    {0, 0, 255, 239},
    {0, 255, 0, 193},
    {255, 0, 0, 131},
    {31, 127, 247, 67},
    {83, 197, 19, 17},
    {213, 37, 91, 1},
    {42, 93, 164, 222},
    {231, 141, 7, 99},
};

static uint8_t expand_2(uint8_t value)
{
    return (uint8_t)(value * 85u);
}

static uint8_t expand_4(uint8_t value)
{
    return (uint8_t)((value << 4) | value);
}

static uint8_t expand_5(uint8_t value)
{
    return (uint8_t)((value << 3) | (value >> 2));
}

static uint8_t expand_6(uint8_t value)
{
    return (uint8_t)((value << 2) | (value >> 4));
}

static uint8_t luma(ref_pixel_t pixel)
{
    return (uint8_t)(((uint32_t)pixel.red * 77u + (uint32_t)pixel.green * 150u +
                      (uint32_t)pixel.blue * 29u + 128u) >> 8);
}

static ref_pixel_t pattern_for(lv_color_format_t format, uint32_t index)
{
    ref_pixel_t pixel = s_pattern[index % ARRAY_SIZE(s_pattern)];
    if(format == LV_COLOR_FORMAT_L8 || format == LV_COLOR_FORMAT_AL88) {
        uint8_t value = (uint8_t)(13u + (index * 37u) % 231u);
        pixel.blue = value;
        pixel.green = value;
        pixel.red = value;
    }
    else if(format == LV_COLOR_FORMAT_A8) {
        pixel.blue = 255;
        pixel.green = 255;
        pixel.red = 255;
    }
    return pixel;
}

static bool ref_read_lv(const uint8_t * src, lv_color_format_t format, ref_pixel_t * pixel)
{
    uint16_t value;
    switch(format) {
        case LV_COLOR_FORMAT_L8:
            pixel->blue = src[0];
            pixel->green = src[0];
            pixel->red = src[0];
            pixel->alpha = 255;
            return true;
        case LV_COLOR_FORMAT_A8:
            pixel->blue = 255;
            pixel->green = 255;
            pixel->red = 255;
            pixel->alpha = src[0];
            return true;
        case LV_COLOR_FORMAT_AL88:
            pixel->blue = src[0];
            pixel->green = src[0];
            pixel->red = src[0];
            pixel->alpha = src[1];
            return true;
        case LV_COLOR_FORMAT_RGB565:
        case LV_COLOR_FORMAT_RGB565_SWAPPED:
            memcpy(&value, src, sizeof(value));
            if(format == LV_COLOR_FORMAT_RGB565_SWAPPED) {
                value = (uint16_t)((value << 8) | (value >> 8));
            }
            pixel->blue = expand_5((uint8_t)(value & 0x1fu));
            pixel->green = expand_6((uint8_t)((value >> 5) & 0x3fu));
            pixel->red = expand_5((uint8_t)((value >> 11) & 0x1fu));
            pixel->alpha = 255;
            return true;
        case LV_COLOR_FORMAT_RGB888:
            pixel->blue = src[0];
            pixel->green = src[1];
            pixel->red = src[2];
            pixel->alpha = 255;
            return true;
        case LV_COLOR_FORMAT_XRGB8888:
            pixel->blue = src[0];
            pixel->green = src[1];
            pixel->red = src[2];
            pixel->alpha = 255;
            return true;
        case LV_COLOR_FORMAT_ARGB8888:
        case LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED:
            pixel->blue = src[0];
            pixel->green = src[1];
            pixel->red = src[2];
            pixel->alpha = src[3];
            if(format == LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED &&
               pixel->alpha != 0u && pixel->alpha != 255u) {
                pixel->blue = (uint8_t)LV_MIN(255u,
                    ((uint32_t)pixel->blue * 255u + pixel->alpha / 2u) / pixel->alpha);
                pixel->green = (uint8_t)LV_MIN(255u,
                    ((uint32_t)pixel->green * 255u + pixel->alpha / 2u) / pixel->alpha);
                pixel->red = (uint8_t)LV_MIN(255u,
                    ((uint32_t)pixel->red * 255u + pixel->alpha / 2u) / pixel->alpha);
            }
            return true;
        case LV_COLOR_FORMAT_ARGB1555:
            memcpy(&value, src, sizeof(value));
            pixel->blue = expand_5((uint8_t)((value >> 11) & 0x1fu));
            pixel->green = expand_5((uint8_t)((value >> 6) & 0x1fu));
            pixel->red = expand_5((uint8_t)((value >> 1) & 0x1fu));
            pixel->alpha = (value & 1u) ? 255 : 0;
            return true;
        case LV_COLOR_FORMAT_ARGB4444:
            memcpy(&value, src, sizeof(value));
            pixel->blue = expand_4((uint8_t)((value >> 12) & 0x0fu));
            pixel->green = expand_4((uint8_t)((value >> 8) & 0x0fu));
            pixel->red = expand_4((uint8_t)((value >> 4) & 0x0fu));
            pixel->alpha = expand_4((uint8_t)(value & 0x0fu));
            return true;
        case LV_COLOR_FORMAT_ARGB2222:
            pixel->blue = expand_2((uint8_t)((src[0] >> 6) & 0x03u));
            pixel->green = expand_2((uint8_t)((src[0] >> 4) & 0x03u));
            pixel->red = expand_2((uint8_t)((src[0] >> 2) & 0x03u));
            pixel->alpha = expand_2((uint8_t)(src[0] & 0x03u));
            return true;
        case LV_COLOR_FORMAT_ARGB8565:
            memcpy(&value, src, sizeof(value));
            pixel->blue = expand_5((uint8_t)(value & 0x1fu));
            pixel->green = expand_6((uint8_t)((value >> 5) & 0x3fu));
            pixel->red = expand_5((uint8_t)((value >> 11) & 0x1fu));
            pixel->alpha = src[2];
            return true;
        default:
            return false;
    }
}

static bool ref_write_lv(uint8_t * dest, lv_color_format_t format, ref_pixel_t pixel)
{
    uint16_t value;
    switch(format) {
        case LV_COLOR_FORMAT_L8:
            dest[0] = luma(pixel);
            return true;
        case LV_COLOR_FORMAT_A8:
            dest[0] = pixel.alpha;
            return true;
        case LV_COLOR_FORMAT_AL88:
            dest[0] = luma(pixel);
            dest[1] = pixel.alpha;
            return true;
        case LV_COLOR_FORMAT_RGB565:
        case LV_COLOR_FORMAT_RGB565_SWAPPED:
            value = (uint16_t)(((uint16_t)(pixel.red >> 3) << 11) |
                               ((uint16_t)(pixel.green >> 2) << 5) |
                               (pixel.blue >> 3));
            if(format == LV_COLOR_FORMAT_RGB565_SWAPPED) {
                value = (uint16_t)((value << 8) | (value >> 8));
            }
            memcpy(dest, &value, sizeof(value));
            return true;
        case LV_COLOR_FORMAT_RGB888:
            dest[0] = pixel.blue;
            dest[1] = pixel.green;
            dest[2] = pixel.red;
            return true;
        case LV_COLOR_FORMAT_XRGB8888:
            dest[0] = pixel.blue;
            dest[1] = pixel.green;
            dest[2] = pixel.red;
            dest[3] = 255;
            return true;
        case LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED:
            pixel.blue = (uint8_t)(((uint32_t)pixel.blue * pixel.alpha + 127u) / 255u);
            pixel.green = (uint8_t)(((uint32_t)pixel.green * pixel.alpha + 127u) / 255u);
            pixel.red = (uint8_t)(((uint32_t)pixel.red * pixel.alpha + 127u) / 255u);
            /* fall through */
        case LV_COLOR_FORMAT_ARGB8888:
            dest[0] = pixel.blue;
            dest[1] = pixel.green;
            dest[2] = pixel.red;
            dest[3] = pixel.alpha;
            return true;
        case LV_COLOR_FORMAT_ARGB1555:
            value = (uint16_t)(((uint16_t)(pixel.blue >> 3) << 11) |
                               ((uint16_t)(pixel.green >> 3) << 6) |
                               ((uint16_t)(pixel.red >> 3) << 1) |
                               (pixel.alpha >> 7));
            memcpy(dest, &value, sizeof(value));
            return true;
        case LV_COLOR_FORMAT_ARGB4444:
            value = (uint16_t)(((uint16_t)(pixel.blue >> 4) << 12) |
                               ((uint16_t)(pixel.green >> 4) << 8) |
                               ((uint16_t)(pixel.red >> 4) << 4) |
                               (pixel.alpha >> 4));
            memcpy(dest, &value, sizeof(value));
            return true;
        case LV_COLOR_FORMAT_ARGB2222:
            dest[0] = (uint8_t)((pixel.blue & 0xc0u) |
                                ((pixel.green >> 2) & 0x30u) |
                                ((pixel.red >> 4) & 0x0cu) |
                                (pixel.alpha >> 6));
            return true;
        case LV_COLOR_FORMAT_ARGB8565:
            value = (uint16_t)(((uint16_t)(pixel.red >> 3) << 11) |
                               ((uint16_t)(pixel.green >> 2) << 5) |
                               (pixel.blue >> 3));
            memcpy(dest, &value, sizeof(value));
            dest[2] = pixel.alpha;
            return true;
        default:
            return false;
    }
}

static bool format_is_expand_source(lv_color_format_t format)
{
    switch(format) {
        case LV_COLOR_FORMAT_L8:
        case LV_COLOR_FORMAT_A8:
        case LV_COLOR_FORMAT_AL88:
        case LV_COLOR_FORMAT_RGB565:
        case LV_COLOR_FORMAT_RGB565_SWAPPED:
        case LV_COLOR_FORMAT_ARGB1555:
        case LV_COLOR_FORMAT_ARGB4444:
        case LV_COLOR_FORMAT_ARGB2222:
        case LV_COLOR_FORMAT_ARGB8565:
            return true;
        default:
            return false;
    }
}

static bool format_has_alpha(lv_color_format_t format)
{
    return format == LV_COLOR_FORMAT_A8 || format == LV_COLOR_FORMAT_AL88 ||
           format == LV_COLOR_FORMAT_ARGB1555 || format == LV_COLOR_FORMAT_ARGB4444 ||
           format == LV_COLOR_FORMAT_ARGB2222 || format == LV_COLOR_FORMAT_ARGB8565 ||
           format == LV_COLOR_FORMAT_ARGB8888 ||
           format == LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED;
}

static bool format_is_high(lv_color_format_t format)
{
    return format == LV_COLOR_FORMAT_RGB888 || format == LV_COLOR_FORMAT_XRGB8888 ||
           format == LV_COLOR_FORMAT_ARGB8888 ||
           format == LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED;
}

static bool format_is_pack_source(lv_color_format_t format)
{
    return format == LV_COLOR_FORMAT_RGB888 || format == LV_COLOR_FORMAT_XRGB8888 ||
           format == LV_COLOR_FORMAT_ARGB8888;
}

static bool format_is_pack_dest(lv_color_format_t format)
{
    return format == LV_COLOR_FORMAT_RGB565 || format == LV_COLOR_FORMAT_RGB565_SWAPPED ||
           format == LV_COLOR_FORMAT_ARGB1555 || format == LV_COLOR_FORMAT_ARGB4444 ||
           format == LV_COLOR_FORMAT_ARGB2222 || format == LV_COLOR_FORMAT_ARGB8565;
}

static const char * expected_backend(lv_color_format_t src, lv_color_format_t dest)
{
    if(format_is_expand_source(src) && format_is_high(dest) &&
       !(dest == LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED && format_has_alpha(src))) {
        return "RVV";
    }
    if(format_is_pack_source(src) && format_is_pack_dest(dest)) {
        return "RVV";
    }
    if(format_is_high(src) && format_is_high(dest)) {
        bool source_opaque = src == LV_COLOR_FORMAT_RGB888 || src == LV_COLOR_FORMAT_XRGB8888;
        bool src_premult = src == LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED;
        bool dest_premult = dest == LV_COLOR_FORMAT_ARGB8888_PREMULTIPLIED;
        if(source_opaque || src_premult == dest_premult) {
            return "RVV";
        }
    }
    return "scalar";
}

static uint32_t aligned_stride(uint32_t width, lv_color_format_t format)
{
    uint32_t bytes = width * lv_color_format_get_size(format) + 5u;
    return (bytes + 7u) & ~7u;
}

static void print_bytes(const uint8_t * bytes, uint32_t count)
{
    for(uint32_t i = 0; i < count; i++) {
        printf("%s%02x", i ? " " : "", bytes[i]);
    }
}

static bool run_rvv_case(const lv_format_case_t * src_case, const lv_format_case_t * dest_case,
                         test_stats_t * stats)
{
    uint32_t src_bpp = lv_color_format_get_size(src_case->format);
    uint32_t dest_bpp = lv_color_format_get_size(dest_case->format);
    uint32_t src_stride = aligned_stride(RVV_TEST_WIDTH, src_case->format);
    uint32_t dest_stride = aligned_stride(RVV_TEST_WIDTH, dest_case->format);
    size_t src_size = (size_t)src_stride * RVV_TEST_HEIGHT;
    size_t dest_size = (size_t)dest_stride * RVV_TEST_HEIGHT;
    uint8_t * src_data = NULL;
    uint8_t * dest_data = NULL;
    uint8_t * expected = NULL;
    bool passed = false;

    if(posix_memalign((void **)&src_data, 64, (src_size + 63u) & ~63u) != 0 ||
       posix_memalign((void **)&dest_data, 64, (dest_size + 63u) & ~63u) != 0 ||
       posix_memalign((void **)&expected, 64, (dest_size + 63u) & ~63u) != 0) {
        printf("FAIL RVV %s -> %s: allocation failed\n", src_case->name, dest_case->name);
        goto done;
    }

    memset(src_data, 0x5a, src_size);
    memset(dest_data, 0xa5, dest_size);
    memset(expected, 0xa5, dest_size);
    for(uint32_t y = 0; y < RVV_TEST_HEIGHT; y++) {
        for(uint32_t x = 0; x < RVV_TEST_WIDTH; x++) {
            ref_pixel_t input = pattern_for(src_case->format, y * RVV_TEST_WIDTH + x);
            uint8_t * src_pixel = src_data + (size_t)y * src_stride + x * src_bpp;
            uint8_t * expected_pixel = expected + (size_t)y * dest_stride + x * dest_bpp;
            ref_write_lv(src_pixel, src_case->format, input);
            ref_read_lv(src_pixel, src_case->format, &input);
            ref_write_lv(expected_pixel, dest_case->format, input);
        }
    }

    lv_draw_buf_t src;
    lv_draw_buf_t dest;
    if(lv_draw_buf_init(&src, RVV_TEST_WIDTH, RVV_TEST_HEIGHT, src_case->format,
                        src_stride, src_data, (uint32_t)src_size) != LV_RESULT_OK ||
       lv_draw_buf_init(&dest, RVV_TEST_WIDTH, RVV_TEST_HEIGHT, dest_case->format,
                        dest_stride, dest_data, (uint32_t)dest_size) != LV_RESULT_OK) {
        printf("FAIL RVV %s -> %s: draw buffer init failed\n", src_case->name, dest_case->name);
        goto done;
    }

    lv_draw_buf_copy(&dest, NULL, &src, NULL);
    const char * wanted_backend = expected_backend(src_case->format, dest_case->format);
    const char * actual_backend = lv_k230_image_convert_backend();
    if(strcmp(actual_backend, wanted_backend) != 0) {
        if(stats->details++ < 20u) {
            printf("FAIL RVV %s -> %s: backend %s, expected %s\n",
                   src_case->name, dest_case->name, actual_backend, wanted_backend);
        }
        goto done;
    }

    for(uint32_t y = 0; y < RVV_TEST_HEIGHT; y++) {
        for(uint32_t x = 0; x < RVV_TEST_WIDTH; x++) {
            const uint8_t * got = dest_data + (size_t)y * dest_stride + x * dest_bpp;
            const uint8_t * want = expected + (size_t)y * dest_stride + x * dest_bpp;
            if(memcmp(got, want, dest_bpp) != 0) {
                if(stats->details++ < 20u) {
                    printf("FAIL RVV %s -> %s at %u,%u: got [", src_case->name,
                           dest_case->name, x, y);
                    print_bytes(got, dest_bpp);
                    printf("], expected [");
                    print_bytes(want, dest_bpp);
                    printf("]\n");
                }
                goto done;
            }
        }
        for(uint32_t i = RVV_TEST_WIDTH * dest_bpp; i < dest_stride; i++) {
            if(dest_data[(size_t)y * dest_stride + i] != 0xa5u) {
                if(stats->details++ < 20u) {
                    printf("FAIL RVV %s -> %s: row padding overwritten at row %u byte %u\n",
                           src_case->name, dest_case->name, y, i);
                }
                goto done;
            }
        }
    }

    passed = true;
    if(strcmp(wanted_backend, "RVV") == 0) stats->rvv++;
    else stats->scalar++;
    if(stats->verbose) {
        printf("PASS RVV %s -> %s (%s)\n", src_case->name, dest_case->name, actual_backend);
    }

done:
    free(src_data);
    free(dest_data);
    free(expected);
    return passed;
}

static void run_rvv_tests(test_stats_t * stats)
{
    unsigned start_total = stats->total;
    unsigned start_passed = stats->passed;
    for(size_t src = 0; src < ARRAY_SIZE(s_lv_formats); src++) {
        for(size_t dest = 0; dest < ARRAY_SIZE(s_lv_formats); dest++) {
            if(src == dest) continue;
            stats->total++;
            if(run_rvv_case(&s_lv_formats[src], &s_lv_formats[dest], stats)) {
                stats->passed++;
            }
        }
    }
    printf("RVV matrix: %u/%u passed (%u RVV, %u scalar fallback)\n",
           stats->passed - start_passed, stats->total - start_total, stats->rvv, stats->scalar);
}

static void put_u16_native(uint8_t * dest, uint16_t value)
{
    memcpy(dest, &value, sizeof(value));
}

static ref_pixel_t csc_pattern(uint32_t x)
{
    ref_pixel_t pixel = s_pattern[(x / 8u) % 8u];
    return pixel;
}

static void fill_csc_packed(const csc_case_t * test, uint8_t * data)
{
    uint32_t stride = CSC_TEST_WIDTH * test->bytes_per_pixel;
    for(uint32_t y = 0; y < CSC_TEST_HEIGHT; y++) {
        for(uint32_t x = 0; x < CSC_TEST_WIDTH; x++) {
            ref_pixel_t p = csc_pattern(x);
            uint8_t * dest = data + (size_t)y * stride + x * test->bytes_per_pixel;
            uint16_t value;
            switch(test->mpp_format) {
                case PIXEL_FORMAT_RGB_565:
                case PIXEL_FORMAT_RGB_565_LE:
                case PIXEL_FORMAT_BGR_565:
                case PIXEL_FORMAT_BGR_565_LE: {
                    bool bgr = test->mpp_format == PIXEL_FORMAT_BGR_565 ||
                               test->mpp_format == PIXEL_FORMAT_BGR_565_LE;
                    bool little = test->mpp_format == PIXEL_FORMAT_RGB_565_LE ||
                                  test->mpp_format == PIXEL_FORMAT_BGR_565_LE;
                    uint8_t high = bgr ? p.blue : p.red;
                    uint8_t low = bgr ? p.red : p.blue;
                    value = (uint16_t)(((uint16_t)(high >> 3) << 11) |
                                       ((uint16_t)(p.green >> 2) << 5) | (low >> 3));
                    dest[little ? 0 : 1] = (uint8_t)value;
                    dest[little ? 1 : 0] = (uint8_t)(value >> 8);
                    break;
                }
                case PIXEL_FORMAT_RGB_888:
                    dest[0] = p.red;
                    dest[1] = p.green;
                    dest[2] = p.blue;
                    break;
                case PIXEL_FORMAT_BGR_888:
                    dest[0] = p.blue;
                    dest[1] = p.green;
                    dest[2] = p.red;
                    break;
                case PIXEL_FORMAT_ARGB_1555:
                    value = (uint16_t)(((uint16_t)(p.blue >> 3) << 11) |
                                       ((uint16_t)(p.green >> 3) << 6) |
                                       ((uint16_t)(p.red >> 3) << 1) | (p.alpha >> 7));
                    put_u16_native(dest, value);
                    break;
                case PIXEL_FORMAT_ARGB_4444:
                    value = (uint16_t)(((uint16_t)(p.blue >> 4) << 12) |
                                       ((uint16_t)(p.green >> 4) << 8) |
                                       ((uint16_t)(p.red >> 4) << 4) | (p.alpha >> 4));
                    put_u16_native(dest, value);
                    break;
                case PIXEL_FORMAT_ARGB_8888:
                    dest[0] = p.alpha;
                    dest[1] = p.red;
                    dest[2] = p.green;
                    dest[3] = p.blue;
                    break;
                case PIXEL_FORMAT_BGRA_8888:
                    dest[0] = p.blue;
                    dest[1] = p.green;
                    dest[2] = p.red;
                    dest[3] = p.alpha;
                    break;
                default:
                    break;
            }
        }
    }
}

typedef struct {
    uint8_t y;
    uint8_t u;
    uint8_t v;
} yuv_pixel_t;

static const yuv_pixel_t s_yuv_pattern[] = {
    {0, 128, 128},
    {255, 128, 128},
    {76, 85, 255},
    {150, 44, 21},
    {29, 255, 107},
    {180, 70, 180},
    {96, 190, 64},
    {48, 160, 210},
};

static void fill_csc_yuv(const csc_case_t * test, uint8_t * data)
{
    uint8_t * y_plane = data;
    uint8_t * u_plane = data + CSC_TEST_WIDTH * CSC_TEST_HEIGHT;
    uint8_t * v_plane = u_plane + CSC_TEST_WIDTH * CSC_TEST_HEIGHT / 4u;

    for(uint32_t y = 0; y < CSC_TEST_HEIGHT; y++) {
        for(uint32_t x = 0; x < CSC_TEST_WIDTH; x++) {
            y_plane[y * CSC_TEST_WIDTH + x] = s_yuv_pattern[(x / 8u) % 8u].y;
        }
    }
    for(uint32_t y = 0; y < CSC_TEST_HEIGHT / 2u; y++) {
        for(uint32_t x = 0; x < CSC_TEST_WIDTH / 2u; x++) {
            yuv_pixel_t p = s_yuv_pattern[((x * 2u) / 8u) % 8u];
            if(test->lv_format == LV_COLOR_FORMAT_NV12 || test->lv_format == LV_COLOR_FORMAT_NV21) {
                uint8_t * uv = data + CSC_TEST_WIDTH * CSC_TEST_HEIGHT +
                               y * CSC_TEST_WIDTH + x * 2u;
                bool nv21 = test->lv_format == LV_COLOR_FORMAT_NV21;
                uv[nv21 ? 1 : 0] = p.u;
                uv[nv21 ? 0 : 1] = p.v;
            }
            else {
                u_plane[y * (CSC_TEST_WIDTH / 2u) + x] = p.u;
                v_plane[y * (CSC_TEST_WIDTH / 2u) + x] = p.v;
            }
        }
    }
}

static ref_pixel_t decode_csc_packed(const csc_case_t * test, const uint8_t * src)
{
    ref_pixel_t p = {0, 0, 0, 255};
    uint16_t value;
    switch(test->mpp_format) {
        case PIXEL_FORMAT_RGB_565:
        case PIXEL_FORMAT_RGB_565_LE:
        case PIXEL_FORMAT_BGR_565:
        case PIXEL_FORMAT_BGR_565_LE: {
            bool bgr = test->mpp_format == PIXEL_FORMAT_BGR_565 ||
                       test->mpp_format == PIXEL_FORMAT_BGR_565_LE;
            bool little = test->mpp_format == PIXEL_FORMAT_RGB_565_LE ||
                          test->mpp_format == PIXEL_FORMAT_BGR_565_LE;
            value = little ? (uint16_t)(src[0] | ((uint16_t)src[1] << 8)) :
                             (uint16_t)(src[1] | ((uint16_t)src[0] << 8));
            uint8_t high = expand_5((uint8_t)((value >> 11) & 0x1fu));
            uint8_t low = expand_5((uint8_t)(value & 0x1fu));
            p.red = bgr ? low : high;
            p.green = expand_6((uint8_t)((value >> 5) & 0x3fu));
            p.blue = bgr ? high : low;
            break;
        }
        case PIXEL_FORMAT_RGB_888:
            p.red = src[0]; p.green = src[1]; p.blue = src[2];
            break;
        case PIXEL_FORMAT_BGR_888:
            p.blue = src[0]; p.green = src[1]; p.red = src[2];
            break;
        case PIXEL_FORMAT_ARGB_1555:
            memcpy(&value, src, sizeof(value));
            p.blue = expand_5((uint8_t)((value >> 11) & 0x1fu));
            p.green = expand_5((uint8_t)((value >> 6) & 0x1fu));
            p.red = expand_5((uint8_t)((value >> 1) & 0x1fu));
            p.alpha = (value & 1u) ? 255 : 0;
            break;
        case PIXEL_FORMAT_ARGB_4444:
            memcpy(&value, src, sizeof(value));
            p.blue = expand_4((uint8_t)((value >> 12) & 0x0fu));
            p.green = expand_4((uint8_t)((value >> 8) & 0x0fu));
            p.red = expand_4((uint8_t)((value >> 4) & 0x0fu));
            p.alpha = expand_4((uint8_t)(value & 0x0fu));
            break;
        case PIXEL_FORMAT_ARGB_8888:
            p.alpha = src[0]; p.red = src[1]; p.green = src[2]; p.blue = src[3];
            break;
        case PIXEL_FORMAT_BGRA_8888:
            p.blue = src[0]; p.green = src[1]; p.red = src[2]; p.alpha = src[3];
            break;
        default:
            break;
    }
    return p;
}

static int rounded_shift_8(int value)
{
    return value >= 0 ? (value + 128) / 256 : -((-value + 128) / 256);
}

static uint8_t clamp_u8(int value)
{
    if(value < 0) return 0;
    if(value > 255) return 255;
    return (uint8_t)value;
}

static ref_pixel_t decode_csc_yuv(const csc_case_t * test, const uint8_t * data,
                                  uint32_t x, uint32_t y)
{
    uint8_t yy = data[y * CSC_TEST_WIDTH + x];
    const uint8_t * chroma = data + CSC_TEST_WIDTH * CSC_TEST_HEIGHT;
    uint8_t u;
    uint8_t v;
    if(test->lv_format == LV_COLOR_FORMAT_NV12 || test->lv_format == LV_COLOR_FORMAT_NV21) {
        const uint8_t * uv = chroma + (y / 2u) * CSC_TEST_WIDTH + (x & ~1u);
        bool nv21 = test->lv_format == LV_COLOR_FORMAT_NV21;
        u = uv[nv21 ? 1 : 0];
        v = uv[nv21 ? 0 : 1];
    }
    else {
        uint32_t plane_size = CSC_TEST_WIDTH * CSC_TEST_HEIGHT / 4u;
        u = chroma[(y / 2u) * (CSC_TEST_WIDTH / 2u) + x / 2u];
        v = chroma[plane_size + (y / 2u) * (CSC_TEST_WIDTH / 2u) + x / 2u];
    }

    ref_pixel_t p;
    p.red = clamp_u8(rounded_shift_8(256 * yy + 359 * v) - 180);
    p.green = clamp_u8(rounded_shift_8(256 * yy - 88 * u - 183 * v) + 135);
    p.blue = clamp_u8(rounded_shift_8(256 * yy + 453 * u) - 227);
    p.alpha = 255;
    return p;
}

static unsigned channel_delta(uint8_t a, uint8_t b)
{
    return a > b ? (unsigned)(a - b) : (unsigned)(b - a);
}

static uint32_t csc_source_size(const csc_case_t * test)
{
    return test->yuv ? CSC_TEST_WIDTH * CSC_TEST_HEIGHT * 3u / 2u :
                       CSC_TEST_WIDTH * CSC_TEST_HEIGHT * test->bytes_per_pixel;
}

static bool init_csc_source(lv_draw_buf_t * source, lv_yuv_buf_t * yuv,
                            const csc_case_t * test, uint8_t * data, uint32_t size)
{
    if(!test->yuv) {
        return lv_draw_buf_init(source, CSC_TEST_WIDTH, CSC_TEST_HEIGHT, test->lv_format,
                                CSC_TEST_WIDTH * test->bytes_per_pixel,
                                data, size) == LV_RESULT_OK;
    }

    memset(source, 0, sizeof(*source));
    memset(yuv, 0, sizeof(*yuv));
    uint8_t * chroma = data + CSC_TEST_WIDTH * CSC_TEST_HEIGHT;
    yuv->planar.y.buf = data;
    yuv->planar.y.stride = CSC_TEST_WIDTH;
    if(test->lv_format == LV_COLOR_FORMAT_NV12 || test->lv_format == LV_COLOR_FORMAT_NV21) {
        yuv->semi_planar.uv.buf = chroma;
        yuv->semi_planar.uv.stride = CSC_TEST_WIDTH;
    }
    else {
        yuv->planar.u.buf = chroma;
        yuv->planar.u.stride = CSC_TEST_WIDTH / 2u;
        yuv->planar.v.buf = chroma + CSC_TEST_WIDTH * CSC_TEST_HEIGHT / 4u;
        yuv->planar.v.stride = CSC_TEST_WIDTH / 2u;
    }

    source->header.magic = LV_IMAGE_HEADER_MAGIC;
    source->header.cf = test->lv_format;
    source->header.w = CSC_TEST_WIDTH;
    source->header.h = CSC_TEST_HEIGHT;
    source->header.stride = CSC_TEST_WIDTH;
    source->data = (uint8_t *)yuv;
    source->unaligned_data = source->data;
    source->data_size = sizeof(*yuv);
    source->handlers = lv_draw_buf_get_handlers();
    return true;
}

static bool run_csc_case(const csc_case_t * test, k_u32 pool_id, test_stats_t * stats)
{
    uint32_t source_size = csc_source_size(test);
    k_vb_blk_handle block = kd_mpi_vb_get_block(pool_id, CSC_BLOCK_SIZE, NULL);
    if(block == VB_INVALID_HANDLE) {
        printf("FAIL CSC %s: no VB block\n", test->name);
        return false;
    }

    k_u64 phys = kd_mpi_vb_handle_to_phyaddr(block);
    uint8_t * data = kd_mpi_sys_mmap(phys, CSC_BLOCK_SIZE);
    lv_draw_buf_t * dest = NULL;
    bool passed = false;
    if(!data) {
        printf("FAIL CSC %s: mmap failed\n", test->name);
        goto done;
    }

    memset(data, 0, CSC_BLOCK_SIZE);
    if(test->yuv) fill_csc_yuv(test, data);
    else fill_csc_packed(test, data);
    lv_k230_vglite_register_mpp_buffer(data, phys, source_size, false, test->mpp_format);

    lv_draw_buf_t source;
    lv_yuv_buf_t yuv;
    if(!init_csc_source(&source, &yuv, test, data, source_size)) {
        printf("FAIL CSC %s: source draw buffer init failed\n", test->name);
        goto unregister;
    }

    lv_color_format_t dest_format = test->has_alpha ? LV_COLOR_FORMAT_ARGB8888 :
                                                       LV_COLOR_FORMAT_XRGB8888;
    dest = lv_draw_buf_create(CSC_TEST_WIDTH, CSC_TEST_HEIGHT, dest_format,
                              CSC_TEST_WIDTH * 4u);
    if(!dest) {
        printf("FAIL CSC %s: destination allocation failed\n", test->name);
        goto unregister;
    }
    memset(dest->data, 0xa5, dest->data_size);
    lv_draw_buf_copy(dest, NULL, &source, NULL);

    const char * backend = lv_k230_image_convert_backend();
    bool expect_csc = test->mpp_format != PIXEL_FORMAT_RGB_565 &&
                      test->mpp_format != PIXEL_FORMAT_BGR_565;
    bool backend_ok = expect_csc ? strncmp(backend, "CSC+", 4) == 0 :
                                   strcmp(backend, "scalar") == 0;
    if(!backend_ok) {
        if(stats->details++ < 20u) {
            printf("FAIL CSC %s: backend %s, expected %s\n", test->name, backend,
                   expect_csc ? "CSC+SDMA or CSC+memcpy" : "scalar fallback");
        }
        goto unregister;
    }

    for(uint32_t y_pos = 0; y_pos < CSC_TEST_HEIGHT; y_pos++) {
        for(uint32_t x = 0; x < CSC_TEST_WIDTH; x++) {
            ref_pixel_t wanted;
            if(test->yuv) {
                wanted = decode_csc_yuv(test, data, x, y_pos);
            }
            else {
                const uint8_t * src_pixel = data +
                    ((size_t)y_pos * CSC_TEST_WIDTH + x) * test->bytes_per_pixel;
                wanted = decode_csc_packed(test, src_pixel);
            }
            const uint8_t * got = dest->data + (size_t)y_pos * dest->header.stride + x * 4u;
            if(channel_delta(got[0], wanted.blue) > test->rgb_tolerance ||
               channel_delta(got[1], wanted.green) > test->rgb_tolerance ||
               channel_delta(got[2], wanted.red) > test->rgb_tolerance ||
               (test->has_alpha && channel_delta(got[3], wanted.alpha) > test->alpha_tolerance)) {
                if(stats->details++ < 20u) {
                    printf("FAIL CSC %s at %u,%u: got BGRA [%u %u %u %u], expected [%u %u %u %u]\n",
                           test->name, x, y_pos, got[0], got[1], got[2], got[3],
                           wanted.blue, wanted.green, wanted.red, wanted.alpha);
                }
                goto unregister;
            }
        }
    }

    passed = true;
    if(stats->verbose) printf("PASS CSC %s (%s)\n", test->name, backend);

unregister:
    if(dest) lv_draw_buf_destroy(dest);
    lv_k230_vglite_unregister_buffer(data);
done:
    if(data) kd_mpi_sys_munmap(data, CSC_BLOCK_SIZE);
    kd_mpi_vb_release_block(block);
    return passed;
}

static bool run_csc_tests(test_stats_t * stats)
{
    k_vb_pool_config config;
    memset(&config, 0, sizeof(config));
    config.blk_cnt = 1;
    config.blk_size = CSC_BLOCK_SIZE;
    config.mode = VB_REMAP_MODE_NOCACHE;
    k_s32 pool = kd_mpi_vb_create_pool(&config);
    if(pool == (k_s32)VB_INVALID_POOLID) {
        printf("CSC suite: failed to create source VB pool\n");
        return false;
    }

    unsigned start_total = stats->total;
    unsigned start_passed = stats->passed;
    for(size_t i = 0; i < ARRAY_SIZE(s_csc_formats); i++) {
        stats->total++;
        if(run_csc_case(&s_csc_formats[i], (k_u32)pool, stats)) stats->passed++;
    }
    kd_mpi_vb_destory_pool((k_u32)pool);
    printf("MPP/CSC formats: %u/%u passed\n", stats->passed - start_passed,
           stats->total - start_total);
    return true;
}

static void usage(const char * program)
{
    printf("Usage: %s [--rvv-only|--csc-only] [--verbose]\n", program);
}

int main(int argc, char ** argv)
{
    bool run_rvv = true;
    bool run_csc = true;
    test_stats_t stats = {0};

    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "--rvv-only") == 0) run_csc = false;
        else if(strcmp(argv[i], "--csc-only") == 0) run_rvv = false;
        else if(strcmp(argv[i], "--verbose") == 0) stats.verbose = true;
        else if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        else {
            usage(argv[0]);
            return 2;
        }
    }

    k_vb_config vb_config;
    memset(&vb_config, 0, sizeof(vb_config));
    vb_config.max_pool_cnt = 64;
    k_s32 ret = kd_mpi_vb_set_config(&vb_config);
    if(ret == K_SUCCESS) ret = kd_mpi_vb_init();
    if(ret != K_SUCCESS) {
        printf("VB initialization failed: %d\n", ret);
        return 2;
    }

    lv_init();
    printf("LVGL K230 image conversion test\n");
    if(run_rvv) run_rvv_tests(&stats);
    if(run_csc && !run_csc_tests(&stats)) stats.total++;

    if(stats.details > 20u) {
        printf("%u additional failure details suppressed\n", stats.details - 20u);
    }
    printf("Total: %u/%u passed\n", stats.passed, stats.total);

    lv_k230_image_convert_deinit();
    lv_deinit();
    kd_mpi_vb_exit();
    return stats.passed == stats.total ? 0 : 1;
}
