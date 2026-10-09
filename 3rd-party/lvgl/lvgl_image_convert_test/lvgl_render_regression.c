/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lv_k230_vglite.h"
#include "lv_k230_image_convert.h"
#include "lvgl.h"
#include "src/misc/cache/instance/lv_image_cache.h"

static bool test_copy_area(lv_color_format_t format, uint32_t padding, bool full_width)
{
    const uint32_t width = 480, height = 128;
    uint32_t pixel_size = lv_color_format_get_size(format);
    uint32_t stride = lv_draw_buf_width_to_stride(width, format) + padding;
    lv_draw_buf_t *src = lv_draw_buf_create(width, height, format, stride);
    lv_draw_buf_t *dst = lv_draw_buf_create(width, height, format, stride);
    bool passed = false;
    if(!src || !dst) goto done;

    memset(src->data, 0x55, src->data_size);
    memset(dst->data, 0xa5, dst->data_size);
    lv_draw_buf_flush_cache(src, NULL);
    lv_draw_buf_flush_cache(dst, NULL);

    lv_area_t area = {full_width ? 0 : 100, 3, full_width ? 479 : 109, 124};
    lv_draw_buf_copy(dst, &area, src, &area);
    lv_draw_buf_flush_cache(dst, NULL);
    lv_draw_buf_invalidate_cache(dst, NULL);

    for(uint32_t y = 0; y < height; y++) {
        for(uint32_t x = 0; x < stride; x++) {
            bool inside = y >= (uint32_t)area.y1 && y <= (uint32_t)area.y2 &&
                          x >= (uint32_t)area.x1 * pixel_size &&
                          x < (uint32_t)(area.x2 + 1) * pixel_size;
            if(dst->data[y * stride + x] != (inside ? 0x55 : 0xa5)) {
                printf("FAIL copy cf=%u padding=%u full=%u at byte %u,%u\n",
                       format, padding, full_width, x, y);
                goto done;
            }
        }
    }
    k_u64 src_phys, dst_phys;
    bool dma_available = lv_k230_vglite_get_buffer_phys(src->data, src->data_size, &src_phys) &&
                         lv_k230_vglite_get_buffer_phys(dst->data, dst->data_size, &dst_phys);
    if(dma_available && full_width && padding == 0 &&
       strcmp(lv_k230_image_convert_backend(), "SDMA") != 0) {
        printf("FAIL full-row copy did not exercise SDMA\n");
        goto done;
    }
    passed = true;
    printf("PASS copy cf=%u padding=%u full=%u\n", format, padding, full_width);
done:
    if(src) lv_draw_buf_destroy(src);
    if(dst) lv_draw_buf_destroy(dst);
    return passed;
}

static void draw_label(lv_layer_t *layer)
{
    lv_draw_label_dsc_t label;
    lv_draw_label_dsc_init(&label);
    label.color = lv_color_white();
    label.font = &lv_font_montserrat_14;
    label.text = "GPU + SW label";
    lv_area_t area = {30, 40, 169, 70};
    lv_draw_label(layer, &label, &area);
}

static bool test_gpu_handoff(void)
{
    enum { WIDTH = 192, HEIGHT = 128 };
    const unsigned batches[] = {1, 2, 31, 32, 33};
    lv_display_t *display = lv_display_create(WIDTH, HEIGHT);
    lv_draw_buf_t *reference = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_XRGB8888, 0);
    lv_draw_buf_t *actual = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_XRGB8888, 0);
    lv_obj_t *canvas = NULL;
    bool passed = false;
    if(!display || !reference || !actual) goto done;
    canvas = lv_canvas_create(lv_display_get_screen_active(display));
    if(!canvas) goto done;

    lv_canvas_set_draw_buf(canvas, reference);
    lv_canvas_fill_bg(canvas, lv_color_black(), LV_OPA_COVER);
    for(uint32_t y = 20; y <= 99; y++) {
        for(uint32_t x = 20; x <= 179; x++) {
            uint8_t *p = reference->data + y * reference->header.stride + x * 4;
            p[0] = p[1] = p[2] = 0x30;
        }
    }
    lv_draw_buf_flush_cache(reference, NULL);
    lv_layer_t layer;
    lv_canvas_init_layer(canvas, &layer);
    draw_label(&layer);
    lv_canvas_finish_layer(canvas, &layer);

    for(size_t i = 0; i < sizeof(batches) / sizeof(batches[0]); i++) {
        lv_canvas_set_draw_buf(canvas, actual);
        lv_canvas_fill_bg(canvas, lv_color_black(), LV_OPA_COVER);
        lv_k230_renderer_reset_stats();
        lv_canvas_init_layer(canvas, &layer);

        lv_draw_fill_dsc_t fill;
        lv_draw_fill_dsc_init(&fill);
        fill.color = lv_color_hex(0x303030);
        lv_area_t area = {20, 20, 179, 99};
        for(unsigned j = 0; j < batches[i]; j++) lv_draw_fill(&layer, &fill, &area);
        draw_label(&layer);
        lv_canvas_finish_layer(canvas, &layer);
        if(!lv_k230_vglite_wait_idle()) goto done;

        lv_k230_renderer_stats_t stats;
        lv_k230_renderer_get_stats(&stats);
        unsigned expected_tasks = lv_k230_renderer_uses_vglite() ? batches[i] : 0;
        if(stats.vglite_tasks != expected_tasks) {
            printf("FAIL GPU handoff: GPU task count %llu, expected %u\n",
                   (unsigned long long)stats.vglite_tasks, expected_tasks);
            goto done;
        }
        for(uint32_t y = 0; y < HEIGHT; y++) {
            for(uint32_t x = 0; x < WIDTH; x++) {
                if(memcmp(actual->data + y * actual->header.stride + x * 4,
                          reference->data + y * reference->header.stride + x * 4, 3)) {
                    printf("FAIL GPU handoff batch=%u at %u,%u\n", batches[i], x, y);
                    goto done;
                }
            }
        }
        printf("PASS render handoff batch=%u gpu_tasks=%u: pixels match software reference\n",
               batches[i], expected_tasks);
    }
    passed = true;
done:
    if(canvas) lv_obj_delete(canvas);
    if(reference) lv_draw_buf_destroy(reference);
    if(actual) lv_draw_buf_destroy(actual);
    if(display) lv_display_delete(display);
    return passed;
}

static bool test_argb_handoff(uint32_t padding)
{
    enum { WIDTH = 192, HEIGHT = 128 };
    const lv_area_t fill_area = {20, 20, 179, 99};
    lv_display_t *display = lv_display_create(WIDTH, HEIGHT);
    lv_draw_buf_t *reference = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_ARGB8888, 0);
    lv_draw_buf_t *actual = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_ARGB8888, WIDTH * 4 + padding);
    lv_draw_buf_t *composite = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_XRGB8888, 0);
    lv_obj_t *canvas = NULL;
    lv_image_dsc_t image = {0};
    bool passed = false;
    if(!display || !reference || !actual || !composite) goto done;
    canvas = lv_canvas_create(lv_display_get_screen_active(display));
    if(!canvas) goto done;

    lv_canvas_set_draw_buf(canvas, reference);
    lv_canvas_fill_bg(canvas, lv_color_black(), LV_OPA_TRANSP);
    for(int32_t y = fill_area.y1; y <= fill_area.y2; y++) {
        for(int32_t x = fill_area.x1; x <= fill_area.x2; x++) {
            uint8_t *p = reference->data + y * reference->header.stride + x * 4;
            p[2] = 255;
            p[3] = 128;
        }
    }
    lv_draw_buf_flush_cache(reference, NULL);
    lv_layer_t layer;
    lv_canvas_init_layer(canvas, &layer);
    draw_label(&layer);
    lv_canvas_finish_layer(canvas, &layer);

    lv_canvas_set_draw_buf(canvas, actual);
    lv_canvas_fill_bg(canvas, lv_color_black(), LV_OPA_TRANSP);
    lv_k230_renderer_reset_stats();
    lv_canvas_init_layer(canvas, &layer);
    lv_draw_fill_dsc_t fill;
    lv_draw_fill_dsc_init(&fill);
    fill.color = lv_color_hex(0xff0000);
    fill.opa = 128;
    lv_draw_fill(&layer, &fill, &fill_area);
    draw_label(&layer);
    lv_canvas_finish_layer(canvas, &layer);
    if(!lv_k230_vglite_wait_idle()) goto done;

    lv_k230_renderer_stats_t stats;
    lv_k230_renderer_get_stats(&stats);
    if(stats.vglite_tasks != 0 || lv_draw_buf_has_flag(actual, LV_IMAGE_FLAGS_PREMULTIPLIED)) {
        printf("FAIL ARGB handoff: straight-alpha target used premultiplied rendering\n");
        goto done;
    }
    unsigned edge_pixels = 0;
    for(unsigned y = 0; y < HEIGHT; y++) {
        for(unsigned x = 0; x < WIDTH; x++) {
            const uint8_t *expected = reference->data + y * reference->header.stride + x * 4;
            const uint8_t *pixel = actual->data + y * actual->header.stride + x * 4;
            if(memcmp(pixel, expected, 4)) {
                printf("FAIL ARGB handoff at %u,%u\n", x, y);
                goto done;
            }
            if(expected[3] > 128 && expected[3] < 255) edge_pixels++;
        }
    }
    if(edge_pixels == 0) goto done;

    /* Composite the image on the GPU, then reuse it through tiled software
     * rendering to check both consumers of the decoded image cache. */
    lv_draw_buf_to_image(actual, &image);
    image.header.flags &= ~(LV_IMAGE_FLAGS_ALLOCATED | LV_IMAGE_FLAGS_MODIFIABLE);
    lv_canvas_set_draw_buf(canvas, composite);
    for(unsigned tiled = 0; tiled <= 1; tiled++) {
        lv_canvas_fill_bg(canvas, lv_color_black(), LV_OPA_COVER);
        lv_k230_renderer_reset_stats();
        lv_canvas_init_layer(canvas, &layer);
        lv_draw_image_dsc_t dsc;
        lv_draw_image_dsc_init(&dsc);
        dsc.src = &image;
        dsc.tile = tiled;
        const lv_area_t whole = {0, 0, WIDTH - 1, HEIGHT - 1};
        dsc.image_area = whole;
        lv_draw_image(&layer, &dsc, &whole);
        lv_canvas_finish_layer(canvas, &layer);
        if(!lv_k230_vglite_wait_idle()) goto done;
        for(unsigned y = 0; y < HEIGHT; y++) {
            for(unsigned x = 0; x < WIDTH; x++) {
                const uint8_t *src = reference->data + y * reference->header.stride + x * 4;
                const uint8_t *pixel = composite->data + y * composite->header.stride + x * 4;
                if(memcmp(actual->data + y * actual->header.stride + x * 4, src, 4)) {
                    printf("FAIL ARGB composite modified source at %u,%u\n", x, y);
                    goto done;
                }
                for(unsigned c = 0; c < 3; c++) {
                    unsigned expected = (unsigned)src[c] * src[3] / 255;
                    unsigned delta = pixel[c] > expected ? pixel[c] - expected : expected - pixel[c];
                    if(delta > 2) {
                        printf("FAIL ARGB composite tiled=%u at %u,%u channel=%u actual=%u expected=%u\n",
                               tiled, x, y, c, pixel[c], expected);
                        goto done;
                    }
                }
            }
        }
        lv_k230_renderer_get_stats(&stats);
        bool ok = stats.vglite_tasks == (!tiled && lv_k230_renderer_uses_vglite() ? 1u : 0u);
        printf("%s ARGB handoff and composite: padding=%u tiled=%u edge_pixels=%u gpu_composites=%llu\n",
               ok ? "PASS" : "FAIL", padding, tiled, edge_pixels, (unsigned long long)stats.vglite_tasks);
        if(!ok) goto done;
    }
    passed = true;
done:
    if(image.data) lv_image_cache_drop(&image);
    if(canvas) lv_obj_delete(canvas);
    if(composite) lv_draw_buf_destroy(composite);
    if(actual) lv_draw_buf_destroy(actual);
    if(reference) lv_draw_buf_destroy(reference);
    if(display) lv_display_delete(display);
    return passed;
}

static bool test_tiled_image(lv_color_format_t format)
{
    enum { WIDTH = 192, HEIGHT = 128, TILE_SIZE = 100 };
    const lv_point_t offsets[] = {{0, 0}, {-37, -53}};
    const lv_area_t image_area = {7, 9, WIDTH - 12, HEIGHT - 8};
    lv_display_t *display = lv_display_create(WIDTH, HEIGHT);
    lv_draw_buf_t *source = lv_draw_buf_create(TILE_SIZE, TILE_SIZE, format, 0);
    lv_draw_buf_t *actual = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_XRGB8888, 0);
    lv_obj_t *canvas = NULL;
    lv_image_dsc_t image = {0};
    bool passed = false;
    if(!display || !source || !actual) goto done;
    canvas = lv_canvas_create(lv_display_get_screen_active(display));
    if(!canvas) goto done;

    /* Match the benchmark's padded 100x100 tiles; padding must never be visible. */
    memset(source->data, 0xa5, source->data_size);
    for(uint32_t y = 0; y < TILE_SIZE; y++) {
        for(uint32_t x = 0; x < TILE_SIZE; x++) {
            unsigned channel = (x / 13 + y / 17) % 3;
            uint8_t *p = source->data + y * source->header.stride +
                         x * lv_color_format_get_size(format);
            if(format == LV_COLOR_FORMAT_RGB565) {
                const uint16_t colors[] = {0x001f, 0x07e0, 0xf800};
                memcpy(p, &colors[channel], sizeof(uint16_t));
            }
            else {
                p[0] = p[1] = p[2] = 0;
                p[channel] = 0xff;
                if(format == LV_COLOR_FORMAT_XRGB8888) p[3] = 0xff;
            }
        }
    }
    lv_draw_buf_flush_cache(source, NULL);
    lv_draw_buf_to_image(source, &image);
    image.header.flags &= ~LV_IMAGE_FLAGS_ALLOCATED;
    lv_canvas_set_draw_buf(canvas, actual);

    for(size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        lv_canvas_fill_bg(canvas, lv_color_black(), LV_OPA_COVER);
        lv_layer_t layer;
        lv_canvas_init_layer(canvas, &layer);
        lv_draw_fill_dsc_t fill;
        lv_draw_fill_dsc_init(&fill);
        fill.color = lv_color_white();
        lv_area_t whole = {0, 0, WIDTH - 1, HEIGHT - 1};
        lv_draw_fill(&layer, &fill, &whole);

        lv_draw_image_dsc_t dsc;
        lv_draw_image_dsc_init(&dsc);
        dsc.src = &image;
        dsc.tile = 1;
        dsc.image_area = (lv_area_t) {offsets[i].x, offsets[i].y,
                                     offsets[i].x + TILE_SIZE - 1,
                                     offsets[i].y + TILE_SIZE - 1};
        /* The image widget clips the layer before submitting a tiled image. */
        layer._clip_area = image_area;
        lv_draw_image(&layer, &dsc, &image_area);
        lv_canvas_finish_layer(canvas, &layer);
        if(!lv_k230_vglite_wait_idle()) goto done;

        unsigned mismatches = 0;
        for(int32_t y = 0; y < HEIGHT; y++) {
            for(int32_t x = 0; x < WIDTH; x++) {
                uint8_t expected[] = {0xff, 0xff, 0xff};
                if(x >= image_area.x1 && x <= image_area.x2 &&
                   y >= image_area.y1 && y <= image_area.y2) {
                    unsigned sx = (x - offsets[i].x) % TILE_SIZE;
                    unsigned sy = (y - offsets[i].y) % TILE_SIZE;
                    memset(expected, 0, sizeof(expected));
                    expected[(sx / 13 + sy / 17) % 3] = 0xff;
                }
                if(memcmp(actual->data + y * actual->header.stride + x * 4,
                          expected, sizeof(expected))) mismatches++;
            }
        }
        printf("%s tiled image cf=%u stride=%u offset=%d,%d mismatches=%u\n",
               mismatches ? "FAIL" : "PASS", format, source->header.stride,
               offsets[i].x, offsets[i].y, mismatches);
        if(mismatches) goto done;
    }
    passed = true;
done:
    if(canvas) lv_obj_delete(canvas);
    if(source) {
        if(image.data) lv_image_cache_drop(&image);
        lv_draw_buf_destroy(source);
    }
    if(actual) lv_draw_buf_destroy(actual);
    if(display) lv_display_delete(display);
    return passed;
}

#if LV_USE_DEMO_BENCHMARK && LV_COLOR_DEPTH == 32
static bool test_benchmark_image(bool copy, bool tiled)
{
    LV_IMAGE_DECLARE(img_benchmark_lvgl_logo_rgb);
    enum { WIDTH = 256, HEIGHT = 160 };
    const lv_point_t positions[] = {{4, 12}, {72, 28}, {180, 44}};
    const lv_image_dsc_t *image = &img_benchmark_lvgl_logo_rgb;
    lv_image_dsc_t copied_image = {0};
    lv_display_t *display = lv_display_create(WIDTH, HEIGHT);
    lv_draw_buf_t *actual = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_XRGB8888, 0);
    lv_draw_buf_t *reference = lv_draw_buf_create(WIDTH, HEIGHT, LV_COLOR_FORMAT_XRGB8888, 0);
    lv_draw_buf_t *source = NULL;
    lv_obj_t *canvas = NULL;
    bool passed = false;
    if(!display || !actual || !reference) goto done;
    if(copy) {
        source = lv_draw_buf_create(image->header.w, image->header.h,
                                    image->header.cf, image->header.stride);
        if(!source) goto done;
        memcpy(source->data, image->data, image->data_size);
        lv_draw_buf_flush_cache(source, NULL);
        lv_draw_buf_to_image(source, &copied_image);
        /* Match the immutable ELF asset when software realigns its stride. */
        copied_image.header.flags &= ~(LV_IMAGE_FLAGS_ALLOCATED | LV_IMAGE_FLAGS_MODIFIABLE);
        image = &copied_image;
    }
    k_u64 phys = 0;
    bool accessible = lv_k230_vglite_get_buffer_phys(image->data, image->data_size, &phys);
    printf("Benchmark image copy=%u tiled=%u cf=%u stride=%u gpu_accessible=%u align=%u\n",
           copy, tiled, image->header.cf, image->header.stride, accessible,
           (unsigned)((uintptr_t)image->data % 64));
    canvas = lv_canvas_create(lv_display_get_screen_active(display));
    if(!canvas) goto done;
    lv_canvas_set_draw_buf(canvas, actual);
    lv_canvas_fill_bg(canvas, lv_color_white(), LV_OPA_COVER);
    memset(reference->data, 0xff, reference->data_size);
    lv_layer_t layer;
    lv_canvas_init_layer(canvas, &layer);
    lv_draw_image_dsc_t dsc;
    lv_draw_image_dsc_init(&dsc);
    dsc.src = image;
    dsc.tile = tiled;
    lv_k230_renderer_reset_stats();
    for(size_t n = 0; n < (tiled ? 1 : sizeof(positions) / sizeof(positions[0])); n++) {
        lv_area_t area = {positions[n].x, positions[n].y,
                          positions[n].x + image->header.w - 1,
                          positions[n].y + image->header.h - 1};
        if(tiled) {
            area = (lv_area_t) {0, 0, WIDTH - 1, HEIGHT - 1};
            dsc.image_area = (lv_area_t) {-37, -53, 62, 46};
        }
        lv_draw_image(&layer, &dsc, &area);
        for(int32_t y = area.y1; y <= area.y2 && y < HEIGHT; y++) {
            for(int32_t x = area.x1; x <= area.x2 && x < WIDTH; x++) {
                unsigned sx = tiled ? (x + 37) % image->header.w : x - area.x1;
                unsigned sy = tiled ? (y + 53) % image->header.h : y - area.y1;
                memcpy(reference->data + y * reference->header.stride + x * 4,
                       image->data + sy * image->header.stride + sx * 4, 3);
            }
        }
    }
    lv_canvas_finish_layer(canvas, &layer);
    if(!lv_k230_vglite_wait_idle()) goto done;
    unsigned mismatches = 0, max_delta = 0;
    for(unsigned y = 0; y < HEIGHT; y++) {
        for(unsigned x = 0; x < WIDTH; x++) {
            const uint8_t *expected = reference->data + y * reference->header.stride + x * 4;
            const uint8_t *pixel = actual->data + y * actual->header.stride + x * 4;
            bool mismatch = false;
            for(unsigned c = 0; c < 3; c++) {
                unsigned delta = pixel[c] > expected[c] ? pixel[c] - expected[c] : expected[c] - pixel[c];
                if(delta > max_delta) max_delta = delta;
                /* VG-Lite blending can round a channel down by one level. */
                if(delta > (tiled ? 0u : 1u)) mismatch = true;
            }
            if(mismatch) mismatches++;
        }
    }
    lv_k230_renderer_stats_t stats;
    lv_k230_renderer_get_stats(&stats);
    passed = mismatches == 0 &&
             stats.vglite_tasks == (!tiled && lv_k230_renderer_uses_vglite() ? 3u : 0u);
    printf("%s benchmark image copy=%u tiled=%u mismatches=%u max_delta=%u gpu_tasks=%llu\n",
           passed ? "PASS" : "FAIL", copy, tiled, mismatches, max_delta,
           (unsigned long long)stats.vglite_tasks);
done:
    if(canvas) lv_obj_delete(canvas);
    lv_image_cache_drop(image);
    if(source) lv_draw_buf_destroy(source);
    if(reference) lv_draw_buf_destroy(reference);
    if(actual) lv_draw_buf_destroy(actual);
    if(display) lv_display_delete(display);
    return passed;
}
#endif

bool lvgl_run_render_regressions(void)
{
    unsigned total = 0, passed = 0;
    const lv_color_format_t formats[] = {LV_COLOR_FORMAT_RGB565, LV_COLOR_FORMAT_RGB888,
                                         LV_COLOR_FORMAT_XRGB8888};
    for(size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
        for(unsigned padding = 0; padding <= 64; padding += 64) {
            for(unsigned full = 0; full <= 1; full++) {
                total++;
                if(test_copy_area(formats[i], padding, full)) passed++;
            }
        }
    }
    total++;
    if(test_gpu_handoff()) passed++;
    for(unsigned cache = 0; cache <= 1; cache++) {
        unsigned cache_size = cache ? 256 * 1024 : 0;
        lv_image_cache_resize(cache_size, true);
        printf("ARGB image cache size=%u\n", cache_size);
        for(uint32_t padding = 0; padding <= 64; padding += 64) {
            total++;
            if(test_argb_handoff(padding)) passed++;
        }
    }
    lv_image_cache_resize(LV_CACHE_DEF_SIZE, true);
    for(size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
        total++;
        if(test_tiled_image(formats[i])) passed++;
    }
#if LV_USE_DEMO_BENCHMARK && LV_COLOR_DEPTH == 32
    for(unsigned cache = 0; cache <= 1; cache++) {
        unsigned cache_size = cache ? 256 * 1024 : 0;
        lv_image_cache_resize(cache_size, true);
        printf("Benchmark image cache size=%u\n", cache_size);
        for(unsigned copy = 0; copy <= 1; copy++) {
            for(unsigned tiled = 0; tiled <= 1; tiled++) {
                total++;
                if(test_benchmark_image(copy, tiled)) passed++;
            }
        }
    }
    lv_image_cache_resize(LV_CACHE_DEF_SIZE, true);
#endif
    printf("LVGL_RENDER_REGRESSION: %u/%u passed\n", passed, total);
    return passed == total;
}
