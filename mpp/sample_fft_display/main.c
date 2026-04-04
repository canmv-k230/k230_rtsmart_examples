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

/*
 * Audio FFT Spectrum Visualizer
 *
 * Captures audio from I2S microphone, runs hardware FFT, and renders
 * a real-time spectrum bar graph on the display OSD layer.
 *
 * You can use this web site to generate test audio samples with specific frequencies:
 * https://www.audiocutter.org/tw/frequency-generator
 *
 * Usage:
 *   sample_fft_display -c <connector_type> [-w <osd_w>] [-h <osd_h>]
 *                      [-r <rotation>] [-s <sample_rate>] [-p <fft_points>]
 *
 * Example (01studio 480x800 screen, connector type 20):
 *   sample_fft_display -c 20 -s 16000 -p 512
 *
 * Ctrl+C to exit.
 */

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k_audio_comm.h"
#include "k_connector_comm.h"
#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"

#include "mpi_ai_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"

#include "drv_fft.h"
#include "kd_display.h"

#define ALIGN_UP(x, a) (((x) + ((a)-1)) & ~((a)-1))

#define AI_DEV    0 /* I2S device */
#define AI_CHN    0 /* channel 0 */
#define OSD_LAYER K_VO_LAYER_OSD0
#define OSD_BPP   4 /* ARGB8888 = 4 bytes/pixel */

/* Colors (ARGB8888) */
#define COLOR_BG       0xFF000000 /* black background */
#define COLOR_BAR_LOW  0xFF00FF00 /* green */
#define COLOR_BAR_MID  0xFFFFFF00 /* yellow */
#define COLOR_BAR_HIGH 0xFFFF0000 /* red */
#define COLOR_GRID     0xFF333333 /* dark gray grid lines */
#define COLOR_LABEL_BG 0xC0000000 /* semi-transparent label background */
#define COLOR_AXIS     0xFF666666
#define COLOR_TEXT     0xFFE8E8E8
#define COLOR_WHITE    0xFFFFFFFF

#define AXIS_TICK_COUNT 6

typedef struct {
    int plot_x;
    int plot_y;
    int plot_w;
    int plot_h;
    int axis_y;
    int label_y;
} spectrum_layout_t;

typedef struct {
    char    ch;
    uint8_t rows[7];
} glyph5x7_t;

static const glyph5x7_t g_font5x7[] = {
    { ' ', { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }, { '.', { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C } },
    { '0', { 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E } }, { '1', { 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E } },
    { '2', { 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F } }, { '3', { 0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E } },
    { '4', { 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 } }, { '5', { 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E } },
    { '6', { 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E } }, { '7', { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 } },
    { '8', { 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E } }, { '9', { 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C } },
    { 'H', { 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 } }, { 'k', { 0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12 } },
    { 'z', { 0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F } },
};

static volatile int g_exit;

static void sig_handler(int sig)
{
    (void)sig;
    g_exit = 1;
}

/* ------------------------------------------------------------------ */
/* Display framebuffer                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    k_u32           width;
    k_u32           height;
    k_u32*          pixels; /* virtual address, ARGB8888 */
    k_u64           phys;
    k_u32           size; /* buffer size in bytes */
    k_s32           pool_id;
    k_vb_blk_handle blk;
} framebuf_t;

static int fb_alloc(framebuf_t* fb)
{
    k_vb_pool_config pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    pcfg.blk_cnt  = 1;
    pcfg.blk_size = ALIGN_UP(fb->width * fb->height * OSD_BPP + 4096, 0x1000);

    fb->pool_id = kd_mpi_vb_create_pool(&pcfg);
    if (fb->pool_id < 0) {
        printf("vb_create_pool failed\n");
        return -1;
    }

    fb->size = pcfg.blk_size;
    fb->blk  = kd_mpi_vb_get_block(fb->pool_id, fb->size, NULL);
    if (fb->blk == VB_INVALID_HANDLE) {
        printf("vb_get_block failed\n");
        kd_mpi_vb_destory_pool(fb->pool_id);
        return -1;
    }

    fb->phys   = kd_mpi_vb_handle_to_phyaddr(fb->blk);
    fb->pixels = (k_u32*)kd_mpi_sys_mmap(fb->phys, fb->size);
    if (!fb->pixels) {
        printf("sys_mmap failed\n");
        kd_mpi_vb_release_block(fb->blk);
        kd_mpi_vb_destory_pool(fb->pool_id);
        return -1;
    }
    return 0;
}

static void fb_free(framebuf_t* fb)
{
    if (fb->pixels) {
        kd_mpi_sys_munmap(fb->pixels, fb->size);
        fb->pixels = NULL;
    }
    if (fb->blk != VB_INVALID_HANDLE) {
        kd_mpi_vb_release_block(fb->blk);
        fb->blk = VB_INVALID_HANDLE;
    }
    if (fb->pool_id >= 0) {
        kd_mpi_vb_destory_pool(fb->pool_id);
        fb->pool_id = -1;
    }
}

static void fb_push(const framebuf_t* fb)
{
    k_video_frame_info vf;
    memset(&vf, 0, sizeof(vf));
    vf.mod_id               = K_ID_VO;
    vf.pool_id              = fb->pool_id;
    vf.v_frame.width        = fb->width;
    vf.v_frame.height       = fb->height;
    vf.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    vf.v_frame.stride[0]    = fb->width * OSD_BPP;
    vf.v_frame.phys_addr[0] = fb->phys;
    kd_display_layer_push_frame(OSD_LAYER, &vf);
}

/* ------------------------------------------------------------------ */
/* Rendering                                                          */
/* ------------------------------------------------------------------ */

static inline void fb_fill(framebuf_t* fb, k_u32 color)
{
    k_u32 n = fb->width * fb->height;
    for (k_u32 i = 0; i < n; i++)
        fb->pixels[i] = color;
}

static inline void fb_put_pixel(framebuf_t* fb, int x, int y, k_u32 color)
{
    if (x < 0 || y < 0 || x >= (int)fb->width || y >= (int)fb->height)
        return;

    fb->pixels[y * fb->width + x] = color;
}

static inline void fb_rect(framebuf_t* fb, int x0, int y0, int w, int h, k_u32 color)
{
    if (x0 < 0) {
        w += x0;
        x0 = 0;
    }
    if (y0 < 0) {
        h += y0;
        y0 = 0;
    }
    int x1 = x0 + w;
    int y1 = y0 + h;
    if (x1 > (int)fb->width)
        x1 = (int)fb->width;
    if (y1 > (int)fb->height)
        y1 = (int)fb->height;
    for (int y = y0; y < y1; y++) {
        k_u32* row = fb->pixels + y * fb->width;
        for (int x = x0; x < x1; x++)
            row[x] = color;
    }
}

static k_u32 bar_color(float ratio)
{
    /* green → yellow → red based on bar height ratio */
    if (ratio < 0.5f)
        return COLOR_BAR_LOW;
    else if (ratio < 0.8f)
        return COLOR_BAR_MID;
    else
        return COLOR_BAR_HIGH;
}

static const uint8_t* font5x7_lookup(char ch)
{
    size_t count = sizeof(g_font5x7) / sizeof(g_font5x7[0]);

    for (size_t i = 0; i < count; i++) {
        if (g_font5x7[i].ch == ch)
            return g_font5x7[i].rows;
    }

    return g_font5x7[0].rows;
}

static int text5x7_width(const char* text, int scale)
{
    int width = 0;

    while (*text) {
        width += 6 * scale;
        text++;
    }

    return (width > 0) ? (width - scale) : 0;
}

static void fb_draw_text5x7(framebuf_t* fb, int x, int y, int scale, k_u32 color, const char* text)
{
    while (*text) {
        const uint8_t* glyph = font5x7_lookup(*text);

        for (int row = 0; row < 7; row++) {
            for (int col = 0; col < 5; col++) {
                if ((glyph[row] & (1u << (4 - col))) == 0)
                    continue;

                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++)
                        fb_put_pixel(fb, x + col * scale + dx, y + row * scale + dy, color);
                }
            }
        }

        x += 6 * scale;
        text++;
    }
}

static k_gdma_rotation_e rotation_from_degrees(int degrees)
{
    if (degrees == 90)
        return GDMA_ROTATE_DEGREE_90;
    if (degrees == 180)
        return GDMA_ROTATE_DEGREE_180;
    if (degrees == 270)
        return GDMA_ROTATE_DEGREE_270;

    return GDMA_ROTATE_DEGREE_0;
}

static int rotation_swaps_axes(int degrees) { return degrees == 90 || degrees == 270; }

static void connector_dimensions(k_connector_type conn_type, k_u32* width, k_u32* height)
{
    k_u32 conn_w = K_CONN_WIDTH(conn_type);
    k_u32 conn_h = K_CONN_HEIGHT(conn_type);

    if (conn_w == 0 || conn_h == 0) {
        *width  = 480;
        *height = 800;
        return;
    }

    *width  = conn_w;
    *height = conn_h;
}

static spectrum_layout_t spectrum_layout_for_fb(const framebuf_t* fb)
{
    spectrum_layout_t layout;
    int               side_margin = ((int)fb->width >= 640) ? 16 : 10;
    int               top_margin  = ((int)fb->height >= 480) ? 12 : 8;
    int               axis_band   = ((int)fb->height >= 480) ? 34 : 26;

    layout.plot_x = side_margin;
    layout.plot_y = top_margin;
    layout.plot_w = (int)fb->width - side_margin * 2;
    layout.plot_h = (int)fb->height - top_margin - axis_band - 8;

    if (layout.plot_w < 32)
        layout.plot_w = 32;
    if (layout.plot_h < 24)
        layout.plot_h = 24;

    layout.axis_y  = layout.plot_y + layout.plot_h + 4;
    layout.label_y = layout.axis_y + 6;

    return layout;
}

static void format_freq_label(int hz, char* buf, size_t buf_size)
{
    if (hz >= 1000) {
        if ((hz % 1000) == 0) {
            snprintf(buf, buf_size, "%dk", hz / 1000);
        } else {
            snprintf(buf, buf_size, "%.1fk", hz / 1000.0f);
        }
    } else {
        snprintf(buf, buf_size, "%d", hz);
    }
}

static void draw_frequency_axis(framebuf_t* fb, const spectrum_layout_t* layout, k_u32 sample_rate)
{
    int label_scale      = (fb->width >= 640) ? 2 : 1;
    int nyquist_hz       = (int)(sample_rate / 2);
    int last_label_right = -10000;

    fb_rect(fb, layout->plot_x, layout->axis_y, layout->plot_w, 1, COLOR_AXIS);

    for (int i = 0; i < AXIS_TICK_COUNT; i++) {
        int  x    = layout->plot_x + ((layout->plot_w - 1) * i) / (AXIS_TICK_COUNT - 1);
        int  freq = (nyquist_hz * i) / (AXIS_TICK_COUNT - 1);
        char label[16];
        int  label_w;
        int  label_x;

        fb_rect(fb, x, layout->axis_y - 3, 1, 7, COLOR_AXIS);
        format_freq_label(freq, label, sizeof(label));

        label_w = text5x7_width(label, label_scale);
        label_x = x - label_w / 2;
        if (i == 0)
            label_x = layout->plot_x;
        if (i == AXIS_TICK_COUNT - 1)
            label_x = layout->plot_x + layout->plot_w - label_w;

        if (label_x < layout->plot_x)
            label_x = layout->plot_x;
        if (label_x + label_w > layout->plot_x + layout->plot_w)
            label_x = layout->plot_x + layout->plot_w - label_w;

        if (label_x <= last_label_right + 4)
            continue;

        fb_draw_text5x7(fb, label_x, layout->label_y, label_scale, COLOR_TEXT, label);
        last_label_right = label_x + label_w;
    }

    fb_draw_text5x7(fb, layout->plot_x + layout->plot_w - text5x7_width("Hz", label_scale), layout->plot_y + 2, label_scale,
                    COLOR_TEXT, "Hz");
}

/*
 * Render spectrum bars into framebuffer.
 * magnitudes[0..num_bars-1] are the FFT bin magnitudes (linear).
 * max_mag is used for normalization.
 */
static void render_spectrum(framebuf_t* fb, const float* magnitudes, int num_bars, float max_mag, k_u32 sample_rate)
{
    spectrum_layout_t layout       = spectrum_layout_for_fb(fb);
    int               bar_gap      = (layout.plot_w >= num_bars * 3) ? 1 : 0;
    int               visible_bars = num_bars;

    if (layout.plot_w <= 0 || layout.plot_h <= 0 || num_bars <= 0)
        return;

    if (visible_bars > layout.plot_w)
        visible_bars = layout.plot_w;
    if (bar_gap != 0) {
        int max_bars = (layout.plot_w + bar_gap) / (bar_gap + 1);
        if (visible_bars > max_bars)
            visible_bars = max_bars;
    }
    if (visible_bars < 1)
        visible_bars = 1;

    int total_gaps = (visible_bars > 1) ? (visible_bars - 1) * bar_gap : 0;
    int bar_w      = (layout.plot_w - total_gaps) / visible_bars;
    if (bar_w < 1) {
        bar_gap      = 0;
        visible_bars = layout.plot_w;
        if (visible_bars > num_bars)
            visible_bars = num_bars;
        if (visible_bars < 1)
            visible_bars = 1;
        total_gaps = 0;
        bar_w      = layout.plot_w / visible_bars;
        if (bar_w < 1)
            bar_w = 1;
    }

    /* clear background */
    fb_fill(fb, COLOR_BG);

    /* draw horizontal grid lines (4 lines) */
    for (int i = 1; i <= 4; i++) {
        int gy = layout.plot_y + layout.plot_h - (layout.plot_h * i / 4);
        fb_rect(fb, layout.plot_x, gy, layout.plot_w, 1, COLOR_GRID);
    }

    fb_rect(fb, layout.plot_x, layout.plot_y + layout.plot_h, layout.plot_w, 1, COLOR_AXIS);

    /* clamp normalization */
    if (max_mag < 1.0f)
        max_mag = 1.0f;

    /* draw bars */
    for (int i = 0; i < visible_bars; i++) {
        int   start_bin  = (i * num_bars) / visible_bars;
        int   end_bin    = ((i + 1) * num_bars) / visible_bars;
        float bucket_mag = 0.0f;
        float ratio;
        int   bar_h;
        int   bx;
        int   by;

        if (end_bin <= start_bin)
            end_bin = start_bin + 1;

        for (int bin = start_bin; bin < end_bin; bin++) {
            if (magnitudes[bin] > bucket_mag)
                bucket_mag = magnitudes[bin];
        }

        ratio = bucket_mag / max_mag;
        if (ratio > 1.0f)
            ratio = 1.0f;

        bar_h = (int)(ratio * layout.plot_h);
        if (bar_h < 0)
            bar_h = 0;

        bx = layout.plot_x + i * (bar_w + bar_gap);
        by = layout.plot_y + layout.plot_h - bar_h;

        fb_rect(fb, bx, by, bar_w, bar_h, bar_color(ratio));
    }

    draw_frequency_axis(fb, &layout, sample_rate);
}

/* ------------------------------------------------------------------ */
/* Audio input                                                        */
/* ------------------------------------------------------------------ */

static int ai_init(k_u32 sample_rate, k_u32 point_num_per_frame)
{
    k_aio_dev_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.audio_type                                 = KD_AUDIO_INPUT_TYPE_I2S;
    attr.kd_audio_attr.i2s_attr.chn_cnt             = 2;
    attr.kd_audio_attr.i2s_attr.sample_rate         = sample_rate;
    attr.kd_audio_attr.i2s_attr.bit_width           = KD_AUDIO_BIT_WIDTH_16;
    attr.kd_audio_attr.i2s_attr.snd_mode            = KD_AUDIO_SOUND_MODE_MONO;
    attr.kd_audio_attr.i2s_attr.i2s_mode            = K_STANDARD_MODE;
    attr.kd_audio_attr.i2s_attr.frame_num           = 15;
    attr.kd_audio_attr.i2s_attr.point_num_per_frame = point_num_per_frame;
    attr.kd_audio_attr.i2s_attr.i2s_type            = K_AIO_I2STYPE_INNERCODEC;

    k_s32 ret = kd_mpi_ai_set_pub_attr(AI_DEV, &attr);
    if (ret) {
        printf("ai_set_pub_attr failed: %d\n", ret);
        return -1;
    }

    ret = kd_mpi_ai_enable(AI_DEV);
    if (ret) {
        printf("ai_enable failed: %d\n", ret);
        return -1;
    }

    ret = kd_mpi_ai_enable_chn(AI_DEV, AI_CHN);
    if (ret) {
        printf("ai_enable_chn failed: %d\n", ret);
        return -1;
    }

    return 0;
}

static void ai_deinit(void)
{
    kd_mpi_ai_disable_chn(AI_DEV, AI_CHN);
    kd_mpi_ai_disable(AI_DEV);
}

/* ------------------------------------------------------------------ */
/* Main loop                                                          */
/* ------------------------------------------------------------------ */

static void print_usage(const char* prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("  -c <type>    Connector type (required), see list_connector\n");
    printf("  -w <width>   OSD width  [default: panel width after rotation]\n");
    printf("  -h <height>  OSD height [default: panel height after rotation]\n");
    printf("  -r <deg>     Rotation: 0, 90, 180, 270 [default: auto landscape]\n");
    printf("  -s <rate>    Audio sample rate [default: 44100], choice in: 8000, 12000, 16000, 24000, 32000, 44100, "
           "48000,96000, 192000\n");
    printf("  -p <points>  FFT points (64-4096, power of 2) [default: 512]\n");
    printf("  -g <gain>    Display gain in dB [default: 0]\n");
}

int main(int argc, char** argv)
{
    k_connector_type  conn_type    = 0;
    k_u32             osd_w        = 0;
    k_u32             osd_h        = 0;
    k_u32             sample_rate  = 44100;
    k_u32             fft_points   = 512;
    float             gain_db      = 0.0f;
    int               conn_set     = 0;
    int               width_set    = 0;
    int               height_set   = 0;
    int               rotation_deg = -1;
    k_u32             panel_w      = 0;
    k_u32             panel_h      = 0;
    k_gdma_rotation_e rotate;

    int opt;
    while ((opt = getopt(argc, argv, "c:w:h:r:s:p:g:")) != -1) {
        switch (opt) {
        case 'c':
            conn_type = (k_connector_type)atoi(optarg);
            conn_set  = 1;
            break;
        case 'w':
            osd_w     = (k_u32)atoi(optarg);
            width_set = 1;
            break;
        case 'h':
            osd_h      = (k_u32)atoi(optarg);
            height_set = 1;
            break;
        case 'r':
            rotation_deg = atoi(optarg);
            break;
        case 's':
            sample_rate = (k_u32)atoi(optarg);
            break;
        case 'p':
            fft_points = (k_u32)atoi(optarg);
            break;
        case 'g':
            gain_db = (float)atof(optarg);
            break;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!conn_set) {
        printf("ERROR: -c <connector_type> is required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Validate FFT points: must be power of 2, 64..4096 */
    if (fft_points < 64 || fft_points > 4096 || (fft_points & (fft_points - 1)) != 0) {
        printf("ERROR: FFT points must be power of 2 in [64, 4096]\n");
        return 1;
    }

    connector_dimensions(conn_type, &panel_w, &panel_h);
    if (rotation_deg != -1 && rotation_from_degrees(rotation_deg) == GDMA_ROTATE_DEGREE_0 && rotation_deg != 0) {
        printf("ERROR: rotation must be one of 0, 90, 180, 270\n");
        return 1;
    }
    if (rotation_deg == -1)
        rotation_deg = (panel_h > panel_w) ? 90 : 0;

    if (!width_set || !height_set) {
        k_u32 effective_w = rotation_swaps_axes(rotation_deg) ? panel_h : panel_w;
        k_u32 effective_h = rotation_swaps_axes(rotation_deg) ? panel_w : panel_h;

        if (!width_set)
            osd_w = effective_w;
        if (!height_set)
            osd_h = effective_h;
    }

    rotate = rotation_from_degrees(rotation_deg);

    float gain_linear = powf(10.0f, gain_db / 20.0f);

    printf("FFT Spectrum: connector=%u, panel=%ux%u, OSD=%ux%u, rotate=%d, rate=%u, points=%u, gain=%.1f dB\n", conn_type,
           panel_w, panel_h, osd_w, osd_h, rotation_deg, sample_rate, fft_points, gain_db);
    printf("main start connector=%u panel=%ux%u osd=%ux%u rotate=%d rate=%u points=%u\n", conn_type, panel_w, panel_h, osd_w,
           osd_h, rotation_deg, sample_rate, fft_points);

    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* 1) Init display */
    if (kd_display_init(conn_type, 0, 0, rotate) != 0) {
        printf("kd_display_init failed\n");
        return 1;
    }

    /* 2) VB system init */
    k_vb_config vb_cfg;
    memset(&vb_cfg, 0, sizeof(vb_cfg));
    vb_cfg.max_pool_cnt = 10;
    if (kd_mpi_vb_set_config(&vb_cfg) || kd_mpi_vb_init()) {
        printf("VB init failed\n");
        kd_display_deinit();
        return 1;
    }

    /* 3) Allocate framebuffer */
    framebuf_t fb = { .width = osd_w, .height = osd_h, .pool_id = -1, .blk = VB_INVALID_HANDLE };
    if (fb_alloc(&fb) != 0) {
        kd_mpi_vb_exit();
        kd_display_deinit();
        return 1;
    }

    /* 4) Configure OSD layer */
    if (kd_display_layer_configure(OSD_LAYER, PIXEL_FORMAT_ARGB_8888, osd_w, osd_h, 0, 0) != 0) {
        printf("layer_configure failed\n");
        fb_free(&fb);
        kd_mpi_vb_exit();
        kd_display_deinit();
        return 1;
    }
    kd_display_layer_enable(OSD_LAYER);

    /* Show initial blank screen */
    fb_fill(&fb, COLOR_BG);
    fb_push(&fb);

    /* 5) Init audio input */
    if (ai_init(sample_rate, fft_points) != 0) {
        kd_display_layer_disable(OSD_LAYER);
        fb_free(&fb);
        kd_mpi_vb_exit();
        kd_display_deinit();
        return 1;
    }

    /* 6) Open FFT device */
    drv_fft_inst_t* fft_inst = NULL;
    printf("before drv_fft_open\n");
    if (drv_fft_open(&fft_inst) != 0) {
        printf("drv_fft_open failed\n");
        ai_deinit();
        kd_display_layer_disable(OSD_LAYER);
        fb_free(&fb);
        kd_mpi_vb_exit();
        kd_display_deinit();
        return 1;
    }
    printf("after drv_fft_open inst=%p\n", fft_inst);

    /* Allocate FFT I/O buffers */
    short* fft_in_real  = (short*)calloc(fft_points, sizeof(short));
    short* fft_in_imag  = (short*)calloc(fft_points, sizeof(short));
    short* fft_out_real = (short*)calloc(fft_points, sizeof(short));
    short* fft_out_imag = (short*)calloc(fft_points, sizeof(short));

    /* Number of visible spectrum bars: only the first half of FFT bins
     * (Nyquist) are meaningful, and we cap to the framebuffer width */
    int    num_bins   = (int)(fft_points / 2);
    float* magnitudes = (float*)calloc(num_bins, sizeof(float));

    if (!fft_in_real || !fft_in_imag || !fft_out_real || !fft_out_imag || !magnitudes) {
        printf("malloc failed\n");
        goto cleanup;
    }

    drv_fft_cfg_t fft_cfg = {
        .point       = fft_points,
        .mode        = FFT_MODE,
        .input_mode  = RIRI,
        .output_mode = RR_II_OUT,
        .shift       = 0x555,
        .timeout_ms  = 0,
    };

    printf("Running... press Ctrl+C to stop\n");

    /* Exponential moving average for peak tracking */
    float peak_mag = 100.0f;

    /* ---- Main loop ---- */
    while (!g_exit) {
        k_audio_frame frame;
        k_s32         ret = kd_mpi_ai_get_frame(AI_DEV, AI_CHN, &frame, 500);
        if (ret != K_SUCCESS)
            continue;

        /* Map audio data */
        k_s16* pcm = (k_s16*)kd_mpi_sys_mmap(frame.phys_addr, frame.len);
        if (!pcm) {
            kd_mpi_ai_release_frame(AI_DEV, AI_CHN, &frame);
            continue;
        }

        /* Copy PCM samples into FFT input (real part).
         * frame.len is bytes per channel; for 16-bit mono that's
         * point_num_per_frame * 2 bytes. */
        k_u32 num_samples = frame.len / sizeof(k_s16);
        if (num_samples > fft_points)
            num_samples = fft_points;

        for (k_u32 i = 0; i < num_samples; i++)
            fft_in_real[i] = pcm[i];
        /* Zero-pad if fewer samples than FFT points */
        for (k_u32 i = num_samples; i < fft_points; i++)
            fft_in_real[i] = 0;
        memset(fft_in_imag, 0, fft_points * sizeof(short));

        kd_mpi_sys_munmap(pcm, frame.len);
        kd_mpi_ai_release_frame(AI_DEV, AI_CHN, &frame);

        /* Run hardware FFT */
        ret = drv_fft_fft(fft_inst, &fft_cfg, fft_in_real, fft_in_imag, fft_out_real, fft_out_imag);
        if (ret != 0) {
            printf("drv_fft_fft failed ret=%d", ret);
            continue;
        }

        /* Compute magnitudes for the first half of FFT bins */
        float frame_peak = 0.0f;
        for (int i = 0; i < num_bins; i++) {
            float re      = (float)fft_out_real[i] * gain_linear;
            float im      = (float)fft_out_imag[i] * gain_linear;
            magnitudes[i] = sqrtf(re * re + im * im);
            if (magnitudes[i] > frame_peak)
                frame_peak = magnitudes[i];
        }

        /* Smooth peak: fast attack, slow decay */
        if (frame_peak > peak_mag)
            peak_mag = frame_peak;
        else
            peak_mag = peak_mag * 0.95f + frame_peak * 0.05f;

        /* Render and push */
        render_spectrum(&fb, magnitudes, num_bins, peak_mag, sample_rate);
        fb_push(&fb);
    }

cleanup:
    printf("\nShutting down...\n");
    free(magnitudes);
    free(fft_out_imag);
    free(fft_out_real);
    free(fft_in_imag);
    free(fft_in_real);

    drv_fft_close(&fft_inst);
    ai_deinit();
    kd_display_layer_disable(OSD_LAYER);
    fb_free(&fb);
    kd_mpi_vb_exit();
    kd_display_deinit();

    printf("Done.\n");
    return 0;
}
