/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
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

#ifndef SAMPLE_DW200_CONFIG_H
#define SAMPLE_DW200_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

#include "k_dewarp_comm.h"

#define BLOCK_SIZE 16
#define VS_PI 3.1415926535897932384626433832795
#define VS_2PI 6.283185307179586476925286766559

#define MAX_MAP_SIZE 0xF0000
#define DW200_INPUT_COUNT 2U
#define DW200_OUTPUT_COUNT 4U
#define DW200_VSE_OUTPUT_COUNT 3U

struct dewarp_buffer_size {
    uint32_t width;
    uint32_t height;
};

struct dw200_resolution {
    uint32_t yuvbit;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t enable;
};

struct dewarp_single_pixel {
    uint8_t y;
    uint8_t u;
    uint8_t v;
};

struct fov_parameter {
    double off_angle_ul;
    double off_angle_ur;
    double off_angle_dl;
    double off_angle_dr;
    double fov_ul;
    double fov_ur;
    double fov_dl;
    double fov_dr;
    int pano_at_win;
    double center_offset_ratio_ul;
    double center_offset_ratio_ur;
    double center_offset_ratio_dl;
    double center_offset_ratio_dr;
    double circle_offset_ratio_ul;
    double circle_offset_ratio_ur;
    double circle_offset_ratio_dl;
    double circle_offset_ratio_dr;
};

struct dw200_parameters {
    struct dw200_resolution input_res[DW200_INPUT_COUNT];
    struct dw200_resolution output_res[DW200_OUTPUT_COUNT];
    struct dewarp_buffer_size roi_start;
    struct dewarp_single_pixel boundary_pixel;
    uint32_t scale_factor;
    uint32_t split_horizon_line;
    uint32_t split_vertical_line_up;
    uint32_t split_vertical_line_down;
    uint32_t dewarp_type;
    bool rotation;
    bool hflip;
    bool vflip;
    bool bypass;
    struct fov_parameter fov;

    uint32_t vse_input_select;
    struct k_vse_crop_size vse_crop_size[DW200_VSE_OUTPUT_COUNT];
    struct k_vse_format_conv_settings vse_format_conv[DW200_VSE_OUTPUT_COUNT];
    struct k_vse_mi_settings mi_settings[DW200_VSE_OUTPUT_COUNT];
};

struct dewarp_distortion_map {
    double camera_matrix[9];
    double perspective_matrix[9];
    double distortion_coeff[8];
};

enum format_t {
    MEDIA_PIX_FMT_YUV422SP = 0,
    MEDIA_PIX_FMT_YUV422I,
    MEDIA_PIX_FMT_YUV420SP,
    MEDIA_PIX_FMT_YUV444,
    MEDIA_PIX_FMT_RGB888,
    MEDIA_PIX_FMT_RGB888P,
    MEDIA_PIX_FMT_RAW8,
    MEDIA_PIX_FMT_RAW12,
};

#define INT_FRAME_DONE      (1 << 0)
#define INT_ERR_STATUS_MASK  0x000000FE
#define INT_ERR_STATUS_SHIFT 1
#define INT_FRAME_BUSY       0x00010000

#endif /* SAMPLE_DW200_CONFIG_H */
