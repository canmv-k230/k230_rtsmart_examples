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

#include "sample_dpu_vo.h"
#include <fcntl.h>

extern sample_dv_cfg_t g_vdd_cfg[];

void sample_dv_vicap_config(k_u32 ch, k_s32 sensor_index)
{
    sample_dv_cfg_t *vdd_cfg;
    k_s32 ret;
    k_vicap_dev vicap_dev = VICAP_DEV_ID_0;
    k_vicap_chn vicap_chn = ch;
    k_vicap_dev_attr dev_attr;
    k_vicap_chn_attr chn_attr;
    k_vicap_sensor_info sensor_info;

    vdd_cfg = &g_vdd_cfg[ch];

    memset(&dev_attr, 0, sizeof(k_vicap_dev_attr));
    memset(&chn_attr, 0, sizeof(k_vicap_chn_attr));
    memset(&sensor_info, 0, sizeof(k_vicap_sensor_info));

    sensor_info.sensor_type = sensor_index;

#if 1
    printf("If sensor type is OV9286, Modify the code.\n");
#else

    k_u8 *cmu;
    k_s32 mem_fd = -1;
    #define CMU_REG_BASE 0x91100000
    #define CMU_REG_SIZE 0x1000

    if (sensor_index == OV_OV9286_MIPI_1280X720_30FPS_10BIT_MCLK_25M_LINEAR_SPECKLE ||
        sensor_index == OV_OV9286_MIPI_1280X720_30FPS_10BIT_MCLK_25M_LINEAR_IR)
    {
        mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (mem_fd < 0)
        {
            printf("open /dev/mem fail\n");
            return ;
        }

        cmu = mmap(NULL, CMU_REG_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, CMU_REG_BASE);
        if (cmu == NULL)
        {
            printf("mmap fail\n");
            return ;
        }

        *(volatile unsigned int *)(cmu + 0x6c) = 0x803b89f7; //config MCLK for sensor
    }
#endif

    ret = kd_mpi_vicap_get_sensor_info(sensor_info.sensor_type, &sensor_info);
    VDD_CHECK_RET(ret, __func__, __LINE__);

    dev_attr.acq_win.width = sensor_info.width;
    dev_attr.acq_win.height = sensor_info.height;
    dev_attr.mode = VICAP_WORK_ONLINE_MODE;

    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(vicap_dev, dev_attr);
    VDD_CHECK_RET(ret, __func__, __LINE__);

    chn_attr.out_win.width = vdd_cfg->img_width;
    chn_attr.out_win.height = vdd_cfg->img_height;

    chn_attr.crop_win = chn_attr.out_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_FALSE;
    chn_attr.scale_enable = K_FALSE;
    chn_attr.chn_enable = K_TRUE;

    //chn_attr.bit_width = ISP_PIXEL_YUV_8_BIT;
    chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num = VDD_INPUT_BUF_CNT;
    chn_attr.buffer_size = vdd_cfg->img_width * vdd_cfg->img_height * 3 / 2;
    chn_attr.buffer_pool_id = VB_INVALID_POOLID;
    //chn_attr.block_type = ISP_BUFQUE_TIMEOUT_TYPE;
    //chn_attr.wait_time = 500;

    ret = kd_mpi_vicap_set_chn_attr(vicap_dev, vicap_chn, chn_attr);
    VDD_CHECK_RET(ret, __func__, __LINE__);

    ret = kd_mpi_vicap_init(vicap_dev);
    VDD_CHECK_RET(ret, __func__, __LINE__);
}

void sample_dv_vicap_start(k_u32 ch)
{
    k_s32 ret;

    ret = kd_mpi_vicap_start_stream(VICAP_DEV_ID_0);
    VDD_CHECK_RET(ret, __func__, __LINE__);
}

void sample_dv_vicap_stop(k_u32 ch)
{
    k_s32 ret;

    ret = kd_mpi_vicap_stop_stream(VICAP_DEV_ID_0);
    VDD_CHECK_RET(ret, __func__, __LINE__);
    ret = kd_mpi_vicap_deinit(VICAP_DEV_ID_0);
    VDD_CHECK_RET(ret, __func__, __LINE__);
}