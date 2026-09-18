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
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "ai_utils.h"
#include "face_detection.h"
#include "face_gender.h"
#include "video_pipeline.h"

using std::cerr;
using std::cout;
using std::endl;

std::atomic<bool> isp_stop(false);

void print_usage(const char *name)
{
    cout << "Usage: " << name << "<kmodel_det> <obj_thres> <nms_thres> <kmodel_fg> <input_mode> <debug_mode>" << endl
         << "Options:" << endl
         << "  kmodel_det      人脸检测kmodel路径\n"
         << "  obj_thres       人脸检测阈值\n"
         << "  nms_thres       人脸检测nms阈值\n"
         << "  kmodel_fg       人脸性别kmodel路径\n"
         << "  input_mode      本地图片(图片路径)/ 摄像头(None) \n"
         << "  debug_mode      是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "  -s <csi_num>    Sensor CSI口编号，默认 2\n"
         << "  -L <2|4>        MIPI lane preference，默认 ANY\n"
         << "\n"
         << endl;
}

static int g_csi_num = 2;
static k_vicap_mipi_lane_pref g_lane_pref = VICAP_MIPI_LANE_PREF_ANY;

void video_proc(char *argv[])
{
    int debug_mode = atoi(argv[6]);
    FrameCHWSize image_size={AI_FRAME_CHANNEL,AI_FRAME_HEIGHT, AI_FRAME_WIDTH};
    // 创建一个空的Mat对象，用于存储绘制的帧
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    // 创建一个空的runtime_tensor对象，用于存储输入数据
    runtime_tensor input_tensor;
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };

    // 创建一个PipeLine对象，用于处理视频流
    PipeLine pl(debug_mode, g_csi_num, g_lane_pref);
    // 初始化PipeLine（probe 失败则退出）
    if (pl.Create() != 0) {
        printf("PipeLine Create failed, exit\n");
        exit(1);
    }
    // 创建一个DumpRes对象，用于存储帧数据
    DumpRes dump_res;
    // 创建FaceDetection实例
    FaceDetection fd(argv[1], atof(argv[2]),atof(argv[3]), image_size,debug_mode);
    FaceGender fa(argv[4], image_size,debug_mode);

    vector<FaceDetectionInfo> det_results;
    FaceGenderInfo fa_result;

    while(!isp_stop){
        // 创建一个ScopedTiming对象，用于计算总时间
        ScopedTiming st("total time", 1);
        // 从PipeLine中获取一帧数据，并创建tensor
        pl.GetFrame(dump_res);
        input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
        //前处理，推理，后处理
        det_results.clear();
        fd.pre_process(input_tensor);
        fd.inference();
        fd.post_process(image_size,det_results);
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
        for (int i = 0; i < det_results.size(); ++i)
        {
            fa.pre_process(input_tensor,det_results[i].bbox);
            fa.inference();
            fa.post_process(fa_result);
            fa.draw_result(draw_frame,det_results[i].bbox,fa_result,false);
        }
        // 将绘制的帧插入到PipeLine中
        pl.InsertFrame(draw_frame.data);
        // 释放帧数据
        pl.ReleaseFrame(dump_res);
    }
    pl.Destroy();
}


/* Parse optional camera options (CSI defaults to 2, lane preference defaults to ANY); strip them so positional argc checks stay valid. */

static int parse_csi_and_compact_argv(int argc, char **argv, int *csi_num,
                                      k_vicap_mipi_lane_pref *lane_pref)
{
    *csi_num = 2;
    *lane_pref = VICAP_MIPI_LANE_PREF_ANY;
    int w = 1;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--csi") == 0) && i + 1 < argc) {
            *csi_num = atoi(argv[++i]);
            continue;
        }
        if ((strcmp(argv[i], "-L") == 0 || strcmp(argv[i], "--lane") == 0 ||
             strcmp(argv[i], "-lane") == 0) && i + 1 < argc) {
            int lane = atoi(argv[++i]);
            if (lane == 2)
                *lane_pref = VICAP_MIPI_LANE_PREF_2LANE;
            else if (lane == 4)
                *lane_pref = VICAP_MIPI_LANE_PREF_4LANE;
            else {
                printf("ERROR: -L/--lane must be 2 or 4\n");
                return -1;
            }
            continue;
        }
        argv[w++] = argv[i];
    }
    argv[w] = nullptr;
    return w;
}

int main(int argc, char *argv[])
{
    argc = parse_csi_and_compact_argv(argc, argv, &g_csi_num, &g_lane_pref);
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 7)
    {
        print_usage(argv[0]);
        return -1;
    }

    if (strcmp(argv[5], "None") == 0)
    {
        std::thread thread_isp(video_proc, argv);
        while (getchar() != 'q')
        {
            usleep(10000);
        }

        isp_stop = true;
        thread_isp.join();
    }
    else
    {

        int debug_mode = atoi(argv[6]);
        // 读取图片
        cv::Mat ori_img = cv::imread(argv[5]);
        FrameCHWSize image_size={ori_img.channels(),ori_img.rows,ori_img.cols};
        // 创建一个空的向量，用于存储chw图像数据,将读入的hwc数据转换成chw数据
        std::vector<uint8_t> chw_vec;
        std::vector<cv::Mat> bgrChannels(3);
        cv::split(ori_img, bgrChannels);
        for (auto i = 2; i > -1; i--)
        {
            std::vector<uint8_t> data = std::vector<uint8_t>(bgrChannels[i].reshape(1, 1));
            chw_vec.insert(chw_vec.end(), data.begin(), data.end());
        }
        // 创建tensor
        dims_t in_shape { 1, 3, ori_img.rows, ori_img.cols };
        runtime_tensor input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("cannot create input tensor");
        auto input_buf = input_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
        memcpy(reinterpret_cast<char *>(input_buf.data()), chw_vec.data(), chw_vec.size());
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("write back input failed");
        
        FaceDetection fd(argv[1], atof(argv[2]),atof(argv[3]), image_size,debug_mode);
        FaceGender fa(argv[4], image_size,debug_mode);

        vector<FaceDetectionInfo> det_results;
        FaceGenderInfo fa_result;
        fd.pre_process(input_tensor);
        fd.inference();
        fd.post_process(image_size,det_results);
        for (int i = 0; i < det_results.size(); ++i)
        {
            fa.pre_process(input_tensor, det_results[i].bbox);
            fa.inference();
            fa.post_process(fa_result);
            fa.draw_result(ori_img,det_results[i].bbox,fa_result);
        }
        cv::imwrite("face_gender_result.jpg", ori_img);
    }
    return 0;
}