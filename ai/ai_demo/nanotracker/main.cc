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
//
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "nanotrack_crop.h"
#include "nanotrack_src.h"
#include "nanotrack_tracker.h"
#include "video_pipeline.h"
#include "ai_utils.h"

using std::cerr;
using std::cout;
using std::endl;

using namespace std;
using namespace cv;

std::atomic<bool> isp_stop(false);

void print_usage(const char *name)
{
    cout << "Usage: " << name << "<crop_kmodel> <src_kmodel> <head_kmodel> <crop_net_len> <src_net_len> <head_thresh>  <debug_mode> " << endl
         << "For example: " << endl
         << " [for isp]  ./nanotracker.elf cropped_test127.kmodel nanotrack_backbone_sim.kmodel nanotracker_head_calib_k230.kmodel 0.1 0" << endl
         << "Options:" << endl
         << " crop_kmodel    模板template kmodel文件路径 \n"
         << " src_kmodel  跟踪目标 kmodel文件路径 \n"
         << " head_kmodel  检测头 kmodel文件路径 \n"
         << " head_thresh    检测头 检测阈值  \n"
         << " debug_mode      是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "  -s <csi_num>    Sensor CSI口编号，默认 2\n"
         << "  -L <2|4>        MIPI lane preference，默认 ANY\n"
         << "\n"
         << endl;
}

static int g_csi_num = 2;
static k_vicap_mipi_lane_pref g_lane_pref = VICAP_MIPI_LANE_PREF_ANY;

void video_proc(char *argv[])
{
    int debug_mode = atoi(argv[5]);
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
    float head_thresh = atof(argv[4]);
    NanoTrackCrop nanotrack_crop(argv[1], image_size,debug_mode);
    NanoTrackSrc nanotrack_src(argv[2], image_size,debug_mode);
    NanoTrackTracker nanotrack_tracker(argv[3],image_size,head_thresh,debug_mode);

    std::vector<float> results_crop;
    std::vector<float> results_src;
    std::vector<float> results_tracker;
    Bbox bbox;
    int count=0;
    while(!isp_stop){
        // 创建一个ScopedTiming对象，用于计算总时间
        ScopedTiming st("total time", 1);
        // 从PipeLine中获取一帧数据，并创建tensor
        pl.GetFrame(dump_res);
        input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
        //前处理，推理，后处理
        if(count<100){
            results_crop.clear();
            nanotrack_crop.pre_process(input_tensor);
            nanotrack_crop.inference();
            nanotrack_crop.post_process(results_crop);
            nanotrack_crop.draw_box(draw_frame);
            count++;
        }
        else{
            draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
            results_src.clear();
            results_tracker.clear();
            nanotrack_src.set_center(nanotrack_tracker.get_center());
            nanotrack_src.set_rect_size(nanotrack_tracker.get_rect_size());
            nanotrack_src.pre_process(input_tensor);
            nanotrack_src.inference();
            nanotrack_src.post_process(results_src);

            nanotrack_tracker.pre_process(results_crop,results_src);
            nanotrack_tracker.inference();
            nanotrack_tracker.post_process(bbox);
            nanotrack_tracker.draw_result(draw_frame,bbox);
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
    if (argc != 6)
    {
        print_usage(argv[0]);
        return -1;
    }

    {
        std::thread thread_isp(video_proc, argv);
        while (getchar() != 'q')
        {
            usleep(10000);
        }
        isp_stop = true;
        thread_isp.join();
    }
    
    return 0;
}
