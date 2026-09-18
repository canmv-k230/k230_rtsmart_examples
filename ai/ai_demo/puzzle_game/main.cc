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
#include <chrono>
#include <fstream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include "ai_utils.h"
#include "video_pipeline.h"
#include "hand_detection.h"
#include "hand_keypoint.h"
#include "sliding_puzzle.h"


std::atomic<bool> isp_stop(false);

void print_usage(const char *name)
{
	cout << "Usage: " << name << "<kmodel_det> <obj_thresh> <nms_thresh> <kmodel_kp> <bin_file> <level> <debug_mode>" << endl
		 << "Options:" << endl
		 << "  kmodel_det      手掌检测kmodel路径\n"
         << "  obj_thresh      手掌检测阈值\n"
         << "  nms_thresh      手掌检测非极大值抑制阈值\n"
		 << "  kmodel_kp       手势关键点检测kmodel路径\n"
         << "  bin_file        拼图文件 (文件名 或者 None(表示排序数字拼图模式))\n"
         << "  level           拼图游戏难度\n"
		 << "  debug_mode      是否需要调试,0、1、2分别表示不调试、简单调试、详细调试\n"
         << "  -s <csi_num>    Sensor CSI口编号，默认 2\n"
         << "  -L <2|4>        MIPI lane preference，默认 ANY\n"
		 << "\n"
		 << endl;
}

static int g_csi_num = 2;
static k_vicap_mipi_lane_pref g_lane_pref = VICAP_MIPI_LANE_PREF_ANY;

void video_proc(char *argv[])
{
    int debug_mode = atoi(argv[7]);
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
    int level = atoi(argv[6]);
    SlidingPuzzle puzzle(level);
    puzzle.initialize(draw_frame);
    HandDetection hd(argv[1], atof(argv[2]), atof(argv[3]),image_size,debug_mode);
    HandKeypoint hk(argv[4],image_size,debug_mode);
    std::vector<BoxInfo> results;
    std::vector<int> two_point;

    while(!isp_stop){
        // 创建一个ScopedTiming对象，用于计算总时间
        ScopedTiming st("total time", 1);
        // 从PipeLine中获取一帧数据，并创建tensor
        pl.GetFrame(dump_res);
        input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
        //前处理，推理，后处理
        results.clear();
        two_point.clear();

        hd.pre_process(input_tensor);
        hd.inference();
        hd.post_process(results);
        if(results.size()==1){
            BoxInfo r=results[0];
            int w = r.x2 - r.x1 + 1;
            int h = r.y2 - r.y1 + 1;
            int length = std::max(w,h)/2;
            int cx = (r.x1+r.x2)/2;
            int cy = (r.y1+r.y2)/2;
            int ratio_num = 1.26*length;
            int x1_1 = std::max(0,cx-ratio_num);
            int y1_1 = std::max(0,cy-ratio_num);
            int x2_1 = std::min(image_size.width-1, cx+ratio_num);
            int y2_1 = std::min(image_size.height-1, cy+ratio_num);
            int w_1 = x2_1 - x1_1 + 1;
            int h_1 = y2_1 - y1_1 + 1;
            Bbox bbox = {x:x1_1,y:y1_1,w:w_1,h:h_1};
            Bbox draw_box={r.x1,r.y1,(r.x2-r.x1),(r.y2-r.y1)};
            hk.pre_process(input_tensor,bbox);
            hk.inference();
            hk.post_process(bbox);
            hk.get_two_point(draw_frame,two_point);
            puzzle.process_hand(two_point, draw_frame);
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
    if (argc != 8)
    {
        print_usage(argv[0]);
        return -1;
    }
    std::thread thread_isp(video_proc, argv);
    while (getchar() != 'q')
    {
        usleep(10000);
    }
    isp_stop = true;
    thread_isp.join();
    return 0;
}