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
#include <chrono>
#include <fstream>
#include <thread>
#include <algorithm>
#include "ai_utils.h"
#include "video_pipeline.h"
#include "hand_detection.h"
#include "hand_keypoint.h"
#include "ocr_box.h"
#include "ocr_reco.h"

#define CNUM 20

std::atomic<bool> isp_stop(false);

void print_usage(const char *name)
{
	cout << "Usage: " << name << "<kmodel_det> <input_mode> <obj_thresh> <nms_thresh> <kmodel_kp> <kmodel_ocrdet> <threshold> <box_thresh> <kmodel_reco> <debug_mode>" << endl
		 << "Options:" << endl
		 << "  kmodel_det      手掌检测kmodel路径\n"
		 << "  input_mode      本地图片(图片路径)/ 摄像头(None) \n"
         << "  obj_thresh      手掌检测kmodel obj阈值\n"
         << "  nms_thresh      手掌检测kmodel nms阈值\n"
		 << "  kmodel_kp       手势关键点检测kmodel路径\n"
         << "  kmodel_ocrdet   ocr检测kmodel路径\n"
         << "  threshold       ocr检测 threshold\n"
         << "  box_thresh      ocr检测 box_thresh\n"
         << "  kmodel_reco     ocr识别kmodel路径 \n"
		 << "  debug_mode      是否需要调试, 0、1、2分别表示不调试、简单调试、详细调试\n"
         << "  -s <csi_num>    Sensor CSI口编号，默认 2\n"
         << "  -L <2|4>        MIPI lane preference，默认 ANY\n"
		 << "\n"
		 << endl;
}

static int g_csi_num = 2;
static k_vicap_mipi_lane_pref g_lane_pref = VICAP_MIPI_LANE_PREF_ANY;

void video_proc(char *argv[])
{
    int debug_mode = atoi(argv[10]);
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

    HandDetection hd(argv[1], atof(argv[3]), atof(argv[4]), image_size,debug_mode);
    HandKeypoint hk(argv[5], image_size,debug_mode);
    OCRBox ocr_det(argv[6], atof(argv[7]), atof(argv[8]), image_size, debug_mode);
    OCRReco ocr_rec(argv[9],debug_mode);
    std::vector<BoxInfo> results;
    vector<ocr_det_res> det_results;
    vector<std::string> rec_results;
    while(!isp_stop){
        // 创建一个ScopedTiming对象，用于计算总时间
        ScopedTiming st("total time", 1);
        // 从PipeLine中获取一帧数据，并创建tensor
        pl.GetFrame(dump_res);
        input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
        //前处理，推理，后处理
        results.clear();
        det_results.clear();
        rec_results.clear();
        hd.pre_process(input_tensor);
        hd.inference();
        hd.post_process(results);
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
        if (results.size()>=2)
        {
            cv::Point2f left_top, right_bottom;
            for(int i=0;i< results.size();i++)
            {
                BoxInfo r = results[i];
                int w = r.x2 - r.x1 + 1;
                int h = r.y2 - r.y1 + 1;
                int rect_x = int((float)r.x1/ image_size.width * draw_frame.cols);
                int rect_y = int((float)r.y1/ image_size.height * draw_frame.rows);
                int rect_w = int((float)w / image_size.width * draw_frame.cols);
                int rect_h = int((float)h / image_size.height * draw_frame.rows);
                cv::rectangle(draw_frame, cv::Rect(rect_x, rect_y, rect_w, rect_h), cv::Scalar( 0,255, 0, 255), 2, 2, 0);

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
                hk.pre_process(input_tensor,bbox);
                hk.inference();
                vector<float> output;
                hk.get_point(output);
                float pred_x = output[0];
                float pred_y = output[1];

                int draw_x,draw_y;
                if(i==0){
                    left_top.x = pred_x * w_1 + x1_1;
                    left_top.y = pred_y * h_1 + y1_1;
                    draw_x = left_top.x / image_size.width * draw_frame.cols;
                    draw_y = left_top.y / image_size.height * draw_frame.rows;
                }else if(i==1){
                    right_bottom.x = pred_x * w_1 + x1_1;
                    right_bottom.y = pred_y * h_1 + y1_1;
                    draw_x = right_bottom.x / image_size.width * draw_frame.cols;
                    draw_y = right_bottom.y / image_size.height * draw_frame.rows;
                }
                cv::circle(draw_frame, cv::Point(draw_x, draw_y), 6, cv::Scalar(0, 255,0,255), 3);
                cv::circle(draw_frame, cv::Point(draw_x, draw_y), 5, cv::Scalar(0, 255,0,255), 3);
            }
            std::string fr_result="";
            int x_min = std::min(left_top.x, right_bottom.x);
            int x_max = std::max(left_top.x, right_bottom.x);
            int y_min = std::min(left_top.y, right_bottom.y);
            int y_max = std::max(left_top.y, right_bottom.y);
            Bbox box_info = {x_min,y_min, (x_max-x_min+1),(y_max-y_min+1)};

            ocr_det.pre_process(input_tensor,box_info);
            ocr_det.inference();
            ocr_det.post_process(det_results);
            
            cv::Mat ori_img;
            void* vaddr=reinterpret_cast<void*>(dump_res.virt_addr);
            cv::Mat ori_img_R = cv::Mat(image_size.height, image_size.width, CV_8UC1, vaddr);
            cv::Mat ori_img_G = cv::Mat(image_size.height, image_size.width, CV_8UC1, vaddr + image_size.width * image_size.height);
            cv::Mat ori_img_B = cv::Mat(image_size.height, image_size.width, CV_8UC1, vaddr + 2 * image_size.width * image_size.height);
            std::vector<cv::Mat> sensor_rgb;
            sensor_rgb.push_back(ori_img_B);
            sensor_rgb.push_back(ori_img_G);
            sensor_rgb.push_back(ori_img_R);
            cv::merge(sensor_rgb, ori_img);
            for(int i = 0; i < det_results.size(); i++)
            {
                vector<cv::Point> ori_vec;
                vector<cv::Point2f> sort_vtd(4);
                for(int j = 0; j < 4; j++)
                {
                    ori_vec.push_back(det_results[i].vertices[j]);
                }
                cv::RotatedRect rect = cv::minAreaRect(ori_vec);
                cv::Point2f ver[4];
                rect.points(ver);
                cv::Mat crop;
                ocr_det.warppersp(ori_img, crop, det_results[i], sort_vtd);
                ocr_rec.pre_process(crop);
                ocr_rec.inference();
                std::string result_reco;
                ocr_rec.post_process(result_reco);
                rec_results.push_back(result_reco);
            }
            ocr_det.draw_result(draw_frame, det_results, rec_results);
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
    if (argc != 11)
    {
        print_usage(argv[0]);
        return -1;
    }

    if (strcmp(argv[2], "None") == 0)
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
        int debug_mode = atoi(argv[10]);
        // 读取图片
        cv::Mat ori_img = cv::imread(argv[2]);
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
        HandDetection hd(argv[1], atof(argv[3]), atof(argv[4]), image_size,debug_mode);
        HandKeypoint hk(argv[5], image_size,debug_mode);
        OCRBox ocr_det(argv[6], atof(argv[7]), atof(argv[8]), image_size, debug_mode);
        OCRReco ocr_rec(argv[9],debug_mode);
        std::vector<BoxInfo> results;
        vector<ocr_det_res> det_results;
        vector<std::string> rec_results;
        results.clear();
        hd.pre_process(input_tensor);
        hd.inference();
        hd.post_process(results);
        if (results.size()>=2)
        {
            cv::Point2f left_top, right_bottom;
            for(int i=0;i< results.size();i++)
            {
                BoxInfo r = results[i];
                int w = r.x2 - r.x1 + 1;
                int h = r.y2 - r.y1 + 1;
                int rect_x = int((float)r.x1/ image_size.width * ori_img.cols);
                int rect_y = int((float)r.y1/ image_size.height * ori_img.rows);
                int rect_w = int((float)w / image_size.width * ori_img.cols);
                int rect_h = int((float)h / image_size.height * ori_img.rows);
                cv::rectangle(ori_img, cv::Rect(rect_x, rect_y, rect_w, rect_h), cv::Scalar( 0,255, 0), 2, 2, 0);
                int length = std::max(w,h)/2;
                int cx = (r.x1+r.x2)/2;
                int cy = (r.y1+r.y2)/2;
                int ratio_num = 1.26*length;
                int x1_1 = std::max(0,cx-ratio_num);
                int y1_1 = std::max(0,cy-ratio_num);
                int x2_1 = std::min(ori_img.cols-1, cx+ratio_num);
                int y2_1 = std::min(ori_img.rows-1, cy+ratio_num);
                int w_1 = x2_1 - x1_1 + 1;
                int h_1 = y2_1 - y1_1 + 1;
                Bbox bbox = {x:x1_1,y:y1_1,w:w_1,h:h_1};
                hk.pre_process(input_tensor,bbox);
                hk.inference();
                vector<float> output;
                hk.get_point(output);
                float pred_x = output[0];
                float pred_y = output[1];

                int draw_x,draw_y;
                if(i==0){
                    left_top.x = pred_x * w_1 + x1_1;
                    left_top.y = pred_y * h_1 + y1_1;
                    draw_x = int(1.0 * left_top.x / image_size.width * ori_img.cols);
                    draw_y = int(1.0 * left_top.y / image_size.height * ori_img.rows);
                }else if(i==1){
                    right_bottom.x = pred_x * w_1 + x1_1;
                    right_bottom.y = pred_y * h_1 + y1_1;
                    draw_x = int(1.0 * right_bottom.x / image_size.width * ori_img.cols);
                    draw_y = int(1.0 * right_bottom.y / image_size.height * ori_img.rows);
                }
                cv::circle(ori_img, cv::Point(draw_x, draw_y), 6, cv::Scalar(0, 255,0), 3);
                cv::circle(ori_img, cv::Point(draw_x, draw_y), 5, cv::Scalar(0, 255,0), 3);
            }
            std::string fr_result="";
            int x_min = std::min(left_top.x, right_bottom.x);
            int x_max = std::max(left_top.x, right_bottom.x);
            int y_min = std::min(left_top.y, right_bottom.y);
            int y_max = std::max(left_top.y, right_bottom.y);
            Bbox box_info = {x_min,y_min, (x_max-x_min+1),(y_max-y_min+1)};

            ocr_det.pre_process(input_tensor,box_info);
            ocr_det.inference();
            ocr_det.post_process(det_results);
            for(int i = 0; i < det_results.size(); i++)
            {
                vector<cv::Point> ori_vec;
                vector<cv::Point2f> sort_vtd(4);
                for(int j = 0; j < 4; j++)
                {
                    ori_vec.push_back(det_results[i].vertices[j]);
                }
                cv::RotatedRect rect = cv::minAreaRect(ori_vec);
                cv::Point2f ver[4];
                rect.points(ver);
                cv::Mat crop;
                ocr_det.warppersp(ori_img, crop, det_results[i], sort_vtd);
                ocr_rec.pre_process(crop);
                ocr_rec.inference();
                std::string result_reco;
                ocr_rec.post_process(result_reco);
                rec_results.push_back(result_reco);
            }
            ocr_det.draw_result(ori_img, det_results, rec_results);
            
        }
        cv::imwrite("hand_kp_ocr_result.jpg", ori_img);
    }
    return 0;
}

