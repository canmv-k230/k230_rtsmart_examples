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
#include <thread>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <chrono>
#include <sys/stat.h>
#include "video_pipeline.h"
#include "self_learning.h"

using std::cerr;
using std::cout;
using std::endl;

using namespace std;
using namespace cv;

std::atomic<bool> isp_stop(false);
std::string regist_name="";
int cur_state=0;

void print_usage(const char *name)
{
    cout << "Usage: " << name << "<kmodel> <topk> <debug_mode> " << endl
         << "Options:" << endl
         << " 1> kmodel         kmodel文件路径 \n"
         << " 2> thres          判别阈值 \n"
         << " 3> topk           识别范围 \n"
         << " 4> debug_mode     是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "\n"
         << endl;
}

void print_help() {
    std::cout << "======== 操作说明 ========\n";
    std::cout << "请输入下列命令并按回车以执行对应操作：\n\n";

    std::cout << "  h/help    : 显示帮助说明（即本页内容）\n";
    std::cout << "  i         : 进入注册模式\n";
    std::cout << "              - 系统将自动截图用于识别注册\n";
    std::cout << "              - 注册后继续输入该物体的名称并回车完成注册\n";
    std::cout << "  q         : 退出程序并清理资源\n\n";
    std::cout << "==========================\n" << std::endl;
}

void video_proc(char *argv[])
{
    int debug_mode = atoi(argv[4]);
    FrameCHWSize image_size={AI_FRAME_CHANNEL,AI_FRAME_HEIGHT, AI_FRAME_WIDTH};
    // 创建一个空的Mat对象，用于存储绘制的帧
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    // 创建一个空的runtime_tensor对象，用于存储输入数据
    runtime_tensor input_tensor;
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };

    // 创建一个PipeLine对象，用于处理视频流
    PipeLine pl(debug_mode);
    // 初始化PipeLine对象
    pl.Create();
    // 创建一个DumpRes对象，用于存储帧数据
    DumpRes dump_res;
    
    SelfLearning sl(argv[1], atof(argv[2]), atoi(argv[3]),image_size,debug_mode);
    std::vector<ClassResult> results;

    int display_state=0;
    std::vector<cv::Mat> sensor_bgr(3);
    cv::Mat dump_img(AI_FRAME_HEIGHT,AI_FRAME_WIDTH , CV_8UC3);

    while(!isp_stop){
        // 创建一个ScopedTiming对象，用于计算总时间
        ScopedTiming st("total time", debug_mode);
        // 从PipeLine中获取一帧数据，并创建tensor
        pl.GetFrame(dump_res);
        input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
        //前处理，推理，后处理
        results.clear();
        if(cur_state==-1){

        }
        else if(cur_state==0){
            //不执行任何操作
            sl.draw_result(draw_frame,results);
        }
        else if (cur_state==1){
            // dump一帧图片，作为待注册图片供用户查看
            draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
            dump_img.setTo(cv::Scalar(0, 0, 0));
            sensor_bgr.clear();
            void* data=reinterpret_cast<void*>(dump_res.virt_addr);
            cv::Mat ori_img_R(AI_FRAME_HEIGHT, AI_FRAME_WIDTH, CV_8UC1, data);
            cv::Mat ori_img_G(AI_FRAME_HEIGHT, AI_FRAME_WIDTH, CV_8UC1, data + AI_FRAME_HEIGHT * AI_FRAME_WIDTH);
            cv::Mat ori_img_B(AI_FRAME_HEIGHT, AI_FRAME_WIDTH, CV_8UC1, data + 2 * AI_FRAME_HEIGHT * AI_FRAME_WIDTH);
            if (ori_img_B.empty() || ori_img_G.empty() || ori_img_R.empty()) {
                std::cout << "One or more of the channel images is empty." << std::endl;
                continue;
            }
            sensor_bgr.push_back(ori_img_B);
            sensor_bgr.push_back(ori_img_G);
            sensor_bgr.push_back(ori_img_R);
            cv::merge(sensor_bgr, dump_img);
            cur_state=0;
            display_state=1;
        }
        else if (cur_state==2){
            FrameCHWSize reg_size={dump_img.channels(),dump_img.rows,dump_img.cols};
            // 创建一个空的向量，用于存储chw图像数据,将读入的hwc数据转换成chw数据
            std::vector<uint8_t> chw_vec;
            std::vector<cv::Mat> bgrChannels(3);
            cv::split(dump_img, bgrChannels);
            for (auto i = 2; i > -1; i--)
            {
                std::vector<uint8_t> data = std::vector<uint8_t>(bgrChannels[i].reshape(1, 1));
                chw_vec.insert(chw_vec.end(), data.begin(), data.end());
            }
            // 创建tensor
            dims_t in_shape { 1, 3, dump_img.rows, dump_img.cols };
            runtime_tensor reg_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("cannot create input tensor");
            auto ref_buf = reg_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
            memcpy(reinterpret_cast<char *>(ref_buf.data()), chw_vec.data(), chw_vec.size());
            hrt::sync(reg_tensor, sync_op_t::sync_write_back, true).expect("write back input failed");
            sl.pre_process(reg_tensor);
            sl.inference();
            sl.register_object(regist_name);
            draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
            cur_state=3;
            display_state=0;
        }
        else if (cur_state==3){
            sl.pre_process(input_tensor);
            sl.inference();
            sl.post_process(results);
            draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
            sl.draw_result(draw_frame,results);
            display_state=0;
        }

        if(display_state==1){
            cv::cvtColor(dump_img, dump_img, cv::COLOR_BGR2BGRA);
            cv::Mat resized_dump;
            cv::resize(dump_img, resized_dump, cv::Size(OSD_WIDTH, OSD_HEIGHT));
            cv::Rect roi(0, 0, resized_dump.cols, resized_dump.rows);
            resized_dump.copyTo(draw_frame(roi));
            display_state=0;
            cur_state==-1;

            int min_len=std::min(draw_frame.cols,draw_frame.rows);
            int draw_w=min_len*0.4;
            int draw_h=min_len*0.4;
            int draw_x=draw_frame.cols/2-draw_w/2;
            int draw_y=draw_frame.rows/2-draw_h/2;
            cv::rectangle(draw_frame,cv::Rect(draw_x,draw_y,draw_w,draw_h),cv::Scalar(0,255,0),2);
        }

        // 将绘制的帧插入到PipeLine中
        pl.InsertFrame(draw_frame.data);
        // 释放帧数据
        pl.ReleaseFrame(dump_res);
    }
    pl.Destroy();
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Press 'q+Enter'  to exit." << std::endl;
    if (argc != 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    std::thread thread_isp(video_proc, argv);
    // 命令行输入处理主循环
    std::string last_input = "";
    sleep(2);
    // 输入提示信息
    std::cout << "输入 'h' 或 'help' 并回车 查看命令说明" << std::endl;
    while(true){
        std::string input;
        std::getline(std::cin, input);  // 获取用户输入

        if (input == "h" || input == "help") {
            print_help();  // 打印帮助信息
        }
        else if (input == "q") {
            usleep(100000);     
            isp_stop.store(true);      
            break;
        }
        else if (input == "i") {
            // 进入注册模式
            cur_state = 1; 
            last_input="i";
        }
        else {
            // 其他输入处理
            if (last_input == "i") {
                // 上一次为注册命令，现在应是注册名称
                regist_name = input;
                cur_state = 2;   // 注册确认阶段
            }
            else {
                // 未进入注册模式却输入名称
                std::cout << "请先输入 'i' 并回车以进入注册模式！" << std::endl;
            }
        }
    }
    thread_isp.join();
    return 0;
}