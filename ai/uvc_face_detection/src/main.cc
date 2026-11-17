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
#include <errno.h>
#include <sys/select.h>
#include <unistd.h>
#include "ai_utils.h"
#include "face_detection.h"
#include "setting.h"
#include "uvc_pipeline.h"

std::atomic<bool> isp_stop(false);
// video_proc 的结果，作为进程退出码
std::atomic<int> pipeline_result(0);
// video_proc 已自行结束（Create 失败、致命错误或放弃），主线程据此退出
std::atomic<bool> video_done(false);

// 连续取帧/送显失败时的退避时间，以及放弃前允许的最长持续失败时间
#define FRAME_RETRY_SLEEP_US        (10 * 1000)
#define FRAME_FAIL_GIVE_UP_MS       10000

void print_usage(const char *name)
{
    cout << "Usage: " << name << "<kmodel_det> <obj_thres> <nms_thres> <input_mode> <debug_mode>" << endl
         << "Options:" << endl
         << "  kmodel_det      人脸检测kmodel路径\n"
         << "  obj_thres       人脸检测kmodel阈值\n"
         << "  nms_thres       人脸检测kmodel nms阈值\n"
         << "  debug_mode      是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "\n"
         << endl;
}

void video_proc(char *argv[])
{
    int debug_mode = atoi(argv[4]);
    FrameCHWSize image_size={AI_FRAME_CHANNEL,AI_FRAME_HEIGHT, AI_FRAME_WIDTH};
    // 创建一个空的runtime_tensor对象，用于存储输入数据
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };
    runtime_tensor input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("cannot create input tensor");
    auto input_buf = input_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();

    // 创建一个PipeLine对象，用于处理视频流
    UVC_PipeLine pl(debug_mode);
    // 初始化PipeLine；失败时也要 Destroy() 清理已完成的部分
    if (pl.Create() != 0) {
        printf("PipeLine Create failed, exit\n");
        if (pl.Destroy() != 0) {
            printf("PipeLine Destroy failed\n");
        }
        pipeline_result = -1;
        return;
    }
    // 创建一个DumpRes对象，用于存储帧数据
    DumpRes dump_res = {};
    // 创建FaceDetection实例
    FaceDetection fd(argv[1], atof(argv[2]),atof(argv[3]), image_size, debug_mode);
    vector<FaceDetectionInfo> results;

    std::vector<uint8_t> chw_vec;
    std::vector<cv::Mat> rgbChannels(3);
    cv::Mat ori_img;
    int ret=0;
    int consecutive_failures = 0;
    std::chrono::steady_clock::time_point first_failure;
    // 处理 GetFrame/ReleaseFrame 的失败（失败时本次迭代的资源都已归还）。返回 false 表示应退出循环
    auto keep_going_after = [&](int status) -> bool {
        if (status == UVC_FRAME_FATAL) {
            printf("pipeline error, stop\n");
            pipeline_result = -1;
            return false;
        }
        // 丢帧（超时或缓冲不足）：短暂退避后重试，长时间没有成功帧则放弃
        auto now = std::chrono::steady_clock::now();
        if (consecutive_failures++ == 0) {
            // 成功后计数归零，下一次失败会重新记录起点。
            first_failure = now;
        }
        auto failing_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - first_failure).count();
        if (failing_ms >= FRAME_FAIL_GIVE_UP_MS) {
            printf("no good frame for %lld ms (%d failures), give up\n",
                   (long long)failing_ms, consecutive_failures);
            pipeline_result = -1;
            return false;
        }
        usleep(FRAME_RETRY_SLEEP_US);
        return true;
    };
    while(!isp_stop){
        // 计算总时间（只打印成功的迭代）
        auto frame_start = std::chrono::steady_clock::now();
        // 从PipeLine中获取一帧数据，并创建tensor
        ret=pl.GetFrame(dump_res);
        if(ret){
            printf("GetFrame fail\n");
            if (!keep_going_after(ret)) {
                break;
            }
            continue;
        }
        {
            ScopedTiming st("create tensor", debug_mode);
            void* vaddr=reinterpret_cast<void*>(dump_res.virt_addr);
            ori_img = cv::Mat(image_size.height, image_size.width, CV_8UC3, vaddr);
            chw_vec.clear();
            rgbChannels.clear();
            cv::split(ori_img, rgbChannels);
            for (auto i = 0; i < 3; i++)
            {
                std::vector<uint8_t> data = std::vector<uint8_t>(rgbChannels[i].reshape(1, 1));
                chw_vec.insert(chw_vec.end(), data.begin(), data.end());
            }
            memcpy(reinterpret_cast<char *>(input_buf.data()), chw_vec.data(), chw_vec.size());
            hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("write back input failed");
        }
        // usleep(30000);
        results.clear();
        //前处理，推理，后处理
        fd.pre_process(input_tensor);
        fd.inference();
        fd.post_process(image_size,results);
        // 绘制检测结果
        fd.draw_result(ori_img,results);
        // 送显并释放帧数据（无论送显是否成功都会归还全部资源，映射随之解除）
        ret = pl.ReleaseFrame(dump_res);
        ori_img.release();
        if (ret) {
            if (!keep_going_after(ret)) {
                break;
            }
            continue;
        }
        if (consecutive_failures) {
            printf("recovered after %d failed frames\n", consecutive_failures);
            consecutive_failures = 0;
        }

        double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frame_start).count();
        std::cout << "total time took " << elapsed_ms << " ms" << std::endl;
    }
    if (pl.Destroy() != 0) {
        printf("PipeLine Destroy failed\n");
        pipeline_result = -1;
    }
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 5)
    {
        print_usage(argv[0]);
        return -1;
    }
    
    std::thread thread_isp([argv]() {
        video_proc(argv);
        video_done = true;
    });
    // 按 q 或 video_proc 自行结束时退出。
    // 轮询 stdin，这样 video_proc 结束时主线程也能退出。
    // stdin 关闭（EOF）或不可读时不退出：无交互启动的 demo 应一直运行到被结束，
    // 此时只停止监听 stdin（EOF 时 select 会一直可读，继续监听会空转）。
    bool watch_stdin = true;
    while (!video_done)
    {
        if (!watch_stdin) {
            usleep(100 * 1000);
            continue;
        }
        fd_set rfds;
        struct timeval tv;
        FD_ZERO(&rfds);
        FD_SET(STDIN_FILENO, &rfds);
        tv.tv_sec = 0;
        tv.tv_usec = 100 * 1000;
        int ret = select(STDIN_FILENO + 1, &rfds, nullptr, nullptr, &tv);
        if (ret < 0) {
            if (errno != EINTR)
                watch_stdin = false;
            continue;
        }
        if (ret > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
            // 直接读取 fd，避免 stdio 预读后 select 看不到缓冲区中的 q。
            char ch;
            ssize_t count = read(STDIN_FILENO, &ch, 1);
            if (count < 0) {
                if (errno != EINTR && errno != EAGAIN)
                    watch_stdin = false;
                continue;
            }
            if (count == 0) {
                printf("stdin closed, running until the pipeline stops\n");
                watch_stdin = false;
                continue;
            }
            if (ch == 'q' || ch == 'Q')
                break;
        }
    }
    isp_stop = true;
    thread_isp.join();
    return pipeline_result;
}
