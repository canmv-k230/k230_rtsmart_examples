#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <thread>

// 项目相关头文件
#include "video_pipeline.h"
#include "ai_utils.h"
#include "yolov8_det.h"
#include "BYTETracker.h"

// 使用标准输出流
using std::cerr;
using std::cout;
using std::endl;

// 原子标志，用于安全地停止 ISP / 视频处理线程
std::atomic<bool> isp_stop(false);

/**
 * @brief 打印程序使用说明
 * @param name 程序名称
 */
void print_usage(const char *name)
{
    cout << "Usage: " << name << " <yolov8_kmodel> <score_thres> <nms_thres> <track_thresh> <high_thresh> <match_thresh> <fps> <buffer> <debug_mode>" << endl
         << "Options:" << endl
         << "  yolov8_kmodel            YOLOv8 检测模型 kmodel 路径\n"
         << "  score_thres              目标检测置信度阈值\n"
         << "  nms_thres                目标检测 NMS 阈值\n"
         << "  track_thresh             跟踪最低置信度阈值\n"
         << "  high_thresh              确认轨迹的高置信度阈值\n"
         << "  match_thresh             匹配时的最大距离 / IoU 阈值\n"
         << "  fps                      输入视频帧率\n" 
         << "  buffer                   丢失缓冲区大小（最大允许丢失帧数）\n"
         << "  debug_mode               调试等级：0（关闭），1（简单），2（详细）\n"
         << "\n"
         << endl;
}

/**
 * @brief 视频处理线程函数
 *        负责图像采集、检测、跟踪、绘制以及显示
 * @param argv 命令行参数
 */
void video_proc(char *argv[])
{
    // 解析调试模式
    int debug_mode = atoi(argv[9]);

    // 输入帧尺寸（CHW 格式）
    FrameCHWSize image_size={AI_FRAME_CHANNEL,AI_FRAME_HEIGHT, AI_FRAME_WIDTH};

    // 用于 OSD 绘制的 RGBA 缓冲区
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // AI 输入运行时 tensor
    runtime_tensor input_tensor;

    // 输入 tensor 形状：NCHW
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };

    // 创建视频管线实例
    PipeLine pl(debug_mode);

    // 初始化管线资源
    pl.Create();

    // 帧缓冲结构体
    DumpRes dump_res;

    // 检测结果容器
    std::vector<YOLOBbox> person_results;

    // 初始化 YOLOv8 检测器
    YOLOv8Det yolo_det_app(argv[1], atof(argv[2]), atof(argv[3]), image_size, debug_mode);

    // 跟踪参数
    float track_thresh = atof(argv[4]);
    float high_thresh  = atof(argv[5]);
    float match_thresh = atof(argv[6]);
    int fps            = atoi(argv[7]);
    int buffer         = atoi(argv[8]);

    // 打印跟踪参数（调试用）
    std::cout<<"==================bytetrack config====================="<<std::endl;
    std::cout<<"track_thresh:"<<track_thresh<<std::endl;
    std::cout<<"high_thresh:"<<high_thresh<<std::endl;
    std::cout<<"match_thresh:"<<match_thresh<<std::endl;
    std::cout<<"fps:"<<fps<<std::endl;
    std::cout<<"buffer:"<<buffer<<std::endl;
    std::cout<<"==================bytetrack config end====================="<<std::endl;

    // 初始化 BYTETracker
    BYTETracker tracker(track_thresh,high_thresh,match_thresh,fps,buffer);

    // 主处理循环
    while(!isp_stop){
        // 统计单帧总处理时间
        ScopedTiming st("total time", debug_mode);

        // 从管线获取一帧图像
        pl.GetFrame(dump_res);

        // 使用物理地址和虚拟地址创建运行时 tensor
        input_tensor = host_runtime_tensor::create(
            typecode_t::dt_uint8,
            in_shape,
            { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },
            false,
            hrt::pool_shared,
            dump_res.phy_addr
        ).expect("cannot create input tensor");

        // 同步输入 tensor 数据
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true)
            .expect("sync write_back failed");

        // 清空上一帧检测结果
        person_results.clear();

        /*******************************************************
        * 行人检测与跟踪
        ********************************************************/

        // 前处理
        yolo_det_app.pre_process(input_tensor);

        // 执行推理
        yolo_det_app.inference();

        // 后处理，获取检测框
        yolo_det_app.post_process(person_results);

        // 将检测结果转换为跟踪器输入格式
        std::vector<Object> objects;
        for (auto res : person_results)
        {            
            Object obj{
                {res.box.x,res.box.y,res.box.width,res.box.height},
                res.index,
                res.confidence
            };
            objects.push_back(obj);
        }

        // 更新跟踪器并获取当前有效轨迹
        std::vector<STrack> output_stracks = tracker.update(objects);

        // 清空 OSD 图层
        draw_frame.setTo(cv::Scalar(0,0,0,0));

        // 绘制跟踪结果
        for (int i = 0; i < output_stracks.size(); i++)
        {
            std::vector<float> tlwh = output_stracks[i].tlwh;

            // 坐标从 AI 空间映射到 OSD 显示空间
            int x1 =  tlwh[0] / image_size.width  * OSD_WIDTH;
            int y1 =  tlwh[1] / image_size.height * OSD_HEIGHT;
            int w  =  tlwh[2] / image_size.width  * OSD_WIDTH;
            int h  =  tlwh[3] / image_size.height * OSD_HEIGHT;

            // 绘制轨迹 ID
            cv::putText(draw_frame, format("%d", output_stracks[i].track_id), Point(x1, y1 - 40), 0, 1, Scalar(255,255, 0, 255), 2, LINE_AA);

            // 绘制跟踪时间和置信度
            cv::putText(draw_frame, format("time:%.2fs score:%.2f", output_stracks[i].track_time, output_stracks[i].score), Point(x1, y1 - 10), 0, 1, Scalar(255,255, 0, 255), 2, LINE_AA);

            // 绘制目标框
            cv::rectangle(draw_frame, Rect(x1,y1,w,h), Scalar(255,255, 0, 255), 2);
        }
        
        // 将 OSD 图像送入显示管线
        pl.InsertFrame(draw_frame.data);

        // 释放当前帧资源
        pl.ReleaseFrame(dump_res);
    }

    // 销毁管线资源
    pl.Destroy();
}

/**
 * @brief 程序入口函数
 */
int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at "
              << __DATE__ << " " << __TIME__ << std::endl;

    // 检查参数数量
    if (argc !=10 )
    {
        print_usage(argv[0]);
        return -1;
    }

    // 启动视频处理线程
    std::thread thread_isp(video_proc, argv);

    // 等待用户输入 'q' 退出
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    // 通知线程停止
    isp_stop.store(true);

    // 等待线程结束
    thread_isp.join();

    return 0;
}
