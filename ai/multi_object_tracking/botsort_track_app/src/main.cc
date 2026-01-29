#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <thread>

// 视频采集、显示及处理流水线抽象
#include "video_pipeline.h"

// AI 工具函数与辅助接口
#include "ai_utils.h"

// YOLOv8 目标检测封装
#include "yolov8_det.h"

// ReID 外观特征提取模块
#include "feature.h"

// BoTSORT 多目标跟踪器
#include "BoTSORT.h"

using std::cerr;
using std::cout;
using std::endl;

// 全局原子标志，用于安全地停止 ISP / 视频处理线程
std::atomic<bool> isp_stop(false);

/**
 * @brief 打印命令行使用说明
 * @param name 程序名称
 */
void print_usage(const char *name)
{
    cout << "Usage: " << name << "<yolov8_kmodel> <score_thres> <nms_thres> <feature_kmodel> <track_high_thresh> <track_low_thresh> <new_track_thresh> <frame_buffer> <match_thresh> <proximity_thresh> <appearance_thresh> <appearance_thresh> <lambda> <debug_mode>" << endl
         << "Options:" << endl
         << "  yolov8_kmodel            YOLOv8 检测模型的 kmodel 路径\n"
         << "  score_thres              目标检测置信度阈值\n"
         << "  nms_thres                目标检测 NMS 阈值\n"
         << "  feature_kmodel           ReID（外观特征）模型的 kmodel 路径\n"
         << "  track_high_thresh        高置信度阈值，高于该值的检测被视为可靠目标\n"
         << "                           用于抑制低分检测引起的 ID 切换\n"
         << "  track_low_thresh         低置信度阈值，低于该值的检测将被丢弃\n"
         << "                           用于过滤 YOLO 的误检\n"
         << "  new_track_thresh         新建轨迹阈值，只有高于该值的检测才能生成新轨迹\n"
         << "                           较大的值可以减少错误轨迹初始化\n"
         << "  frame_buffer             轨迹缓冲大小（未匹配情况下轨迹的最大保留帧数）\n"
         << "  match_thresh             最大匹配代价阈值（IOU / 距离），超过该值视为不匹配\n"
         << "  proximity_thresh         邻近匹配阈值（中心距离或 IOU）\n"
         << "                           数值越小，匹配条件越严格\n"
         << "  appearance_thresh        ReID 外观特征距离阈值\n"
         << "                           数值越小，外观匹配越严格\n"
         << "  lambda                   IOU / 距离 与 ReID 特征之间的权重因子\n"
         << "                           越接近 1：越依赖 IOU；越接近 0：越依赖外观特征\n"
         << "  debug_mode               调试模式：0 = 关闭，1 = 简单调试，2 = 详细调试\n"
         << "\n"
         << endl;
}

/**
 * @brief 视频处理线程函数
 *        负责视频采集、目标检测、特征提取、
 *        BoTSORT 跟踪以及 OSD 绘制
 * @param argv 命令行参数
 */
void video_proc(char *argv[])
{
    // 调试级别
    int debug_mode = atoi(argv[13]);

    // 输入图像尺寸（CHW 格式）
    FrameCHWSize image_size = {AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH};

    // 用于绘制跟踪结果的 OSD 帧（RGBA）
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // 作为 AI 模型输入的运行时张量
    runtime_tensor input_tensor;

    // 输入张量形状：NCHW
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };

    // 视频流水线实例（ISP + 显示）
    PipeLine pl(debug_mode);

    // 创建并初始化流水线资源
    pl.Create();

    // 保存采集帧缓冲区信息的结构体
    DumpRes dump_res;

    // 人体检测结果
    std::vector<YOLOBbox> person_results;

    // YOLOv8 目标检测应用
    YOLOv8Det yolo_det_app(
        argv[1],                  // YOLOv8 kmodel 路径
        atof(argv[2]),             // 检测置信度阈值
        atof(argv[3]),             // NMS 阈值
        image_size,
        debug_mode
    );

    // ReID 特征提取应用
    Feature feature_app(argv[4], image_size, debug_mode);

    // 是否启用 ReID 特征匹配
    bool reid_enabled = true;

    // BoTSORT 配置参数
    float track_high_thresh = atof(argv[5]);   // 高置信度阈值
    float track_low_thresh  = atof(argv[6]);   // 低置信度阈值
    float new_track_thresh  = atof(argv[7]);   // 新轨迹创建阈值
    long  track_buffer      = atol(argv[8]);   // 轨迹缓冲长度
    float match_thresh      = atof(argv[9]);   // 匹配代价阈值
    float proximity_thresh  = atof(argv[10]);  // IOU / 邻近阈值
    float appearance_thresh = atof(argv[11]);  // ReID 外观阈值
    long  frame_rate        = 30;               // 视频帧率
    float lambda            = atof(argv[12]);  // IOU / 外观融合权重

    // 打印 BoTSORT 配置信息
    std::cout << "=============BotTrack Config===============" << std::endl;
    std::cout << "track_high_thresh: " << track_high_thresh << std::endl;
    std::cout << "track_low_thresh: "  << track_low_thresh  << std::endl;
    std::cout << "new_track_thresh: "  << new_track_thresh  << std::endl;
    std::cout << "track_buffer: "      << track_buffer      << std::endl;
    std::cout << "match_thresh: "       << match_thresh       << std::endl;
    std::cout << "proximity_thresh: "   << proximity_thresh   << std::endl;
    std::cout << "appearance_thresh: "  << appearance_thresh  << std::endl;
    std::cout << "frame_rate: "         << frame_rate         << std::endl;
    std::cout << "lambda: "             << lambda             << std::endl;
    std::cout << "============BotTrack Config End===============" << std::endl;

    // 初始化 BoTSORT 跟踪器
    BoTSORT tracker(
        reid_enabled,
        track_high_thresh,
        track_low_thresh,
        new_track_thresh,
        track_buffer,
        match_thresh,
        proximity_thresh,
        appearance_thresh,
        frame_rate,
        lambda
    );

    // 主处理循环
    while (!isp_stop)
    {
        // 统计整体处理耗时（可选调试）
        ScopedTiming st("total time", debug_mode);

        // 从流水线中获取一帧
        pl.GetFrame(dump_res);

        // 将 ISP 缓冲区封装为运行时张量（零拷贝）
        input_tensor = host_runtime_tensor::create(
            typecode_t::dt_uint8,
            in_shape,
            { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },
            false,
            hrt::pool_shared,
            dump_res.phy_addr
        ).expect("cannot create input tensor");

        // 同步缓冲区，保证设备可访问
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true)
            .expect("sync write_back failed");

        // 清空上一帧的检测结果
        person_results.clear();

        /*******************************************************
         * 行人检测与跟踪
         *******************************************************/
        yolo_det_app.pre_process(input_tensor);
        yolo_det_app.inference();
        yolo_det_app.post_process(person_results);

        // 为 BoTSORT 准备检测结果
        std::vector<Detection> track_detections;

        for (YOLOBbox res : person_results)
        {
            Detection tmpRow;

            // TLWH 格式的边界框
            tmpRow.bbox_tlwh = cv::Rect_<float>(
                (float)res.box.x,
                (float)res.box.y,
                (float)res.box.width,
                (float)res.box.height
            );

            tmpRow.confidence = res.confidence;
            tmpRow.class_id   = 0;  // 行人类别

            // 为该检测目标提取 ReID 外观特征
            feature_app.pre_process(input_tensor, res.box);
            feature_app.inference();
            feature_app.get_feature(tmpRow.feature);

            track_detections.push_back(tmpRow);
        }

        // 执行 BoTSORT 跟踪
        std::vector<std::shared_ptr<Track>> tracks = tracker.track(track_detections);

        // 清空 OSD 画面
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));

        // 绘制跟踪结果
        for (const std::shared_ptr<Track> &track : tracks)
        {
            std::vector<float> bbox_tlwh = track->get_tlwh();

            // 坐标转换到 OSD 分辨率
            int x = int(bbox_tlwh[0] / image_size.width  * OSD_WIDTH);
            int y = int(bbox_tlwh[1] / image_size.height * OSD_HEIGHT);
            int w = int(bbox_tlwh[2] / image_size.width  * OSD_WIDTH);
            int h = int(bbox_tlwh[3] / image_size.height * OSD_HEIGHT);

            // 绘制轨迹 ID
            cv::putText(draw_frame, cv::format("%d", track->track_id), cv::Point(x, y - 40), 0, 1, cv::Scalar(255, 255, 0, 255), 2, cv::LINE_AA);

            // 绘制驻留时间与置信度
            cv::putText(draw_frame, cv::format("time:%.2fs score:%.2f",track->stay_time,track->get_score()), cv::Point(x, y - 10), 0, 1, cv::Scalar(255, 255, 0, 255), 2, cv::LINE_AA);

            // 绘制边界框
            cv::rectangle(draw_frame, cv::Rect(x, y, w, h), cv::Scalar(255, 255, 0, 255), 2);
        }

        //将 OSD 帧送入显示流水线
        pl.InsertFrame(draw_frame.data);

        //释放采集到的帧缓冲区
        pl.ReleaseFrame(dump_res);
    }

    // 销毁流水线资源
    pl.Destroy();
}

/**
 * @brief 程序入口
 */
int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0]
              << " built at " << __DATE__ << " " << __TIME__
              << std::endl;

    // 检查参数个数
    if (argc != 14)
    {
        print_usage(argv[0]);
        return -1;
    }

    // 启动视频处理线程
    std::thread thread_isp(video_proc, argv);

    // 等待按下 'q' 键退出
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    // 通知处理线程退出
    isp_stop.store(true);

    // 等待线程结束
    thread_isp.join();

    return 0;
}
