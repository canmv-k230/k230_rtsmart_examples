#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <thread>
#include "setting.h"
#include "video_stream_pipeline.h"
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
    cout << "Usage: " << name << "<yolov8_kmodel> <score_thres> <nms_thres> <feature_kmodel> <track_high_thresh> <track_low_thresh> <new_track_thresh> <frame_buffer> <match_thresh> <proximity_thresh> <appearance_thresh> <appearance_thresh> <lambda> <debug_mode> <video_path>" << endl
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
         << "  video_path               视频源路径，支持以下类型：\n"
         << "                           - \"realtime\"   实时摄像头采集\n"
         << "                           - \"*.mp4\"      MP4 视频文件\n"
         << "                           - \"UVC\"        UVC 摄像头\n"
         << "                           - \"rtsp://*\"   RTSP 网络视频流\n"
         << endl;
}

template <typename T>
int process_pipeline(char *argv[], T& pl)
{
    // 1. 参数解析 (统一从 argv 获取)
    int debug_mode = atoi(argv[13]);
    FrameCHWSize image_size = {AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH};
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };

    // 2. 初始化应用 (Detection, ReID, Tracker)
    YOLOv8Det yolo_det_app(argv[1], atof(argv[2]), atof(argv[3]), image_size, debug_mode);
    Feature feature_app(argv[4], image_size, debug_mode);

    BoTSORT tracker(
        true, // reid_enabled
        atof(argv[5]), atof(argv[6]), atof(argv[7]), atol(argv[8]),
        atof(argv[9]), atof(argv[10]), atof(argv[11]), 30, atof(argv[12])
    );

    // 3. 资源创建
    if (pl.Create()){
        print_usage(argv[0]);
        return -1;
    }

    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    runtime_tensor input_tensor;
    DumpRes dump_res;
    std::vector<YOLOBbox> person_results;

    // 4. 主处理循环
    while (!isp_stop)
    {
        ScopedTiming st("total time", debug_mode);

        if(pl.IsFinished()){
            isp_stop.store(true);
            break;
        }

        if(0 != pl.GetFrame(dump_res)) continue;

        // 零拷贝封装张量
        input_tensor = host_runtime_tensor::create(
            typecode_t::dt_uint8, in_shape,
            { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },
            false, hrt::pool_shared, dump_res.phy_addr
        ).expect("cannot create input tensor");

        hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync failed");

        // AI 推理与跟踪
        person_results.clear();
        yolo_det_app.pre_process(input_tensor);
        yolo_det_app.inference();
        yolo_det_app.post_process(person_results);

        std::vector<Detection> track_detections;
        for (auto &res : person_results) {
            Detection tmpRow;
            tmpRow.bbox_tlwh = cv::Rect_<float>(res.box.x, res.box.y, res.box.width, res.box.height);
            tmpRow.confidence = res.confidence;
            tmpRow.class_id = 0;

            feature_app.pre_process(input_tensor, res.box);
            feature_app.inference();
            feature_app.get_feature(tmpRow.feature);
            track_detections.push_back(tmpRow);
        }

        std::vector<std::shared_ptr<Track>> tracks = tracker.track(track_detections);

        // OSD 绘制
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
        for (const auto &track : tracks) {
            auto bbox = track->get_tlwh();
            int x = int(bbox[0] / image_size.width * OSD_WIDTH);
            int y = int(bbox[1] / image_size.height * OSD_HEIGHT);
            int w = int(bbox[2] / image_size.width * OSD_WIDTH);
            int h = int(bbox[3] / image_size.height * OSD_HEIGHT);

            cv::putText(draw_frame, cv::format("%d", track->track_id), cv::Point(x, y - 40), 0, 1, cv::Scalar(255, 255, 0, 255), 2);
            cv::putText(draw_frame, cv::format("time:%.2fs score:%.2f", track->stay_time, track->get_score()), cv::Point(x, y - 10), 0, 1, cv::Scalar(255, 255, 0, 255), 2);
            cv::rectangle(draw_frame, cv::Rect(x, y, w, h), cv::Scalar(255, 255, 0, 255), 2);
        }

        pl.InsertFrame(draw_frame.data);
        pl.ReleaseFrame(dump_res);
    }

    pl.Destroy();

    return 0;
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
    string video_path = argv[14];
    if (video_path == "realtime"){
        PipeLine pl(debug_mode);
        process_pipeline(argv, pl);
    }
    else{
        DisplayType display_type = (DISPLAY_MODE == 1) ? DISPLAY_LCD : DISPLAY_HDMI;
        VideoStreamPipeline pl(video_path, display_type, AI_FRAME_WIDTH, AI_FRAME_HEIGHT, 30, -1,RTSP_RTP_OVER_TCP);
        process_pipeline(argv, pl);
    }
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
    if (argc != 15)
    {
        print_usage(argv[0]);
        return -1;
    }

    std::thread thread_isp;

    // 启动视频处理线程
    thread_isp = std::thread(video_proc, argv);

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
