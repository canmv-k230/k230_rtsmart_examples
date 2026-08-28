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

// 视频处理线程函数（定义在下方）
void video_proc(const AppSettings& settings);

// video_proc 线程入口（pthread 包装，用于显式指定线程栈大小）
static void* video_proc_entry(void *arg)
{
    AppSettings *settings = static_cast<AppSettings*>(arg);
    video_proc(*settings);
    return nullptr;
}

template <typename T>
int process_pipeline(const AppSettings& settings, T& pl)
{
    // 1. 参数解析 (从 settings 获取)
    int debug_mode = settings.debug_mode;

    // 2. 资源创建（先于模型初始化：帧尺寸要等管线建立后才能确定）
    if (pl.Create()){
        CmdLineParser::print_usage("multisource_ai_analyzer");
        return -1;
    }

    // 帧尺寸运行时确定（Create 之后才能取到）：
    // 实时源=显示分辨率，文件/网络源=实际解码分辨率
    int frame_w, frame_h;
    pl.GetFrameSize(frame_w, frame_h);
    FrameCHWSize image_size = {(size_t)AI_FRAME_CHANNEL, (size_t)frame_h, (size_t)frame_w};
    dims_t in_shape { 1, (size_t)AI_FRAME_CHANNEL, (size_t)frame_h, (size_t)frame_w };

    // 3. 初始化应用 (Detection, ReID, Tracker)
    YOLOv8Det yolo_det_app(settings.det_model_path.c_str(),
                           settings.score_thresh,
                           settings.nms_thresh,
                           image_size, debug_mode);
    Feature feature_app(settings.reid_model_path.c_str(),
                        image_size, debug_mode);

    BoTSORT tracker(
        true, // reid_enabled
        settings.track_high_thresh,
        settings.track_low_thresh,
        settings.new_track_thresh,
        settings.frame_buffer,
        settings.match_thresh,
        settings.proximity_thresh,
        settings.appearance_thresh,
        30,
        settings.lambda
    );

    cv::Mat draw_frame(frame_h, frame_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
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
            int x = int(bbox[0] / image_size.width * frame_w);
            int y = int(bbox[1] / image_size.height * frame_h);
            int w = int(bbox[2] / image_size.width * frame_w);
            int h = int(bbox[3] / image_size.height * frame_h);

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
 * @param settings 应用配置参数
 */
void video_proc(const AppSettings& settings)
{
    // 调试级别
    int debug_mode = settings.debug_mode;
    string video_path = settings.video_path;
    if (video_path == "realtime"){
        // 实时采集：CSI 口与显示器类型由 -s/-c 指定
        PipeLine pl(debug_mode, settings.csi_num, settings.connector_type);
        process_pipeline(settings, pl);
    }
    else{
        // 文件/网络源：流媒体管线实际只支持 ST7701（LCD）与 LT9611（HDMI）两种输出，
        // 其余面板（如 HX8377）归入任一分支都会灌错面板时序，这里保持 HDMI 并明确告警
        DisplayType display_type = DISPLAY_HDMI;
        if (settings.connector_type == ST7701_V1_MIPI_2LAN_480X800_30FPS) {
            display_type = DISPLAY_LCD;
        } else if (settings.connector_type != LT9611_MIPI_4LAN_1920X1080_30FPS &&
                   settings.connector_type != LT9611_MIPI_4LAN_1920X1080_60FPS) {
            printf("WARNING: connector %u not supported by the stream pipeline (only ST7701 LCD / LT9611 HDMI), falling back to HDMI\n",
                   (unsigned int)settings.connector_type);
        }
        int out_w, out_h;
        GetConnectorDisplaySize(settings.connector_type, out_w, out_h);
        VideoStreamPipeline pl(video_path, display_type, out_w, out_h, 30, -1, RTSP_RTP_OVER_TCP);
        process_pipeline(settings, pl);
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

    // 解析命令行参数
    AppSettings settings;
    if (!CmdLineParser::parse(argc, argv, settings)) {
        return -1;
    }
    
    // 可选：打印配置信息
    settings.print();

    // 启动视频处理线程（VICAP/ISP 初始化 + nncase 模型加载调用链深，
    // musl 默认 pthread 栈仅 128KB 会溢出，显式加大到 1MB）
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1024 * 1024);

    pthread_t thread_isp;
    if (pthread_create(&thread_isp, &attr, video_proc_entry, &settings) != 0) {
        std::cerr << "Failed to create video thread" << std::endl;
        return -1;
    }
    pthread_attr_destroy(&attr);

    // 等待按下 'q' 键退出
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    // 通知处理线程退出
    isp_stop.store(true);

    // 等待线程结束
    pthread_join(thread_isp, nullptr);

    return 0;
}
