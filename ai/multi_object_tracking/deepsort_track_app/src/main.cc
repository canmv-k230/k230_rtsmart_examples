#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <thread>

/* 视频采集与显示管线 */
#include "video_pipeline.h"

/* AI 通用工具函数 */
#include "ai_utils.h"

/* YOLOv8 目标检测模块 */
#include "yolov8_det.h"

/* ReID 外观特征提取模块 */
#include "feature.h"

/* DeepSORT 跟踪算法实现 */
#include "tracker.h"

using std::cerr;
using std::cout;
using std::endl;

/* 全局标志位，用于控制 ISP / 视频处理线程 */
std::atomic<bool> isp_stop(false);

/**
 * @brief 打印命令行参数使用说明
 * @param name 程序名称
 */
void print_usage(const char *name)
{
    cout << "Usage: " << name << "<yolov8_kmodel> <score_thres> <nms_thres> <feature_kmodel> <max_cosine_distance> <nn_budget> <max_iou_distance> <max_age> <n_init> <debug_mode> " << endl
         << "Options:" << endl
         << "  yolov8_kmodel            YOLOv8 目标检测模型 kmodel 路径\n"
         << "  score_thres              检测置信度阈值\n"
         << "  nms_thres                检测阶段 NMS 阈值\n"
         << "  feature_kmodel           ReID（外观特征）模型 kmodel 路径\n"
         << "  max_cosine_distance      ReID 特征匹配的最大余弦距离\n"
         << "  nn_budget                每个轨迹最多保存的特征数量\n"
         << "  max_iou_distance         基于空间匹配的最大 IoU 距离\n"
         << "  max_age                  轨迹允许丢失的最大帧数\n"
         << "  n_init                   轨迹确认所需的最少连续检测次数\n"
         << "  debug_mode               调试等级：0=关闭，1=基础，2=详细\n"
         << "\n"
         << endl;
}

/**
 * @brief 视频主处理线程
 *        包含图像采集、目标检测、特征提取、
 *        DeepSORT 跟踪以及 OSD 绘制显示
 * @param argv 命令行参数
 */
void video_proc(char *argv[])
{
    /* 调试模式等级 */
    int debug_mode = atoi(argv[10]);

    /* 输入帧尺寸（CHW 格式） */
    FrameCHWSize image_size={AI_FRAME_CHANNEL,AI_FRAME_HEIGHT, AI_FRAME_WIDTH};

    /* 用于绘制跟踪结果的 OSD 缓冲区 */
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    /* AI 输入运行时 Tensor */
    runtime_tensor input_tensor;

    /* 输入 Tensor 形状：NCHW */
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };

    /* 视频管线实例 */
    PipeLine pl(debug_mode);

    /* 初始化管线资源 */
    pl.Create();

    /* 帧数据结构 */
    DumpRes dump_res;

    /* YOLO 检测结果 */
    std::vector<YOLOBbox> person_results;

    /* YOLOv8 检测器实例 */
    YOLOv8Det yolo_det_app(argv[1], atof(argv[2]), atof(argv[3]), image_size, debug_mode);

    /* ReID 特征提取器实例 */
    Feature feature_app(argv[4], image_size, debug_mode);

    /* ReID 特征匹配的最大余弦距离 */
    float max_cosine_distance = atof(argv[5]);

    /* 每个轨迹保存的最大特征数量 */
    int nn_budget = atoi(argv[6]);

    /* 空间匹配的最大 IoU 距离 */
    float max_iou_distance = atof(argv[7]);

    /* 轨迹允许丢失的最大帧数 */
    int max_age = atoi(argv[8]);

    /* 确认轨迹所需的最少检测次数 */
    int n_init = atoi(argv[9]);

    /* 打印 DeepSORT 配置参数 */
    std::cout<<"===========deepsort config=================="<<std::endl;
    std::cout<<"max_cosine_distance:"<<max_cosine_distance<<std::endl;
    std::cout<<"nn_budget:"<<nn_budget<<std::endl;
    std::cout<<"max_iou_distance:"<<max_iou_distance<<std::endl;
    std::cout<<"max_age:"<<max_age<<std::endl;
    std::cout<<"n_init:"<<n_init<<std::endl;
    std::cout<<"===========deepsort config end============="<<std::endl;

    /* 初始化 DeepSORT 跟踪器 */
    tracker deepsort_tracker(max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init);

    /* 主处理循环 */
    while(!isp_stop){
        /* 统计整帧处理时间 */
        ScopedTiming st("total time", debug_mode);

        /* 从视频管线获取一帧 */
        pl.GetFrame(dump_res);

        /* 使用物理内存创建输入 Tensor */
        input_tensor = host_runtime_tensor::create(
            typecode_t::dt_uint8,
            in_shape,
            { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },
            false,
            hrt::pool_shared,
            dump_res.phy_addr
        ).expect("cannot create input tensor");

        /* 推理前同步输入 Tensor */
        hrt::sync(input_tensor, sync_op_t::sync_write_back, true)
            .expect("sync write_back failed");

        /* 清空检测结果 */
        person_results.clear();

        /*******************************************************
         * 行人检测与跟踪流程
         *******************************************************/
        yolo_det_app.pre_process(input_tensor);
        yolo_det_app.inference();
        yolo_det_app.post_process(person_results);

        /* 将检测结果转换为 DeepSORT 输入格式 */
        DETECTIONS detections;
        for (YOLOBbox res : person_results)
        {
            DETECTION_ROW tmpRow;

            /* tlwh 格式的目标框 */
            tmpRow.tlwh = DETECTBOX(
                (float)res.box.x,
                (float)res.box.y,
                (float)res.box.width,
                (float)res.box.height
            );

            /* 检测置信度 */
            tmpRow.confidence = res.confidence;

            /* 对该检测框提取 ReID 特征 */
            feature_app.pre_process(input_tensor, res.box);
            feature_app.inference();

            std::vector<float> feature;
            feature_app.get_feature(feature);

            /* 将特征向量转换为 Eigen 格式 */
            FEATURE feature_e = Eigen::Map<FEATURE>(feature.data());
            tmpRow.feature = feature_e;

            detections.push_back(tmpRow);
        }

        /* 预测现有轨迹状态 */
        deepsort_tracker.predict();

        /* 使用当前检测结果更新轨迹 */
        deepsort_tracker.update(detections);

        /* 清空 OSD 缓冲区 */
        draw_frame.setTo(cv::Scalar(0,0,0,0));

        /* 绘制已确认的轨迹 */
        for(Track& track : deepsort_tracker.tracks) {
            if(!track.is_confirmed() || track.time_since_update > 1)
                continue;

            /* 获取轨迹对应的目标框 */
            DETECTBOX tmp = track.to_tlwh();

            /* 将坐标映射到 OSD 分辨率 */
            int x = int(tmp(0)/image_size.width*OSD_WIDTH);
            int y = int(tmp(1)/image_size.height*OSD_HEIGHT);
            int w = int(tmp(2)/image_size.width*OSD_WIDTH);
            int h = int(tmp(3)/image_size.height*OSD_HEIGHT);

            /* 绘制轨迹 ID */ 
            cv::putText(draw_frame, cv::format("%d", track.track_id), cv::Point(x, y - 40), 0, 1, cv::Scalar(255,255, 0, 255), 2, cv::LINE_AA);
            /* 绘制驻留时间和置信度 */ 
            cv::putText(draw_frame, cv::format("time:%.2fs, score:%.2f", track.stay_time, track.score), cv::Point(x, y - 10), 0, 1, cv::Scalar(255,255, 0, 255), 2, cv::LINE_AA);
            /* 绘制目标框 */ 
            cv::rectangle(draw_frame, cv::Rect(x,y,w,h), cv::Scalar(255,255, 0, 255), 2);
        }

        /* 将 OSD 图像送入显示管线 */
        pl.InsertFrame(draw_frame.data);

        /* 释放当前帧资源 */
        pl.ReleaseFrame(dump_res);
    }

    /* 销毁视频管线资源 */
    pl.Destroy();
}

/**
 * @brief 程序入口函数
 */
int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at "
              << __DATE__ << " " << __TIME__ << std::endl;

    /* 检查参数数量是否正确 */
    if (argc != 11)
    {
        print_usage(argv[0]);
        return -1;
    }

    /* 启动视频处理线程 */
    std::thread thread_isp(video_proc, argv);

    /* 等待用户按下 'q' 键退出 */
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    /* 通知线程停止处理 */
    isp_stop.store(true);

    /* 等待线程退出 */
    thread_isp.join();

    return 0;
}
