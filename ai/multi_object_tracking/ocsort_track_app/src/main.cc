#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <thread>

/* 视频管线抽象：ISP → AI → OSD */
#include "video_pipeline.h"

/* AI 通用工具（tensor、计时等） */
#include "ai_utils.h"

/* YOLOv8 目标检测封装 */
#include "yolov8_det.h"

/* OCSort 多目标跟踪实现 */
#include "OCSort.hpp"

using std::cerr;
using std::cout;
using std::endl;

/* 全局原子标志，用于安全停止 ISP 处理线程 */
std::atomic<bool> isp_stop(false);

/**
 * @brief 将 std::vector<std::vector<float>> 转换为 Eigen 矩阵
 *
 * 每一行表示一个检测结果：
 * [x1, y1, x2, y2, score, class_id]
 *
 * @param data 输入检测数据
 * @return Eigen::Matrix<float, Dynamic, 6>
 */
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        for (int j = 0; j < data[0].size(); ++j)
            matrix(i, j) = data[i][j];
    return matrix;
}

/**
 * @brief 打印命令行用法及参数说明
 *
 * @param name 程序名称
 */
void print_usage(const char *name)
{
    cout << "Usage: " << name
         << "<yolov8_kmodel> <score_thres> <nms_thres> <det_thresh> <max_age> <min_hits> <iou_thresh> <delta_t> <inertia> <debug_mode> "<< endl
         << "Options:" << endl
         << "  yolov8_kmodel            YOLOv8 检测模型 kmodel 路径\n"
         << "  score_thres              检测置信度阈值\n"
         << "  nms_thres                检测阶段 NMS 阈值\n"
         << "  det_thresh               低于该置信度的检测结果不会送入跟踪器\n"
         << "                           用于过滤低置信度目标，减少虚假轨迹\n"
         << "  max_age                  轨迹在被删除前允许丢失的最大帧数\n"
         << "                           较大 → ID 更稳定，但可能产生更多虚假轨迹\n"
         << "                           较小 → 响应更快，但更容易丢失 ID\n"
         << "  min_hits                 轨迹被确认前所需的最小连续检测次数\n"
         << "  iou_thresh               匈牙利匹配时使用的 IOU 阈值\n"
         << "                           高 → 匹配严格，ID 稳定但易断裂\n"
         << "                           低 → 更容易匹配，但 ID 切换更多\n"
         << "  delta_t                  卡尔曼滤波运动模型的时间步长\n"
         << "  inertia                  惯性权重（OCSort 的核心改进）\n"
         << "                           较大 → 速度预测占主导（适合快速目标）\n"
         << "                           较小 → 检测结果占主导（适合慢速目标）\n"
         << "  debug_mode               调试等级：0=关闭，1=基础，2=详细\n"
         << endl;
}

/**
 * @brief 视频处理线程函数
 *
 * 功能包括：
 *  - 从 ISP 获取图像帧
 *  - YOLOv8 推理
 *  - OCSort 目标跟踪
 *  - OSD 绘制
 *
 * @param argv 命令行参数
 */
void video_proc(char *argv[])
{
    /* 调试等级 */
    int debug_mode = atoi(argv[10]);

    /* AI 输入图像尺寸（CHW 格式） */
    FrameCHWSize image_size = {
        AI_FRAME_CHANNEL,
        AI_FRAME_HEIGHT,
        AI_FRAME_WIDTH
    };

    /* OSD 图像缓冲区（RGBA） */
    cv::Mat draw_frame(
        OSD_HEIGHT,
        OSD_WIDTH,
        CV_8UC4,
        cv::Scalar(0, 0, 0, 0)
    );

    /* AI 输入运行时 Tensor */
    runtime_tensor input_tensor;

    /* 输入 Tensor 形状：NCHW */
    dims_t in_shape { 1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH };

    /* 创建视频管线实例 */
    PipeLine pl(debug_mode);

    /* 初始化管线（ISP、显示、缓冲区等） */
    pl.Create();

    /* 原始帧数据结构 */
    DumpRes dump_res;

    /* YOLOv8 检测结果 */
    std::vector<YOLOBbox> person_results;

    /* YOLOv8 检测应用实例 */
    YOLOv8Det yolo_det_app(
        argv[1],                 // kmodel 路径
        atof(argv[2]),            // 检测置信度阈值
        atof(argv[3]),            // NMS 阈值
        image_size,
        debug_mode
    );

    /* OCSort 参数 */
    float det_thresh = atof(argv[4]);   // 送入跟踪器的最小置信度
    int   max_age    = atoi(argv[5]);   // 最大丢失帧数
    int   min_hits   = atoi(argv[6]);   // 轨迹确认所需的最小命中次数
    float iou_thresh = atof(argv[7]);   // IOU 匹配阈值
    int   delta_t    = atoi(argv[8]);   // 卡尔曼滤波时间步长
    std::string asso_func = "giou";      // 关联函数（"iou" 或 "giou"）
    float inertia   = atof(argv[9]);     // 运动惯性权重
    bool  use_byte  = true;              // 启用 BYTE 风格的两阶段匹配

    /* 打印 OCSort 配置参数 */
    std::cout << "=================ocsort config==================" << std::endl;
    std::cout << "det_thresh:" << det_thresh << std::endl;
    std::cout << "max_age:"    << max_age    << std::endl;
    std::cout << "min_hits:"   << min_hits   << std::endl;
    std::cout << "iou_thresh:" << iou_thresh << std::endl;
    std::cout << "delta_t:"    << delta_t    << std::endl;
    std::cout << "asso_func:"  << asso_func  << std::endl;
    std::cout << "inertia:"    << inertia    << std::endl;
    std::cout << "use_byte:"   << use_byte   << std::endl;
    std::cout << "=================ocsort config end==================" << std::endl;

    /* 初始化 OCSort 跟踪器 */
    ocsort::OCSort tracker(
        det_thresh,
        max_age,
        min_hits,
        iou_thresh,
        delta_t,
        asso_func,
        inertia,
        use_byte
    );

    /* 主处理循环 */
    while (!isp_stop) {

        /* 统计整帧处理时间 */
        ScopedTiming st("total time", debug_mode);

        /* 从管线获取一帧图像 */
        pl.GetFrame(dump_res);

        /* 使用物理地址和虚拟地址创建运行时 Tensor */
        input_tensor = host_runtime_tensor::create(
            typecode_t::dt_uint8,
            in_shape,
            { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },
            false,
            hrt::pool_shared,
            dump_res.phy_addr
        ).expect("cannot create input tensor");

        /* 推理前进行内存同步 */
        hrt::sync(
            input_tensor,
            sync_op_t::sync_write_back,
            true
        ).expect("sync write_back failed");

        /* 清空上一帧检测结果 */
        person_results.clear();

        /*******************************************************
         * 目标检测 + 跟踪
         *******************************************************/

        /* YOLOv8 前处理 */
        yolo_det_app.pre_process(input_tensor);

        /* YOLOv8 推理 */
        yolo_det_app.inference();

        /* YOLOv8 后处理 */
        yolo_det_app.post_process(person_results);

        /* 准备送入跟踪器的检测数据 */
        std::vector<std::vector<float>> data;
        for (YOLOBbox res : person_results)
        {
            data.push_back({
                (float)res.box.x,
                (float)res.box.y,
                (float)(res.box.x + res.box.width),
                (float)(res.box.y + res.box.height),
                res.confidence,
                (float)0          // 类别 ID 占位
            });
        }

        /* 清空 OSD 缓冲区 */
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));

        /* 若存在检测结果，则执行跟踪 */
        if (!data.empty()) {

            /* 更新跟踪器并获取跟踪结果 */
            std::vector<Eigen::RowVectorXf> results =
                tracker.update(Vector2Matrix(data));

            for (auto& res : results) {

                /* 将 AI 坐标映射到 OSD 坐标 */
                int x = int(res[0] / image_size.width  * OSD_WIDTH);
                int y = int(res[1] / image_size.height * OSD_HEIGHT);
                int w = int(res[2] / image_size.width  * OSD_WIDTH)  - x;
                int h = int(res[3] / image_size.height * OSD_HEIGHT) - y;

                int   ID        = int(res[4]);  // 轨迹 ID
                int   class_id  = int(res[5]);  // 类别 ID
                float score     = res[6];       // 检测置信度
                double stay_time= res[7];       // 累计跟踪时间

                /* 绘制轨迹 ID */
                cv::putText(draw_frame, cv::format("%d", ID), cv::Point(x, y - 40), 0, 1, cv::Scalar(255, 255, 0, 255), 2, cv::LINE_AA);

                /* 绘制驻留时间和置信度 */
                cv::putText(draw_frame, cv::format("time:%.2fs, score:%.2f", stay_time, score), cv::Point(x, y - 10), 0, 1, cv::Scalar(255, 255, 0, 255), 2, cv::LINE_AA);

                /* 绘制目标框 */
                cv::rectangle(draw_frame, cv::Rect(x, y, w, h), cv::Scalar(255, 255, 0, 255), 2);
            }
        }

        /* 将 OSD 图像送入显示管线 */
        pl.InsertFrame(draw_frame.data);

        /* 释放 ISP 帧缓冲 */
        pl.ReleaseFrame(dump_res);
    }

    /* 销毁管线资源 */
    pl.Destroy();
}

/**
 * @brief 程序入口函数
 */
int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0]
              << " built at " << __DATE__ << " " << __TIME__
              << std::endl;

    /* 检查参数数量 */
    if (argc != 11)
    {
        print_usage(argv[0]);
        return -1;
    }

    /* 启动视频处理线程 */
    std::thread thread_isp(video_proc, argv);

    /* 等待按下 'q' 键退出 */
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    /* 通知线程停止并等待退出 */
    isp_stop.store(true);
    thread_isp.join();

    return 0;
}
