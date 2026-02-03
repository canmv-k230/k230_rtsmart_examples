#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <tuple>
#include <csignal>
#include <unistd.h>

#include "ai_utils.h"
#include "face_detection.h"
#include "hand_detection.h"
#include "yolov8_detect.h"
#include "setting.h"
#include "video_pipeline.h"

using namespace std;

// ================= 全局控制 =================
std::atomic<bool> face_det_isp_stop(false);
std::atomic<bool> person_det_isp_stop(false);
std::atomic<bool> hand_det_isp_stop(false);

// 仅保护 NPU / KPU inference
std::mutex g_infer_mutex;

std::mutex g_insert_mutex;


// ================= 使用说明 =================
void print_usage(const char *name)
{
    cout << "Usage: " << name
         << " <face_kmodel_det> <face_obj_thres> <face_nms_thres> <hand_kmodel_det> <hand_conf_thres> <hand_nms_thres> <yolov8_kmodel_det> <yolov8_conf_thres> <yolov8_nms_thres> <input_mode> <debug_mode>\n"
         << "Options:\n"
         << "  face_kmodel_det      人脸检测 kmodel 路径\n"
         << "  face_obj_thres       人脸检测目标阈值\n"
         << "  face_nms_thres       人脸检测NMS 阈值\n"
         << "  hand_kmodel_det      手掌检测kmodel 路径\n"
         << "  hand_conf_thres    手掌检测目标阈值\n"
         << "  hand_nms_thres     手掌检测NMS 阈值\n"
         << "  yolov8_kmodel_det    80分类目标检测kmodel 路径\n"
         << "  yolov8_conf_thres    80分类目标检测目标阈值\n"
         << "  yolov8_nms_thres     80分类目标检测NMS 阈值\n"
         << "  debug_mode      0/1/2：不调试 / 简单 / 详细\n"
         << endl;
}

// ================= 人脸检测视频处理线程 =================
void face_det_video_proc(PipeLine &pl,char **argv,int isp_id,int osd_id)
{
    int debug_mode = atoi(argv[10]);
    FrameCHWSize image_size = {AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};
    // OSD 绘制帧（线程私有）
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4,cv::Scalar(0, 0, 0, 0));
    // ISP 帧缓存
    DumpRes dump_res;

    // 输入 tensor shape
    dims_t in_shape = {1,AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};

    FaceDetection fd(argv[1], atof(argv[2]),atof(argv[3]), image_size, debug_mode);
    std::vector<FaceDetectionInfo> results;
    while (!face_det_isp_stop.load())
    {
        // 1. 获取 ISP 帧（阻塞）
        pl.GetFrame(dump_res, isp_id);
        // 2. 构建输入 tensor（共享物理内存）
        runtime_tensor input_tensor =host_runtime_tensor::create(typecode_t::dt_uint8,in_shape,{ (gsl::byte *)dump_res.virt_addr,compute_size(in_shape) },false,hrt::pool_shared,dump_res.phy_addr).expect("create input tensor failed");
        hrt::sync(input_tensor,sync_op_t::sync_write_back,true).expect("sync write_back failed");
        results.clear();
        // 3. 预处理（线程私有，无锁）
        fd.pre_process(input_tensor);
        // 4. 推理（仅 inference 加锁）
        {
            std::lock_guard<std::mutex> lock(g_infer_mutex);
            fd.inference();
        }
        // 5. 后处理（线程私有，无锁）
        fd.post_process(image_size, results);
        // 6. OSD 绘制
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
        fd.draw_result(draw_frame,results,false);
        cv::putText(draw_frame, "CAM-"+std::to_string(isp_id)+" Face Detect", {30,30}, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0 , 255, 255), 2, 8, 0);
        // 7. 输出到显示通道
        pl.InsertFrame(draw_frame.data, osd_id);
        // 8. 释放 ISP 帧
        pl.ReleaseFrame(dump_res, isp_id);
    }

    printf("[ISP %d] thread exit\n", isp_id);
}

// ================= 手掌检测视频处理线程 =================
void hand_det_video_proc(PipeLine &pl,char **argv,int isp_id,int osd_id)
{
    int debug_mode = atoi(argv[10]);
    FrameCHWSize image_size = {AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};
    // OSD 绘制帧（线程私有）
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4,cv::Scalar(0, 0, 0, 0));
    // ISP 帧缓存
    DumpRes dump_res;

    // 输入 tensor shape
    dims_t in_shape = {1,AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};

    HandDetection hd(argv[4], atof(argv[5]), atof(argv[6]), image_size,debug_mode);
    vector<BoxInfo> results;

    while (!person_det_isp_stop.load())
    {
        // 1. 获取 ISP 帧（阻塞）
        pl.GetFrame(dump_res, isp_id);
        // 2. 构建输入 tensor（共享物理内存）
        runtime_tensor input_tensor =host_runtime_tensor::create(typecode_t::dt_uint8,in_shape,{ (gsl::byte *)dump_res.virt_addr,compute_size(in_shape) },false,hrt::pool_shared,dump_res.phy_addr).expect("create input tensor failed");
        hrt::sync(input_tensor,sync_op_t::sync_write_back,true).expect("sync write_back failed");
        results.clear();
        // 3. 预处理（线程私有，无锁）
        hd.pre_process(input_tensor);
        // 4. 推理（仅 inference 加锁）
        {
            std::lock_guard<std::mutex> lock(g_infer_mutex);
            hd.inference();
        }
        // 5. 后处理（线程私有，无锁）
        hd.post_process(results);
        // 6. OSD 绘制
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
        hd.draw_result(draw_frame,results);
        cv::putText(draw_frame, "CAM-"+std::to_string(isp_id)+" Hand Detect", {30,30}, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0 , 255, 255), 2, 8, 0);
        // 7. 输出到显示通道
        pl.InsertFrame(draw_frame.data, osd_id);
        // 8. 释放 ISP 帧
        pl.ReleaseFrame(dump_res, isp_id);
    }
    printf("[ISP %d] thread exit\n", isp_id);
}

// ================= 80分类目标检测视频处理线程 =================
void yolov8_det_video_proc(PipeLine &pl,char **argv,int isp_id,int osd_id)
{
    int debug_mode = atoi(argv[10]);
    FrameCHWSize image_size = {AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};
    // OSD 绘制帧（线程私有）
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4,cv::Scalar(0, 0, 0, 0));
    // ISP 帧缓存
    DumpRes dump_res;

    // 输入 tensor shape
    dims_t in_shape = {1,AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};

    OBDet od(argv[7], atof(argv[8]), atof(argv[9]), image_size, debug_mode);
    vector<YOLOBbox> results;

    while (!hand_det_isp_stop.load())
    {
        // 1. 获取 ISP 帧（阻塞）
        pl.GetFrame(dump_res, isp_id);
        // 2. 构建输入 tensor（共享物理内存）
        runtime_tensor input_tensor =host_runtime_tensor::create(typecode_t::dt_uint8,in_shape,{ (gsl::byte *)dump_res.virt_addr,compute_size(in_shape) },false,hrt::pool_shared,dump_res.phy_addr).expect("create input tensor failed");
        hrt::sync(input_tensor,sync_op_t::sync_write_back,true).expect("sync write_back failed");
        results.clear();
        // 3. 预处理（线程私有，无锁）
        od.pre_process(input_tensor);
        // 4. 推理（仅 inference 加锁）
        {
            std::lock_guard<std::mutex> lock(g_infer_mutex);
            od.inference();
        }
        // 5. 后处理（线程私有，无锁）
        od.post_process(results);
        // 6. OSD 绘制
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
        od.draw_result(draw_frame,results);
        cv::putText(draw_frame, "CAM-"+std::to_string(isp_id)+" YOLOv8 Detect", {30,30}, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0 , 255, 255), 2, 8, 0);
        // 7. 输出到显示通道
        pl.InsertFrame(draw_frame.data, osd_id);
        // 8. 释放 ISP 帧
        pl.ReleaseFrame(dump_res, isp_id);
    }
    printf("[ISP %d] thread exit\n", isp_id);
}

// ================= main =================
int main(int argc, char *argv[])
{
    cout << "case " << argv[0]
         << " built at " << __DATE__ << " " << __TIME__ << endl;

    if (argc != 11)
    {
        print_usage(argv[0]);
        return -1;
    }

    int debug_mode = atoi(argv[5]);

    // 1. 创建视频管线
    PipeLine pl(debug_mode);
    pl.Create();

    // 2. 摄像头模式
    std::thread t0(face_det_video_proc, std::ref(pl),argv, 0, 4);

    std::thread t1(hand_det_video_proc, std::ref(pl),argv, 1, 5);

    std::thread t2(yolov8_det_video_proc, std::ref(pl),argv, 2, 6);

    // 主线程等待退出指令
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    // 通知线程退出
    face_det_isp_stop.store(true);
    t0.join();
    person_det_isp_stop.store(true);
    t1.join();
    hand_det_isp_stop.store(true);
    t2.join();
    // 3. 销毁管线
    pl.Destroy();
    cout << "exit success" << endl;
    return 0;
}
