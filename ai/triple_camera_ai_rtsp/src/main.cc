#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <tuple>
#include <csignal>
#include <unistd.h>
#include "ai_utils.h"
#include "yolov8_detect.h"
#include "setting.h"
#include "video_pipeline.h"

using namespace std;

// ================= 全局控制 =================
std::atomic<bool> sensor0_stop(false);
std::atomic<bool> sensor1_stop(false);
std::atomic<bool> sensor2_stop(false);

// ai2d和kpu均为独占设备，添加互斥锁
std::mutex mut;

// ================= 使用说明 =================
void print_usage(const char *name)
{
    cout << "Usage: " << name
         << " <yolov8_kmodel_det> <yolov8_conf_thres> <yolov8_nms_thres> <debug_mode>\n"
         << "Options:\n"
         << "  yolov8_kmodel_det        80分类目标检测kmodel 路径\n"
         << "  yolov8_conf_thres        80分类目标检测目标阈值\n"
         << "  yolov8_nms_thres         80分类目标检测NMS 阈值\n"
         << "  debug_mode               0/1/2：不调试 / 简单 / 详细\n"
         << endl;
}


void yolov8_det_video_proc(PipeLine &pl,char **argv,int isp_id,int osd_id)
{
    int debug_mode = atoi(argv[4]);
    FrameCHWSize image_size = {AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};

    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4,cv::Scalar(0, 0, 0, 0));
   
    DumpRes dump_res;
    dims_t in_shape = {1,AI_FRAME_CHANNEL,AI_FRAME_HEIGHT,AI_FRAME_WIDTH};

    OBDet od(argv[1], atof(argv[2]), atof(argv[3]), image_size, debug_mode);
    vector<YOLOBbox> results;

    int ret=0;
    while (!sensor0_stop.load())
    {
        ScopedTiming st("yolov8 program 1 total time", debug_mode);
        ret=pl.GetFrame(dump_res, isp_id);
        if(ret!=0){
            std::cerr << "GetFrame failed, ret: " << ret << std::endl;
            continue;
        }
        runtime_tensor input_tensor =host_runtime_tensor::create(typecode_t::dt_uint8,in_shape,{ (gsl::byte *)dump_res.virt_addr,compute_size(in_shape) },false,hrt::pool_shared,dump_res.phy_addr).expect("create input tensor failed");
        hrt::sync(input_tensor,sync_op_t::sync_write_back,true).expect("sync write_back failed");
        results.clear();
        {
            mut.lock();
            od.pre_process(input_tensor);
            od.inference();
            mut.unlock();
        }
        od.post_process(results);
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
        od.draw_result(draw_frame,results);
        cv::putText(draw_frame, "CAM-"+std::to_string(isp_id)+" YOLOv8 Detect", {30,30}, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0 , 255, 255), 2, 8, 0);
        pl.InsertFrame(draw_frame.data, osd_id);
        pl.ReleaseFrame(dump_res, isp_id);
        printf("[ISP %d] yolov8_det_video_proc_0 running normally, get %d objects\n", isp_id, results.size());
    }
    printf("[ISP %d] thread exit\n", isp_id);
    return;
}

// ================= main =================
int main(int argc, char *argv[])
{
    cout << "case " << argv[0]
         << " built at " << __DATE__ << " " << __TIME__ << endl;

    if (argc != 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    int debug_mode = atoi(argv[4]);

    // 1. 创建视频管线
    PipeLine pl(debug_mode);
    pl.Create();

    // 2. 摄像头应用线程
    std::thread t0(yolov8_det_video_proc, std::ref(pl),argv, 0,4);
        
    std::thread t1(yolov8_det_video_proc, std::ref(pl),argv, 1,5);

    std::thread t2(yolov8_det_video_proc, std::ref(pl),argv, 2,6);

    // 主线程等待退出指令
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    // 通知线程退出
    sensor0_stop.store(true);
    t0.join();
    sensor1_stop.store(true);
    t1.join();
    sensor2_stop.store(true);
    t2.join();
    // 3. 销毁管线
    pl.Destroy();
    cout << "exit success" << endl;
    return 0;
}
