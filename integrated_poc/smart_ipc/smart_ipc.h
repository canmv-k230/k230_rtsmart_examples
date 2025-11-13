#ifndef _SMART_IPC_H
#define _SMART_IPC_H
#include "rtsp_server.h"
#include "media.h"
#include <atomic>
#include "face_detection.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

class MySmartIPC : public IOnAEncData, public IOnVEncData, public IOnAIFrameData {
public:
    MySmartIPC();
    // 初始化
    int Init(const KdMediaInputConfig &config, const std::string &stream_url = "test", int port = 8554);
    // 反初始化
    int DeInit();
    // 启动
    int Start();
    // 停止
    int Stop();

protected:
    // 音频编码数据回调
    virtual void OnAEncData(k_u32 chn_id, k_u8* pdata, size_t size, k_u64 time_stamp);
    // 视频编码数据回调
    virtual void OnVEncData(k_u32 chn_id, void *data, size_t size, k_venc_pack_type type, uint64_t timestamp);
    // AI帧数据回调
    virtual void OnAIFrameData(k_u32 chn_id, k_video_frame_info* frame_info);

private:
    // AI分析初始化
    int _ai_analyse_init();
    // RTSP线程主函数
    void RtspThreadMain();

private:
    KdMediaFeatureConfig feature_config_;
    KdMediaInputConfig input_config_;
    KdRtspServer rtsp_server_;       // RTSP服务器
    KdMedia media_;                  // 媒体MPI接口
    FaceDetection* face_detection_;  // 人脸检测
    std::string stream_url_;
    int rtsp_port_;                  // RTSP端口
    std::atomic<bool> started_{false};
    std::atomic<bool> rtsp_running_{false};  // RTSP线程运行标志
    std::atomic<bool> rtsp_ready_{false};    // RTSP就绪标志（初始化并启动完成）

    // 线程与同步
    std::thread rtsp_thread_;
    std::mutex rtsp_mutex_;
    std::condition_variable rtsp_cv_;

    // AI帧数据相关
    size_t ai_frame_paddr_{0};
    size_t ai_frame_size_{0};
    void *ai_frame_vaddr_{nullptr};
    void *osd_vaddr_{nullptr};
    std::vector<FaceDetectionInfo> detect_result_;
};

#endif // _SMART_IPC_H