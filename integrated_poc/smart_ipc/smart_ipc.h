#ifndef _SMART_IPC_H
#define _SMART_IPC_H
#include "media.h"
#include <atomic>
#include "face_detection.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdint>

/* Streaming mode: RTSP or WebRTC, mutually exclusive */
enum class StreamingMode {
    kRtsp = 0,
    kWebRtc,
};

/* WebRTC (libpeer) */
extern "C" {
#include "peer.h"
}
#include "http_server.h"   /* http_response_t for OnHttpRequest */

class MySmartIPC : public IOnAEncData, public IOnVEncData, public IOnAIFrameData {
public:
    MySmartIPC();
    // 初始化
    int Init(const KdMediaInputConfig &config, const std::string &stream_url = "test",
             StreamingMode mode = StreamingMode::kRtsp, int port = 8554);
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

    // ── RTSP ──
    void RtspThreadMain();

    // ── WebRTC ──
    void WebRtcThreadMain();

public:
    // ── WebRTC (public for C-linkage wrapper) ──
    static void OnHttpRequest(const char* method, const char* path,
                              const char* body, int body_len,
                              http_response_t* response);

private:
    KdMediaFeatureConfig feature_config_;
    KdMediaInputConfig input_config_;
    KdMedia media_;                  // 媒体MPI接口
    FaceDetection* face_detection_;  // 人脸检测
    std::string stream_url_;
    StreamingMode streaming_mode_ = StreamingMode::kRtsp;
    int port_;                       // RTSP端口 或 WebRTC HTTP端口
    std::atomic<bool> started_{false};
    std::atomic<bool> streaming_running_{false}; // 流媒体线程运行标志
    std::atomic<bool> streaming_ready_{false};   // 流媒体就绪标志

    // 线程与同步
    std::thread streaming_thread_;
    std::mutex streaming_mutex_;
    std::condition_variable streaming_cv_;

    // AI帧数据相关
    size_t ai_frame_paddr_{0};
    size_t ai_frame_size_{0};
    void *ai_frame_vaddr_{nullptr};
    void *osd_vaddr_{nullptr};
    std::vector<FaceDetectionInfo> detect_result_;

    // ── RTSP specific ──
    void* rtsp_server_ = nullptr;   // KdRtspServer* (void* to avoid header in .h)

    // ── WebRTC specific ──
    PeerConnection* webrtc_pc_{nullptr};
    std::atomic<PeerConnectionState> webrtc_state_{PEER_CONNECTION_CLOSED};
    std::thread webrtc_peer_thread_;     // peer_connection_loop thread

    // Drain flags: set while a worker thread is inside webrtc_pc_ internals,
    // cleared on exit. HTTP thread drains on these after close() before create_offer().
    std::atomic<int> peer_in_pc_{0};
    std::atomic<int> venc_in_pc_{0};

    // SDP offer/answer synchronization
    pthread_mutex_t offer_mutex_ = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t offer_cond_ = PTHREAD_COND_INITIALIZER;
    char* offer_sdp_{nullptr};
    int offer_ready_{0};

    // H.264 SPS/PPS cache (prepend to every I-frame for WebRTC)
    uint8_t* sps_pps_buf_{nullptr};
    size_t sps_pps_size_{0};
};

#endif // _SMART_IPC_H
