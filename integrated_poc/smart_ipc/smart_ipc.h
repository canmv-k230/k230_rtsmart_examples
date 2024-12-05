#ifndef _SMART_IPC_H
#define _SMART_IPC_H
#include "rtsp_server.h"
#include "media.h"
#include <atomic>
#include "face_detection.h"
#include <vector>

class MySmartIPC :public IOnBackChannel,  public IOnAEncData, public IOnVEncData,public IOnAIFrameData {
  public:
    MySmartIPC();

    virtual void OnBackChannelData(std::string &session_name, const uint8_t *data, size_t size, uint64_t timestamp) override {
        ;
    }
    // IOnAEncData
    virtual void OnAEncData(k_u32 chn_id, k_u8*pdata,size_t size,k_u64 time_stamp);
    // IOnVEncData
    virtual void OnVEncData(k_u32 chn_id, void *data, size_t size, k_venc_pack_type type,uint64_t timestamp);
    // IOnAIFrameData
    virtual void OnAIFrameData(k_u32 chn_id, k_video_frame_info*frame_info);
    //init
    int Init(const KdMediaInputConfig &config, const std::string &stream_url = "test", int port = 8554);
    //deinit
    int DeInit();
    //start
    int Start();
    //stop
    int Stop();

  private:
    int _ai_analyse_init();

  private:
    KdMediaFeatureConfig feature_config_;
    KdMediaInputConfig input_config_;
    KdRtspServer rtsp_server_;//rtsp server
    KdMedia media_;//media mpi interface
    FaceDetection* face_detection_;//face detection
    std::string stream_url_;
    std::atomic<bool> started_{false};

    size_t ai_frame_paddr_{0};
    size_t ai_frame_size_{0};
    void * ai_frame_vaddr_{nullptr};
    void * osd_vaddr_{nullptr};
    vector<FaceDetectionInfo> detect_result_;
};

#endif // _SMART_IPC_H


