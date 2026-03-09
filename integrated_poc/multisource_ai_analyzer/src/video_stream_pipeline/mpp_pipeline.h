#ifndef MPP_PIPELINE_H
#define MPP_PIPELINE_H

#include <cstdint>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <atomic>
#include "idata_source.h"
#include "k_vo_comm.h"
#include "k_video_comm.h"
#include "k_vb_comm.h"


// RGB平面格式数据（供AI模块）
struct RgbFrame {
    uint8_t* data[3];      // R/G/B平面指针
    int stride[3];         // 行跨度
    int width;             // 宽度
    int height;            // 高度
    int64_t pts;           // 时间戳
};

// MPP管道状态
enum class MppPipelineStatus {
    UNINITIALIZED,
    INITIALIZING,
    RUNNING,
    STOPPED,
    ERROR
};

enum MppVoType{
    EM_VO_HDMI = 0,
    EM_VO_LCD,
};

typedef void (*RgbFrameCallback)(const k_video_frame_info& frame, void* user_data);

class MppPipeline {
    friend class VideoStreamPipeline;
    friend class  VideoStreamPipelineImpl;
public:
    // 构造函数初始化成员
    MppPipeline();

    ~MppPipeline();

    // 禁用拷贝
    MppPipeline(const MppPipeline&) = delete;
    MppPipeline& operator=(const MppPipeline&) = delete;

    void set_rgb_callback(RgbFrameCallback callback,void* user_data);
    int init(int w, int h, bool enable_dsl,bool rotation_90, EncType type,MppVoType vo_type);

    int start();
    int decode_stream(const EncStream& stream);
    int decode_stream(const k_vdec_stream& stream);
    int stop();
    int deinit();


private:
    // 初始化视频缓冲池（VB）
    int _init_vb();

    int _init_vb_module();

    // 初始化解码器（VDEC）
    int _init_vdec();

    // 初始化显示输出（VO）
    int _init_vo();

    int _init_osd(k_vo_layer_id osd_id);

    // 初始化色彩转换（CSC）
    int _init_csc();

    // 绑定管道组件
    int _bind_pipeline();

    int _unbind_pipeline();

    // 处理解码后的数据帧
    void _process_decoded_frames();

    int _osd_alloc_frame(void **osd_vaddr);
    int _osd_draw_frame(void* osd_data);
    int _release_rgb_frame(k_video_frame_info* rgb_frame_info_);

    static void* _frame_process_thread_entry(void* arg);

    // 成员变量
    mutable std::mutex mtx_;       // 线程安全锁
    int width_;                    // 视频宽度
    int height_;                   // 视频高度
    bool enable_dsl_;
    bool rotation_90_;
    EncType enc_type_;             // 编码类型
    MppPipelineStatus status_;     // 管道状态
    RgbFrameCallback rgb_frame_callback_;     // RGB回调函数
    mutable std::mutex release_frame_mtx_;       // 线程安全锁
    void* user_data_;

    k_u32 stream_pool_id_{VB_INVALID_POOLID};
    k_u8* stream_virt_addr_;
    k_u64 stream_phys_addr_;

    k_u32 vdec_pool_id_{VB_INVALID_POOLID};
    k_u32 csc_pool_id_{VB_INVALID_POOLID};
    k_u32 vdec_chn_id_{0};
    k_u32 csc_chn_id_{0};
    MppVoType connector_type_;
    k_vo_layer_id vo_layer_chn_id_{K_VO_LAYER_VIDEO1};
    k_vo_layer_id osd_id_{K_VO_LAYER_OSD3}; // OSD ID
    k_u32 osd_pool_id_{VB_INVALID_POOLID}; // OSD pool ID
    k_u32 osd_vb_handle_{VB_INVALID_HANDLE}; // OSD video buffer handle
    void* osd_vaddr_;
    k_video_frame_info rgb_frame_info_;
    k_video_frame_info osd_vf_info_; // OSD video frame information

    pthread_t frame_process_thread_;
    std::atomic<bool> thread_running_;
    bool      vb_exit_{false};
};

#endif  // MPP_PIPELINE_H