#include "rtsp_server.h"
#include "k_type.h"
#include "k_video_comm.h"
#include "k_venc_comm.h"
#include "k_vb_comm.h"

class EncodedRtspServer {
public:
    // 构造与析构
    EncodedRtspServer();
    ~EncodedRtspServer();

    // 初始化：分配编码器资源、RTSP 服务上下文等
    int init(int width, int height, int bitrate_kbps, const std::string &stream_url = "test", int port = 8554);

    // 启动 RTSP 服务（开始监听端口，准备推流）
    int start();

    // 停止 RTSP 服务（停止监听，断开所有客户端）
    int stop();

    // 反初始化：释放编码器、RTSP 资源
    int deinit();

    // 发送一帧原始图像数据，内部自动编码并推流(非绑定模式使用)
    int send_raw_frame(k_video_frame_info *frame, k_s32 milli_sec);

    // 绑定和解绑 VI 通道到编码器
    int bind_vi_chn(k_u32 vi_dev_id, k_u32 vi_chn_id);

    // 解绑 VI 通道从编码器
    int unbind_vi_chn();

    // 获取 RTSP 流地址
    char* get_rtsp_url();

protected:
    k_u32 _venc_vb_create_pool();
    int _init_venc();
    int _deinit_venc();
    int _init_rtsp_server();
    int _deinit_rtsp_server();
    static void *venc_stream_thread(void *arg);
    int _do_venc_stream(k_u32 chn_id, unsigned char *data, size_t size, k_venc_pack_type type, uint64_t timestamp);

private:
    k_u32 venc_chn_id_{0}; // 编码通道 ID
    k_u32 venc_attach_pool_id_{VB_INVALID_POOLID};

    int rtsp_port_;          // RTSP 监听端口
    std::string stream_url_; // RTSP 流名称
    int width_{1280};        // 视频宽度
    int height_{720};        // 视频高度
    int bitrate_kbps_{2000}; // 视频码率（kbps）
    int rotation_90_{0};
    bool binded_vi_{false};
    k_u32 vi_dev_id_{0};
    k_u32 vi_chn_id_{0};

    KdRtspServer rtsp_server_;// RTSP服务器
    pthread_t venc_tid_;     // Video encoder thread ID
    bool start_get_video_stream_{false}; // Flag to start getting video stream
};