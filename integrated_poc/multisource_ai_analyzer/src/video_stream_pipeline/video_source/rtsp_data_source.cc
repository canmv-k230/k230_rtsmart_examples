// rtsp_data_source.cpp
#include "rtsp_data_source.h"
#include <thread>
#include <atomic>
#include <cstring>
#include <unistd.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/time.h>
}

#define RTSP_READ_FRAME_TIMEOUT 5000

// 全局初始化
class FFmpegInitializer {
public:
    FFmpegInitializer() { avformat_network_init(); }
};
static FFmpegInitializer ffmpeg_init_;

struct TimeoutContext {
    std::chrono::steady_clock::time_point start;
    int timeoutMs; // 超时毫秒数，例如 10000 = 10秒
};

static int interruptCallback(void* ctx) {
    TimeoutContext* tctx = static_cast<TimeoutContext*>(ctx);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - tctx->start).count();
    return (elapsed > tctx->timeoutMs) ? 1 : 0; // 返回1表示中断
}

struct RTSPDataSource::Impl {
    std::string url;
    bool isOpened = false;
    std::atomic<bool> isDelivering{false};
    std::thread pullThread;

    int width = 0;
    int height = 0;
    int videoCodecId = -1; // 0: H264, 1: H265
    int videoStreamIndex = -1;

    StreamCallback callback = nullptr;
    void* userData = nullptr;

    AVFormatContext* fmtCtx = nullptr;
    bool enableRtpOverTcp = false;
};

RTSPDataSource::RTSPDataSource(bool enable_rtp_over_tcp) : m_impl(std::make_unique<Impl>()) {
    m_impl->enableRtpOverTcp = enable_rtp_over_tcp;
}

RTSPDataSource::~RTSPDataSource() {
}

int RTSPDataSource::open(const std::string& url, int fps,const DataSourceCallbacks& callbacks) {
    if (m_impl->isOpened) return -1;

    m_impl->url = url;
    m_impl->callback = callbacks.stream_cb;
    m_impl->userData = callbacks.user_data;

    if (!_open_and_probe()) {
        printf("open rtsp url:%s failed\n",url.c_str());
        return -1;
    }

    printf("open rtsp url:%s ok\n",url.c_str());

    m_impl->isOpened = true;
    return 0;
}

void RTSPDataSource::close() {
    // 关闭 AVFormatContext
    if (m_impl->fmtCtx) {
        if (m_impl->fmtCtx->interrupt_callback.opaque) {
            TimeoutContext* tctx = static_cast<TimeoutContext*>(m_impl->fmtCtx->interrupt_callback.opaque);
            delete tctx; // 释放内存
            m_impl->fmtCtx->interrupt_callback.opaque = nullptr;
        }

        avformat_close_input(&m_impl->fmtCtx);
        m_impl->fmtCtx = nullptr;
    }
    m_impl->isOpened = false;
    m_impl->width = 0;
    m_impl->height = 0;
    m_impl->videoCodecId = -1;
    m_impl->videoStreamIndex = -1;
}

int RTSPDataSource::start() {
    if (!m_impl->isOpened || m_impl->isDelivering.exchange(true)) {
        return -1;
    }
    m_impl->pullThread = std::thread(&RTSPDataSource::_ffmpeg_pull_loop, this);
    return 0;
}

int RTSPDataSource::stop() {
    if (!m_impl->isOpened) return -1;
    m_impl->isDelivering.store(false);
    if (m_impl->pullThread.joinable()) {
        m_impl->pullThread.join();
    }
    return 0;
}

int RTSPDataSource::get_width() const {
    return m_impl->width;
}

int RTSPDataSource::get_height() const {
    return m_impl->height;
}

EncType RTSPDataSource::get_stream_type() const {
    return (EncType)m_impl->videoCodecId;
}

bool RTSPDataSource::_open_and_probe() {
    AVDictionary* options = nullptr;
    if (m_impl->enableRtpOverTcp)
    {
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
    }
    av_dict_set(&options, "stimeout", "5000000", 0); // 5秒超时
    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);
    av_dict_set(&options, "analyzeduration", "0", 0);
    av_dict_set(&options, "probesize", "32768", 0);

    TimeoutContext tctx;
    tctx.timeoutMs = RTSP_READ_FRAME_TIMEOUT; //超时

    AVFormatContext* fmt_ctx = avformat_alloc_context();
    TimeoutContext* userCtx = new TimeoutContext{std::chrono::steady_clock::now(), RTSP_READ_FRAME_TIMEOUT};
    fmt_ctx->interrupt_callback.callback = interruptCallback;
    fmt_ctx->interrupt_callback.opaque = userCtx;

    //AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, m_impl->url.c_str(), nullptr, &options);
    av_dict_free(&options);
    if (ret != 0) {
        return false;
    }

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        avformat_close_input(&fmt_ctx);
        return false;
    }

    int video_stream_idx = -1;
    AVCodecParameters* codecpar = nullptr;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            codecpar = fmt_ctx->streams[i]->codecpar;
            break;
        }
    }

    if (video_stream_idx == -1) {
        avformat_close_input(&fmt_ctx);
        return false;
    }

    m_impl->fmtCtx = fmt_ctx;
    m_impl->videoStreamIndex = video_stream_idx;
    m_impl->width = codecpar->width;
    m_impl->height = codecpar->height;

    if (codecpar->codec_id == AV_CODEC_ID_H264) {
        m_impl->videoCodecId = 0;
    } else if (codecpar->codec_id == AV_CODEC_ID_HEVC) {
        m_impl->videoCodecId = 1;
    } else {
        m_impl->videoCodecId = -1;
    }

    return true;
}

void RTSPDataSource::_ffmpeg_pull_loop() {
    AVFormatContext* fmt_ctx = m_impl->fmtCtx;
    if (!fmt_ctx || m_impl->videoStreamIndex < 0) {
        m_impl->isDelivering.store(false, std::memory_order_relaxed);
        if (m_impl->callback) {
            m_impl->callback(EncStream{}, m_impl->userData, true); // EOF
        }
        return;
    }

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        m_impl->isDelivering.store(false, std::memory_order_relaxed);
        if (m_impl->callback) {
            m_impl->callback(EncStream{}, m_impl->userData, true);
        }
        return;
    }

    while (m_impl->isDelivering.load(std::memory_order_relaxed)) {

        static_cast<TimeoutContext*>(fmt_ctx->interrupt_callback.opaque)->start =
        std::chrono::steady_clock::now();

        int ret = av_read_frame(fmt_ctx, pkt);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                printf("RTSP stream ended (EOF).\n");
            }
            else if (ret == AVERROR_EXIT) {
                printf("FFmpeg context was forced to exit.\n");
            }
            else if (ret == AVERROR(EAGAIN) || ret == AVERROR(EWOULDBLOCK)) {
                printf("Read timeout or would block (EAGAIN).\n");
            }
            else {
                char errbuf[256];
                av_strerror(ret, errbuf, sizeof(errbuf));
                printf("av_read_frame failed: %s (error code: %d)\n", errbuf, ret);
            }

            break;
        }

        if (pkt->stream_index == m_impl->videoStreamIndex) {
            EncStream frame;
            frame.data = pkt->data;
            frame.size = pkt->size;
            frame.pts = pkt->pts;
            frame.enctype = static_cast<EncType>(m_impl->videoCodecId);
            _deliver_frame(frame);
        }
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    m_impl->isDelivering.store(false, std::memory_order_relaxed);

    // 发送 EOF
    if (m_impl->callback) {
        EncStream eof;
        m_impl->callback(eof, m_impl->userData, true);
    }
}

void RTSPDataSource::_deliver_frame(const EncStream& frame) {
    if (m_impl->callback) {
        m_impl->callback(frame, m_impl->userData, false);
    }
}