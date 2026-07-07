#include "smart_ipc.h"
#include <unistd.h>
#include "mpi_sys_api.h"
#include "k_vb_comm.h"
#include "scoped_timing.hpp"
#include <cstdio>
#include <chrono>
#include <thread>
#include <cerrno>
#include <cstring>
#include <cstdlib>

/* Streaming mode specific headers */
#include "rtsp_server.h"   /* RTSP */
#include "http_server.h"   /* WebRTC signaling */
#include "web_page.h"      /* WebRTC embedded page */

/* Global instance for C-style HTTP callback (no userdata in http_server API) */
static MySmartIPC* g_smart_ipc_instance = nullptr;

/* C-linkage wrapper so http_server_start gets a proper C function pointer.
 * OnHttpRequest is a C++ static member and may not be safely passed as a C callback. */
extern "C" void http_request_handler(const char* method, const char* path,
                                      const char* body, int body_len,
                                      http_response_t* response) {
    MySmartIPC::OnHttpRequest(method, path, body, body_len, response);
}

MySmartIPC::MySmartIPC() : port_(8554) {}

void MySmartIPC::OnAEncData(k_u32 chn_id, k_u8* pdata, size_t size, k_u64 time_stamp) {
    if (!started_ || !streaming_ready_) return;

    if (streaming_mode_ == StreamingMode::kRtsp) {
        auto* srv = static_cast<KdRtspServer*>(rtsp_server_);
        if (srv) srv->SendAudioData(stream_url_, (const uint8_t*)pdata, size, time_stamp);
    }
    /* WebRTC mode: no audio support in this demo */
}

static k_u64 get_ticks() {
    volatile k_u64 time_elapsed;
    __asm__ __volatile__(
        "rdtime %0"
        : "=r"(time_elapsed));
    return time_elapsed;
}

void MySmartIPC::OnVEncData(k_u32 chn_id, void *data, size_t size, k_venc_pack_type type, uint64_t timestamp) {
    if (!started_ || !streaming_ready_) return;

    if (streaming_mode_ == StreamingMode::kRtsp) {
        /* ── RTSP path ── */
        auto* srv = static_cast<KdRtspServer*>(rtsp_server_);
        if (srv) srv->SendVideoData(stream_url_, (const uint8_t*)data, size, timestamp);

    } else {
        /* ── WebRTC path ── */
        /* Mark in-use BEFORE the state check so the reconnect path (GET /offer)
         * sees us and waits. If we checked state first, close() could happen
         * between the check and setting the flag — the drain would miss us. */
        venc_in_pc_ = 1;
        if (!webrtc_pc_ || (webrtc_state_ != PEER_CONNECTION_COMPLETED && webrtc_state_ != PEER_CONNECTION_CONNECTED)) {
            venc_in_pc_ = 0;
            return;
        }

        if (type == K_VENC_HEADER) {
            /* Cache SPS/PPS for prepending to I-frames */
            if (sps_pps_buf_) free(sps_pps_buf_);
            sps_pps_buf_ = (uint8_t*)malloc(size);
            if (sps_pps_buf_) {
                memcpy(sps_pps_buf_, data, size);
                sps_pps_size_ = size;
            }
            venc_in_pc_ = 0;
            return;
        }
        /* Prepend SPS/PPS before every I-frame */
        if (type == K_VENC_I_FRAME && sps_pps_buf_ && sps_pps_size_ > 0) {
            peer_connection_send_video(webrtc_pc_, sps_pps_buf_, sps_pps_size_, timestamp);
        }
        peer_connection_send_video(webrtc_pc_, (const uint8_t*)data, size, timestamp);
        venc_in_pc_ = 0;
    }
}

void MySmartIPC::OnAIFrameData(k_u32 chn_id, k_video_frame_info* frame_info) {
    if (!started_) return;

    ScopedTiming st("ai total time", 0);
    // 复制AI帧数据到本地内存
    {
        ScopedTiming st("isp copy", 0);
        auto vbvaddr = kd_mpi_sys_mmap_cached(frame_info->v_frame.phys_addr[0], ai_frame_size_);
        memcpy(ai_frame_vaddr_, (void *)vbvaddr, ai_frame_size_);
        kd_mpi_sys_munmap(vbvaddr, ai_frame_size_);
    }

    // AI分析获取检测结果
    detect_result_.clear();
    face_detection_->pre_process();
    face_detection_->inference();
    face_detection_->post_process({input_config_.ai_width, input_config_.ai_height}, detect_result_);

    // 绘制OSD
    cv::Mat osd_frame(input_config_.osd_height, input_config_.osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    {
        ScopedTiming st("osd draw", 0);
        if (input_config_.vo_connect_type == LT9611_MIPI_4LAN_1920X1080_30FPS) {
            face_detection_->draw_result(osd_frame, detect_result_, false);
        } else if (input_config_.vo_connect_type == ST7701_V1_MIPI_2LAN_480X800_30FPS) {
            face_detection_->draw_result(osd_frame, detect_result_, false);
        } else if (input_config_.vo_connect_type == HX8377_V2_MIPI_4LAN_1080X1920_30FPS) {
            face_detection_->draw_result(osd_frame, detect_result_, false);
        }
    }

    // 复制OSD数据并显示
    if (osd_vaddr_ != NULL) {
        ScopedTiming st("osd copy", 0);
        memcpy(osd_vaddr_, osd_frame.data, input_config_.osd_width * input_config_.osd_height * 4);
        media_.osd_draw_frame();
    }
}


#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>

/**
 * @brief 获取指定网络接口的IP地址（IPv4）
 */
int get_interface_ip(const char *ifname, char *ip_str, int str_len) {
    int sock_get_ip;
    struct sockaddr_in *sin;
    struct ifreq ifr_ip;

    if (ifname == NULL || ip_str == NULL || str_len < 16) {
        printf("Invalid parameters for get_interface_ip\n");
        return -1;
    }

    if ((sock_get_ip = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        printf("Failed to create socket for getting IP\n");
        return -1;
    }

    memset(&ifr_ip, 0, sizeof(ifr_ip));
    strncpy(ifr_ip.ifr_name, ifname, sizeof(ifr_ip.ifr_name) - 1);

    if (ioctl(sock_get_ip, SIOCGIFADDR, &ifr_ip) < 0) {
        printf("ioctl(SIOCGIFADDR) failed for interface %s\n", ifname);
        close(sock_get_ip);
        return -1;
    }

    sin = (struct sockaddr_in *)&ifr_ip.ifr_addr;
    strncpy(ip_str, inet_ntoa(sin->sin_addr), str_len - 1);
    ip_str[str_len - 1] = '\0';

    close(sock_get_ip);
    return 0;
}

static int get_valid_ip_with_retry(const char *ifname, char *ip_str, int str_len,
                           int max_retry, int interval_ms) {
    if (ifname == NULL || ip_str == NULL || str_len < 16 || max_retry <= 0 || interval_ms <= 0) {
        return -1;
    }

    for (int i = 0; i < max_retry; i++) {
        if (get_interface_ip(ifname, ip_str, str_len) == 0) {
            if (strcmp(ip_str, "0.0.0.0") != 0) {
                return 0;
            }
        }
        printf("IP is 0.0.0.0, retrying (%d/%d)...\n", i + 1, max_retry);
        usleep(interval_ms * 1000);
    }
    return -1;
}

static char* replace_rtsp_ip(const char* original_url, const char* new_ip, char* new_url, int new_url_len) {
    if (original_url == NULL || new_ip == NULL || new_url == NULL || new_url_len <= 0) {
        return NULL;
    }

    const char* rtsp_prefix = "rtsp://";
    size_t prefix_len = strlen(rtsp_prefix);
    if (strncmp(original_url, rtsp_prefix, prefix_len) != 0) {
        return NULL;
    }

    const char* ip_start = original_url + prefix_len;
    const char* ip_end = strpbrk(ip_start, ":/");
    if (ip_end == NULL) {
        ip_end = original_url + strlen(original_url);
    }

    size_t ip_len = ip_end - ip_start;
    size_t suffix_len = strlen(ip_end);
    size_t new_ip_len = strlen(new_ip);

    if (prefix_len + new_ip_len + suffix_len + 1 > new_url_len) {
        return NULL;
    }

    strncpy(new_url, rtsp_prefix, prefix_len);
    new_url[prefix_len] = '\0';

    strncat(new_url, new_ip, new_ip_len);
    strncat(new_url, ip_end, suffix_len);

    return new_url;
}

/* ── RTSP thread ─────────────────────────────────────────────────────── */

void MySmartIPC::RtspThreadMain() {
    const int RETRY_INTERVAL_MS = 3000;
    streaming_ready_ = false;

    auto* srv = new KdRtspServer();
    rtsp_server_ = srv;

    while (streaming_running_) {
        if (srv->Init(port_, nullptr) < 0) {
            printf("RTSP initialization failed, retrying in %d milliseconds...\n", RETRY_INTERVAL_MS);
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
            continue;
        }
        printf("rtsp server init ok\n");

        SessionAttr session_attr;
        session_attr.with_audio = true;
        session_attr.with_audio_backchannel = false;
        session_attr.with_video = true;

        if (input_config_.video_type == KdMediaVideoType::kVideoTypeH264) {
            session_attr.video_type = VideoType::kVideoTypeH264;
        } else if (input_config_.video_type == KdMediaVideoType::kVideoTypeH265) {
            session_attr.video_type = VideoType::kVideoTypeH265;
        } else {
            printf("Unsupported video codec type, retrying in %d milliseconds...\n", RETRY_INTERVAL_MS);
            srv->DeInit();
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
            continue;
        }

        if (srv->CreateSession(stream_url_, session_attr) < 0) {
            printf("RTSP session creation failed, retrying in %d milliseconds...\n", RETRY_INTERVAL_MS);
            srv->DeInit();
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
            continue;
        }

        srv->Start();
        streaming_ready_ = true;

        char ip[16];
        if (get_valid_ip_with_retry("u0", ip, sizeof(ip), 10, 1000) == 0) {
            const char* original_rtsp_url = srv->GetRtspUrl(stream_url_);
            if (original_rtsp_url == NULL) {
                printf("Failed to get original RTSP URL\n");
            } else {
                char new_rtsp_url[128] = {0};
                if (replace_rtsp_ip(original_rtsp_url, ip, new_rtsp_url, sizeof(new_rtsp_url)) != NULL) {
                    printf("RTSP service started successfully: %s\n", new_rtsp_url);
                } else {
                    printf("RTSP service started successfully (original URL): %s\n", original_rtsp_url);
                }
            }
        } else {
            printf("Failed to get local IP address\n");
            printf("RTSP service started successfully: %s\n", srv->GetRtspUrl(stream_url_));
        }

        while (streaming_running_ && started_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        srv->Stop();
        srv->DeInit();
        streaming_ready_ = false;
        printf("RTSP service stopped\n");

        if (streaming_running_) {
            std::unique_lock<std::mutex> lock(streaming_mutex_);
            streaming_cv_.wait(lock, [this]() { return !streaming_running_ || started_; });
        }
    }

    delete srv;
    rtsp_server_ = nullptr;
    printf("RTSP thread exited\n");
}

/* ── WebRTC HTTP signaling handler ─────────────────────────────────── */

void MySmartIPC::OnHttpRequest(const char* method, const char* path,
                               const char* body, int body_len,
                               http_response_t* response) {
    extern MySmartIPC* g_smart_ipc_instance;

    if (strcmp(method, "GET") == 0 && (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0)) {
        response->status = 200;
        response->content_type = "text/html; charset=utf-8";
        response->body = WEB_PAGE_HTML;
        response->body_len = strlen(WEB_PAGE_HTML);

    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/offer") == 0) {
        auto* ipc = g_smart_ipc_instance;
        if (!ipc) {
            response->status = 500;
            response->content_type = "text/plain";
            response->body = "No IPC instance";
            response->body_len = strlen(response->body);
            return;
        }

        /* Close existing connection before new offer.
         * Drain: wait for both worker threads to exit webrtc_pc_ internals.
         * close() sets pc->state=CLOSED so no new loop/send_video calls
         * will enter the dtls_srtp code path, but a call already past the
         * state check may still be using srtp_out. Wait until both
         * in-use flags are 0 — guaranteed within ~1ms. */
        if (ipc->webrtc_state_ == PEER_CONNECTION_COMPLETED || ipc->webrtc_state_ == PEER_CONNECTION_CONNECTED) {
            peer_connection_close(ipc->webrtc_pc_);
            ipc->webrtc_state_ = PEER_CONNECTION_CLOSED;
            while (ipc->peer_in_pc_ || ipc->venc_in_pc_) {
                usleep(1000);
            }
        }

        ipc->offer_ready_ = 0;
        if (ipc->offer_sdp_) {
            free(ipc->offer_sdp_);
            ipc->offer_sdp_ = nullptr;
        }

        peer_connection_create_offer(ipc->webrtc_pc_);

        pthread_mutex_lock(&ipc->offer_mutex_);
        while (!ipc->offer_ready_ && ipc->streaming_running_) {
            pthread_cond_wait(&ipc->offer_cond_, &ipc->offer_mutex_);
        }

        if (ipc->offer_sdp_) {
            response->status = 200;
            response->content_type = "application/sdp";
            response->body = ipc->offer_sdp_;
            response->body_len = strlen(ipc->offer_sdp_);
        } else {
            response->status = 500;
            response->content_type = "text/plain";
            response->body = "Failed to create offer";
            response->body_len = strlen(response->body);
        }
        pthread_mutex_unlock(&ipc->offer_mutex_);

    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/answer") == 0) {
        auto* ipc = g_smart_ipc_instance;
        if (!ipc || !body || body_len <= 0) {
            response->status = 400;
            response->content_type = "text/plain";
            response->body = "Missing SDP body";
            response->body_len = strlen(response->body);
            return;
        }

        char* answer_copy = (char*)calloc(1, body_len + 1);
        memcpy(answer_copy, body, body_len);
        answer_copy[body_len] = '\0';

        peer_connection_set_remote_description(ipc->webrtc_pc_, answer_copy, SDP_TYPE_ANSWER);

        response->status = 200;
        response->content_type = "text/plain";
        response->body = "OK";
        response->body_len = strlen(response->body);

        free(answer_copy);

    } else {
        response->status = 404;
        response->content_type = "text/plain";
        response->body = "Not Found";
        response->body_len = strlen(response->body);
    }
}

/* ── WebRTC thread: peer_connection_loop + HTTP signaling ──────────── */

void MySmartIPC::WebRtcThreadMain() {
    /* Initialize libpeer */
    PeerConfiguration config;
    config.datachannel = DATA_CHANNEL_NONE;
    config.video_codec = (input_config_.video_type == KdMediaVideoType::kVideoTypeH265) ? CODEC_H265 : CODEC_H264;
    config.audio_codec = CODEC_NONE;

    peer_init();
    webrtc_pc_ = peer_connection_create(&config);

    /* Register callbacks — libpeer does NOT pass userdata, use g_smart_ipc_instance */
    peer_connection_oniceconnectionstatechange(webrtc_pc_,
        [](PeerConnectionState state, void* /*data*/) {
            auto* ipc = g_smart_ipc_instance;
            printf("[WebRTC] State: %s\n", peer_connection_state_to_string(state));
            if (ipc) ipc->webrtc_state_ = state;
        });

    peer_connection_onicecandidate(webrtc_pc_,
        [](char* sdp, void* /*data*/) {
            auto* ipc = g_smart_ipc_instance;
            if (!ipc) return;
            pthread_mutex_lock(&ipc->offer_mutex_);
            if (ipc->offer_sdp_) { free(ipc->offer_sdp_); ipc->offer_sdp_ = nullptr; }
            ipc->offer_sdp_ = strdup(sdp);
            ipc->offer_ready_ = 1;
            pthread_cond_signal(&ipc->offer_cond_);
            pthread_mutex_unlock(&ipc->offer_mutex_);
        });

    /* Set global instance for HTTP handler */
    g_smart_ipc_instance = this;

    /* Start peer_connection_loop in a separate thread */
    webrtc_peer_thread_ = std::thread([this]() {
        while (streaming_running_) {
            peer_in_pc_ = 1;
            peer_connection_loop(webrtc_pc_);
            peer_in_pc_ = 0;
            usleep(1000);
        }
    });

    /* Start HTTP signaling server (blocks in accept loop until stopped) */
    streaming_ready_ = true;

    char webrtc_ip[16];
    if (get_valid_ip_with_retry("u0", webrtc_ip, sizeof(webrtc_ip), 10, 1000) == 0) {
        printf("[WebRTC] Open in browser: http://%s:%d\n", webrtc_ip, port_);
    } else {
        printf("[WebRTC] HTTP signaling server starting on port %d (IP unavailable)\n", port_);
    }

    http_server_start(port_, http_request_handler);

    /* Wait until started_ becomes true (main.cpp calls Start() after Init()),
     * then keep running until streaming_running_ becomes false or started_ drops. */
    {
        std::unique_lock<std::mutex> lock(streaming_mutex_);
        streaming_cv_.wait(lock, [this]() { return started_ || !streaming_running_; });
    }

    while (streaming_running_ && started_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    /* Stop */
    streaming_ready_ = false;
    streaming_running_ = false;
    http_server_stop();

    if (webrtc_peer_thread_.joinable()) {
        webrtc_peer_thread_.join();
    }

    if (webrtc_pc_) {
        peer_connection_destroy(webrtc_pc_);
        webrtc_pc_ = nullptr;
    }
    peer_deinit();

    if (sps_pps_buf_) {
        free(sps_pps_buf_);
        sps_pps_buf_ = nullptr;
    }
    if (offer_sdp_) {
        free(offer_sdp_);
        offer_sdp_ = nullptr;
    }

    printf("[WebRTC] Thread exited\n");
}

/* ── Init / DeInit / Start / Stop ──────────────────────────────────── */

int MySmartIPC::Init(const KdMediaInputConfig &config, const std::string &stream_url,
                     StreamingMode mode, int port) {
    // 初始化媒体部分
    feature_config_.enable_video_encoder = config.enable_video_encoding;
    feature_config_.on_venc_data = this;
    feature_config_.enable_ai_analysis = config.enable_ai_analysis;
    feature_config_.on_ai_frame_data = this;
    feature_config_.enable_render = config.enable_video_output;
    feature_config_.enable_audio_encoder = config.enable_capture_audio;
    feature_config_.on_aenc_data = this;

    if (0 != media_.configure_media_features(config, feature_config_)) {
        return -1;
    }

    // 保存配置
    input_config_ = config;
    stream_url_ = stream_url;
    streaming_mode_ = mode;
    port_ = port;

    // 初始化AI分析
    if (_ai_analyse_init() != 0) {
        DeInit();
        return -1;
    }

    // 启动流媒体线程（二选一）
    streaming_running_ = true;
    if (streaming_mode_ == StreamingMode::kRtsp) {
        streaming_thread_ = std::thread(&MySmartIPC::RtspThreadMain, this);
    } else {
        streaming_thread_ = std::thread(&MySmartIPC::WebRtcThreadMain, this);
    }

    return 0;
}

int MySmartIPC::DeInit() {
    Stop();

    // 停止流媒体线程
    streaming_running_ = false;
    streaming_cv_.notify_one();
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }

    // 清理媒体资源
    media_.destroy_media_features();

    // 清理AI资源
    if (face_detection_) {
        delete face_detection_;
        face_detection_ = nullptr;
    }
    if (ai_frame_vaddr_) {
        kd_mpi_sys_mmz_free(ai_frame_paddr_, ai_frame_vaddr_);
        ai_frame_vaddr_ = nullptr;
        ai_frame_paddr_ = 0;
    }

    return 0;
}

int MySmartIPC::Start() {
    ScopedTiming st = ScopedTiming("MySmartIPC::start", 1);
    if (started_) return 0;

    // 启动媒体服务
    media_.enable_media_features();
    started_ = true;

    // 通知流媒体线程可以启动服务
    streaming_cv_.notify_one();
    return 0;
}

int MySmartIPC::Stop() {
    if (!started_) return 0;

    // 停止媒体服务
    started_ = false;
    media_.disable_media_features();

    // 通知流媒体线程停止服务
    streaming_cv_.notify_one();
    return 0;
}

int MySmartIPC::_ai_analyse_init() {
    // 分配OSD帧缓存
    media_.osd_alloc_frame(&osd_vaddr_);

    // 分配AI输入帧内存
    size_t size = 3 * input_config_.ai_width * input_config_.ai_height; // RGB888格式
    ai_frame_size_ = size;
    int ret = kd_mpi_sys_mmz_alloc_cached(&ai_frame_paddr_, &ai_frame_vaddr_, "allocate", "anonymous", size);
    if (ret) {
        std::cerr << "内存分配失败: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    // 初始化人脸检测
    {
        ScopedTiming st("@@@@@@@@face detection init", 1);
        face_detection_ = new FaceDetection(
            input_config_.kmodel_file.c_str(),
            input_config_.obj_thresh,
            input_config_.nms_thresh,
            {3, input_config_.ai_height, input_config_.ai_width},
            reinterpret_cast<uintptr_t>(ai_frame_vaddr_),
            reinterpret_cast<uintptr_t>(ai_frame_paddr_),
            0
        );
    }

    return 0;
}
