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

MySmartIPC::MySmartIPC() : rtsp_port_(8554) {}

void MySmartIPC::OnAEncData(k_u32 chn_id, k_u8* pdata, size_t size, k_u64 time_stamp) {
    // 仅在RTSP就绪且整体启动后才推流
    if (started_ && rtsp_ready_) {
        rtsp_server_.SendAudioData(stream_url_, (const uint8_t*)pdata, size, time_stamp);
    }
}

static k_u64 get_ticks() {
    volatile k_u64 time_elapsed;
    __asm__ __volatile__(
        "rdtime %0"
        : "=r"(time_elapsed));
    return time_elapsed;
}

void MySmartIPC::OnVEncData(k_u32 chn_id, void *data, size_t size, k_venc_pack_type type, uint64_t timestamp) {
    // 仅在RTSP就绪且整体启动后才推流
    if (started_ && rtsp_ready_) {
        rtsp_server_.SendVideoData(stream_url_, (const uint8_t*)data, size, timestamp);
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
 * @param ifname 网络接口名称（如"eth0"、"wlan0"）
 * @param ip_str 输出缓冲区（至少16字节，用于存储"xxx.xxx.xxx.xxx"）
 * @param str_len 输出缓冲区长度（建议传入16）
 * @return 0：成功；-1：失败
 */
int get_interface_ip(const char *ifname, char *ip_str, int str_len) {
    int sock_get_ip;
    struct sockaddr_in *sin;
    struct ifreq ifr_ip;

    // 参数合法性校验
    if (ifname == NULL || ip_str == NULL || str_len < 16) {
        printf("Invalid parameters for get_interface_ip\n");
        return -1;
    }

    // 创建socket（用于ioctl调用，无需实际连接）
    if ((sock_get_ip = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        printf("Failed to create socket for getting IP\n");
        return -1;
    }

    // 初始化接口请求结构体
    memset(&ifr_ip, 0, sizeof(ifr_ip));
    strncpy(ifr_ip.ifr_name, ifname, sizeof(ifr_ip.ifr_name) - 1); // 避免越界

    // 获取接口IP地址
    if (ioctl(sock_get_ip, SIOCGIFADDR, &ifr_ip) < 0) {
        printf("ioctl(SIOCGIFADDR) failed for interface %s\n", ifname);
        close(sock_get_ip);
        return -1;
    }

    // 提取IP地址并转换为字符串
    sin = (struct sockaddr_in *)&ifr_ip.ifr_addr;
    strncpy(ip_str, inet_ntoa(sin->sin_addr), str_len - 1); // 确保不越界
    ip_str[str_len - 1] = '\0'; // 手动添加字符串结束符

    // 清理资源
    close(sock_get_ip);
    return 0;
}

// 带重试机制的IP获取（直到获取有效IP或超时）
static int get_valid_ip_with_retry(const char *ifname, char *ip_str, int str_len,
                           int max_retry, int interval_ms) {
    if (ifname == NULL || ip_str == NULL || str_len < 16 || max_retry <= 0 || interval_ms <= 0) {
        return -1;
    }

    for (int i = 0; i < max_retry; i++) {
        if (get_interface_ip(ifname, ip_str, str_len) == 0) {
            // 检查是否为有效IP（非0.0.0.0）
            if (strcmp(ip_str, "0.0.0.0") != 0) {
                return 0; // 成功获取有效IP
            }
        }

        // 未获取到有效IP，等待重试
        printf("IP is 0.0.0.0, retrying (%d/%d)...\n", i + 1, max_retry);
        usleep(interval_ms * 1000); // 毫秒转微秒
    }

    return -1; // 达到最大重试次数仍未获取有效IP
}

// 辅助函数：替换RTSP URL中的IP地址
static char* replace_rtsp_ip(const char* original_url, const char* new_ip, char* new_url, int new_url_len) {
    if (original_url == NULL || new_ip == NULL || new_url == NULL || new_url_len <= 0) {
        return NULL;
    }

    // 检查原始URL是否以"rtsp://"开头
    const char* rtsp_prefix = "rtsp://";
    size_t prefix_len = strlen(rtsp_prefix);
    if (strncmp(original_url, rtsp_prefix, prefix_len) != 0) {
        return NULL; // 不是合法的RTSP URL
    }

    // 定位原始IP的起始位置（跳过"rtsp://"）
    const char* ip_start = original_url + prefix_len;
    // 定位原始IP的结束位置（寻找第一个 ':' 或 '/'，即端口或路径的分隔符）
    const char* ip_end = strpbrk(ip_start, ":/");
    if (ip_end == NULL) {
        ip_end = original_url + strlen(original_url); // 若没有分隔符，IP到URL末尾
    }

    // 计算各部分长度
    size_t ip_len = ip_end - ip_start;          // 原始IP的长度
    size_t suffix_len = strlen(ip_end);         // IP后面的部分（端口+路径等）
    size_t new_ip_len = strlen(new_ip);         // 新IP的长度

    // 检查新URL缓冲区是否足够（前缀 + 新IP + 后缀 + 结束符）
    if (prefix_len + new_ip_len + suffix_len + 1 > new_url_len) {
        return NULL; // 缓冲区不足
    }

    // 拼接新URL：rtsp:// + 新IP + 后缀
    strncpy(new_url, rtsp_prefix, prefix_len);
    new_url[prefix_len] = '\0';

    strncat(new_url, new_ip, new_ip_len);
    strncat(new_url, ip_end, suffix_len);

    return new_url;
}

// RTSP线程主函数：包含初始化、启动和循环等待
void MySmartIPC::RtspThreadMain() {
    const int RETRY_INTERVAL_MS = 3000; // 重试间隔1秒
    rtsp_ready_ = false;

    // 线程循环：直到收到停止信号
    while (rtsp_running_) {
        // 初始化RTSP服务器
        if (rtsp_server_.Init(rtsp_port_, nullptr) < 0) {
            printf("RTSP initialization failed, retrying in %d milliseconds...\n", RETRY_INTERVAL_MS);
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
            continue;
        }
        else{
            printf("rtsp server init ok\n");
        }

        // 配置会话属性
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
            rtsp_server_.DeInit();
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
            continue;
        }

        // 创建RTSP会话
        if (rtsp_server_.CreateSession(stream_url_, session_attr) < 0) {
            printf("RTSP session creation failed, retrying in %d milliseconds...\n", RETRY_INTERVAL_MS);
            rtsp_server_.DeInit();
            std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
            continue;
        }

        // 启动RTSP服务
        rtsp_server_.Start();
        rtsp_ready_ = true; // 标记RTSP就绪

        //获取rtsp 地址
        char ip[16];
        if (get_valid_ip_with_retry("u0", ip, sizeof(ip), 10, 1000) == 0) {

            // 获取原始RTSP URL
            const char* original_rtsp_url = rtsp_server_.GetRtspUrl(stream_url_);
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
            // 若IP获取失败，直接输出原始URL
            printf("RTSP service started successfully: %s\n", rtsp_server_.GetRtspUrl(stream_url_));
        }

        // 等待停止信号（循环检测，避免阻塞线程退出）
        while (rtsp_running_ && started_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 停止RTSP服务
        rtsp_server_.Stop();
        rtsp_server_.DeInit();
        rtsp_ready_ = false;
        printf("RTSP service stopped\n");

        // 如果是临时停止（不是销毁），等待重启信号
        if (rtsp_running_) {
            std::unique_lock<std::mutex> lock(rtsp_mutex_);
            rtsp_cv_.wait(lock, [this]() { return !rtsp_running_ || started_; });
        }
    }

    printf("RTSP thread exited\n");
}

int MySmartIPC::Init(const KdMediaInputConfig &config, const std::string &stream_url, int port) {
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
    rtsp_port_ = port;

    // 初始化AI分析
    if (_ai_analyse_init() != 0) {
        DeInit();
        return -1;
    }

    // 启动RTSP线程（此时仅初始化线程，不启动服务）
    rtsp_running_ = true;
    rtsp_thread_ = std::thread(&MySmartIPC::RtspThreadMain, this);

    return 0;
}

int MySmartIPC::DeInit() {
    Stop();

    // 停止RTSP线程
    rtsp_running_ = false;
    rtsp_cv_.notify_one(); // 唤醒可能等待的线程
    if (rtsp_thread_.joinable()) {
        rtsp_thread_.join();
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

    // 通知RTSP线程可以启动服务
    rtsp_cv_.notify_one();
    return 0;
}

int MySmartIPC::Stop() {
    if (!started_) return 0;

    // 停止媒体服务
    started_ = false;
    media_.disable_media_features();

    // 通知RTSP线程停止服务
    rtsp_cv_.notify_one();
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