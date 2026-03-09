#include "mp4_data_source.h"
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#include "k_type.h"

MP4DataSource::MP4DataSource(){

}

MP4DataSource::~MP4DataSource() {
    close();
}

int MP4DataSource::open(const std::string& url, int fps,const DataSourceCallbacks& callbacks) {
    if (m_isOpened) {
        std::cerr << "MP4 data source already opened" << std::endl;
        return -1;
    }

    // 检查文件格式
    size_t len = url.length();
    if (len < 4 || url.substr(len - 4) != ".mp4") {
        std::cerr << "Invalid MP4 file: " << url << std::endl;
        return -1;
    }

    // 初始化MP4配置
    k_mp4_config_s mp4_config;
    memset(&mp4_config, 0, sizeof(mp4_config));
    mp4_config.config_type = K_MP4_CONFIG_DEMUXER;
    strncpy(mp4_config.demuxer_config.file_name, url.c_str(),
            sizeof(mp4_config.demuxer_config.file_name) - 1);
    mp4_config.muxer_config.fmp4_flag = 0;

    // 创建MP4解封装器
    k_s32 ret = kd_mp4_create(&m_mp4_handle, &mp4_config);
    if (ret < 0) {
        std::cerr << "Failed to create mp4 demuxer" << std::endl;
        return -1;
    }

    // 获取文件信息
    k_mp4_file_info_s file_info;
    memset(&file_info, 0, sizeof(file_info));
    ret = kd_mp4_get_file_info(m_mp4_handle, &file_info);
    if (ret < 0) {
        std::cerr << "Failed to get file info" << std::endl;
        kd_mp4_destroy(m_mp4_handle);
        m_mp4_handle = nullptr;
        return -1;
    }

    // 获取视频轨道信息（宽度、高度、编码格式）
    for (int i = 0; i < file_info.track_num; ++i) {
        k_mp4_track_info_s track_info;
        memset(&track_info, 0, sizeof(track_info));
        ret = kd_mp4_get_track_by_index(m_mp4_handle, i, &track_info);
        if (ret < 0) {
            std::cerr << "Failed to get track " << i << " info" << std::endl;
            continue;
        }

        if (track_info.track_type == K_MP4_STREAM_VIDEO) {
            m_width = track_info.video_info.width;
            m_height = track_info.video_info.height;
            m_videoCodecId = track_info.video_info.codec_id;
            break;
        }
    }

    if (m_width == 0 || m_height == 0) {
        std::cerr << "No valid video track found" << std::endl;
        kd_mp4_destroy(m_mp4_handle);
        m_mp4_handle = nullptr;
        return -1;
    }

    // 保存回调和用户数据
    m_url = url;
    m_callback = callbacks.stream_cb;
    m_userData = callbacks.user_data;
    m_isOpened = true;
    if (fps <= 0){
        fps = 30;
    }
    m_output_fps = fps;

    return 0;
}

void MP4DataSource::close() {
    if (!m_isOpened) return;

    // 销毁MP4句柄
    if (m_mp4_handle != nullptr) {
        kd_mp4_destroy(m_mp4_handle);
        m_mp4_handle = nullptr;
    }

    // 重置成员变量
    m_isOpened = false;
    m_callback = nullptr;
    m_userData = nullptr;
    m_width = 0;
    m_height = 0;
    m_videoCodecId = -1;
}

int MP4DataSource::start()
{
    // 启动工作线程
    m_isRunning = true;

    int ret = pthread_create(&m_workerThread, nullptr,
                            &MP4DataSource::_worker_thread_entry, this);

    if (ret != 0) {
        printf("pthread_create failed, ret=%d\n", ret);
        return -1;
    }

    return 0;
}

int MP4DataSource::stop()
{
    // 停止工作线程
    m_isRunning = false;
    pthread_join(m_workerThread, nullptr);
    return 0;
}

int MP4DataSource::get_width() const {
    return m_width;
}

int MP4DataSource::get_height() const {
    return m_height;
}

EncType MP4DataSource::get_stream_type() const {
    return (EncType)m_videoCodecId;
}

// 辅助函数：获取当前微秒时间
static int64_t get_current_time_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + (int64_t)tv.tv_usec;
}

void* MP4DataSource::_worker_thread_entry(void* arg)
{
    MP4DataSource* self = static_cast<MP4DataSource*>(arg);
    self->working();
    return nullptr;
}

void MP4DataSource::working() {
    k_mp4_frame_data_s frame_data;
    bool firstVideoFrame = true;
    int64_t lastVideoOutputTimeUs = 0;

    while (m_isRunning) {
        memset(&frame_data, 0, sizeof(frame_data));
        k_s32 ret = kd_mp4_get_frame(m_mp4_handle, &frame_data);
        if (ret < 0) {
            std::cerr << "Failed to get frame data" << std::endl;
            break;
        }

        if (frame_data.eof) {
            std::cout << "MP4 demux finished" << std::endl;
            memset(&m_encstream, 0, sizeof(m_encstream));
            m_callback(m_encstream, m_userData, true);
            break;
        }

        bool isVideo = (frame_data.data_length > 0 &&
                       (frame_data.codec_id == K_MP4_CODEC_ID_H264 ||
                        frame_data.codec_id == K_MP4_CODEC_ID_H265));

        if (m_callback) {
            if (isVideo) {
                // --- 视频帧：限速 ---
                int64_t nowUs = get_current_time_us();
                int fps = (m_output_fps > 0) ? m_output_fps : 30;
                int64_t minIntervalUs = 1000000LL / fps;

                if (!firstVideoFrame) {
                    int64_t elapsedUs = nowUs - lastVideoOutputTimeUs;
                    if (elapsedUs < minIntervalUs) {
                        int64_t delayUs = minIntervalUs - elapsedUs;
                        if (delayUs > 0 && delayUs < 500000) {
                            usleep((useconds_t)delayUs);
                            // 注意：usleep 后时间已推进，但 lastVideoOutputTimeUs 仍按“目标时间”更新更稳
                            nowUs = get_current_time_us(); // 可选：重新获取时间
                        }
                    }
                }

                // 更新时间戳
                lastVideoOutputTimeUs = nowUs;
                firstVideoFrame = false;

                // 回调视频帧
                m_encstream.data = frame_data.data;
                m_encstream.size = frame_data.data_length;
                m_encstream.pts = frame_data.time_stamp;
                m_encstream.enctype = (EncType)frame_data.codec_id;
                m_callback(m_encstream, m_userData, false);

            } else {
                // --- 非视频帧（如音频）：直接回调，不限速 ---
                // 注意：你的原始代码没有处理音频回调，这里假设你可能需要
                // 如果不需要音频，可直接跳过
                m_encstream.data = frame_data.data;
                m_encstream.size = frame_data.data_length;
                m_encstream.pts = frame_data.time_stamp;
                m_encstream.enctype = (EncType)frame_data.codec_id; // 注意：音频 codec_id 不是 H264/H265
                m_callback(m_encstream, m_userData, false);
            }
        }
    }

    if (!frame_data.eof) {
        memset(&m_encstream, 0, sizeof(m_encstream));
        m_callback(m_encstream, m_userData, true);
    }
}
