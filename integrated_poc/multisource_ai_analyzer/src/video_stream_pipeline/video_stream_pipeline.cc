#include "video_stream_pipeline.h"
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <memory>
#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <semaphore.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>

#include "mpp_pipeline.h"
#include "mp4_data_source.h"
#include "rtsp_data_source.h"
#include "uvc_data_source.h"
#include "mpi_sys_api.h"


// Internal implementation class (PIMPL)
struct k_video_frame_info_ex {
    k_video_frame_info frame_info;
    unsigned int frame_id;
};

class VideoStreamPipelineImpl {
public:
    int out_width_;
    int out_height_;
    int dec_width_;
    int dec_height_;
    int rtsp_rtp_over_tcp;
    IDataSource* data_source_ = nullptr;
    MppPipeline mpp_pipeline_;
    std::string video_path_;
    DisplayType display_type_;
    void* osd_vaddr_{nullptr};
    std::atomic<bool> m_isRunning{false};
    std::queue<k_video_frame_info_ex> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    int stream_fps_;
    int ai_fps_;
    k_video_frame_info dump_res_;
    int frame_index_{0};
    bool is_uvc_{false};

    // Callbacks as member functions (or static with user_data)
    static void on_stream_cb(const EncStream& stream, void* user_data, bool is_end);
    static void on_mpp_stream_cb(const k_vdec_stream& stream, void* user_data, bool is_end);
    static void on_rgb_frame_cb(const k_video_frame_info& frame, void* user_data);

    void _process_rgb_frame(const k_video_frame_info& frame);
};

// Implement VideoStreamPipeline using pimpl
VideoStreamPipeline::VideoStreamPipeline(const std::string& video_path,
                                         DisplayType display_type,
                                         int out_width,
                                         int out_height,
                                         int stream_fps,
                                         int ai_fps,
                                         int rtsp_rtp_over_tcp)
    : pimpl_(new VideoStreamPipelineImpl) {
    pimpl_->video_path_ = video_path;
    pimpl_->out_width_ = out_width;
    pimpl_->out_height_ = out_height;
    pimpl_->stream_fps_ = stream_fps;
    pimpl_->ai_fps_ = ai_fps;
    pimpl_->display_type_ = display_type;
    pimpl_->rtsp_rtp_over_tcp = rtsp_rtp_over_tcp;
}

VideoStreamPipeline::~VideoStreamPipeline() {
    delete pimpl_;
}

static bool is_rtsp_stream(const std::string& path) {
    return path.rfind("rtsp://", 0) == 0; // 检查是否以 "rtsp://" 开头
}

static bool is_mp4_file_by_extension(const std::string& filepath) {
    if (filepath.size() < 4) return false;
    std::string ext = filepath.substr(filepath.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "mp4";
}

int VideoStreamPipeline::Create() {
    auto* pl = pimpl_;

    bool enable_dsl;
    bool rotation_90 = false;
    int stream_width;
    int stream_height;
    DataSourceCallbacks cbs;

    if (is_mp4_file_by_extension(pl->video_path_)) {
        pl->data_source_ = new MP4DataSource();
    } else if (pl->video_path_.rfind("rtsp://", 0) == 0) {
        pl->data_source_ = new RTSPDataSource(pimpl_->rtsp_rtp_over_tcp);
    }
    else if (pl->video_path_.rfind("uvc", 0) == 0) {
        pl->data_source_ = new UVCDataSource(pimpl_->out_width_,pimpl_->out_height_);
        pl->is_uvc_ = true;
    }
    else {
        printf("Invalid video path (not MP4 or RTSP or realtime or UVC): %s\n", pl->video_path_.c_str());
        return -1;
    }

    if (!pl->data_source_) {
        std::cerr << "Failed to create DataSource instance" << std::endl;
        return -1;
    }

    cbs.user_data = pl;
    if (pl->is_uvc_){
        cbs.mpp_cb = VideoStreamPipelineImpl::on_mpp_stream_cb;
    }
    else{
        cbs.stream_cb = VideoStreamPipelineImpl::on_stream_cb;
    }

    int open_success = pl->data_source_->open(pl->video_path_, pl->stream_fps_, cbs);
    if (open_success) {
        std::cerr << "Failed to open url: " << pl->video_path_ << std::endl;
        return -1;
    }

    stream_width = pl->data_source_->get_width();
    stream_height = pl->data_source_->get_height();
    EncType enc_type = pl->data_source_->get_stream_type();

    if (stream_width <= 0 || stream_height <= 0) {
        std::cerr << "Invalid video resolution: " << stream_width << "x" << stream_height << std::endl;
        return -1;
    } else {
        std::cout << "stream video resolution: " << stream_width << "x" << stream_height << ",enc type:" << enc_type << std::endl;
    }

    MppVoType vo_type = EM_VO_HDMI;
    if (pl->display_type_ == DISPLAY_HDMI) {
        vo_type = EM_VO_HDMI;
    } else if (pl->display_type_ == DISPLAY_LCD) {
        vo_type = EM_VO_LCD;
    }
    printf("vo_type: %d (%s)\n", vo_type, (vo_type == EM_VO_HDMI) ? "hdmi" : (vo_type == EM_VO_LCD) ? "lcd" : "unknown");

    if (stream_width >= pl->out_width_ && stream_height >= pl->out_height_) {
        pl->dec_width_ = pl->out_width_;
        pl->dec_height_ = pl->out_height_;
        enable_dsl = true;
    } else {
        pl->dec_width_ = stream_width;
        pl->dec_height_ = stream_height;
        enable_dsl = false;
    }

    if (vo_type == EM_VO_LCD) {
        rotation_90 = true;
    }

    if (pl->mpp_pipeline_.init(pl->dec_width_, pl->dec_height_, enable_dsl, rotation_90, enc_type, vo_type) != 0) {
        std::cerr << "Failed to initialize MppPipeline" << std::endl;
    }

    pl->mpp_pipeline_.set_rgb_callback(VideoStreamPipelineImpl::on_rgb_frame_cb, pl);
    pl->mpp_pipeline_._osd_alloc_frame(&pl->osd_vaddr_);
    pl->m_isRunning.store(true);

    if (pl->mpp_pipeline_.start() != 0) {
        std::cerr << "Failed to start MppPipeline" << std::endl;
        pl->m_isRunning.store(false);
        return -1;
    }

    pl->data_source_->start();
    return 0;
}


void VideoStreamPipeline::GetFrameSize(int &width, int &height) {
    width = pimpl_->dec_width_;
    height = pimpl_->dec_height_;
}

bool VideoStreamPipeline::IsFinished() {
    return !pimpl_->m_isRunning.load();
}

int VideoStreamPipeline::GetFrame(DumpRes &dump_res) {
    auto* pl = pimpl_;
    bool bfind = false;
    std::unique_lock<std::mutex> lock(pl->queue_mutex_);
    const auto timeout_duration = std::chrono::milliseconds(1000);
    bool has_frame_or_stopped = pl->queue_cv_.wait_for(
        lock, timeout_duration,
        [pl]() { return !pl->frame_queue_.empty() || !pl->m_isRunning.load(); }
    );

    if (!has_frame_or_stopped) {
        return -1;
    }

    if (!pl->m_isRunning.load()) {
        while (!pl->frame_queue_.empty()) {
            auto& front_frame = pl->frame_queue_.front();
            pl->frame_queue_.pop();

            pl->mpp_pipeline_._release_rgb_frame(&front_frame.frame_info);
        }
        return -1;
    }

    if (pl->ai_fps_ <= 0){
        //Keep only the latest frame
        while (pl->frame_queue_.size() > 1) {
            auto& old_frame = pl->frame_queue_.front();
            pl->frame_queue_.pop();
            pl->mpp_pipeline_._release_rgb_frame(&old_frame.frame_info);
        }
    }
    else{
        int skip_interval = (pl->ai_fps_ > 0 && pl->stream_fps_ >= pl->ai_fps_)
        ? static_cast<int>(std::round(static_cast<double>(pl->stream_fps_) / pl->ai_fps_))
        : 1;
        while (!pl->frame_queue_.empty()) {
            auto& old_frame = pl->frame_queue_.front();

            if (( old_frame.frame_id - 1) % skip_interval != 0){
                pl->frame_queue_.pop();
                pl->mpp_pipeline_._release_rgb_frame(&old_frame.frame_info);
            }
            else
            {
                bfind = true;
                break;
            }
        }
        if (!bfind)
        {
            return -1;
        }
    }

    // {
    //     printf("%s frame id:%d\n",__func__,pl->frame_queue_.front().frame_id);
    // }
    pl->dump_res_ = pl->frame_queue_.front().frame_info;
    dump_res.virt_addr = reinterpret_cast<uintptr_t>(
        kd_mpi_sys_mmap(pl->dump_res_.v_frame.phys_addr[0],
                        3 * pl->dump_res_.v_frame.width * pl->dump_res_.v_frame.height)
    );
    dump_res.phy_addr = reinterpret_cast<uintptr_t>(pl->dump_res_.v_frame.phys_addr[0]);
    pl->frame_queue_.pop();
    return 0;
}

int VideoStreamPipeline::ReleaseFrame(DumpRes &dump_res) {
    auto* pl = pimpl_;

    kd_mpi_sys_munmap(reinterpret_cast<void*>(dump_res.virt_addr),
                      3 * pl->dump_res_.v_frame.width * pl->dump_res_.v_frame.height);

    pl->mpp_pipeline_._release_rgb_frame(&pl->dump_res_);

    return 0;
}

int VideoStreamPipeline::InsertFrame(void* osd_data) {
    if (pimpl_->osd_vaddr_ != nullptr) {
        pimpl_->mpp_pipeline_._osd_draw_frame(osd_data);
    }
    return 0;
}

int VideoStreamPipeline::Destroy() {
    auto* pl = pimpl_;
    pl->mpp_pipeline_.set_rgb_callback(nullptr, nullptr);
    sleep(1);

    {
        std::unique_lock<std::mutex> lock(pl->queue_mutex_);
        while (!pl->frame_queue_.empty()) {
            auto frame = pl->frame_queue_.front();
            pl->frame_queue_.pop();

            pl->mpp_pipeline_._release_rgb_frame(&frame.frame_info);
        }
    }

    if (pl->data_source_) {
        pl->data_source_->stop();
        pl->data_source_->close();
        delete pl->data_source_;
        pl->data_source_ = nullptr;
    }

    pl->mpp_pipeline_.stop();
    pl->mpp_pipeline_.deinit();
    pl->frame_index_ = 0;

    return 0;
}

// --- Callback implementations in impl ---

void VideoStreamPipelineImpl::on_stream_cb(const EncStream& stream, void* user_data, bool is_end) {
    if (!user_data) return;
    auto* pl = static_cast<VideoStreamPipelineImpl*>(user_data);
    if (!is_end) {
        pl->mpp_pipeline_.decode_stream(stream);
    } else {
        usleep(1000*500);
        printf("%s stream is end\n",__func__);
        pl->m_isRunning.store(false);
    }
}

void VideoStreamPipelineImpl::on_mpp_stream_cb(const k_vdec_stream& stream, void* user_data, bool is_end) {
    if (!user_data) return;
    auto* pl = static_cast<VideoStreamPipelineImpl*>(user_data);
    if (!is_end) {
        pl->mpp_pipeline_.decode_stream(stream);
    } else {
        usleep(1000*500);
        printf("%s stream is end\n",__func__);
        pl->m_isRunning.store(false);
    }
}

void VideoStreamPipelineImpl::on_rgb_frame_cb(const k_video_frame_info& frame, void* user_data) {
    if (!user_data) return;
    auto* pl = static_cast<VideoStreamPipelineImpl*>(user_data);
    if (!pl->m_isRunning.load()) {
        pl->mpp_pipeline_._release_rgb_frame(const_cast<k_video_frame_info*>(&frame));
        return;
    }

    pl->_process_rgb_frame(frame);
}

void VideoStreamPipelineImpl::_process_rgb_frame(const k_video_frame_info& frame) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    frame_index_++;
    // Avoid concurrent calls to release from multiple threads without guaranteeing the release order.
    // if (!frame_queue_.empty()) {
    //     auto& old_frame = frame_queue_.front();
    //     frame_queue_.pop();
    //     mpp_pipeline_._release_rgb_frame(&old_frame.frame_info);
    // }
    k_video_frame_info_ex frame_ex;
    frame_ex.frame_info = frame;
    frame_ex.frame_id = frame_index_;
    frame_queue_.push(frame_ex);
    queue_cv_.notify_one();
}