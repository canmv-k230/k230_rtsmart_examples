#include "smart_ipc.h"
#include <unistd.h>
#include "mpi_sys_api.h"
#include "k_vb_comm.h"
#include "scoped_timing.hpp"

MySmartIPC::MySmartIPC()
{

}

void MySmartIPC::OnAEncData(k_u32 chn_id, k_u8*pdata,size_t size,k_u64 time_stamp) {
    if (started_) {
        rtsp_server_.SendAudioData(stream_url_, (const uint8_t*)pdata, size, time_stamp);
    }
}

void MySmartIPC::OnVEncData(k_u32 chn_id, void *data, size_t size, k_venc_pack_type type,uint64_t timestamp)
{
    if (started_) {
        rtsp_server_.SendVideoData(stream_url_, (const uint8_t*)data, size, timestamp);
    }
}

void MySmartIPC::OnAIFrameData(k_u32 chn_id, k_video_frame_info*frame_info)
{
    if (!started_) return;

    ScopedTiming st("ai total time", 0);
    //copy AI frame(ai_frame_vaddr_) data from vicap chn to this memory, then do AI analysis
    {
        ScopedTiming st("isp copy", 0);
        auto vbvaddr = kd_mpi_sys_mmap_cached(frame_info->v_frame.phys_addr[0], ai_frame_size_);
        memcpy(ai_frame_vaddr_, (void *)vbvaddr, ai_frame_size_);  // 这里以后可以去掉，不用copy
        kd_mpi_sys_munmap(vbvaddr, ai_frame_size_);
    }

    // ai analyse to get detect result
    detect_result_.clear();
    face_detection_->pre_process();
    face_detection_->inference();
    // 旋转后图像
    face_detection_->post_process({input_config_.ai_width, input_config_.ai_height}, detect_result_);

    //copy detect result to osd frame
    cv::Mat osd_frame(input_config_.osd_height, input_config_.osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    {
        ScopedTiming st("osd draw", 0);
        if (input_config_.vo_connect_type == LT9611_MIPI_4LAN_1920X1080_30FPS)
        {
            face_detection_->draw_result(osd_frame,detect_result_,false);
        }
        else if (input_config_.vo_connect_type == ST7701_V1_MIPI_2LAN_480X800_30FPS)
        {
            cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
            face_detection_->draw_result(osd_frame,detect_result_,false);
            cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
        }
    }

    //copy osd frame data to osd_vaddr_,for osd draw
    {
        ScopedTiming st("osd copy", 0);
        memcpy(osd_vaddr_, osd_frame.data, input_config_.osd_width * input_config_.osd_height * 4);
        // 显示通道插入帧
        media_.osd_draw_frame();
    }
}

int MySmartIPC::Init(const KdMediaInputConfig &config, const std::string &stream_url/*= "test"*/, int port/*= 8554*/)
{
    ScopedTiming st = ScopedTiming("MySmartIPC::Init", 1);
    //init rtsp server
    input_config_ = config;
    if (rtsp_server_.Init(port, nullptr) < 0) {
        return -1;
    }
    // enable audio-track
    SessionAttr session_attr;
    session_attr.with_audio = true;
    session_attr.with_audio_backchannel = false;
    session_attr.with_video = true;

    if (config.video_type == KdMediaVideoType::kVideoTypeH264) {
        session_attr.video_type = VideoType::kVideoTypeH264;
    } else if (config.video_type == KdMediaVideoType::kVideoTypeH265) {
        session_attr.video_type = VideoType::kVideoTypeH265;
    } else {
        printf("video codec type not supported yet\n");
        return -1;
    }

    if (rtsp_server_.CreateSession(stream_url, session_attr) < 0) {
        return -1;
    }
    stream_url_ = stream_url;

    //init media
    feature_config_.enable_video_encoder = true;
    feature_config_.on_venc_data = this;
    feature_config_.enable_ai_analysis = true;
    feature_config_.on_ai_frame_data = this;
    feature_config_.enable_render = true;
    feature_config_.enable_audio_encoder = true;
    feature_config_.on_aenc_data = this;

    media_.configure_media_features(config, feature_config_);

    //init ai analyse
    _ai_analyse_init();

    return 0;
}

int MySmartIPC::DeInit()
{
    Stop();
    media_.destroy_media_features();
    rtsp_server_.DeInit();
    return 0;
}

int MySmartIPC::Start()
{
    ScopedTiming st = ScopedTiming("MySmartIPC::start", 1);
    if(started_) return 0;
    media_.enable_media_features();
    rtsp_server_.Start();
    started_ = true;
    return 0;
}

int MySmartIPC::Stop()
{
    if (!started_) return 0;
    rtsp_server_.Stop();
    started_ = false;
    media_.disable_media_features();
    return 0;
}


int MySmartIPC::_ai_analyse_init()
{
    //alloc one osd frame for draw osd
    media_.osd_alloc_frame(&osd_vaddr_);

    // Allocate memory for AI input frame to copy AI frame data from ISP to this memory
    size_t size = 3 * input_config_.ai_width * input_config_.ai_height;//3 for rgb888p
    ai_frame_size_ = size;
    int ret = kd_mpi_sys_mmz_alloc_cached(&ai_frame_paddr_, &ai_frame_vaddr_, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    {
        ScopedTiming st("@@@@@@@@face detection init", 1);
        //init face detection
        face_detection_ = new FaceDetection(input_config_.kmodel_file.c_str(),input_config_.obj_thresh, input_config_.nms_thresh,{3, input_config_.ai_height,input_config_.ai_width},reinterpret_cast<uintptr_t>(ai_frame_vaddr_), reinterpret_cast<uintptr_t>(ai_frame_paddr_),0);
    }

    return 0;
}