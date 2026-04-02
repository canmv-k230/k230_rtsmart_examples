#include "encoded_rtsp_server.h"
#include <fcntl.h>
#include <cstring>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic>

#include "mpi_sys_api.h"
#include "mpi_venc_api.h"
#include "k_vb_comm.h"
#include "mpi_vb_api.h"

EncodedRtspServer::EncodedRtspServer()
{

}

EncodedRtspServer::~EncodedRtspServer()
{

}

int EncodedRtspServer::init(int width, int height, int bitrate_kbps, const std::string &stream_url /*= "test"*/, int port /*= 8554*/)
{
    k_s32 ret ;
    rtsp_port_ = port;
    stream_url_ = stream_url;
    width_ = width;
    height_ = height;
    bitrate_kbps_ = bitrate_kbps;

    ret = kd_mpi_venc_request_chn(&venc_chn_id_);
    if (ret != 0) {
        printf("Failed to request venc channel, ret=%d\n", ret);
        return ret;
    }

    if (_init_venc() < 0) {
        printf("Failed to init venc\n");
        return -1;
    }

    if (_init_rtsp_server() < 0) {
        printf("Failed to init rtsp server\n");
        return -1;
    }

    return 0;
}

int EncodedRtspServer::start()
{
    k_s32 ret ;
    rtsp_server_.Start();

    // venc start
    ret = kd_mpi_venc_start_chn(venc_chn_id_);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_venc_start_chn failed:0x%x\n", ret);
        return -1;
    }

    if (!start_get_video_stream_)
    {
        start_get_video_stream_ = true;
        pthread_create(&venc_tid_, NULL, venc_stream_thread, this);
    }

    return 0;
}

int EncodedRtspServer::stop()
{
    unbind_vi_chn();

    if (start_get_video_stream_)
    {
        start_get_video_stream_ = false;
        pthread_join(venc_tid_, NULL);
    }

    // stop  encoder channel
    kd_mpi_venc_stop_chn(venc_chn_id_);

    rtsp_server_.Stop();
    return 0;
}

int EncodedRtspServer::deinit()
{
    _deinit_rtsp_server();
    _deinit_venc();

    return 0;
}

int EncodedRtspServer::send_raw_frame(k_video_frame_info *frame, k_s32 milli_sec)
{
    return kd_mpi_venc_send_frame(venc_chn_id_, frame,milli_sec);
}

int EncodedRtspServer::_init_venc()
{
    k_s32 ret;
    k_venc_chn_attr chn_attr;
    k_u32 venc_chn_id = venc_chn_id_;

    venc_attach_pool_id_ = _venc_vb_create_pool();
    if (venc_attach_pool_id_ == VB_INVALID_POOLID){
        printf("%s _venc_vb_create_pool failed\n",__func__);
        return -1;
    }
    kd_mpi_venc_attach_vb_pool(venc_chn_id_,venc_attach_pool_id_);

    memset(&chn_attr, 0, sizeof(chn_attr));
    k_u64 stream_size = width_ * height_ / 2;
    chn_attr.venc_attr.pic_width = width_;
    chn_attr.venc_attr.pic_height = height_;
    chn_attr.rc_attr.rc_mode = K_VENC_RC_MODE_CBR;
    chn_attr.rc_attr.cbr.src_frame_rate = 30;
    chn_attr.rc_attr.cbr.dst_frame_rate = 30;
    chn_attr.rc_attr.cbr.bit_rate = bitrate_kbps_;
    chn_attr.venc_attr.type = K_PT_H264;
    chn_attr.venc_attr.profile = VENC_PROFILE_H264_MAIN;

    ret = kd_mpi_venc_create_chn(venc_chn_id, &chn_attr);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_venc_create_chn failed:0x%x\n", ret);
        return -1;
    }

    ret = kd_mpi_venc_enable_idr(venc_chn_id, K_TRUE);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_venc_enable_idr failed:0x%x\n", ret);
        return -1;
    }

    if (rotation_90_ != 0){
        k_rotation rotaion = rotation_90_ ? K_VPU_ROTATION_90 : K_VPU_ROTATION_0;
        ret = kd_mpi_venc_set_rotation(venc_chn_id, rotaion);
        if (ret != K_SUCCESS)
        {
            printf("kd_mpi_venc_set_rotation failed:0x%x\n", ret);
            return -1;
        }
    }

    return 0;
}

int EncodedRtspServer::_deinit_venc()
{
    k_s32 ret = -1;

    ret = kd_mpi_venc_detach_vb_pool(venc_chn_id_);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_venc_detach_vb_pool failed:0x%x\n", ret);
    }

    ret = kd_mpi_venc_destroy_chn(venc_chn_id_);
    if (ret != K_SUCCESS)
    {
        printf("kd_mpi_venc_destroy_chn failed:0x%x\n", ret);
        return -1;
    }

    kd_mpi_venc_release_chn(venc_chn_id_);

    return 0;
}

int EncodedRtspServer::_init_rtsp_server()
{
    if (rtsp_server_.Init(rtsp_port_, nullptr) < 0) {
        printf("RTSP initialization failed\n");
        return -1;
    }

    // 配置会话属性
    SessionAttr session_attr;
    session_attr.with_audio = true;
    session_attr.with_audio_backchannel = false;
    session_attr.with_video = true;
    session_attr.video_type = VideoType::kVideoTypeH264;

    // 创建RTSP会话
    if (rtsp_server_.CreateSession(stream_url_, session_attr) < 0) {
        printf("RTSP session creation failed\n");
        return -1;
    }

    return 0;
}

int EncodedRtspServer::_deinit_rtsp_server()
{
    rtsp_server_.DeInit();
    return 0;
}

k_u32 EncodedRtspServer::_venc_vb_create_pool()
{
    k_u32 private_pool_id;
    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));
    k_u64 stream_size = width_ * height_ / 2;
    pool_config.blk_cnt = 4;
    pool_config.blk_size =  ((stream_size + 0xfff) & ~0xfff);
    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    private_pool_id = kd_mpi_vb_create_pool(&pool_config);
    printf("%s poolid %d\n", __func__,private_pool_id);

    return private_pool_id;
}

char* EncodedRtspServer::get_rtsp_url()
{
    return rtsp_server_.GetRtspUrl(stream_url_);
}

// void *EncodedRtspServer::venc_stream_thread(void *arg)
// {
//     EncodedRtspServer *pthis = (EncodedRtspServer*)arg;
//     k_u32 venc_chn = pthis->venc_chn_id_;
//     k_s32 ret;

//     k_venc_stream output;
//     k_venc_chn_status status;
//     k_u8 *pData;
//     while (pthis->start_get_video_stream_)
//     {
//         kd_mpi_venc_query_status(venc_chn, &status);

//         if (status.cur_packs > 0)
//             output.pack_cnt = status.cur_packs;
//         else
//             output.pack_cnt = 1;

//         output.pack = (k_venc_pack *)malloc(sizeof(k_venc_pack) * output.pack_cnt);

//         ret = kd_mpi_venc_get_stream(venc_chn, &output, 1000);
//         if (ret != K_SUCCESS){
//             free(output.pack);
//             continue;
//         }

//         for (int i = 0; i < output.pack_cnt; i++)
//         {
//             pData = (k_u8 *)kd_mpi_sys_mmap(output.pack[i].phys_addr, output.pack[i].len);

//             pthis->_do_venc_stream(venc_chn, pData, output.pack[i].len, output.pack[i].type, output.pack[i].pts);

//             kd_mpi_sys_munmap(pData, output.pack[i].len);
//         }

//         kd_mpi_venc_release_stream(venc_chn, &output);

//         free(output.pack);
//     }

//     //Empty all data in the encoder.
//     while(1)
//     {
//         kd_mpi_venc_query_status(venc_chn, &status);

//         if (status.cur_packs > 0)
//             output.pack_cnt = status.cur_packs;
//         else
//             break;

//         output.pack = (k_venc_pack *)malloc(sizeof(k_venc_pack) * output.pack_cnt);
//         ret = kd_mpi_venc_get_stream(venc_chn, &output, 1000);
//         if (ret != K_SUCCESS){
//             free(output.pack);
//             break;
//         }

//         kd_mpi_venc_release_stream(venc_chn, &output);
//         free(output.pack);
//     }

//     return NULL;
// }
void *EncodedRtspServer::venc_stream_thread(void *arg)
{
    EncodedRtspServer *pthis = (EncodedRtspServer*)arg;
    k_u32 venc_chn = pthis->venc_chn_id_;
    k_s32 ret;

    k_venc_stream output;
    k_venc_chn_status status;
    k_u8 *pData = nullptr;
    
    while (pthis->start_get_video_stream_)
    {
        memset(&output, 0, sizeof(output));  // 初始化output，避免野指针
        kd_mpi_venc_query_status(venc_chn, &status);

        output.pack_cnt = status.cur_packs > 0 ? status.cur_packs : 1;
        output.pack = (k_venc_pack *)malloc(sizeof(k_venc_pack) * output.pack_cnt);
        if (!output.pack) {  // 内存分配失败保护
            usleep(10000);  // 休眠10ms重试
            continue;
        }

        ret = kd_mpi_venc_get_stream(venc_chn, &output, 1000);
        if (ret != K_SUCCESS){
            free(output.pack);
            output.pack = nullptr;
            usleep(1000);  // 缩短重试间隔，避免空等
            continue;
        }

        for (int i = 0; i < output.pack_cnt; i++)
        {
            pData = (k_u8 *)kd_mpi_sys_mmap(output.pack[i].phys_addr, output.pack[i].len);
            if (pData) {  // mmap成功才处理
                pthis->_do_venc_stream(venc_chn, pData, output.pack[i].len, output.pack[i].type, output.pack[i].pts);
                kd_mpi_sys_munmap(pData, output.pack[i].len);
                pData = nullptr;
            }
        }

        kd_mpi_venc_release_stream(venc_chn, &output);
        free(output.pack);
        output.pack = nullptr;
    }

    // 清空剩余数据（优化：增加防泄漏）
    memset(&output, 0, sizeof(output));
    while(1)
    {
        kd_mpi_venc_query_status(venc_chn, &status);
        if (status.cur_packs <= 0) break;

        output.pack_cnt = status.cur_packs;
        output.pack = (k_venc_pack *)malloc(sizeof(k_venc_pack) * output.pack_cnt);
        if (!output.pack) break;

        ret = kd_mpi_venc_get_stream(venc_chn, &output, 1000);
        if (ret != K_SUCCESS){
            free(output.pack);
            break;
        }

        kd_mpi_venc_release_stream(venc_chn, &output);
        free(output.pack);
        output.pack = nullptr;
    }

    return NULL;
}
int EncodedRtspServer::_do_venc_stream(k_u32 chn_id, unsigned char *data, size_t size, k_venc_pack_type type, uint64_t timestamp)
{
    return  rtsp_server_.SendVideoData(stream_url_, (const uint8_t*)data, size, timestamp);
}

int EncodedRtspServer::bind_vi_chn(k_u32 vi_dev_id, k_u32 vi_chn_id)
{
    if (binded_vi_) {
        printf("VI channel already binded to encoder\n");
        return -1;
    }

    k_mpp_chn vi_mpp_chn;
    k_mpp_chn venc_mpp_chn;
    k_s32 ret;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = vi_dev_id;
    vi_mpp_chn.chn_id = vi_chn_id;
    venc_mpp_chn.mod_id = K_ID_VENC;
    venc_mpp_chn.dev_id = 0;
    venc_mpp_chn.chn_id = venc_chn_id_;

    ret = kd_mpi_sys_bind(&vi_mpp_chn, &venc_mpp_chn);
    if (ret == 0)
    {
        binded_vi_ = true;
        vi_dev_id_ = vi_dev_id;
        vi_chn_id_ = vi_chn_id;
    }

    return ret;
}


int EncodedRtspServer::unbind_vi_chn()
{
    k_s32 ret;
    if (!binded_vi_) {
        return -1;
    }

    k_mpp_chn vi_mpp_chn;
    k_mpp_chn venc_mpp_chn;

    vi_mpp_chn.mod_id = K_ID_VI;
    vi_mpp_chn.dev_id = vi_dev_id_;
    vi_mpp_chn.chn_id = vi_chn_id_;
    venc_mpp_chn.mod_id = K_ID_VENC;
    venc_mpp_chn.dev_id = 0;
    venc_mpp_chn.chn_id = venc_chn_id_;

    ret = kd_mpi_sys_unbind(&vi_mpp_chn, &venc_mpp_chn);
    if (ret == 0)
    {
        binded_vi_ = false;
    }
    return ret;
}