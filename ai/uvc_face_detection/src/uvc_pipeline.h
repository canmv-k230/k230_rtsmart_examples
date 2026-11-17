#ifndef _UVC_PIPELINE_H
#define _UVC_PIPELINE_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <signal.h>

#include "k_module.h"
#include "k_type.h"
#include "k_gsdma_comm.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "k_vo_comm.h"
#include "k_vicap_comm.h"
#include "k_connector_comm.h"
#include "k_autoconf_comm.h"
#include "k_nonai_2d_comm.h"

#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_isp_api.h"
#include "mpi_sys_api.h"
#include "mpi_vo_api.h"
#include "mpi_connector_api.h"
#include "mpi_sensor_api.h"
#include "mpi_vdec_api.h"
#include "mpi_uvc_api.h"
#include "mpi_nonai_2d_api.h"
#include "scoped_timing.h"
#include "setting.h"


typedef struct DumpRes
{
    uintptr_t virt_addr;
    uintptr_t phy_addr;
} DumpRes;

/**
 * @brief GetFrame()/ReleaseFrame() 的返回值。
 */
enum UVC_FrameStatus
{
    UVC_FRAME_OK      = 0,   ///< 成功
    UVC_FRAME_DROPPED = 1,   ///< 本帧被丢弃（超时、缓冲不足等背压），持有的资源已全部归还，可重试
    UVC_FRAME_FATAL   = -1,  ///< 不可恢复（归还资源失败、映射失败、尺寸不符等），应退出并调用 Destroy()
};


/**
 * @brief UVC_PipeLine 类封装了 UVC 视频采集、解码、绑定、显示等完整处理流程。
 */
class UVC_PipeLine
{
public:
    /**
     * @brief 构造函数，初始化 UVC_PipeLine 实例。
     * @param debug_mode 是否启用调试模式（0：关闭，非0：开启）
     */
    UVC_PipeLine(int debug_mode);

    /**
     * @brief 析构函数，释放资源。
     */
    ~UVC_PipeLine();

    /**
     * @brief 创建并初始化 UVC 管道，包括设备、缓冲区、解码器等。
     * @return 0 表示成功，其他值表示失败。
     */
    int Create();

    /**
     * @brief 获取一帧图像数据。
     * @param dump_res 输出图像帧数据。
     * @return UVC_FrameStatus。非 UVC_FRAME_OK 时本次迭代已不持有任何资源，无需 ReleaseFrame()。
     */
    int GetFrame(DumpRes &dump_res);

    /**
     * @brief 把当前帧送显，并归还本次迭代持有的全部资源（无论送显是否成功）。
     * @return UVC_FrameStatus。送显失败返回 UVC_FRAME_DROPPED，归还失败返回 UVC_FRAME_FATAL。
     */
    int ReleaseFrame(DumpRes &dump_res);

    /**
     * @brief 插入一帧图像帧。
     * @param data 指向数据的指针。
     * @return 0 表示成功，其他值表示失败。
     */
    int InsertFrame(void* data);

    /**
     * @brief 销毁 UVC 管道，释放所有资源。Create() 失败后也必须调用。
     * 按创建的逆序执行每一步，失败不提前返回。
     * @return 0 表示成功，否则为第一个错误码。
     */
    int Destroy();

private:
    /**
     * @brief 按获取的逆序归还本次迭代持有的资源（munmap、RGB、VDEC、UVC）。
     * 每项只尝试一次，无论成功与否都清除持有标志。
     * @return 0 表示全部成功，否则为第一个错误码。
     */
    int ReleaseHeld();

    /**
     * @brief 丢弃 VDEC 与 nonai_2d chn0 中因先前超时而迟到的输出帧，
     * 避免下一次提交取回的帧与提交的帧错位。
     */
    void DrainStaleOutputs();

    // ------------------- 每次迭代的帧所有权 -------------------
    struct FrameHold
    {
        bool uvc = false;            ///< cur_frame 需 uvc_host_put_frame()
        bool vdec = false;           ///< vdec_frame_info 需 kd_mpi_vdec_release_frame()
        bool rgb = false;            ///< rgb888_frame_info 需经 chn0 kd_mpi_nonai_2d_release_frame()
        void *rgb_vaddr = nullptr;   ///< rgb888_frame_info 的 CPU 映射
        k_u32 rgb_map_len = 0;       ///< 映射长度
    };
    FrameHold held_;

    // ------------------- Create() 各阶段完成标志，Destroy() 据此逆序清理 -------------------
    bool vb_inited_ = false;
    k_s32 connector_fd_ = -1;        ///< 保持打开，Destroy() 中给面板断电后关闭
    bool connector_on_ = false;      ///< connector init 成功即已上电
    bool vo_layer_enabled_ = false;
    bool vdec_attached_ = false;
    bool vdec_created_ = false;
    bool vdec_started_ = false;
    bool rgb888_attached_ = false;
    bool rgb888_created_ = false;
    bool rgb888_started_ = false;
    bool yuv420sp_attached_ = false;
    bool yuv420sp_created_ = false;
    bool yuv420sp_started_ = false;
    bool bound_ = false;
    bool uvc_inited_ = false;

    // ------------------- UVC 相关 -------------------
    uvc_format init_format;     ///< 初始的 UVC 图像格式
    unsigned char is_jpeg;      ///< 是否为 JPEG 编码格式
    uvc_frame cur_frame;        ///< 当前 UVC 图像帧结构体

    // ------------------- VB（视频缓冲）相关 -------------------
    k_vb_config config;         ///< 视频缓冲区配置结构体

    // ------------------- 屏幕/显示相关 -------------------
    k_connector_type connector_type; ///< 显示屏连接类型（LT9611, ST7701等）

    // ------------------- VO（视频输出）相关 -------------------
    k_vo_layer_id uvc_vo_id;       ///< 视频输出通道 ID
    k_vo_layer_attr uvc_vo_attr; ///< 视频输出通道属性配置
    k_s32 vo_dev_id;            ///< 视频输出设备 ID

    // ------------------- 解码器（VDEC）相关 -------------------
    k_vdec_chn_attr vdec_attr;  ///< 解码器通道属性配置
    k_s32 vdec_dev_id;          ///< 解码器设备 ID
    k_s32 vdec_bind_chn_id;     ///< 解码器绑定通道 ID
    k_s32 vdec_pool_id;         ///< 解码器使用的缓冲池 ID
    k_video_frame_info vdec_frame_info; ///< 解码器输出的视频帧信息
    k_vdec_supplement_info supplement_info; ///< 解码器补充信息

    // ------------------- nonai 2D 相关 -------------------
    k_u32 nonai2d_dev_id;                   ///< nonai 2D 设备 ID
    k_u32 nonai2d_rgb888_chn_id;            ///< nonai 2D RGB888 通道 ID
    k_nonai_2d_chn_attr attr_2d_rgb888;     ///< nonai 2D RGB888 通道属性
    k_video_frame_info rgb888_frame_info;   ///< nonai 2D RGB888 视频帧信息
    k_s32 nonai2d_rgb888_pool_id;
    k_u32 nonai2d_yuv420sp_chn_id;          ///< nonai 2D YUV420SP 通道 ID
    k_nonai_2d_chn_attr attr_2d_yuv420sp;   ///< nonai 2D YUV420SP 通道属性
    k_video_frame_info yuv420sp_frame_info; ///< nonai 2D YUV420SP 视频帧信息
    k_s32 nonai2d_yuv420sp_pool_id;

    // ------------------- 模块绑定相关 -------------------
    k_mpp_chn nonai2d_mpp_chn;     ///< 解码器模块通道
    k_mpp_chn vo_mpp_chn;       ///< 视频输出模块通道

    // ------------------- 调试相关 -------------------
    int debug_mode_ = 0;        ///< 调试模式标志，非0启用更多日志或功能
};


#endif
