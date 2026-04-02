#ifndef _PIPELINE_H
#define _PIPELINE_H

// ==============================
// C++ 标准库头文件
// ==============================
// 用于时间统计、日志输出、多线程与原子操作（保证并发安全）
#include <chrono>      // 高精度计时，用于帧率统计、性能分析
#include <fstream>     // 文件流（日志、调试数据输出等）
#include <iostream>    // 标准 IO 输出（调试打印）
#include <thread>      // 线程管理（采集、推理、显示可分线程）
#include <atomic>      // 原子变量（多线程同步标志）

// ==============================
// C 标准库 / 系统调用
// ==============================
// 主要用于底层内存管理、系统接口
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>  // mmap/munmap：将物理内存映射到用户态虚拟地址

// ==============================
// K230 / MPP 多媒体相关头文件
// ==============================
// K230 媒体处理平台（MPP）接口：涵盖内存池、视频采集、显示、DMA、传感器等
#include "k_module.h"
#include "k_type.h"
#include "k_gsdma_comm.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_isp_api.h"
#include "mpi_sys_api.h"
#include "k_vo_comm.h"
#include "k_vicap_comm.h"
#include "mpi_vo_api.h"
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "k_autoconf_comm.h"
#include "mpi_sensor_api.h"
#include "mpi_gsdma_api.h"
#include "encoded_rtsp_server.h"
#include "mpi_nonai_2d_api.h"

// ==============================
// 项目内工具头文件
// ==============================
// 项目自定义工具与配置
#include "scoped_timing.h"   // RAII 计时工具，用于函数/代码块耗时统计
#include "setting.h"         // 全局配置宏：分辨率、通道数量、OSD 开关等

// ========================================================================================
// DumpRes
// ----------------------------------------------------------------------------------------
// 用途：
//   用于保存从 VICAP dump 出来的一帧图像的地址信息，
//   供 AI 推理模块（CPU 访问）和硬件模块（DMA/显示）同时使用。
//
// 说明：
//   - virt_addr：用户态虚拟地址，可直接被 CPU 访问（OpenCV/AI 推理等）
//   - phy_addr ：物理地址，供硬件模块（GSDMA、VO、VICAP）使用
// ========================================================================================
typedef struct DumpRes
{
    uintptr_t virt_addr;   // 用户态虚拟地址（CPU 可访问）
    uintptr_t phy_addr;    // 物理地址（供 DMA/硬件模块使用）
} DumpRes;


// ========================================================================================
// PipeLine
// ----------------------------------------------------------------------------------------
// 功能：
//   封装一条完整的视频处理管线：
//   【视频采集 + 显示 + AI 推理 + OSD 叠加】
//
// 典型数据流：
//   Sensor --> VICAP
//                 |--> 通道0 --> VO（实时原始视频显示）
//                 |--> 通道1 --> Dump --> AI 推理
//                                   |
//                                   +--> OSD 绘制 --> VO OSD Layer（结果叠加显示）
//
// 设计目标：
//   - 隔离 MPP 底层细节，对外提供统一接口
//   - 支持多路传感器（0/1/2）
//   - 方便后续扩展 AI 算法或显示布局
// ========================================================================================
class PipeLine
{
public:

    // 构造函数
    // @param debug_mode: 调试模式开关
    //                    非 0 时启用 ScopedTiming，输出性能统计信息
    PipeLine(int debug_mode);

    // 析构函数
    // 负责资源释放，避免 VB 泄漏、句柄未销毁等问题
    ~PipeLine();

    // 创建整条视频管线
    // 功能包括：
    //   1. 初始化 VB 内存池
    //   2. 配置 Connector（MIPI/HDMI/LCD）
    //   3. 创建并启动 VO Layer（视频显示层）
    //   4. 创建 OSD Layer（ARGB 叠加层）
    //   5. 初始化 Sensor + VICAP
    //   6. 绑定 VICAP -> VO 的 MPP 通道关系
    int Create();

    // 从 VICAP 的 AI 通道获取一帧图像（用于 AI 推理）
    // @param dump_res: 返回该帧的物理地址和虚拟地址
    // @param sensor_id: 传感器索引（0/1/2）
    int GetFrame(DumpRes &dump_res, int sensor_id);

    // 释放通过 GetFrame() 获取的帧资源
    // 说明：
    //   - 必须成对调用，否则会造成 VB 缓冲区泄漏
    // @param dump_res: 需要释放的帧地址信息
    // @param sensor_id: 传感器索引（0/1/2）
    int ReleaseFrame(DumpRes &dump_res, int sensor_id);

    // 向 OSD layer 插入一帧（叠加 AI 检测结果）
    // @param osd_data: 外部准备好的 OSD 像素数据指针（通常为 ARGB 格式）
    // @param vo_layer_id: 对应的 VO/OSD Layer ID
    int InsertFrame(void* osd_data, int vo_layer_id);

    // 销毁管线，释放所有资源
    // 包括：
    //   - OSD Layer
    //   - VICAP 设备
    //   - VO Layer
    //   - VB 内存池
    //   - mmap 映射的虚拟地址
    int Destroy();

private:
    // ============================
    // Video Buffer（VB）相关
    // ============================
    // 全局 VB 配置结构体：
    //   - 指定内存池数量
    //   - 每个池的块大小和块数
    //   - 所有视频帧、OSD 缓冲区均来自 VB
    k_vb_config config;

    // ============================
    // 屏幕 / Connector 相关
    // ============================
    // 显示接口类型：
    //   - MIPI、HDMI、LCD 等
    //   - 在 Create() 中根据 setting.h 中的宏选择
    k_connector_type connector_type;

    // ============================
    // VO（Video Output）显示模块
    // ============================
    k_s32 vo_dev_id;            // VO 设备 ID（通常为 0）

    // 三路视频显示 Layer（对应三路摄像头或多窗口显示）
    k_vo_layer_id vi_vo_id_0;
    k_vo_layer_attr vi_vo_attr_0; // 分辨率、像素格式、旋转、缩放等

    k_vo_layer_id vi_vo_id_1;
    k_vo_layer_attr vi_vo_attr_1;

    k_vo_layer_id vi_vo_id_2;
    k_vo_layer_attr vi_vo_attr_2;

    // ============================
    // OSD（On-Screen Display 叠加层）
    // ============================
    // 用于绘制 AI 检测框、文字、关键点等信息
    // 每一路视频对应一层 OSD

    // -------- OSD 通道 0 --------
    k_vo_layer_id osd_vo_id_0;
    k_vo_layer_attr osd_vo_attr_0;   // ARGB、分辨率等
    k_u32 osd_pool_id_0;             // OSD 使用的 VB 内存池 ID
    k_vb_blk_handle handle_0;        // 从 VB 池申请的内存块句柄
    k_video_frame_info osd_frame_info_0; // OSD 帧描述信息
    void *insert_osd_vaddr_0;        // 映射后的虚拟地址（CPU 写入绘制）

    // -------- OSD 通道 1 --------
    k_vo_layer_id osd_vo_id_1;
    k_vo_layer_attr osd_vo_attr_1;
    k_u32 osd_pool_id_1;
    k_vb_blk_handle handle_1;
    k_video_frame_info osd_frame_info_1;
    void *insert_osd_vaddr_1;

    // -------- OSD 通道 2 --------
    k_vo_layer_id osd_vo_id_2;
    k_vo_layer_attr osd_vo_attr_2;
    k_u32 osd_pool_id_2;
    k_vb_blk_handle handle_2;
    k_video_frame_info osd_frame_info_2;
    void *insert_osd_vaddr_2;

    // ============================
    // Sensor & VICAP（视频采集）
    // ============================
    // 每一路包含：
    //   - Sensor 类型
    //   - VICAP 设备
    //   - 一个输出到 VO 的通道
    //   - 一个输出到 AI 的通道（dump）
    //   - 一份 dump 帧信息

    // -------- 摄像头 0 --------
    k_vicap_sensor_type sensor_type_0;
    k_vicap_dev vicap_dev_0;
    k_vicap_chn vicap_chn_to_vo_0;
    k_vicap_chn vicap_chn_to_ai_0;
    k_video_frame_info dump_info_0;

    // -------- 摄像头 1 --------
    k_vicap_sensor_type sensor_type_1;
    k_vicap_dev vicap_dev_1;
    k_vicap_chn vicap_chn_to_vo_1;
    k_vicap_chn vicap_chn_to_ai_1;
    k_video_frame_info dump_info_1;

    // -------- 摄像头 2 --------
    k_vicap_sensor_type sensor_type_2;
    k_vicap_dev vicap_dev_2;
    k_vicap_chn vicap_chn_to_vo_2;
    k_vicap_chn vicap_chn_to_ai_2;
    k_vicap_chn vicap_chn_to_venc_2;
    k_video_frame_info dump_info_2;

    // ============================
    // 模块绑定（MPP 通道关系）
    // ============================
    // 用于建立：
    //   VICAP(源) --> VO(目的)
    // 的硬件直连关系，避免 CPU 拷贝

    k_mpp_chn vicap_mpp_chn_0;
    k_mpp_chn vo_mpp_chn_0;

    k_mpp_chn vicap_mpp_chn_1;
    k_mpp_chn vo_mpp_chn_1;

    k_mpp_chn vicap_mpp_chn_2;
    k_mpp_chn vo_mpp_chn_2;

    // ============================
    // 调试控制
    // ============================
    // 控制是否输出性能日志、调试信息
    int debug_mode_ = 0;

    EncodedRtspServer  rtsp_server_;
};

#endif
