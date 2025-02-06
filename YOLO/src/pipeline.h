#ifndef _PIPELINE_H
#define _PIPELINE_H

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
#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_isp_api.h"
#include "mpi_sys_api.h"
#include "k_vo_comm.h"
#include "mpi_vo_api.h"
#include "vo_test_case.h"
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "k_autoconf_comm.h"
#include "mpi_sensor_api.h"
#include "scoped_timing.hpp"

typedef struct DumpRes
{
    uintptr_t virt_addr;
    uintptr_t phy_addr;
} DumpRes;

typedef struct GeneralConfig
{
    int ISP_WIDTH=1920;
    int ISP_HEIGHT=1080;
    int ISP_FPS=30;
    int DISPLAY_MODE=0.;    
    int DISPLAY_WIDTH=1920;
    int DISPLAY_HEIGHT=1080;
    int DISPLAY_ROTATE=0;
    int AI_FRAME_WIDTH=640;
    int AI_FRAME_HEIGHT=360;
    int AI_FRAME_CHANNEL=3;
    int USE_OSD=1;
    int OSD_WIDTH=1920;
    int OSD_HEIGHT=1080;
    int OSD_CHANNEL=4;
} GeneralConfig;

typedef struct YoloConfig
{
    char* model_type="yolov8";
    char* task_type="detect";
    char* task_mode="video";
    char* image_path="test.jpg";
    char* kmodel_path="yolov8n.kmodel";
    float conf_thres=0.35;
    float nms_thres=0.65;
    float mask_thres=0.5;
    std::string labels_txt_filepath="coco_labels.txt";
    int debug_mode=0;
} YoloConfig;

class PipeLine
{
public:
    
    PipeLine(GeneralConfig &config,int debug_mode);

    ~PipeLine();

    int Create();

    void GetFrame(DumpRes &dump_res);

    int ReleaseFrame();

    int InsertFrame(void* osd_data);
    
    int Destroy();

private:

    GeneralConfig general_config_;
    // vb相关
    k_vb_config config;
    // 屏幕相关
    k_connector_type connector_type;
    // vo相关
    k_vo_layer vo_chn_id;
    k_s32 vo_dev_id;
    k_s32 vo_bind_chn_id;

    //osd相关
    k_vo_osd osd_chn_id;
    k_u32 osd_pool_id;
    k_vb_blk_handle handle;
    k_video_frame_info osd_frame_info;
    void *insert_osd_vaddr;

    //sensor&vicap相关
    k_vicap_sensor_type sensor_type;
    k_vicap_dev vicap_dev;
    k_vicap_chn vicap_chn_to_vo;
    k_vicap_chn vicap_chn_to_ai;
    k_video_frame_info dump_info;

    //bind相关
    k_mpp_chn vicap_mpp_chn;
    k_mpp_chn vo_mpp_chn;
    int debug_mode_=0;
};

#endif
