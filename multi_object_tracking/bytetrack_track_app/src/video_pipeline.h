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
#include "k_dma_comm.h"
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
#include "vo_test_case.h"
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "k_autoconf_comm.h"
#include "mpi_sensor_api.h"
#include "mpi_dma_api.h"
#include "scoped_timing.h"
#include "setting.h"

/**
 * @brief Structure used to dump frame buffer information
 *        including virtual and physical addresses.
 */
typedef struct DumpRes
{
    uintptr_t virt_addr;  ///< Virtual address of the frame buffer
    uintptr_t phy_addr;   ///< Physical address of the frame buffer
} DumpRes;


/**
 * @brief Video processing pipeline class
 *
 * This class encapsulates the complete video pipeline, including:
 * - Video buffer (VB) configuration and management
 * - Sensor and VICAP (video capture) initialization
 * - Video output (VO) configuration and binding
 * - OSD (On-Screen Display) handling
 * - GDMA configuration for frame insertion
 *
 * It provides unified interfaces for frame acquisition, insertion,
 * and resource cleanup.
 */
class PipeLine
{
public:
    /**
     * @brief Constructor
     * @param debug_mode Debug mode:
     *        0 - no debug output
     *        1 - timing information
     *        2 - full debug information
     */
    PipeLine(int debug_mode);

    /**
     * @brief Destructor
     */
    ~PipeLine();

    /**
     * @brief Create and initialize the entire pipeline
     * @return 0 on success, negative value on failure
     */
    int Create();

    /**
     * @brief Acquire a frame from VICAP
     * @param dump_res Output structure containing frame addresses
     */
    void GetFrame(DumpRes &dump_res);

    /**
     * @brief Release a previously acquired frame
     * @param dump_res Frame information to be released
     * @return 0 on success, negative value on failure
     */
    int ReleaseFrame(DumpRes &dump_res);

    /**
     * @brief Insert an OSD frame into the display pipeline
     * @param osd_data Pointer to OSD pixel data
     * @return 0 on success, negative value on failure
     */
    int InsertFrame(void* osd_data);
    
    /**
     * @brief Destroy and release all pipeline resources
     * @return 0 on success, negative value on failure
     */
    int Destroy();

private:
    // Video buffer (VB) related configuration
    k_vb_config config;

    // Display / connector related
    k_connector_type connector_type;

    // Video output (VO) related
    k_vo_layer vo_chn_id;
    k_s32 vo_dev_id;
    k_s32 vo_bind_chn_id;

    // On-Screen Display (OSD) related
    k_vo_osd osd_chn_id;
    k_u32 osd_pool_id;
    k_vb_blk_handle handle;
    k_video_frame_info osd_frame_info;
    void *insert_osd_vaddr;

    // Sensor and VICAP related
    k_vicap_sensor_type sensor_type;
    k_vicap_dev vicap_dev;
    k_vicap_chn vicap_chn_to_vo;
    k_vicap_chn vicap_chn_to_ai;
    k_video_frame_info dump_info;

    // Module binding related
    k_mpp_chn vicap_mpp_chn;
    k_mpp_chn vo_mpp_chn;

    // Debug configuration
    int debug_mode_ = 0;

    // GDMA related
    k_dma_dev_attr_t gdma_dev_attr;
    k_dma_chn_attr_u dma_attr;
    k_u32 gdma_pool_id;
    k_vb_blk_handle gdma_handle;
    k_video_frame_info gdma_frame_info;
    void *insert_gdma_vaddr;
};

#endif

