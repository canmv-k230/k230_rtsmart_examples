/**
 * @file mpp_pipeline.h
 * @brief K230 MPP video pipeline interface for WebRTC demo
 *
 * Manages the hardware video pipeline:
 *   Camera (VICAP) → Display (VO) + Encode (VENC H.264)
 *
 * Usage:
 *   1. mpp_pipeline_init()  — set up all hardware blocks
 *   2. mpp_pipeline_start() — begin capture + encode
 *   3. mpp_pipeline_get_venc_chn() — get VENC channel for polling
 *   4. mpp_pipeline_deinit() — stop and release all resources
 */

#ifndef MPP_PIPELINE_H_
#define MPP_PIPELINE_H_

#include <stdint.h>
#include "k_type.h"
#include "k_vicap_comm.h"
#include "k_vo_comm.h"
#include "k_connector_comm.h"
#include "k_venc_comm.h"

/** Max NAL units per VENC frame (SPS+PPS+slice slices) */
#define VENC_MAX_PACK_CNT   (5)

/** Video encoder type */
typedef enum {
  VENC_TYPE_H264 = 0,
  VENC_TYPE_H265,
} VencType;

/** Get encoder type name string for printing */
static inline const char* venc_type_name(VencType t) {
  return t == VENC_TYPE_H265 ? "H265" : "H264";
}

/**
 * MPP Pipeline configuration.
 * Passed to mpp_pipeline_init() to configure camera, display, and encoder.
 */
typedef struct {
  k_u32 csi_num;                    /**< CSI device number (0-2, selects camera connector) */
  k_connector_type connector_type;  /**< Display connector type (LCD or HDMI) */
  k_u32 venc_width;                 /**< VENC encode width in pixels (e.g. 1280) */
  k_u32 venc_height;                /**< VENC encode height in pixels (e.g. 720) */
  k_u32 venc_bitrate_kbps;          /**< VENC target bitrate in kbps (e.g. 2000) */
  VencType venc_type;               /**< VENC encode type: H.264 or H.265 (default H.264) */
} MppPipelineConfig;

/**
 * Initialize the MPP video pipeline.
 *
 * Sets up in order: VB → Connector → VO Layer → VICAP (CHN0→VO, CHN1→VENC) → VENC
 * Binds: VICAP-CHN0 → VO (display), VICAP-CHN1 → VENC (encode)
 *
 * After init, call mpp_pipeline_start() to begin capture+encode+display.
 *
 * @param config Pipeline configuration
 * @return 0 on success, negative on failure
 */
int mpp_pipeline_init(const MppPipelineConfig* config);

/**
 * Start VICAP streaming and VENC encoding.
 * After this, encoded H.264 frames can be polled via kd_mpi_venc_get_stream().
 *
 * @return 0 on success, negative on failure
 */
int mpp_pipeline_start(void);

/**
 * Get VENC channel ID for stream polling.
 * Use this with kd_mpi_venc_get_stream() to read encoded frames.
 *
 * @return VENC channel ID
 */
k_u32 mpp_pipeline_get_venc_chn(void);

/**
 * Get the encoder type configured at init time.
 *
 * @return Encoder type (VENC_TYPE_H264 or VENC_TYPE_H265)
 */
VencType mpp_pipeline_get_venc_type(void);

/**
 * Stop and deinitialize the MPP pipeline.
 * Releases all resources in reverse order of init.
 * Safe to call even if mpp_pipeline_start() was not called.
 */
void mpp_pipeline_deinit(void);

#endif /* MPP_PIPELINE_H_ */
