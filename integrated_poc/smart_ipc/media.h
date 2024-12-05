#ifndef _KD_MEDIA_H
#define _KD_MEDIA_H

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include "k_vicap_comm.h"
#include "k_venc_comm.h"
#include "k_connector_comm.h"
#include "k_audio_comm.h"
#include "k_vb_comm.h"

enum class KdMediaVideoType {
  kVideoTypeH264,
  kVideoTypeH265,
  kVideoTypeMjpeg,
  kVideoTypeButt
};

struct KdMediaInputConfig {
  bool video_valid = false;
  k_vicap_sensor_type sensor_type = SENSOR_TYPE_MAX;
  int sensor_num = 1;
  KdMediaVideoType video_type = KdMediaVideoType::kVideoTypeH265;
  int venc_width = 1280;
  int venc_height = 720;
  int bitrate_kbps = 4000;

  int audio_samplerate = 8000;
  int audio_channel_cnt = 1;
  int pitch_shift_semitones = 0; // [-12, 12]

  int vo_width = 1920;
  int vo_height = 1080;

  int ai_width = 1280;
  int ai_height = 720;

  int osd_width = 1920;
  int osd_height = 1080;

  k_connector_type vo_connect_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
};

class IOnAEncData {
  public:
  virtual ~IOnAEncData() {}
  virtual void OnAEncData(k_u32 chn_id, k_u8 *pdata, size_t size, k_u64 time_stamp) = 0;
};

class IOnVEncData {
  public:
  virtual ~IOnVEncData() {}
  virtual void OnVEncData(k_u32 chn_id, void *data, size_t size, k_venc_pack_type type, uint64_t timestamp) = 0;
};

class IOnAIFrameData {
  public:
  virtual ~IOnAIFrameData() {}
  virtual void OnAIFrameData(k_u32 chn_id, k_video_frame_info *frame_info) = 0;
};

struct KdMediaFeatureConfig {
  bool enable_video_encoder = true;
  IOnVEncData *on_venc_data = nullptr;

  bool enable_ai_analysis = true;
  IOnAIFrameData *on_ai_frame_data = nullptr;

  bool enable_render = true;

  bool enable_audio_encoder = true;
  IOnAEncData *on_aenc_data = nullptr;
};

/**
 * @class KdMedia
 * @brief Class to manage media features and operations.
 */
class KdMedia {
  public:
  /**
   * @brief Configure media features.
   * @param input_config Configuration for media input.
   * @param feature_config Configuration for media features.
   * @return Status of the configuration operation.
   */
  int configure_media_features(const KdMediaInputConfig &input_config, const KdMediaFeatureConfig &feature_config);

  /**
   * @brief Enable media features.
   * @return Status of the enable operation.
   */
  int enable_media_features();

  /**
   * @brief Disable media features.
   * @return Status of the disable operation.
   */
  int disable_media_features();

  /**
   * @brief Destroy media features.
   * @return Status of the destroy operation.
   */
  int destroy_media_features();

  int osd_alloc_frame(void **osd_vaddr);
  int osd_draw_frame();

  private:
  /**
   * @brief Initialize video buffer pool.
   * @return Status of the initialization operation.
   */
  int _init_vb_pool();

  /**
   * @brief Deinitialize video buffer pool.
   * @return Status of the deinitialization operation.
   */
  int _deinit_vb_pool();

  /**
   * @brief Initialize video capture.
   * @return Status of the initialization operation.
   */
  int _init_vi_cap();

  /**
   * @brief Deinitialize video capture.
   * @return Status of the deinitialization operation.
   */
  int _deinit_vi_cap();

  /**
   * @brief Start video capture.
   * @return Status of the start operation.
   */
  int _start_vi_cap();

  /**
   * @brief Stop video capture.
   * @return Status of the stop operation.
   */
  int _stop_vi_cap();

  /**
   * @brief Initialize connector.
   * @return Status of the initialization operation.
   */
  int _init_connector();

  /**
   * @brief Initialize video output layer.
   * @param chn_id Channel ID for the video output layer.
   * @return Status of the initialization operation.
   */
  int _init_layer(k_vo_layer chn_id);

  /**
   * @brief Initialize On-Screen Display (OSD).
   * @param osd_id OSD ID.
   * @return Status of the initialization operation.
   */
  int _init_osd(k_vo_osd osd_id);

  /**
   * @brief Initialize video output layer and OSD.
   * @return Status of the initialization operation.
   */
  int _init_vo_layer_osd();

  /**
   * @brief Deinitialize video output layer and OSD.
   * @return Status of the deinitialization operation.
   */
  int _deinit_vo_layer_osd();

  /**
   * @brief Initialize video encoder.
   * @return Status of the initialization operation.
   */
  int _init_venc();

  /**
   * @brief Deinitialize video encoder.
   * @return Status of the deinitialization operation.
   */
  int _deinit_venc();

  /**
   * @brief Start video encoder.
   * @return Status of the start operation.
   */
  int _start_venc();

  /**
   * @brief Stop video encoder.
   * @return Status of the stop operation.
   */
  int _stop_venc();

  /**
   * @brief Initialize audio input and encoder.
   * @return Status of the initialization operation.
   */
  int _init_ai_aenc();

  /**
   * @brief Deinitialize audio input and encoder.
   * @return Status of the deinitialization operation.
   */
  int _deinit_ai_aenc();

  /**
   * @brief Start audio input and encoder.
   * @return Status of the start operation.
   */
  int _start_ai_aenc();

  /**
   * @brief Stop audio input and encoder.
   * @return Status of the stop operation.
   */
  int _stop_ai_aenc();

  /**
   * @brief Start dumping frames for AI analysis.
   * @return Status of the start operation.
   */
  int _start_dump_frame_for_ai_analysis();

  /**
   * @brief Stop dumping frames for AI analysis.
   * @return Status of the stop operation.
   */
  int _stop_dump_frame_for_ai_analysis();

  /**
   * @brief Thread function to get audio encoder stream.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *aenc_chn_get_stream_threads(void *arg);

  /**
   * @brief Thread function to get video encoder stream.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *venc_stream_threads(void *arg);

  /**
   * @brief Thread function to analyze AI frames.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *ai_analysis_frame_threads(void *arg);

  private:
  KdMediaInputConfig input_config_;
  KdMediaFeatureConfig feature_config_;

  k_u32 osd_vb_handle_{VB_INVALID_HANDLE};
  int osd_pool_id_{-1};
  k_vo_layer vo_layer_chn_id_{K_VO_LAYER1};
  k_vo_osd osd_id_{K_VO_OSD3};
  k_video_frame_info osd_vf_info_;

  k_vicap_dev_set_info vcap_dev_info_;
  k_vicap_dev vi_dev_id_{VICAP_DEV_ID_0};
  k_vicap_chn vi_chn_render_id_{VICAP_CHN_ID_0};
  k_vicap_chn vi_chn_ai_id_{VICAP_CHN_ID_0};
  k_vicap_chn vi_chn_venc_id_{VICAP_CHN_ID_0};

  k_u32 venc_chn_id_{0};
  pthread_t venc_tid_;
  bool start_get_video_stream_{false};

  pthread_t ai_analysis_frame_tid_{0};
  bool start_dump_ai_analysis_frame_{false};

  int audio_frame_divisor_{25};
  k_u32 ai_dev_{0};
  k_u32 ai_chn_{0};
  k_handle aenc_handle_{0};
  k_audio_stream audio_stream_;
  pthread_t get_audio_stream_tid_{0};
  bool start_get_audio_stream_{false};
  bool ai_started_{false};
};

#endif // _KD_MEDIA_H
