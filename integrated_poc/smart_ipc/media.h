#ifndef _KD_MEDIA_H
#define _KD_MEDIA_H

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <string>
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

typedef enum {
    DATA_SOURCE_SENSOR_CHANNEL = 0, /**< 数据源为Sensor的通道 */
    DATA_SOURCE_VO_WBC,             /**< 数据源为VO模块的WBC回写数据 */
} VencDataSourceType;

struct KdMediaInputConfig {
  k_vicap_sensor_type sensor_type = SENSOR_TYPE_MAX; // Sensor type
  KdMediaVideoType video_type = KdMediaVideoType::kVideoTypeH264; // Venc type
  int venc_width = 1280; // Video encoder width
  int venc_height = 720; // Video encoder height
  int bitrate_kbps = 2000; // Bitrate in kbps
  VencDataSourceType venc_data_source_type = DATA_SOURCE_SENSOR_CHANNEL; // Video encoder data source

  int audio_samplerate = 8000; // Audio sample rate
  int audio_channel_cnt = 1; // Number of audio channels

  int vo_width = 1920; // Video output width
  int vo_height = 1080; // Video output height

  int ai_width = 1280; // AI analysis width
  int ai_height = 720; // AI analysis height

  int osd_width = 1920; // OSD width
  int osd_height = 1080; // OSD height

  k_connector_type vo_connect_type = LT9611_MIPI_4LAN_1920X1080_30FPS; // Video output connector type

  std::string kmodel_file="face_detection_320.kmodel"; // Kmodel file path
  float obj_thresh = 0.6; // Object detection threshold
  float nms_thresh = 0.4; // Non-maximum suppression threshold

  bool enable_video_output = true; // Enable rendering
  bool enable_ai_analysis = true; // Enable AI analysis
  bool enable_video_encoding = true; // Enable video encoder
  bool enable_capture_audio = true; // Enable capture audio
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
  int osd_send_venc_frame();

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
   * @brief Deinitialize connector.
   * @return Status of the initialization operation.
   */
  int _deinit_connector();

  /**
   * @brief Initialize video output layer.
   * @param chn_id Channel ID for the video output layer.
   * @return Status of the initialization operation.
   */
  int _init_layer(k_vo_layer_id chn_id);

  /**
   * @brief Deinitialize video output layer.
   * @param chn_id Channel ID for the video output layer.
   * @return Status of the initialization operation.
   */
  int _deinit_layer(k_vo_layer_id chn_id);

  /**
   * @brief Initialize On-Screen Display (OSD).
   * @param osd_id OSD ID.
   * @return Status of the initialization operation.
   */
  int _init_osd(k_vo_layer_id osd_id);

  /**
   * @brief Deinitialize On-Screen Display (OSD).
   * @param osd_id OSD ID.
   * @return Status of the initialization operation.
   */
  int _deinit_osd(k_vo_layer_id osd_id);

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
   * @brief Initialize wbc
   * @return Status of the initialization operation.
   */
  int _init_wbc();

  /**
   * @brief Deinitialize wbc
   * @return Status of the deinitialization operation.
   */
  int _deinit_wbc();

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
   * @brief Start dumping wbc frames.
   * @return Status of the start operation.
   */
  int _start_dump_wbc_frame();

  /**
   * @brief Stop dumping wbc frames.
   * @return Status of the stop operation.
   */
  int _stop_dump_wbc_frame();

  /**
   * @brief Thread function to get audio encoder stream.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *aenc_chn_get_stream_thread(void *arg);

  /**
   * @brief Thread function to get video encoder stream.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *venc_stream_thread(void *arg);

  /**
   * @brief Thread function to analyze AI frames.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *ai_analysis_frame_thread(void *arg);

    /**
   * @brief Thread function to get vo wbc frame.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *wbc_frame_thread(void *arg);

  /**
   * @brief Thread function to start ai,aenc.
   * Audio codec initialization takes too long, so this part is started in a thread.
   * @param arg Arguments for the thread function.
   * @return Pointer to the result.
   */
  static void *start_ai_aenc_thread(void *arg);

  k_u32 _venc_vb_create_pool();

  private:
  KdMediaInputConfig input_config_; // Media input configuration
  KdMediaFeatureConfig feature_config_; // Media feature configuration

  k_u32 osd_vb_handle_{VB_INVALID_HANDLE}; // OSD video buffer handle
  int osd_pool_id_{-1}; // OSD pool ID
  k_vo_layer_id vo_layer_chn_id_{K_VO_LAYER_VIDEO1}; // Video output layer channel ID
  k_vo_layer_id osd_id_{K_VO_LAYER_OSD3}; // OSD ID
  k_pixel_format osd_format_{PIXEL_FORMAT_ARGB_8888}; // OSD pixel format
  k_video_frame_info osd_vf_info_; // OSD video frame information

  k_vicap_dev_set_info vcap_dev_info_; // Video capture device settings
  k_vicap_dev vi_dev_id_{VICAP_DEV_ID_0}; // Video capture device ID
  k_vicap_chn vi_chn_render_id_{VICAP_CHN_ID_0}; // Video capture render channel ID
  k_pixel_format vi_chn_render_pixel_format_{PIXEL_FORMAT_YUV_SEMIPLANAR_420}; // Render channel pixel format
  k_vicap_chn vi_chn_ai_id_{VICAP_CHN_ID_0}; // AI channel ID
  k_pixel_format vi_chn_ai_pixel_format_{PIXEL_FORMAT_RGB_888_PLANAR}; // AI channel pixel format
  k_vicap_chn vi_chn_venc_id_{VICAP_CHN_ID_0}; // Video encoder channel ID
  k_pixel_format vi_chn_venc_pixel_format_{PIXEL_FORMAT_YUV_SEMIPLANAR_420}; // Video encoder pixel format

  k_u32 venc_chn_id_{0}; // Video encoder channel ID
  pthread_t venc_tid_; // Video encoder thread ID
  bool start_get_video_stream_{false}; // Flag to start getting video stream

  pthread_t ai_analysis_frame_tid_{0}; // AI analysis frame thread ID
  bool start_dump_ai_analysis_frame_{false}; // Flag to start dumping AI frames

  pthread_t wbc_frame_tid_{0}; // get vo wbc frame thread ID
  bool start_dump_wbc_frame_{false}; // Flag to start dumping wbc frames

  int audio_frame_divisor_{25}; // Audio frame divisor
  k_u32 ai_dev_{0}; // Audio input device ID
  k_u32 ai_chn_{0}; // Audio input channel ID
  k_handle aenc_handle_{0}; // Audio encoder handle
  k_audio_stream audio_stream_; // Audio stream data
  pthread_t get_audio_stream_tid_{0}; // Audio stream thread ID
  pthread_t start_ai_aenc_tid_{0}; // Start AI and AENC thread ID
  bool start_get_audio_stream_{false}; // Flag to start getting audio stream
  bool ai_started_{false}; // AI started flag
  bool ai_initialized_{false}; // AI initialized flag

  int wbc_width_{0}; // WBC width
  int wbc_height_{0}; // WBC height
  char *connector_name_{nullptr}; // Connector name

  k_u32 venc_attach_pool_id_{VB_INVALID_POOLID};
  int rotation_90_{0};
};

#endif // _KD_MEDIA_H
