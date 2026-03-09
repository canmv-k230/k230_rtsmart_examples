#ifndef _VIDEO_STREAM_PIPELINE_H
#define _VIDEO_STREAM_PIPELINE_H

#include <string>
#include <cstdint>
#include "pipeline_types.h"

// Forward declaration of internal implementation
class VideoStreamPipelineImpl;
class VideoStreamPipeline {
public:
    /**
    * @brief Constructor for the VideoStreamPipeline class.
    *
    * Supports multiple types of video sources, including local video files (e.g., MP4),
    * network streams (e.g., RTSP), and USB cameras (UVC devices).
    *
    * @param video_path    Path or URI of the input video source. Supported formats include:
    *                      - Local video file (e.g., "/path/to/video.mp4")
    *                      - Network stream URL (e.g., "rtsp://192.168.1.100:554/stream")
    *                      - UVC device path
    * @param display_type  Display type
    *                      - LCD
    *                      - HDMI
    * @param out_width     Target output width (in pixels) for both video display (VO) and AI input.
    *                      If set to -1, the original aspect ratio is preserved and no resizing is applied.
    *                      Otherwise, all output frames are scaled to the specified width.
    *
    * @param out_height    Target output height (in pixels). Behaves consistently with out_width.
    *
    * @param stream_fps    Output frame rate (frames per second) for video rendering/display (VO).
    *                      - If > 0: decoded frames are throttled or interpolated to maintain this frame rate.
    *                      - If <= 0 (default -1): uses the native frame rate of the source or maximum device capability.
    *
    * @param ai_fps        Frame rate (fps) at which frames are delivered to the AI inference module.
    *                      - If > 0: frames are sampled at a fixed interval to ensure exactly ai_fps frames per second.
    *                      - If <= 0 (default -1): employs a "latest-frame-first" strategy—intermediate frames are dropped
    *                        to always feed the most recent available frame to AI, optimizing for low latency
    *                        in real-time applications (e.g., live object detection).
    * @param rtsp_rtp_over_tcp  rtp over tcp
    *                      - 0:UDP
    *                      - 1:TCP
    */
    VideoStreamPipeline(const std::string& video_path,
                        DisplayType display_type = DISPLAY_LCD,
                        int out_width = -1,
                        int out_height = -1,
                        int stream_fps = -1,
                        int ai_fps = -1,
                        int rtsp_rtp_over_tcp = 0);

    ~VideoStreamPipeline();

    /**
     * @brief Create and initialize the entire pipeline
     * @return 0 on success, negative value on failure
     */
    int Create();

    /**
     * @brief Acquire a frame from Streampipeline for AI
     * @param dump_res Output structure containing frame addresses
     */
    int GetFrame(DumpRes &dump_res);

    /**
     * @brief Release a previously acquired frame
     * @param dump_res Frame information to be released
     * @return 0 on success, negative value on failure
     */
    int ReleaseFrame(DumpRes &dump_res);

    /**
     * @brief Insert an OSD frame into the display pipeline,for ai result
     * @param osd_data Pointer to OSD pixel data
     * @return 0 on success, negative value on failure
     */
    int InsertFrame(void* osd_data);

    /**
     * @brief Destroy and release all pipeline resources
     * @return 0 on success, negative value on failure
     */
    int Destroy();

    /**
     * @brief Check if the video data stream reading is completed
     * @return True if the data stream has been read to the end (e.g., EOF of video file, stream closed),
     *         false if the stream is still running (e.g., live camera stream)
     */
    bool IsFinished();

    /**
     * @brief Get the actual target output resolution of the video pipeline
     * @param width [out] Output parameter to store the actual output frame width (in pixels)
     * @param height [out] Output parameter to store the actual output frame height (in pixels)
     * @note Valid only after successful call of Create() function, otherwise width/height will be set to -1
     */
    void GetFrameSize(int &width, int &height);

private:
    // PIMPL idiom: hide all implementation details
    VideoStreamPipelineImpl* pimpl_;
};

#endif // _VIDEO_STREAM_PIPELINE_H