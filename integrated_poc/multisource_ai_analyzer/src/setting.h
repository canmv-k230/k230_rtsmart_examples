#ifndef __SETTING_H__
#define __SETTING_H__

    #define DISPLAY_TYPE 'st7701'   // Select display type: 'st7701' or 'lt9611'
    #define RTSP_RTP_OVER_TCP 0     // 1=TCP, 0=UDP

#if DISPLAY_TYPE == 'st7701'

    // ISP (Image Signal Processor) output resolution
    #define ISP_WIDTH 1920
    #define ISP_HEIGHT 1080

    // Display configuration
    #define DISPLAY_MODE 1       // Display mode selector (platform-specific)
    #define DISPLAY_WIDTH 800    // Physical display width
    #define DISPLAY_HEIGHT 480   // Physical display height
    #define DISPLAY_ROTATE 1     // Display rotation enable (e.g., 90/180 degrees)

    // AI input frame configuration
    #define AI_FRAME_WIDTH 800
    #define AI_FRAME_HEIGHT 480
    #define AI_FRAME_CHANNEL 3   // Number of color channels (e.g., RGB)

    // OSD (On-Screen Display) configuration
    #define USE_OSD 1            // Enable OSD overlay
    #define OSD_WIDTH 800
    #define OSD_HEIGHT 480
    #define OSD_CHANNEL 4        // OSD color channels (e.g., ARGB)

#elif DISPLAY_TYPE == 'lt9611'

    // ISP (Image Signal Processor) output resolution
    #define ISP_WIDTH 1920
    #define ISP_HEIGHT 1080

    // Display configuration
    #define DISPLAY_MODE 0       // Display mode selector (platform-specific)
    #define DISPLAY_WIDTH 1920   // Physical display width
    #define DISPLAY_HEIGHT 1080  // Physical display height
    #define DISPLAY_ROTATE 0     // Display rotation disabled

    // AI input frame configuration
    #define AI_FRAME_WIDTH 1920
    #define AI_FRAME_HEIGHT 1080
    #define AI_FRAME_CHANNEL 3   // Number of color channels (e.g., RGB)

    // OSD (On-Screen Display) configuration
    #define USE_OSD 1            // Enable OSD overlay
    #define OSD_WIDTH 1920
    #define OSD_HEIGHT 1080
    #define OSD_CHANNEL 4        // OSD color channels (e.g., ARGB)

#endif

#include <string>

// 默认参数配置结构体
struct AppSettings {
    // 模型路径
    std::string det_model_path = "yolov8n_320.kmodel";     // YOLOv8 检测模型
    std::string reid_model_path = "feature.kmodel";        // ReID 特征模型
    
    // 检测参数
    float score_thresh = 0.4f;         // 检测置信度阈值
    float nms_thresh = 0.6f;           // NMS 阈值
    
    // 跟踪参数
    float track_high_thresh = 0.6f;    // 高置信度阈值
    float track_low_thresh = 0.2f;     // 低置信度阈值
    float new_track_thresh = 0.75f;    // 新建轨迹阈值
    int frame_buffer = 600;            // 轨迹缓冲大小
    float match_thresh = 0.9f;         // 匹配代价阈值
    float proximity_thresh = 0.3f;     // 邻近匹配阈值
    float appearance_thresh = 0.2f;    // 外观特征距离阈值
    float lambda = 0.99f;              // IOU/ReID 权重因子
    
    // 运行参数
    int debug_mode = 0;                // 调试模式：0/1/2
    std::string video_path;            // 视频源路径（必需）
    
    // 打印当前配置
    void print() const;
};

// 命令行参数解析器
class CmdLineParser {
public:
    static bool parse(int argc, char* argv[], AppSettings& settings);
    static void print_usage(const char* name);
};

#endif // __SETTING_H__
