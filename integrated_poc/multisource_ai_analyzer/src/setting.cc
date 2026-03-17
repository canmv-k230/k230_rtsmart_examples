#include "setting.h"
#include <iostream>
#include <cstring>

using std::cout;
using std::endl;
using std::string;
using std::cerr;

void AppSettings::print() const {
    cout << "=== App Settings ===" << endl;
    cout << "Det Model: " << det_model_path << endl;
    cout << "ReID Model: " << reid_model_path << endl;
    cout << "Score Thresh: " << score_thresh << endl;
    cout << "NMS Thresh: " << nms_thresh << endl;
    cout << "Track High Thresh: " << track_high_thresh << endl;
    cout << "Track Low Thresh: " << track_low_thresh << endl;
    cout << "New Track Thresh: " << new_track_thresh << endl;
    cout << "Frame Buffer: " << frame_buffer << endl;
    cout << "Match Thresh: " << match_thresh << endl;
    cout << "Proximity Thresh: " << proximity_thresh << endl;
    cout << "Appearance Thresh: " << appearance_thresh << endl;
    cout << "Lambda: " << lambda << endl;
    cout << "Debug Mode: " << debug_mode << endl;
    cout << "Video Path: " << video_path << endl;
    cout << "====================" << endl;
}

void CmdLineParser::print_usage(const char* name)
{
    cout << "Usage: " << name << " [OPTIONS] <video_path>" << endl
         << endl
         << "Required:" << endl
         << "  <video_path>              视频源路径，支持以下类型：" << endl
         << "                            - \"realtime\"   实时摄像头采集" << endl
         << "                            - \"*.mp4\"      MP4 视频文件" << endl
         << "                            - \"UVC\"        UVC 摄像头" << endl
         << "                            - \"rtsp://*\"   RTSP 网络视频流" << endl
         << endl
         << "Optional:" << endl
         << "  --det-model <path>        YOLOv8 检测模型路径 (默认：yolov8n_320.kmodel)" << endl
         << "  --score-thres <float>     检测置信度阈值 (默认：0.4)" << endl
         << "  --nms-thres <float>       NMS 阈值 (默认：0.6)" << endl
         << "  --reid-model <path>       ReID 特征模型路径 (默认：feature.kmodel)" << endl
         << "  --track-high <float>      高置信度阈值 (默认：0.6)" << endl
         << "  --track-low <float>       低置信度阈值 (默认：0.2)" << endl
         << "  --new-track <float>       新建轨迹阈值 (默认：0.75)" << endl
         << "  --frame-buffer <int>      轨迹缓冲大小 (默认：600)" << endl
         << "  --match-thresh <float>    匹配代价阈值 (默认：0.9)" << endl
         << "  --proximity <float>       邻近匹配阈值 (默认：0.3)" << endl
         << "  --appearance <float>      外观特征距离阈值 (默认：0.2)" << endl
         << "  --lambda <float>          IOU/ReID 权重因子 (默认：0.99)" << endl
         << "  --debug <0|1|2>           调试模式 (默认：0)" << endl
         << "  --help                    显示此帮助信息" << endl
         << endl
         << "Examples:" << endl
         << "  " << name << " \"rtsp://192.168.1.100:554/stream1\"" << endl
         << "  " << name << " --det-model yolov8n_320.kmodel --score-thres 0.5 \"realtime\"" << endl
         << "  " << name << " --debug 1 --track-high 0.7 test.mp4" << endl
         << endl;
}

bool CmdLineParser::parse(int argc, char* argv[], AppSettings& settings)
{
    int i = 1;  // 跳过程序名
    
    // 解析可选参数
    while (i < argc) {
        string arg = argv[i];
        
        // 检查是否到达位置参数（视频路径）
        if (arg[0] != '-' || arg == "-") {
            break;
        }
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;  // 显示帮助后退出
        }
        else if (arg == "--det-model") {
            if (i + 1 >= argc) {
                cerr << "Error: --det-model requires a path argument" << endl;
                return false;
            }
            settings.det_model_path = argv[++i];
        }
        else if (arg == "--score-thres") {
            if (i + 1 >= argc) {
                cerr << "Error: --score-thres requires a float argument" << endl;
                return false;
            }
            settings.score_thresh = atof(argv[++i]);
        }
        else if (arg == "--nms-thres") {
            if (i + 1 >= argc) {
                cerr << "Error: --nms-thres requires a float argument" << endl;
                return false;
            }
            settings.nms_thresh = atof(argv[++i]);
        }
        else if (arg == "--reid-model") {
            if (i + 1 >= argc) {
                cerr << "Error: --reid-model requires a path argument" << endl;
                return false;
            }
            settings.reid_model_path = argv[++i];
        }
        else if (arg == "--track-high") {
            if (i + 1 >= argc) {
                cerr << "Error: --track-high requires a float argument" << endl;
                return false;
            }
            settings.track_high_thresh = atof(argv[++i]);
        }
        else if (arg == "--track-low") {
            if (i + 1 >= argc) {
                cerr << "Error: --track-low requires a float argument" << endl;
                return false;
            }
            settings.track_low_thresh = atof(argv[++i]);
        }
        else if (arg == "--new-track") {
            if (i + 1 >= argc) {
                cerr << "Error: --new-track requires a float argument" << endl;
                return false;
            }
            settings.new_track_thresh = atof(argv[++i]);
        }
        else if (arg == "--frame-buffer") {
            if (i + 1 >= argc) {
                cerr << "Error: --frame-buffer requires an integer argument" << endl;
                return false;
            }
            settings.frame_buffer = atol(argv[++i]);
        }
        else if (arg == "--match-thresh") {
            if (i + 1 >= argc) {
                cerr << "Error: --match-thresh requires a float argument" << endl;
                return false;
            }
            settings.match_thresh = atof(argv[++i]);
        }
        else if (arg == "--proximity") {
            if (i + 1 >= argc) {
                cerr << "Error: --proximity requires a float argument" << endl;
                return false;
            }
            settings.proximity_thresh = atof(argv[++i]);
        }
        else if (arg == "--appearance") {
            if (i + 1 >= argc) {
                cerr << "Error: --appearance requires a float argument" << endl;
                return false;
            }
            settings.appearance_thresh = atof(argv[++i]);
        }
        else if (arg == "--lambda") {
            if (i + 1 >= argc) {
                cerr << "Error: --lambda requires a float argument" << endl;
                return false;
            }
            settings.lambda = atof(argv[++i]);
        }
        else if (arg == "--debug") {
            if (i + 1 >= argc) {
                cerr << "Error: --debug requires an integer argument (0/1/2)" << endl;
                return false;
            }
            settings.debug_mode = atoi(argv[++i]);
            if (settings.debug_mode < 0 || settings.debug_mode > 2) {
                cerr << "Error: debug mode must be 0, 1, or 2" << endl;
                return false;
            }
        }
        else {
            cerr << "Error: Unknown option: " << arg << endl;
            print_usage(argv[0]);
            return false;
        }
        
        i++;
    }
    
    // 解析必需的视频路径
    if (i >= argc) {
        cerr << "Error: Missing required video_path argument" << endl;
        print_usage(argv[0]);
        return false;
    }
    
    settings.video_path = argv[i];
    
    // 检查是否有额外参数
    if (i + 1 < argc) {
        cerr << "Error: Unexpected argument: " << argv[i + 1] << endl;
        print_usage(argv[0]);
        return false;
    }
    
    return true;
}
