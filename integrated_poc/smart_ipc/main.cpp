#include <iostream>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <signal.h>
#include <thread>
#include "smart_ipc.h"
#include "scoped_timing.hpp"

using namespace std::chrono_literals;

std::atomic<bool> g_exit_flag{false};

static void sigHandler(int sig_no) {
    g_exit_flag.store(true);
    printf("exit_flag true\n");
}

static void Usage() {
    std::cout << "Usage: ./smart_ipc.elf [-H] [-a <audio_sample>] [-c <channel_count>] [-t <codec_type>] [-w <width>] [-h <height>] [-b <bitrate_kbps>] [-C <connector_type>] [-A <ai_input_width>] [-I <ai_input_height>] [-K <kmodel_file>] [-T <obj_thresh>] [-N <nms_thresh>]" << std::endl;
    std::cout << "-H: display this help message" << std::endl;
    std::cout << "-a: the audio sample rate, default 8000" << std::endl;
    std::cout << "-c: the audio channel count, default 1" << std::endl;
    std::cout << "-t: the video encoder type: h264/h265, default h26" << std::endl;
    std::cout << "-w: the video encoder width, default 1280" << std::endl;
    std::cout << "-h: the video encoder height, default 720" << std::endl;
    std::cout << "-b: the video encoder bitrate(kbps), default 2000" << std::endl;
    std::cout << "-C: the video output connector type(0:HDMI,1:LCD), default HDMI" << std::endl;
    std::cout << "-A: the AI analysis input width, default 1280" << std::endl;
    std::cout << "-I: the AI analysis input height, default 720" << std::endl;
    std::cout << "-K: the kmodel file path,default face_detection_320.kmodel" << std::endl;
    std::cout << "-T: the face detection threshold,default 0.6" << std::endl;
    std::cout << "-N: the face detection NMS threshold,default 0.4" << std::endl;

    exit(-1);
}

int parse_config(int argc, char *argv[], KdMediaInputConfig &config) {
    int result;
    opterr = 0;
    while ((result = getopt(argc, argv, "Ha:c:t:w:h:b:C:A:I:K:T:N:")) != -1) {
        switch(result) {
        case 'H' : {
            Usage(); break;
        }
        case 'a': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.audio_samplerate = n;
            break;
        }
        case 'c': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.audio_channel_cnt = n;
            break;
        }
        case 't': {
            std::string s = optarg;
            if (s == "h264") config.video_type = KdMediaVideoType::kVideoTypeH264;
            else if (s == "h265") config.video_type = KdMediaVideoType::kVideoTypeH264;
            else Usage();
            break;
        }
        case 'w': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.venc_width = n;
            break;
        }
        case 'h': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.venc_height = n;
            break;
        }
        case 'b': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.bitrate_kbps = n;
            break;
        }
        case 'C': {
            int n = atoi(optarg);
            if (n == 0) {
                config.vo_connect_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
                config.osd_width = 1920;
                config.osd_height = 1080;
                config.vo_width = 1920;
                config.vo_height = 1080;
            } else if (n == 1) {
                config.vo_connect_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
                config.osd_width = 480;
                config.osd_height = 800;
                config.vo_width = 800;
                config.vo_height = 480;
            } else {
                Usage();
            }
            break;
        }

        case 'W': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.vo_width = n;
            break;
        }
        case 'D': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.vo_height = n;
            break;
        }
        case 'O': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.osd_width = n;
            break;
        }
        case 'P': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.osd_height = n;
            break;
        }
        case 'A': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.ai_width = n;
            break;
        }
        case 'I': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.ai_height = n;
            break;
        }
        case 'K': {
            config.kmodel_file = optarg;
            break;
        }
        case 'T': {
            float n = atof(optarg);
            if (n < 0) Usage();
            config.obj_thresh = n;
            break;
        }
        case 'N': {
            float n = atof(optarg);
            if (n < 0) Usage();
            config.nms_thresh = n;
            break;
        }
        default: Usage(); break;
        }
    }

    printf("Config parameters:\n");
    printf("Audio sample rate: %d\n", config.audio_samplerate);
    printf("Audio channel count: %d\n", config.audio_channel_cnt);
    printf("Video encoder type: %s\n", (config.video_type == KdMediaVideoType::kVideoTypeH264) ? "h264" : "h265");
    printf("Video encoder width: %d\n", config.venc_width);
    printf("Video encoder height: %d\n", config.venc_height);
    printf("Video encoder bitrate (kbps): %d\n", config.bitrate_kbps);
    printf("Video output connector type: %s\n", (config.vo_connect_type == LT9611_MIPI_4LAN_1920X1080_30FPS) ? "HDMI" : "LCD");
    printf("OSD width: %d\n", config.osd_width);
    printf("OSD height: %d\n", config.osd_height);
    printf("Video output width: %d\n", config.vo_width);
    printf("Video output height: %d\n", config.vo_height);
    printf("AI input width: %d\n", config.ai_width);
    printf("AI input height: %d\n", config.ai_height);
    printf("Kmodel file: %s\n", config.kmodel_file.c_str());
    printf("Face detection threshold: %f\n", config.obj_thresh);
    printf("Face detection NMS threshold: %f\n", config.nms_thresh);
    printf("\n");
    return 0;
}

int main(int argc, char *argv[]) {
    ScopedTiming * st = new ScopedTiming("total test", 1);
    signal(SIGINT, sigHandler);
    signal(SIGPIPE, SIG_IGN);
    g_exit_flag.store(false);

    KdMediaInputConfig config;
    parse_config(argc, argv, config);

    MySmartIPC *smartIPC = new MySmartIPC();
    if (!smartIPC || smartIPC->Init(config) < 0) {
        std::cout << "KdRtspServer Init failed." << std::endl;
        return -1;
    }

    smartIPC->Start();
    printf("SmartIPC started.\n");
    delete st;

    while (!g_exit_flag) {
        std::this_thread::sleep_for(100ms);
    }

    smartIPC->Stop();
    smartIPC->DeInit();
    delete smartIPC;
    return 0;
}
