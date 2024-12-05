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
    std::cout << "Usage: ./sample_rtspsever.elf [-v] [-s <sensor_type>] [-t <codec_type>] [-w <width>] [-h <height>] [-b <bitrate_kbps>] [-a <semitones>]" << std::endl;
    std::cout << "-v: enable video session" << std::endl;
    std::cout << "-s: the sensor type, default 7 :" << std::endl;
    std::cout << "       see camera sensor doc." << std::endl;
    std::cout << "-t: the video encoder type: h264/h265, default h265" << std::endl;
    std::cout << "-w: the video encoder width, default 1280" << std::endl;
    std::cout << "-h: the video encoder height, default 720" << std::endl;
    std::cout << "-b: the video encoder bitrate(kbps), default 2000" << std::endl;
    std::cout << "-a: pitch shift semitones [-12,12], default 0" << std::endl;
    exit(-1);
}

int parse_config(int argc, char *argv[], KdMediaInputConfig &config) {
    int result;
    opterr = 0;
    while ((result = getopt(argc, argv, "Hvs:n:t:w:h:b:a:")) != -1) {
        switch(result) {
        case 'H' : {
            Usage(); break;
        }
        case 'v' : {
            config.video_valid = true;
            break;
        }
        case 's' : {
            int n = atoi(optarg);
            if (n < 0 || n > 27) Usage();
            config.sensor_type = (k_vicap_sensor_type)n;
            config.video_valid = true;
            break;
        }
        case 't': {
            std::string s = optarg;
            if (s == "h264") config.video_type = KdMediaVideoType::kVideoTypeH264;
            else if (s == "h265") config.video_type = KdMediaVideoType::kVideoTypeH264;
            else Usage();
            config.video_valid = true;
            break;
        }
        case 'w': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.venc_width = n;
            config.video_valid = true;
            break;
        }
        case 'h': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.venc_height = n;
            config.video_valid = true;
            break;
        }
        case 'b': {
            int n = atoi(optarg);
            if (n < 0) Usage();
            config.bitrate_kbps = n;
            config.video_valid = true;
            break;
        }
        case 'a': {
            int n = atoi(optarg);
            if (n < -12 || n > 12) Usage();
            config.pitch_shift_semitones = n;
            break;
        }
        default: Usage(); break;
        }
    }
    if (config.video_valid) {
        // validate the parameters... TODO
        std::cout << "Validate the input config, not implemented yet, TODO." << std::endl;
    }
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
