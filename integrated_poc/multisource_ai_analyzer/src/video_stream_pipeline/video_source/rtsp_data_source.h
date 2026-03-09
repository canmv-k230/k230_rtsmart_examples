// rtsp_data_source.h
#ifndef RTSP_DATA_SOURCE_H
#define RTSP_DATA_SOURCE_H

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <queue>
#include <string>
#include "idata_source.h"

class RTSPDataSource : public IDataSource {
public:
    RTSPDataSource(bool enable_rtp_over_tcp = true);
    ~RTSPDataSource() override;

    int open(const std::string& url, int fps,const DataSourceCallbacks& callbacks);
    void close();
    int start();
    int stop();

    int get_width() const;
    int get_height() const;
    EncType get_stream_type() const; // 0: H264, 1: H265

private:
    bool _open_and_probe();
    void _ffmpeg_pull_loop();
    void _deliver_frame(const EncStream& frame);

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // RTSP_DATA_SOURCE_H