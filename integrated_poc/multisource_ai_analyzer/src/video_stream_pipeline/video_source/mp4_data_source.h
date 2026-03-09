#ifndef MP4_DATA_SOURCE_H
#define MP4_DATA_SOURCE_H

#include "idata_source.h"
#include <string>
#include <pthread.h>
#include <atomic>

#include "mp4_format.h"


class MP4DataSource : public IDataSource {
public:
    MP4DataSource() ;
    ~MP4DataSource() override;

    // 实现基类接口
    int open(const std::string& url, int fps,const DataSourceCallbacks& callbacks) override;
    void close() override;
    int start() override;
    int stop() override;
    int get_width() const override;
    int get_height() const override;
    EncType get_stream_type() const override;

private:
    // 工作线程函数
    void working();
    static void* _worker_thread_entry(void* arg);

    std::string m_url;
    StreamCallback m_callback = nullptr;
    void* m_userData = nullptr;
    int m_width = 0;
    int m_height = 0;
    bool m_isOpened = false;
    KD_HANDLE m_mp4_handle = nullptr;
    std::atomic<bool> m_isRunning{false};
    pthread_t m_workerThread;
    int m_videoCodecId = -1;
    EncStream m_encstream;
    int m_output_fps{30};

};

#endif // MP4_DATA_SOURCE_H