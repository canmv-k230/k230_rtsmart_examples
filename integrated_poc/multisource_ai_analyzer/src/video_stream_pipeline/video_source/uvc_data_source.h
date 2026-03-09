#ifndef UVC_DATA_SOURCE_H
#define UVC_DATA_SOURCE_H

#include "idata_source.h"
#include <string>
#include <thread>
#include <atomic>
#include <memory>

class UVCDataSource : public IDataSource {
public:
    UVCDataSource(int width,int height);
    ~UVCDataSource() override;

    // 实现基类接口
    int open(const std::string& url,int fps, const DataSourceCallbacks& callbacks) override;
    void close() override;
    int start() override;
    int stop() override;
    int get_width() const override;
    int get_height() const override;
    EncType get_stream_type() const override;

private:
    void _uvc_pull_loop();
    void _deliver_frame(const k_vdec_stream& stream);
    std::string m_url;                  // 数据源URL
    MppStreamCallback m_callback{nullptr}; // 帧回调函数
    void* m_userData{nullptr};         // 用户数据

    int m_width{0};                    // 视频宽度
    int m_height{0};                   // 视频高度
    int m_fps{0};
    bool m_isOpened{false};            // 是否已打开
    bool m_isJpeg{false};

    std::atomic<bool> m_isDelivering{false};
    std::thread m_pullThread;
};

#endif // UVC_DATA_SOURCE_H