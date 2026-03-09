#include "uvc_data_source.h"
#include "mpi_uvc_api.h"
#include <cstdio>

UVCDataSource::UVCDataSource(int width,int height)
{
    m_width = width;
    m_height = height;
}

UVCDataSource::~UVCDataSource() {
    stop();
    close();
}

int UVCDataSource::open(const std::string& url,int fps, const DataSourceCallbacks& callbacks) {
    int ret = -1;
    if (m_isOpened) {
        return false; // 已打开状态下不重复打开
    }

    m_isJpeg = true;
    m_fps = fps;
    // Store the original expected resolution from constructor
    int origin_w = m_width;
    int origin_h = m_height;

    struct uvc_format format = {m_width, m_height, m_isJpeg, 0};
    //Initial attempt to initialize UVC
    ret = uvc_init(&format);
    if (ret == 0 && (format.width < origin_w || format.height < origin_h)) {
        printf("UVCDataSource: Resolution %dx%d < target %dx%d. Retrying...\n",
               format.width, format.height, origin_w, origin_h);

        // Release resources from the failed attempt
        uvc_exit();

        // Force reset to standard 1080P as requested and retry
        format.width = 1920;
        format.height = 1080;
        ret = uvc_init(&format);
    }

    if (ret != 0) {
        printf("UVCDataSource: uvc_init failed\n");
        return -1;
    }

    printf("UVC camera resolution (init/actual) - %dx%d/%dx%d\n",
       m_width, m_height, format.width, format.height);

    // 更新实际协商的宽高
    m_width = format.width;
    m_height = format.height;

    m_url = url;
    m_callback = callbacks.mpp_cb;
    m_userData = callbacks.user_data;
    m_isOpened = true;

    return 0;
}

void UVCDataSource::close() {
    stop();
    if (m_isOpened) {
        uvc_exit();
        m_isOpened = false;
    }
}

int UVCDataSource::start()
{
    if (!m_isOpened || m_isDelivering.exchange(true)) {
        return -1;
    }

    if (uvc_start_stream() != 0) {
        m_isDelivering.store(false);
        return -1;
    }

    m_pullThread = std::thread(&UVCDataSource::_uvc_pull_loop, this);
    return 0;
}

int UVCDataSource::stop()
{
    m_isDelivering.store(false);
    if (m_pullThread.joinable()) {
        m_pullThread.join();
    }
    return 0;
}

int UVCDataSource::get_width() const {
    return m_width;
}

int UVCDataSource::get_height() const {
    return m_height;
}

EncType UVCDataSource::get_stream_type() const {
    return em_enc_jpeg;
}

void UVCDataSource::_uvc_pull_loop() {
    int ret = 0;
    while (m_isDelivering.load(std::memory_order_relaxed)) {
        struct uvc_frame frame;
        // 500ms timeout
        ret = uvc_get_frame(&frame, 500);
        if (ret != 0) {
            continue;
        }

        _deliver_frame(frame.v_stream);

        uvc_put_frame(&frame);
    }

    // 发送 EOF 标志
    if (m_callback) {
        k_vdec_stream eof;
        m_callback(eof, m_userData, true);
    }
}

void UVCDataSource::_deliver_frame(const k_vdec_stream& stream) {
    if (m_callback) {
        m_callback(stream, m_userData, false);
    }
}