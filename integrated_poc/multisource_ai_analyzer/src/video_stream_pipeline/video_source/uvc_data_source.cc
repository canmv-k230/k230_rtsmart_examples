#include "uvc_data_source.h"
#include "mpi_uvc_api.h"
#include <cstdio>

namespace {
constexpr unsigned int kFallbackWidth = 1920;
constexpr unsigned int kFallbackHeight = 1080;
}

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
        return -1; // 已打开状态下不重复打开
    }

    if (m_width <= 0 || m_height <= 0) {
        printf("UVCDataSource: Invalid target resolution %dx%d\n", m_width, m_height);
        return -1;
    }

    static_cast<void>(fps);
    m_isJpeg = true;

    // Store the original expected resolution from constructor
    const unsigned int origin_w = static_cast<unsigned int>(m_width);
    const unsigned int origin_h = static_cast<unsigned int>(m_height);

    struct uvc_format format = {
        origin_w,
        origin_h,
        m_isJpeg ? USBH_VIDEO_FOURCC_MJPEG : USBH_VIDEO_FOURCC_YUY2,
        0
    };
    //Initial attempt to initialize UVC
    ret = uvc_host_init(&format);
    if (ret == 0 && (format.width < origin_w || format.height < origin_h)) {
        printf("UVCDataSource: Resolution %ux%u < target %ux%u. Retrying...\n",
               format.width, format.height, origin_w, origin_h);

        // Release resources from the failed attempt
        uvc_host_exit();

        // Force reset to standard 1080P as requested and retry
        format.width = kFallbackWidth;
        format.height = kFallbackHeight;
        ret = uvc_host_init(&format);
    }

    if (ret != 0) {
        printf("UVCDataSource: uvc_host_init failed\n");
        return -1;
    }

    printf("UVC camera resolution (init/actual) - %dx%d/%ux%u\n",
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
        uvc_host_exit();
        m_isOpened = false;
    }
}

int UVCDataSource::start()
{
    if (!m_isOpened || m_isDelivering.exchange(true)) {
        return -1;
    }

    if (uvc_host_start_stream() != 0) {
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
        ret = uvc_host_get_frame(&frame, 500);
        if (ret != 0) {
            continue;
        }

        _deliver_frame(frame.v_stream);

        uvc_host_put_frame(&frame);
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
