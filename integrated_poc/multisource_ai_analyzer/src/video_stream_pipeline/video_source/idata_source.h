#ifndef I_DATA_SOURCE_H
#define I_DATA_SOURCE_H
#include <string>
#include "k_vdec_comm.h"

enum EncType{
    em_enc_264,
    em_enc_265,
    em_enc_jpeg,
};

struct EncStream {
    uint8_t* data;
    size_t size;
    int64_t pts;
    bool is_key_frame;
    EncType enctype;
};

using StreamCallback = void (*)(const EncStream& stream, void* user_data, bool is_end);
using MppStreamCallback = void (*)(const k_vdec_stream& stream, void* user_data, bool is_end);
struct DataSourceCallbacks {
    StreamCallback stream_cb = nullptr;      // 标准编码流回调
    MppStreamCallback mpp_cb = nullptr;      // MPP 专用流回调
    void* user_data = nullptr;               // 用户自定义数据指针
};

class IDataSource {
public:
    virtual ~IDataSource() = default;

    // 打开数据源并设置帧回调
    virtual int open(const std::string& url, int fps,const DataSourceCallbacks& callbacks) = 0;

    virtual int start() = 0;
    virtual int stop() = 0;

    // 关闭数据源（停止回调）
    virtual void close() = 0;

    // 获取视频宽高（需在open成功后有效）
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
    virtual EncType get_stream_type()const = 0;
};

#endif // I_DATA_SOURCE_H