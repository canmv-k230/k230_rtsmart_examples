#include "mpp_pipeline.h"
#include <iostream>
#include <unistd.h>
#include "mpi_vb_api.h"
#include "mpi_connector_api.h"
#include "mpi_vdec_api.h"
#include "mpi_nonai_2d_api.h"
#include "mpi_sys_api.h"
#include "mpi_vo_api.h"

#define MAX_WIDTH 1920
#define MAX_HEIGHT 1088
#define STREAM_BUF_SIZE MAX_WIDTH*MAX_HEIGHT
#define FRAME_BUF_SIZE MAX_WIDTH*MAX_HEIGHT*2
#define INPUT_BUF_CNT   4
#define OUTPUT_BUF_CNT  6
#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align)-1))
#define ALIGN_1K 0x400

typedef struct {
    k_vo_layer_id layer_id;
    int width;
    int height;
    int ratation_90;
    k_pixel_format format;
} sample_vo_info;

static k_s32 osd_vb_create_pool(int width,int height)
{
    k_vb_pool_config pool_config;

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = 1;
    pool_config.blk_size = ALIGN_UP((width * height * 4), 0x1000);
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

static k_s32 stream_vb_create_pool()
{
    k_vb_pool_config pool_config;

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = INPUT_BUF_CNT;
    pool_config.blk_size = STREAM_BUF_SIZE;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

static k_s32 vdec_vb_create_pool()
{
    k_vb_pool_config pool_config;

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = OUTPUT_BUF_CNT;
    pool_config.blk_size = FRAME_BUF_SIZE;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

static k_s32 csc_vb_create_pool()
{
    k_vb_pool_config pool_config;

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = 8;
    pool_config.blk_size = ALIGN_UP((MAX_WIDTH * MAX_HEIGHT * 3), 0x400);
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

static k_s32 init_layer_ex(sample_vo_info* vo_info)
{
    k_vo_pub_attr attr;
    k_u32 ret = 0;

    kd_mpi_vo_disable_layer(vo_info->layer_id);

    k_vo_layer_attr vo_layer_attr;
    memset(&vo_layer_attr, 0, sizeof(vo_layer_attr));
    vo_layer_attr.layer_id           = vo_info->layer_id;
    vo_layer_attr.position.x         = 0;
    vo_layer_attr.position.y         = 0;
    vo_layer_attr.img_size.width     = vo_info->width;
    vo_layer_attr.img_size.height    = vo_info->height;
    vo_layer_attr.pixel_format       = vo_info->format;
    vo_layer_attr.func               = (vo_info->ratation_90 == 1) ? GDMA_ROTATE_DEGREE_90 :GDMA_ROTATE_DEGREE_0;
    vo_layer_attr.rot_buf_nr         = (vo_info->ratation_90 == 1) ? 2:0; /* 旋转时使用少量 GSDMA buffer */
    vo_layer_attr.global_alpha       = 0xff;

    printf("sample_vo_init: layer_id %d, width %d, height %d, pixel_format %d, func %d,buf_nr:%d\n",
           vo_layer_attr.layer_id, vo_layer_attr.img_size.width, vo_layer_attr.img_size.height,
           vo_layer_attr.pixel_format, vo_layer_attr.func,vo_layer_attr.rot_buf_nr);

    ret = kd_mpi_vo_set_layer_attr(vo_info->layer_id,&vo_layer_attr);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_set_layer_attr failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vo_enable_layer(vo_info->layer_id);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_enable_layer failed, ret=%d\n", ret);
        return ret;
    }
    //exit ;
    return 0;
}

MppPipeline::MppPipeline(){
    width_ = 0;
    height_ = 0;
    enc_type_ = em_enc_264;

    vdec_pool_id_ = VB_INVALID_POOLID;
    csc_pool_id_ = VB_INVALID_POOLID;
    osd_pool_id_ = VB_INVALID_POOLID;
    status_ = MppPipelineStatus::UNINITIALIZED;
    connector_type_ = EM_VO_HDMI;
    stream_pool_id_ = VB_INVALID_POOLID;
    rgb_frame_callback_ = nullptr;
    user_data_ = nullptr;

    _init_vb();
    vb_exit_ = false;
}

MppPipeline::~MppPipeline()
{
    stop();
    if (!vb_exit_)
    {
        vb_exit_ = true;
        kd_mpi_vb_exit();
    }
}

int MppPipeline::init(int w, int h, bool enable_dsl,bool rotation_90, EncType type,MppVoType vo_type) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (status_ != MppPipelineStatus::UNINITIALIZED) {
        std::cerr << "Pipeline already initialized!" << std::endl;
        return -1;
    }

    width_ = w;
    height_ = h;
    enc_type_ = type;
    connector_type_ = vo_type;
    enable_dsl_ = enable_dsl;
    rotation_90_ = rotation_90;

    printf("%s width:%d,height:%d,rotation_90:%d,dsl:%d\n",__func__, width_,height_,rotation_90_,enable_dsl_);

    // 1. 初始化视频缓冲池（VB）
    if (_init_vb_module()) {
        status_ = MppPipelineStatus::ERROR;
        return -1;
    }

    // 2. 初始化解码器（VDEC）
    if (_init_vdec()) {
        status_ = MppPipelineStatus::ERROR;
        return -1;
    }

    // 3. 初始化显示输出（VO）
    if (_init_vo()) {
        status_ = MppPipelineStatus::ERROR;
        return -1;
    }

    if (_init_osd(osd_id_)){
        status_ = MppPipelineStatus::ERROR;
        return -1;
    }

    // 4. 初始化色彩转换（CSC）
    if (_init_csc()) {
        status_ = MppPipelineStatus::ERROR;
        return -1;
    }

    status_ = MppPipelineStatus::INITIALIZING;
    std::cout << "MPP pipeline initialized successfully" << std::endl;
    return 0;
}

void MppPipeline::set_rgb_callback(RgbFrameCallback callback,void* user_data)
{
    rgb_frame_callback_ = callback;
    user_data_ = user_data;
}

int MppPipeline::start()
{
    k_s32 ret;
    std::lock_guard<std::mutex> lock(mtx_);
    if (status_ != MppPipelineStatus::INITIALIZING) {
        std::cerr << "Pipeline is not  initialized!" << std::endl;
        return -1;
    }

    // 绑定管道组件
    if (_bind_pipeline()) {
        status_ = MppPipelineStatus::ERROR;
        return -1;
    }

    //vdec start
    ret = kd_mpi_vdec_start_chn(vdec_chn_id_);
    if (ret) {
        printf("kd_mpi_vdec_start_chn fail, ret = %d\n", ret);
        return -1;
    }

    //csc start
    ret = kd_mpi_nonai_2d_start_chn(csc_chn_id_);
    if (ret) {
        printf("kd_mpi_nonai_2d_start_chn fail, ret = %d\n", ret);
        return -1;
    }

    // start get csc frame
    thread_running_ = true;

    ret = pthread_create(&frame_process_thread_, nullptr,
                            &MppPipeline::_frame_process_thread_entry, this);
    if (ret != 0) {
        printf("pthread_create failed, ret=%d\n", ret);
        return -1;
    }

    status_ = MppPipelineStatus::RUNNING;
    return 0;
}

int MppPipeline::_init_vb() {
    k_s32 ret = 0;
    k_u32 pool_id;
    k_vb_config config;

    // ---------------------------- 配置视频缓冲区（Video Buffer） -----------------------------------
    memset(&config, 0, sizeof(k_vb_config));
    config.max_pool_cnt = 64;

    // 设置 VB 配置
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        //return ret;
    }

    // 设置 VB 附加配置
    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("vb_set_supplement_config failed ret:%d\n", ret);
        //return ret;
    }

    // 初始化 VB 系统
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        //return ret;
    }

    return 0;
}

int MppPipeline::_init_vb_module() {
    k_s32 ret = 0;

    stream_pool_id_ = stream_vb_create_pool();
    if (stream_pool_id_ == VB_INVALID_POOLID) {
        printf("fail to create stream pool\n");
        return -1;
    }

    osd_pool_id_ = osd_vb_create_pool(width_,height_);
    if (osd_pool_id_ == VB_INVALID_POOLID) {
        printf("fail to create osd pool\n");
        return -1;
    }

    vdec_pool_id_ = vdec_vb_create_pool();
    if (vdec_pool_id_ == VB_INVALID_POOLID) {
        printf("fail to create vdec pool\n");
        return -1;
    }

    csc_pool_id_ = csc_vb_create_pool();
    if (csc_pool_id_ == VB_INVALID_POOLID) {
        printf("fail to create csc pool\n");
        return -1;
    }

    printf("osd_pool_id_: %u, stream_pool_id_: %u, vdec_pool_id_: %u, csc_pool_id_: %u\n",
           osd_pool_id_, stream_pool_id_, vdec_pool_id_,  csc_pool_id_);

    return 0;
}

int MppPipeline::_init_vdec() {
    k_vdec_chn_attr vdec_attr;
    k_payload_type codec_type;
    k_s32 ret = 0;

    // 根据编码类型设置解码器类型
    switch (enc_type_) {
        case em_enc_264:  codec_type = K_PT_H264; break;
        case em_enc_265:  codec_type = K_PT_H265; break;
        case em_enc_jpeg: codec_type = K_PT_JPEG; break;
        default:
            std::cerr << "Unsupported encoding type!" << std::endl;
            return false;
    }

    //attach pool vb
    kd_mpi_vdec_attach_vb_pool(vdec_chn_id_,vdec_pool_id_);

    vdec_attr.pic_width = MAX_WIDTH;
    vdec_attr.pic_height = MAX_HEIGHT;
    vdec_attr.stream_buf_size = STREAM_BUF_SIZE;
    vdec_attr.type = codec_type;

    // 创建解码通道
    ret = kd_mpi_vdec_create_chn(vdec_chn_id_, &vdec_attr);
    if (ret) {
        printf("kd_mpi_vdec_create_chn fail, ret = %d\n", ret);
        return -1;
    }

    if (enable_dsl_){
        k_vdec_downscale dsl;
        dsl.dsl_mode = K_VDEC_DSL_MODE_BY_SIZE;
        dsl.dsl_size.dsl_frame_width = width_;
        dsl.dsl_size.dsl_frame_height = height_;
        kd_mpi_vdec_set_downscale(vdec_chn_id_,&dsl);
    }

    // if (rotation_90_){
    //     kd_mpi_vdec_set_rotation(vdec_chn_id_,K_VPU_ROTATION_90);
    // }

    printf("vdec(%d) initialized ok,codec type:%d\n",vdec_chn_id_,codec_type);
    return 0;
}

int MppPipeline::_init_osd(k_vo_layer_id osd_id)
{
    sample_vo_info vo_info;
    vo_info.layer_id = osd_id_;
    vo_info.width = width_;
    vo_info.height = height_;
    vo_info.format = PIXEL_FORMAT_ARGB_8888;
    if (connector_type_ == EM_VO_LCD){
        vo_info.ratation_90 = 1;
    }
    else{
        vo_info.ratation_90 = 0;
    }

    printf("vo layer init: layer_id %d, width %d, height %d, ratation_90 %d\n",
           vo_info.layer_id, vo_info.width, vo_info.height, vo_info.ratation_90);
    init_layer_ex(&vo_info);
    return 0;
}

int MppPipeline::_init_vo() {
    k_connector_info connector_info;
    k_connector_type connector_type;
    k_s32 ret;

    memset(&connector_info, 0, sizeof(k_connector_info));

    if (connector_type_ == EM_VO_HDMI){
        connector_type = LT9611_MIPI_4LAN_1920X1080_60FPS;
        //connector_type = LT9611_MIPI_4LAN_1920X1080_60FPS;
    }
    else if (connector_type_ == EM_VO_LCD){
        connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
    }

    // 获取屏幕配置信息（如名称、分辨率等）
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("the connector type not supported!\n");
        return -1;
    }

    // 打开屏幕连接器设备
    k_s32 connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return -1;
    }

    // 打开屏幕电源并初始化连接器
    kd_mpi_connector_power_set(connector_fd, K_TRUE);
    kd_mpi_connector_init(connector_fd, connector_info);

    sample_vo_info vo_info;
    vo_info.layer_id = vo_layer_chn_id_;
    vo_info.width = width_;
    vo_info.height = height_;
    vo_info.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    if (connector_type == ST7701_V1_MIPI_2LAN_480X800_30FPS || connector_type == HX8377_V2_MIPI_4LAN_1080X1920_30FPS){
        vo_info.ratation_90 = 1;
    }
    else{
        vo_info.ratation_90 = 0;
    }

    printf("vo layer init: layer_id %d, width %d, height %d, ratation_90 %d\n",
           vo_info.layer_id, vo_info.width, vo_info.height, vo_info.ratation_90);
    init_layer_ex(&vo_info);

    return 0;
}

int MppPipeline::_init_csc() {
    k_s32 ret = 0;
    k_nonai_2d_chn_attr attr_2d;

    kd_mpi_nonai_2d_attach_vb_pool(csc_chn_id_,csc_pool_id_);

    attr_2d.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d.dst_fmt = PIXEL_FORMAT_RGB_888_PLANAR;
    ret = kd_mpi_nonai_2d_create_chn(csc_chn_id_, &attr_2d);
    if (ret) {
        printf("kd_mpi_nonai_2d_create_chn fail, ret = %d\n", ret);
        return -1;
    }

    printf("csc(%d) initialized ok\n",csc_chn_id_);
    return 0;
}

int MppPipeline::_bind_pipeline() {
    k_s32 ret;
    k_mpp_chn vdec_mpp_chn;
    k_mpp_chn vo_mpp_chn;
    k_mpp_chn csc_mpp_chn;

    vdec_mpp_chn.mod_id = K_ID_VDEC;
    vdec_mpp_chn.dev_id = 0;
    vdec_mpp_chn.chn_id = vdec_chn_id_;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = 0;
    vo_mpp_chn.chn_id = vo_layer_chn_id_;
    csc_mpp_chn.mod_id = K_ID_NONAI_2D;
    csc_mpp_chn.dev_id = 0;
    csc_mpp_chn.chn_id = csc_chn_id_;

    ret = kd_mpi_sys_bind(&vdec_mpp_chn, &vo_mpp_chn);
    if (ret){
        printf("kd_mpi_sys_bind vdec->vo failed\n");
        return -1;
    }

    ret = kd_mpi_sys_bind(&vdec_mpp_chn, &csc_mpp_chn);
    if (ret){
        printf("kd_mpi_sys_bind vdec->csc failed\n");
        return -1;
    }

    return 0;
}


int MppPipeline::_unbind_pipeline() {
    k_s32 ret;
    k_mpp_chn vdec_mpp_chn;
    k_mpp_chn vo_mpp_chn;
    k_mpp_chn csc_mpp_chn;

    vdec_mpp_chn.mod_id = K_ID_VDEC;
    vdec_mpp_chn.dev_id = 0;
    vdec_mpp_chn.chn_id = vdec_chn_id_;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = 0;
    vo_mpp_chn.chn_id = vo_layer_chn_id_;
    csc_mpp_chn.mod_id = K_ID_NONAI_2D;
    csc_mpp_chn.dev_id = 0;
    csc_mpp_chn.chn_id = csc_chn_id_;

    ret = kd_mpi_sys_unbind(&vdec_mpp_chn, &vo_mpp_chn);
    if (ret){
        printf("kd_mpi_sys_bind vdec->vo failed\n");
        return -1;
    }

    ret = kd_mpi_sys_unbind(&vdec_mpp_chn, &csc_mpp_chn);
    if (ret){
        printf("kd_mpi_sys_bind vdec->csc failed\n");
        return -1;
    }

    return 0;
}

int MppPipeline::decode_stream(const EncStream& stream) {
    k_s32 ret;
    k_vdec_stream vdec_stream;
    k_vb_blk_handle handle;

    std::lock_guard<std::mutex> lock(mtx_);
    if (status_ != MppPipelineStatus::RUNNING) {
        std::cerr << "Pipeline not running!" << std::endl;
        return -1;
    }

    while (1) {
        handle = kd_mpi_vb_get_block(stream_pool_id_, STREAM_BUF_SIZE, NULL);
        if (handle != VB_INVALID_HANDLE) {
            break;
        }
        usleep(1000*5);
    }

    stream_phys_addr_ = kd_mpi_vb_handle_to_phyaddr(handle);
    stream_virt_addr_ = (k_u8 *)kd_mpi_sys_mmap_cached(stream_phys_addr_, STREAM_BUF_SIZE);

    memcpy(stream_virt_addr_,stream.data,stream.size);
    ret = kd_mpi_sys_mmz_flush_cache(stream_phys_addr_, stream_virt_addr_, stream.size);
    if (ret){
        printf("%s kd_mpi_sys_mmz_flush_cache failed\n",__func__);
        return -1;
    }

    vdec_stream.end_of_stream = K_FALSE;
    vdec_stream.phy_addr = stream_phys_addr_;
    vdec_stream.len = stream.size;
    vdec_stream.pts = stream.pts;

    ret = kd_mpi_vdec_send_stream(vdec_chn_id_, &vdec_stream, -1);
    if (ret){
        printf("%s kd_mpi_vdec_send_stream failed\n",__func__);
        return -1;
    }

    ret = kd_mpi_sys_munmap((void *)stream_virt_addr_, STREAM_BUF_SIZE);
    if (ret){
        printf("%s kd_mpi_sys_munmap failed\n",__func__);
        return -1;
    }

    ret = kd_mpi_vb_release_block(handle);
    if (ret){
        printf("%s kd_mpi_vb_release_block failed\n",__func__);
        return -1;
    }

    return 0;
}

int MppPipeline::decode_stream(const k_vdec_stream& stream)
{
    k_s32 ret;
    k_vdec_stream& vdec_stream = const_cast<k_vdec_stream&>(stream);

    std::lock_guard<std::mutex> lock(mtx_);
    if (status_ != MppPipelineStatus::RUNNING) {
        std::cerr << "Pipeline not running!" << std::endl;
        return -1;
    }

    ret = kd_mpi_vdec_send_stream(vdec_chn_id_, &vdec_stream, -1);
    if (ret){
        printf("%s kd_mpi_vdec_send_stream failed\n",__func__);
        return -1;
    }

    return 0;
}

void* MppPipeline::_frame_process_thread_entry(void* arg)
{
    MppPipeline* self = static_cast<MppPipeline*>(arg);
    self->_process_decoded_frames();
    return nullptr;
}

void MppPipeline::_process_decoded_frames() {
    k_s32 ret;

    // FPS 统计相关变量
    std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();
    uint64_t frame_count = 0;
    const double kPrintIntervalSec = 1.0; // 每秒打印一次

    while(thread_running_){
        ret = kd_mpi_nonai_2d_get_frame(csc_chn_id_, &rgb_frame_info_, 500);
        if (ret) {
            //printf("kd_mpi_nonai_2d_get_frame failed. %d\n", ret);
            continue;
        }

        if (rgb_frame_callback_ != nullptr){
            rgb_frame_callback_(rgb_frame_info_,user_data_);
        }
        else{
            ret = kd_mpi_nonai_2d_release_frame(csc_chn_id_,&rgb_frame_info_);
            if (ret) {
                printf("kd_mpi_nonai_2d_release_frame failed. %d\n", ret);
            }
        }

        // FPS 统计
        ++frame_count;
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - last_time;
        if (elapsed.count() >= kPrintIntervalSec) {
            double fps = static_cast<double>(frame_count) / elapsed.count();
            printf("[MPP Pipeline] Output FPS: %.2f\n", fps);
            frame_count = 0;
            last_time = now;
        }
    }

    //clean
    while(1){
        std::lock_guard<std::mutex> lock(release_frame_mtx_);
        ret = kd_mpi_nonai_2d_get_frame(csc_chn_id_, &rgb_frame_info_, 0);
        if (ret){
            break;
        }

        kd_mpi_nonai_2d_release_frame(csc_chn_id_,&rgb_frame_info_);
    }

}

int MppPipeline::_release_rgb_frame(k_video_frame_info* rgb_finfo)
{
    k_s32 ret = 0;
    {
        std::lock_guard<std::mutex> lock(release_frame_mtx_);
        ret = kd_mpi_nonai_2d_release_frame(csc_chn_id_,rgb_finfo);
    }

    return ret;
}

int MppPipeline::stop() {
    k_s32 ret;
    k_vdec_stream vdec_stream;
    k_vb_blk_handle handle;
    k_vdec_chn_status status;
    memset(&status,0,sizeof(status));

    std::lock_guard<std::mutex> lock(mtx_);
    if (status_ == MppPipelineStatus::STOPPED || status_ == MppPipelineStatus::UNINITIALIZED) return -1;

    while (1) {
        handle = kd_mpi_vb_get_block(stream_pool_id_, STREAM_BUF_SIZE, NULL);
        if (handle != VB_INVALID_HANDLE) {
            break;
        }
        usleep(1000*5);
    }
    stream_phys_addr_ = kd_mpi_vb_handle_to_phyaddr(handle);

    vdec_stream.end_of_stream = K_TRUE;
    vdec_stream.phy_addr = stream_phys_addr_;
    vdec_stream.len = 1024;

    ret = kd_mpi_vdec_send_stream(vdec_chn_id_, &vdec_stream, -1);
    if (ret){
        printf("%s kd_mpi_vdec_send_stream failed\n",__func__);
        return -1;
    }

    ret = kd_mpi_vb_release_block(handle);
    if (ret){
        printf("%s kd_mpi_vb_release_block failed\n",__func__);
        return -1;
    }

    while(1){
        kd_mpi_vdec_query_status(vdec_chn_id_,&status);
        if (status.end_of_stream == K_TRUE){
            break;
        }
        else
        {
            usleep(1000*100);
        }
    }

    kd_mpi_vdec_stop_chn(vdec_chn_id_);

    _unbind_pipeline();

    //stop thread
    thread_running_ = false;
    pthread_join(frame_process_thread_, nullptr);
    kd_mpi_nonai_2d_stop_chn(csc_chn_id_);

    status_ = MppPipelineStatus::STOPPED;
    std::cout << "MPP pipeline stopped" << std::endl;

    return 0;
}

int MppPipeline::deinit(){
    k_s32 ret;
    if (status_ == MppPipelineStatus::UNINITIALIZED) return -1;
    kd_mpi_vo_disable_layer(osd_id_);
    kd_mpi_vo_disable_layer(vo_layer_chn_id_);

    kd_mpi_nonai_2d_detach_vb_pool(csc_chn_id_);
    kd_mpi_nonai_2d_destroy_chn(csc_chn_id_);
    kd_mpi_vdec_detach_vb_pool(vdec_chn_id_);
    kd_mpi_vdec_destroy_chn(vdec_chn_id_);

    if (osd_vb_handle_ != VB_INVALID_HANDLE)
    {
        kd_mpi_vb_release_block(osd_vb_handle_);
        osd_vb_handle_ = VB_INVALID_HANDLE;
    }

    if (vdec_pool_id_ != VB_INVALID_POOLID){
        kd_mpi_vb_destory_pool(vdec_pool_id_);
        vdec_pool_id_ = VB_INVALID_POOLID;
    }

    if (stream_pool_id_ != VB_INVALID_POOLID){
        kd_mpi_vb_destory_pool(stream_pool_id_);
        stream_pool_id_ = VB_INVALID_POOLID;
    }

    if (osd_pool_id_ != VB_INVALID_POOLID){
        kd_mpi_vb_destory_pool(osd_pool_id_);
        osd_pool_id_ = VB_INVALID_POOLID;
    }

    if (csc_pool_id_ != VB_INVALID_POOLID){
        kd_mpi_vb_destory_pool(csc_pool_id_);
        csc_pool_id_ = VB_INVALID_POOLID;
    }

    ret = kd_mpi_vb_exit();
    if (ret) {
        printf("kd_mpi_vb_exit failed.\n");
    }
    vb_exit_ = true;

    status_ = MppPipelineStatus::UNINITIALIZED;
    return 0;
}

int MppPipeline::_osd_alloc_frame(void **osd_vaddr){
     k_u64 phys_addr = 0;
    k_u32 *virt_addr;
    k_vb_blk_handle handle;
    k_s32 size;

    memset(&osd_vf_info_, 0, sizeof(osd_vf_info_));
    osd_vf_info_.v_frame.width = width_;
    osd_vf_info_.v_frame.height = height_;
    osd_vf_info_.v_frame.stride[0] = width_*4;
    osd_vf_info_.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
    size = ALIGN_UP(width_ * height_ * 4, ALIGN_1K);

    handle = kd_mpi_vb_get_block(osd_pool_id_, size, NULL);
    if (handle == VB_INVALID_HANDLE)
    {
        printf("%s get vb block error\n", __func__);
        return -1;
    }

    phys_addr = kd_mpi_vb_handle_to_phyaddr(handle);
    if (phys_addr == 0)
    {
        printf("%s get phys addr error\n", __func__);
        return -1;
    }

    virt_addr = (k_u32 *)kd_mpi_sys_mmap(phys_addr, size);
    // virt_addr = (k_u32 *)kd_mpi_sys_mmap_cached(phys_addr, size);

    if (virt_addr == NULL)
    {
        printf("%s mmap error\n", __func__);
        return -1;
    }

    osd_vf_info_.mod_id = K_ID_VO;
    osd_vf_info_.pool_id = osd_pool_id_;
    osd_vf_info_.v_frame.phys_addr[0] = phys_addr;
    if (osd_vf_info_.v_frame.pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        osd_vf_info_.v_frame.phys_addr[1] = phys_addr + (osd_vf_info_.v_frame.height * osd_vf_info_.v_frame.stride[0]);
    *osd_vaddr = virt_addr;
    osd_vaddr_ = virt_addr;

    printf("phys_addr is %lx osd_pool_id is %d \n", phys_addr, osd_pool_id_);

    osd_vb_handle_ = handle;
    return 0;
}

int MppPipeline::_osd_draw_frame(void* osd_data){

    memcpy(osd_vaddr_,osd_data, width_ * height_ * 4);
    return kd_mpi_vo_insert_frame(osd_id_, &osd_vf_info_);
}
