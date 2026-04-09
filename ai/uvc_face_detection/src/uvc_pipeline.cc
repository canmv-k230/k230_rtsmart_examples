#include "uvc_pipeline.h"

#define ALIGN_UP_16(x)  (((x) + 15) & ~15)
#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align)-1))

/**
 * @brief UVC_PipeLine 构造函数，初始化视频显示与解码相关配置。
 * @param debug_mode 是否启用调试模式（0：关闭，非0：开启）
 */
UVC_PipeLine::UVC_PipeLine(int debug_mode)
{
    // 根据 DISPLAY_MODE 宏定义选择连接的显示屏类型（LCD 屏或 HDMI）
    if(DISPLAY_MODE == 0) {
        // 外接 HDMI 转接板 LT9611，1080P，4 LAN，30fps
        connector_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
    } else if(DISPLAY_MODE == 1) {
        // MIPI 屏 ST7701，480x800 分辨率，2 LAN，30fps
        connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
    } else if(DISPLAY_MODE == 2) {
        // MIPI 屏 HX8377，1080x1920 分辨率，4 LAN，30fps
        connector_type = HX8377_V2_MIPI_4LAN_1080X1920_30FPS;
    } else {
        // 默认配置为 HDMI 转接板
        connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
    }

    // 视频输出层 ID（使用 Layer1）
    uvc_vo_id = K_VO_LAYER_VIDEO1;
    // 视频输出设备 ID（通常是 0）
    vo_dev_id = K_VO_DISPLAY_DEV_ID;
    
    // 解码器设备 ID 和通道 ID（通常默认 0）
    vdec_dev_id = 0;
    vdec_bind_chn_id = 0;

    // nonai 2D 相关
    nonai2d_dev_id = 0;
    // NV12转RGB888通道
    nonai2d_rgb888_chn_id = 0;
    // RGB888转YUV420SP通道
    nonai2d_yuv420sp_chn_id = 1;

    // 设置为 JPEG 解码模式（1 表示 JPEG 格式）
    is_jpeg = 1;
    // 初始化 UVC 格式结构体
    init_format = { UVC_WIDTH, UVC_HEIGHT, is_jpeg, 0 };

    // 存储调试模式标志
    debug_mode_ = debug_mode;

    nonai2d_rgb888_pool_id = VB_INVALID_POOLID;
    nonai2d_yuv420sp_pool_id = VB_INVALID_POOLID;
    vdec_pool_id = VB_INVALID_POOLID;
}


UVC_PipeLine::~UVC_PipeLine()
{
}

static k_s32 vdec_vb_create_pool(int width, int height)
{
    k_vb_pool_config pool_config;

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = 6;
    pool_config.blk_size = ALIGN_UP(width * height, 0x1000) * 2;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

static k_s32 nonai_2d_vb_create_pool()
{
    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));

    pool_config.blk_cnt =  2;
    pool_config.blk_size = ALIGN_UP((DISPLAY_WIDTH * DISPLAY_HEIGHT * 3), VICAP_ALIGN_1K);
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}
/**
 * @brief 初始化并创建 UVC 视频处理管线，包括 vb 缓冲、屏幕配置、视频输出、解码器和 UVC 启动等。
 * @return 0 表示成功，其他错误码表示失败。
 */
int UVC_PipeLine::Create()
{
    // 用于统计耗时的调试工具类（自动析构时打印耗时）
    ScopedTiming st("PipeLine::Create", debug_mode_);

    k_s32 ret = 0;
    k_u32 pool_id;
    k_vb_pool_config pool_config;

    // ---------------------------- 配置视频缓冲区（Video Buffer） -----------------------------------
    memset(&config, 0, sizeof(k_vb_config));
    config.max_pool_cnt = 64;

    // 设置 VB 配置
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        return ret;
    }

    // 设置 VB 附加配置
    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("vb_set_supplement_config failed ret:%d\n", ret);
        return ret;
    }

    // 初始化 VB 系统
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        return ret;
    }
    // -----------------------------------------------------------------------------------------------

    // ---------------------------- 配置屏幕连接器 ---------------------------------------------------
    k_connector_info connector_info;
    memset(&connector_info, 0, sizeof(k_connector_info));

    // 获取屏幕配置信息（如名称、分辨率等）
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("the connector type not supported!\n");
        return ret;
    }

    // 打开屏幕连接器设备
    k_s32 connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return K_ERR_VO_NOTREADY;
    }

    // 初始化 connector（配置时序、分辨率等）
    ret = kd_mpi_connector_init(connector_fd, connector_info);
    if (ret) {
        printf("ERROR: kd_mpi_connector_init failed, ret=%d\n", ret);
        return ret;
    }

    // 打开电源
    ret = kd_mpi_connector_power_set(connector_fd, K_TRUE);
    if (ret) {
        printf("ERROR: kd_mpi_connector_power_set failed, ret=%d\n", ret);
        return ret;
    }

    // 关闭设备句柄（配置完成即可关闭）
    ret = kd_mpi_connector_close(connector_fd);
    if (ret) {
        printf("ERROR: kd_mpi_connector_close failed, ret=%d\n", ret);
        return ret;
    }
    // -----------------------------------------------------------------------------------------------

    // ---------------------------- 配置视频输出 VO Layer --------------------------------------------
    kd_mpi_vo_disable_layer(uvc_vo_id);  // 先关闭 layer，避免旧配置干扰

    memset(&uvc_vo_attr, 0, sizeof(uvc_vo_attr));
    uvc_vo_attr.layer_id        = uvc_vo_id;
    uvc_vo_attr.position.x      = 0;
    uvc_vo_attr.position.y      = 0;
    uvc_vo_attr.img_size.width  = DISPLAY_WIDTH;
    uvc_vo_attr.img_size.height = DISPLAY_HEIGHT;
    uvc_vo_attr.pixel_format    = PIXEL_FORMAT_YUV_SEMIPLANAR_420; // NV12
    uvc_vo_attr.global_alpha    = 0xFF;                            // 不透明
    // 根据 DISPLAY_MODE 是否需要旋转
    uvc_vo_attr.func            = DISPLAY_MODE? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;
    // 若旋转，需要额外的 DMA buffer
    uvc_vo_attr.rot_buf_nr      = DISPLAY_MODE? 1 : 0;
    uvc_vo_attr.rot_buf_bpp     = 0;

    ret = kd_mpi_vo_set_layer_attr(uvc_vo_id, &uvc_vo_attr);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_set_layer_attr failed, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vo_enable_layer(uvc_vo_id);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_enable_layer failed, ret=%d\n", ret);
        return ret;
    }
    // ---------------------------------------------------------------------------------------------

    // ---------------------------- 配置 JPEG 解码器（VDEC） ----------------------------------------
    vdec_pool_id = vdec_vb_create_pool(UVC_WIDTH, UVC_HEIGHT);
    if (vdec_pool_id == VB_INVALID_POOLID) {
        printf("fail to create vdec pool\n");
        return -1;
    }

    ret = kd_mpi_vdec_attach_vb_pool(vdec_bind_chn_id,vdec_pool_id);
    if (ret) {
        printf("kd_mpi_vdec_attach_vb_pool fail, ret = %d\n", ret);
        return -1;
    }

    // 解码器属性配置，JPEG->YUV420
    vdec_attr.pic_width = UVC_WIDTH;
    vdec_attr.pic_height = UVC_HEIGHT;
    vdec_attr.stream_buf_size = ALIGN_UP(UVC_WIDTH * UVC_HEIGHT, 0x1000);  // 输入缓冲大小
    vdec_attr.type = K_PT_JPEG;                                            // 解码类型：JPEG

    // 创建解码通道
    ret = kd_mpi_vdec_create_chn(vdec_bind_chn_id, &vdec_attr);
    if (ret) {
        printf("kd_mpi_vdec_create_chn fail, ret = %d\n", ret);
        return -1;
    }

    // 启动解码通道
    ret = kd_mpi_vdec_start_chn(vdec_bind_chn_id);
    if (ret) {
        printf("kd_mpi_vdec_start_chn fail, ret = %d\n", ret);
        return -1;
    }

    // 解码帧初始化、附加信息参数配置
    memset(&vdec_frame_info, 0, sizeof(k_video_frame_info));
    memset(&supplement_info, 0, sizeof(k_vdec_supplement_info));
    supplement_info.type=K_PT_JPEG;
    supplement_info.is_valid_frame=K_TRUE;
    supplement_info.end_of_stream=K_FALSE;
    // -----------------------------------------------------------------------------------------------

    // ----------------------------nonai_2d 格式转换 -------------------------------------------------
    nonai2d_rgb888_pool_id = nonai_2d_vb_create_pool();
    if (nonai2d_rgb888_pool_id == VB_INVALID_POOLID) {
        printf("fail to create nonai2d pool\n");
        return -1;
    }

    ret = kd_mpi_nonai_2d_attach_vb_pool(nonai2d_rgb888_chn_id,nonai2d_rgb888_pool_id);
    if (ret){
        printf("kd_mpi_nonai_2d_attach_vb_pool fail, ret = %d\n", ret);
        return -1;
    }

    // 配置nonai2d_rgb888_chn_id通道，实现YUV420->RGB888，给AI模型使用
    attr_2d_rgb888.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d_rgb888.dst_fmt = PIXEL_FORMAT_RGB_888;
    ret = kd_mpi_nonai_2d_create_chn(nonai2d_rgb888_chn_id, &attr_2d_rgb888);
    if (ret) {
        printf("kd_mpi_nonai_2d_create_chn fail, ret = %d\n", ret);
        return -1;
    }
    ret = kd_mpi_nonai_2d_start_chn(nonai2d_rgb888_chn_id);
    if (ret) {
        printf("kd_mpi_nonai_2d_start_chn fail, ret = %d\n", ret);
        return -1;
    }
    memset(&rgb888_frame_info, 0, sizeof(k_video_frame_info));

    // 配置nonai2d_yuv420sp_chn_id通道，实现RGB888->YUV420SP，该通道nonai2d绑定到VO显示
    nonai2d_yuv420sp_pool_id = nonai_2d_vb_create_pool();
    if (nonai2d_yuv420sp_pool_id == VB_INVALID_POOLID) {
        printf("fail to create nonai2d pool\n");
        return -1;
    }

    ret = kd_mpi_nonai_2d_attach_vb_pool(nonai2d_yuv420sp_chn_id,nonai2d_yuv420sp_pool_id);
    if (ret){
        printf("kd_mpi_nonai_2d_attach_vb_pool fail, ret = %d\n", ret);
        return -1;
    }

    attr_2d_yuv420sp.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d_yuv420sp.dst_fmt = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ret = kd_mpi_nonai_2d_create_chn(nonai2d_yuv420sp_chn_id, &attr_2d_yuv420sp);
    if (ret) {
        printf("kd_mpi_nonai_2d_create_chn fail, ret = %d\n", ret);
        return -1;
    }
    ret = kd_mpi_nonai_2d_start_chn(nonai2d_yuv420sp_chn_id);
    if (ret) {
        printf("kd_mpi_nonai_2d_start_chn fail, ret = %d\n", ret);
        return -1;
    }
    memset(&yuv420sp_frame_info, 0, sizeof(k_video_frame_info));

    // -----------------------------------------------------------------------------------------------

    // -------------------------------------------处理绑定配置-----------------------------------------
    // 配置系统模块绑定：NONAI_2D -> VO
    nonai2d_mpp_chn.mod_id = K_ID_NONAI_2D;
    nonai2d_mpp_chn.dev_id = nonai2d_dev_id;
    nonai2d_mpp_chn.chn_id = nonai2d_yuv420sp_chn_id;

    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = vo_dev_id;
    vo_mpp_chn.chn_id = uvc_vo_id;

    // 执行模块绑定（解码器输出绑定到视频输出模块）
    ret = kd_mpi_sys_bind(&nonai2d_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("sample_vdec_bind_vo fail, ret = %d\n", ret);
        return -1;
    }
    // -----------------------------------------------------------------------------------------------

    // ---------------------------- 初始化并启动 UVC 采集 -------------------------------------------
    ret = uvc_host_init(&init_format);
    if (ret) {
        printf("uvc_host_init fail\n");
        return -1;
    }

    ret = uvc_host_start_stream();
    if (ret) {
        printf("uvc start stream fail\n");
        return -1;
    }
    // -----------------------------------------------------------------------------------------------

    return ret;
}


/**
 * @brief 获取一帧 UVC 数据并送入解码器。
 * @param dump_res 用于后续扩展，暂未使用。
 */
int UVC_PipeLine::GetFrame(DumpRes &dump_res) {
    // 用于性能调试
    ScopedTiming st("PipeLine::GetFrame", debug_mode_);
    int ret = 0;

    memset(&vdec_frame_info, 0, sizeof(k_video_frame_info));
    memset(&rgb888_frame_info, 0, sizeof(k_video_frame_info));
    memset(&yuv420sp_frame_info, 0, sizeof(k_video_frame_info));

    // 获取一帧 UVC 视频帧（带超时 5000ms）
    ret = uvc_host_get_frame(&cur_frame, 5000);
    if (ret) {
        printf("uvc_host_get_frame fail\n");
        return -1;
    }

    // 将帧数据发送给 JPEG 解码器
    ret = kd_mpi_vdec_send_stream(vdec_bind_chn_id, &cur_frame.v_stream, 1000);
    if (ret) {
        printf("kd_mpi_vdec_send_stream fail\n");
        return -1;
    }

    // 获取解码数据帧
    ret=kd_mpi_vdec_get_frame(vdec_bind_chn_id,&vdec_frame_info,&supplement_info,1000);
    if (ret) {
        printf("kd_mpi_vdec_get_frame failed. %d\n", ret);
        return -1;
    }

    // 发送解码数据帧到nonai2d_rgb888_chn_id通道
    ret = kd_mpi_nonai_2d_send_frame(nonai2d_rgb888_chn_id, &vdec_frame_info, 1000);
    if (ret) {
        printf("kd_mpi_nonai_2d_send_frame failed. %d\n", ret);
        return -1;
    }

    // 从nonai2d_rgb888_chn_id通道获取处理后的RGB888数据帧
    ret = kd_mpi_nonai_2d_get_frame(nonai2d_rgb888_chn_id, &rgb888_frame_info, 1000);
    if (ret) {
        printf("kd_mpi_nonai_2d_get_frame failed. %d\n", ret);
        return -1;
    }

    // 映射RGB888数据帧物理地址到虚拟地址
    dump_res.virt_addr=reinterpret_cast<uintptr_t>(kd_mpi_sys_mmap(rgb888_frame_info.v_frame.phys_addr[0], AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH));
    dump_res.phy_addr=reinterpret_cast<uintptr_t>(rgb888_frame_info.v_frame.phys_addr[0]);
    return 0;
}


/**
 * @brief 释放视频帧资源（预留函数，当前无操作）。
 * @return 总是返回 0，表示成功。
 */
int UVC_PipeLine::ReleaseFrame(DumpRes &dump_res) {
    ScopedTiming st("PipeLine::ReleaseFrame", debug_mode_);
    int ret = 0;
    kd_mpi_sys_munmap(reinterpret_cast<void*>(dump_res.virt_addr), AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH);
    // 发送RGB888数据帧到nonai2d_yuv420sp_chn_id通道
    ret = kd_mpi_nonai_2d_send_frame(nonai2d_yuv420sp_chn_id, &rgb888_frame_info, 1000);
    if (ret) {
        printf("kd_mpi_nonai_2d_send_frame failed. %d\n", ret);
        return -1;
    }
    // 从nonai2d_yuv420sp_chn_id通道获取处理后的YUV420SP数据帧
    ret = kd_mpi_nonai_2d_release_frame(nonai2d_rgb888_chn_id, &rgb888_frame_info);
    if (ret) {
        printf("kd_mpi_nonai_2d_release_frame failed. %d\n", ret);
        return -1;
    }

    // 释放 VDEC 帧资源
    ret = kd_mpi_vdec_release_frame(vdec_bind_chn_id, &vdec_frame_info);
    if (ret) {
        printf("kd_mpi_vdec_release_frame failed. %d\n", ret);
    }
    //释放 UVC 帧资源
    ret = uvc_host_put_frame(&cur_frame);
    if (ret) {
        printf("uvc_host_put_frame fail\n");
        return -1;
    }
    return ret;
}


/**
 * @brief 向视频输出插入一帧图像（预留函数，当前无操作）。
 * @param data 指向图像数据的指针。
 * @return 总是返回 0，表示成功。
 */
int UVC_PipeLine::InsertFrame(void* data) {
    ScopedTiming st("PipeLine::InsertFrame", debug_mode_);
    int ret = 0;
    return ret;
}


/**
 * @brief 销毁并清理整个视频处理管线，释放资源。
 * @return 0 表示成功，其他值表示失败。
 */
int UVC_PipeLine::Destroy() {
    ScopedTiming st("PipeLine::Destroy", debug_mode_);
    int ret = 0;

    // 停止 UVC 采集
    uvc_host_exit();

    // 禁用 VO 视频层
    ret = kd_mpi_vo_disable_layer(uvc_vo_id);
    if (ret) {
        printf("kd_mpi_vo_disable_layer failed.\n");
        return ret;
    }
    // 解除系统模块绑定（NONAI_2D -> VO）
    ret = kd_mpi_sys_unbind(&nonai2d_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("kd_mpi_sys_unbind failed:0x%x\n", ret);
    }
    // 停止并销毁非 AI 2D 通道
    kd_mpi_nonai_2d_stop_chn(nonai2d_rgb888_chn_id);
    kd_mpi_nonai_2d_detach_vb_pool(nonai2d_rgb888_chn_id);
    kd_mpi_nonai_2d_destroy_chn(nonai2d_rgb888_chn_id);
    kd_mpi_nonai_2d_stop_chn(nonai2d_yuv420sp_chn_id);
    kd_mpi_nonai_2d_detach_vb_pool(nonai2d_yuv420sp_chn_id);
    kd_mpi_nonai_2d_destroy_chn(nonai2d_yuv420sp_chn_id);

    // 停止并销毁解码通道
    kd_mpi_vdec_stop_chn(vdec_bind_chn_id);
    kd_mpi_vdec_detach_vb_pool(vdec_bind_chn_id);
    kd_mpi_vdec_destroy_chn(vdec_bind_chn_id);

    if (vdec_pool_id != VB_INVALID_POOLID){
        kd_mpi_vb_destory_pool(vdec_pool_id);
        vdec_pool_id = VB_INVALID_POOLID;
    }

    if (nonai2d_rgb888_pool_id != VB_INVALID_POOLID){
        kd_mpi_vb_destory_pool(nonai2d_rgb888_pool_id);
        nonai2d_rgb888_pool_id = VB_INVALID_POOLID;
    }

    if (nonai2d_yuv420sp_pool_id != VB_INVALID_POOLID){
        kd_mpi_vb_destory_pool(nonai2d_yuv420sp_pool_id);
        nonai2d_yuv420sp_pool_id = VB_INVALID_POOLID;
    }

    // 延迟一帧时间，等待 VO 释放 VB 缓冲
    k_u32 display_ms = 1000 / 33; // 大约一帧的时间（33fps）
    usleep(1000 * display_ms);

    // 反初始化视频缓冲系统
    ret = kd_mpi_vb_exit();
    if (ret) {
        printf("kd_mpi_vb_exit failed.\n");
        return ret;
    }

    return 0;
}
