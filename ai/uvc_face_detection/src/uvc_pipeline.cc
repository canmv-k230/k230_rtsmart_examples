#include "uvc_pipeline.h"
#include <errno.h>

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
    uvc_vo_id = K_VO_LAYER_VIDEO0;
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
    bool m_isJpeg = true;
    // 初始化 UVC 格式结构体
    init_format = {
        UVC_WIDTH,
        UVC_HEIGHT,
        m_isJpeg ? USBH_VIDEO_FOURCC_MJPEG : USBH_VIDEO_FOURCC_YUY2,
        0
    };

    // 存储调试模式标志
    debug_mode_ = debug_mode;

    nonai2d_rgb888_pool_id = VB_INVALID_POOLID;
    nonai2d_yuv420sp_pool_id = VB_INVALID_POOLID;
    vdec_pool_id = VB_INVALID_POOLID;
}


UVC_PipeLine::~UVC_PipeLine()
{
}

// 是否使用 GDMA 旋转，与 Create() 中 uvc_vo_attr.func 的取值一致
#define UVC_VO_ROTATE (DISPLAY_MODE != 0)

// VDEC 与 nonai_2d CSC 都不缩放：UVC 输出、AI 输入、显示层三者尺寸必须一致，
// 下面的缓冲池大小也是按 AI_FRAME_* 计算的。
static_assert(UVC_WIDTH == AI_FRAME_WIDTH && UVC_HEIGHT == AI_FRAME_HEIGHT,
              "UVC size must match AI frame size (no scaling in the pipeline)");
static_assert(DISPLAY_WIDTH == AI_FRAME_WIDTH && DISPLAY_HEIGHT == AI_FRAME_HEIGHT,
              "display layer size must match AI frame size (no scaling in the pipeline)");

// RGB888 池（chn0 输出，供 AI 使用）：应用持有 1 块 + chn1 仍在读取的上一块 1 块，再留 1 块余量
#define RGB888_POOL_BLK_CNT     3
// NV12 池（chn1 输出，绑定 VO VIDEO1）：
// 不旋转时 VO 直接扫描这些块：2 个 RDMA bank + 2 个退役后再保持 VO_RDMA_HOLD_VSYNCS 的块
// + 1 个待显示的 mailbox，再加 nonai_2d 正在写的 1 块，共 6 块（见 mpp kernel/vo/BUFFER_OWNERSHIP.md）；
// 旋转时 VO 显示的是自己的 rot_buf，只在 GDMA 旋转期间同步借用源块，2 块即可，取 3 块留余量。
#define YUV420SP_POOL_BLK_CNT   (UVC_VO_ROTATE ? 3 : 6)

// 取帧超时。q 退出最坏要等完正在进行的这些超时（正常出帧时远小于此）。
#define UVC_GET_FRAME_TIMEOUT_MS    1000
#define VDEC_TIMEOUT_MS             1000
#define NONAI_2D_TIMEOUT_MS         1000
// 每次迭代最多丢弃的迟到输出帧数
#define STALE_DRAIN_MAX             8
// Destroy()：等待已停止通道中在途的任务完成并送达 VO
#define STOP_SETTLE_US              (50 * 1000)

static k_s32 vdec_vb_create_pool(int width, int height)
{
    k_vb_pool_config pool_config;

    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = 6;
    pool_config.blk_size = ALIGN_UP(width * height, 0x1000) * 2;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

static k_s32 nonai_2d_vb_create_pool(k_u32 blk_cnt, k_u64 blk_size)
{
    k_vb_pool_config pool_config;
    memset(&pool_config, 0, sizeof(pool_config));

    pool_config.blk_cnt =  blk_cnt;
    pool_config.blk_size = blk_size;
    pool_config.mode = VB_REMAP_MODE_NOCACHE;

    return kd_mpi_vb_create_pool(&pool_config);
}

// RGB888 packed：w * h * 3
static k_u64 rgb888_blk_size(k_u32 width, k_u32 height)
{
    return ALIGN_UP((k_u64)width * height * 3, 0x1000);
}

// NV12：nonai_2d config_job() 把 UV 平面放在 ALIGN_UP(w * h, 4K) 处，UV 长度 w * ((h + 1) / 2) + (w & 1)
static k_u64 yuv420sp_blk_size(k_u32 width, k_u32 height)
{
    k_u64 y_size = ALIGN_UP((k_u64)width * height, 0x1000);
    k_u64 uv_size = (k_u64)width * ((height + 1) / 2) + (width & 1);

    return ALIGN_UP(y_size + uv_size, 0x1000);
}

// VB_INVALID_POOLID 是无符号常量，统一在这里比较
static bool pool_alive(k_s32 pool_id)
{
    return pool_id != (k_s32)VB_INVALID_POOLID;
}

static void note_error(int &first_err, int ret)
{
    if (ret && !first_err) {
        first_err = ret;
    }
}

static int destroy_pool(k_s32 &pool_id, const char *name)
{
    int ret;

    if (!pool_alive(pool_id)) {
        return 0;
    }
    ret = kd_mpi_vb_destory_pool(pool_id);
    if (ret) {
        printf("kd_mpi_vb_destory_pool %s pool %d failed, ret=0x%x\n", name, pool_id, ret);
        return ret;
    }
    pool_id = VB_INVALID_POOLID;
    return 0;
}

/**
 * @brief 初始化并创建 UVC 视频处理管线，包括 vb 缓冲、屏幕配置、视频输出、解码器和 UVC 启动等。
 * 失败时直接返回，已完成的阶段由调用者随后调用 Destroy() 清理。
 * @return 0 表示成功，其他错误码表示失败。
 */
int UVC_PipeLine::Create()
{
    // 用于统计耗时的调试工具类（自动析构时打印耗时）
    ScopedTiming st("PipeLine::Create", debug_mode_);

    k_s32 ret = 0;

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
    vb_inited_ = true;
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
    connector_fd_ = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd_ < 0) {
        printf("%s, connector open failed.\n", __func__);
        connector_fd_ = -1;
        return K_ERR_VO_NOTREADY;
    }

    // 初始化 connector（配置时序、分辨率等）
    ret = kd_mpi_connector_init(connector_fd_, connector_info);
    if (ret) {
        printf("ERROR: kd_mpi_connector_init failed, ret=%d\n", ret);
        return ret;
    }
    // init 已给面板上电；关闭 fd 不会关屏，需在 Destroy() 中显式 deinit
    connector_on_ = true;

    // 打开电源
    ret = kd_mpi_connector_power_set(connector_fd_, K_TRUE);
    if (ret) {
        printf("ERROR: kd_mpi_connector_power_set failed, ret=%d\n", ret);
        return ret;
    }
    // 设备句柄保持打开，Destroy() 中用它给面板断电
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
    vo_layer_enabled_ = true;
    // ---------------------------------------------------------------------------------------------

    // ---------------------------- 配置 JPEG 解码器（VDEC） ----------------------------------------
    vdec_pool_id = vdec_vb_create_pool(UVC_WIDTH, UVC_HEIGHT);
    if (!pool_alive(vdec_pool_id)) {
        printf("fail to create vdec pool\n");
        return -1;
    }

    ret = kd_mpi_vdec_attach_vb_pool(vdec_bind_chn_id,vdec_pool_id);
    if (ret) {
        printf("kd_mpi_vdec_attach_vb_pool fail, ret = %d\n", ret);
        return -1;
    }
    vdec_attached_ = true;

    // 解码器属性配置，JPEG->YUV420
    memset(&vdec_attr, 0, sizeof(vdec_attr));
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
    vdec_created_ = true;

    // 启动解码通道
    ret = kd_mpi_vdec_start_chn(vdec_bind_chn_id);
    if (ret) {
        printf("kd_mpi_vdec_start_chn fail, ret = %d\n", ret);
        return -1;
    }
    vdec_started_ = true;

    // 解码帧初始化、附加信息参数配置
    memset(&vdec_frame_info, 0, sizeof(k_video_frame_info));
    memset(&supplement_info, 0, sizeof(k_vdec_supplement_info));
    supplement_info.type=K_PT_JPEG;
    supplement_info.is_valid_frame=K_TRUE;
    supplement_info.end_of_stream=K_FALSE;
    // -----------------------------------------------------------------------------------------------

    // ----------------------------nonai_2d 格式转换 -------------------------------------------------
    nonai2d_rgb888_pool_id = nonai_2d_vb_create_pool(RGB888_POOL_BLK_CNT,
                                                     rgb888_blk_size(AI_FRAME_WIDTH, AI_FRAME_HEIGHT));
    if (!pool_alive(nonai2d_rgb888_pool_id)) {
        printf("fail to create nonai2d pool\n");
        return -1;
    }

    ret = kd_mpi_nonai_2d_attach_vb_pool(nonai2d_rgb888_chn_id,nonai2d_rgb888_pool_id);
    if (ret){
        printf("kd_mpi_nonai_2d_attach_vb_pool fail, ret = %d\n", ret);
        return -1;
    }
    rgb888_attached_ = true;

    // 配置nonai2d_rgb888_chn_id通道，实现YUV420->RGB888，给AI模型使用
    memset(&attr_2d_rgb888, 0, sizeof(attr_2d_rgb888));
    attr_2d_rgb888.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d_rgb888.dst_fmt = PIXEL_FORMAT_RGB_888;
    ret = kd_mpi_nonai_2d_create_chn(nonai2d_rgb888_chn_id, &attr_2d_rgb888);
    if (ret) {
        printf("kd_mpi_nonai_2d_create_chn fail, ret = %d\n", ret);
        return -1;
    }
    rgb888_created_ = true;
    ret = kd_mpi_nonai_2d_start_chn(nonai2d_rgb888_chn_id);
    if (ret) {
        printf("kd_mpi_nonai_2d_start_chn fail, ret = %d\n", ret);
        return -1;
    }
    rgb888_started_ = true;
    memset(&rgb888_frame_info, 0, sizeof(k_video_frame_info));

    // 配置nonai2d_yuv420sp_chn_id通道，实现RGB888->YUV420SP，该通道nonai2d绑定到VO显示
    nonai2d_yuv420sp_pool_id = nonai_2d_vb_create_pool(YUV420SP_POOL_BLK_CNT,
                                                       yuv420sp_blk_size(AI_FRAME_WIDTH, AI_FRAME_HEIGHT));
    if (!pool_alive(nonai2d_yuv420sp_pool_id)) {
        printf("fail to create nonai2d pool\n");
        return -1;
    }

    ret = kd_mpi_nonai_2d_attach_vb_pool(nonai2d_yuv420sp_chn_id,nonai2d_yuv420sp_pool_id);
    if (ret){
        printf("kd_mpi_nonai_2d_attach_vb_pool fail, ret = %d\n", ret);
        return -1;
    }
    yuv420sp_attached_ = true;

    memset(&attr_2d_yuv420sp, 0, sizeof(attr_2d_yuv420sp));
    attr_2d_yuv420sp.mode = K_NONAI_2D_CALC_MODE_CSC;
    attr_2d_yuv420sp.dst_fmt = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ret = kd_mpi_nonai_2d_create_chn(nonai2d_yuv420sp_chn_id, &attr_2d_yuv420sp);
    if (ret) {
        printf("kd_mpi_nonai_2d_create_chn fail, ret = %d\n", ret);
        return -1;
    }
    yuv420sp_created_ = true;
    ret = kd_mpi_nonai_2d_start_chn(nonai2d_yuv420sp_chn_id);
    if (ret) {
        printf("kd_mpi_nonai_2d_start_chn fail, ret = %d\n", ret);
        return -1;
    }
    yuv420sp_started_ = true;
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

    // 执行模块绑定（nonai_2d chn1 输出绑定到视频输出模块）
    ret = kd_mpi_sys_bind(&nonai2d_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("sample_vdec_bind_vo fail, ret = %d\n", ret);
        return -1;
    }
    bound_ = true;
    // -----------------------------------------------------------------------------------------------

    // ---------------------------- 初始化并启动 UVC 采集 -------------------------------------------
    ret = uvc_host_init(&init_format);
    if (ret) {
        printf("uvc_host_init fail\n");
        return -1;
    }
    // 从这里起 fd 与 UVC 驱动的 VB 池已存在，即使启动失败也要 uvc_host_exit()
    uvc_inited_ = true;

    // uvc_host_init 会回写实际协商的格式；管线尺寸是编译期固定的，不一致无法工作
    if (init_format.width != UVC_WIDTH || init_format.height != UVC_HEIGHT ||
        init_format.fourcc != USBH_VIDEO_FOURCC_MJPEG) {
        printf("uvc negotiated %ux%u fourcc 0x%08x, need %dx%d MJPEG\n",
               init_format.width, init_format.height, init_format.fourcc, UVC_WIDTH, UVC_HEIGHT);
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
 * @brief 按获取的逆序归还本次迭代持有的资源。
 */
int UVC_PipeLine::ReleaseHeld()
{
    int first_err = 0;
    int ret;

    if (held_.rgb_vaddr) {
        ret = kd_mpi_sys_munmap(held_.rgb_vaddr, held_.rgb_map_len);
        if (ret) {
            printf("kd_mpi_sys_munmap failed. %d\n", ret);
        }
        note_error(first_err, ret);
        held_.rgb_vaddr = nullptr;
        held_.rgb_map_len = 0;
    }

    // RGB 帧是 chn0 的输出，无论是否已送入 chn1 都经 chn0 释放：
    // 送入成功时 chn1 任务另持有自己的引用，送入失败时内核已丢弃它取的引用
    if (held_.rgb) {
        held_.rgb = false;
        ret = kd_mpi_nonai_2d_release_frame(nonai2d_rgb888_chn_id, &rgb888_frame_info);
        if (ret) {
            printf("kd_mpi_nonai_2d_release_frame failed. %d\n", ret);
        }
        note_error(first_err, ret);
    }

    // 释放 VDEC 帧资源
    if (held_.vdec) {
        held_.vdec = false;
        ret = kd_mpi_vdec_release_frame(vdec_bind_chn_id, &vdec_frame_info);
        if (ret) {
            printf("kd_mpi_vdec_release_frame failed. %d\n", ret);
        }
        note_error(first_err, ret);
    }

    //释放 UVC 帧资源
    if (held_.uvc) {
        held_.uvc = false;
        ret = uvc_host_put_frame(&cur_frame);
        if (ret) {
            printf("uvc_host_put_frame fail\n");
        }
        note_error(first_err, ret);
    }

    return first_err;
}


/**
 * @brief 丢弃先前超时后才完成的输出帧（超时 0 取帧并立即释放）。
 */
void UVC_PipeLine::DrainStaleOutputs()
{
    k_video_frame_info frame;
    k_vdec_supplement_info supplement;

    for (int i = 0; i < STALE_DRAIN_MAX; i++) {
        memset(&frame, 0, sizeof(frame));
        memset(&supplement, 0, sizeof(supplement));
        if (kd_mpi_vdec_get_frame(vdec_bind_chn_id, &frame, &supplement, 0) != K_SUCCESS) {
            break;
        }
        printf("drop stale vdec frame\n");
        if (kd_mpi_vdec_release_frame(vdec_bind_chn_id, &frame) != K_SUCCESS) {
            printf("kd_mpi_vdec_release_frame (stale) failed\n");
            break;
        }
    }

    for (int i = 0; i < STALE_DRAIN_MAX; i++) {
        memset(&frame, 0, sizeof(frame));
        if (kd_mpi_nonai_2d_get_frame(nonai2d_rgb888_chn_id, &frame, 0) != K_SUCCESS) {
            break;
        }
        printf("drop stale nonai_2d rgb888 frame\n");
        if (kd_mpi_nonai_2d_release_frame(nonai2d_rgb888_chn_id, &frame) != K_SUCCESS) {
            printf("kd_mpi_nonai_2d_release_frame (stale) failed\n");
            break;
        }
    }
}


/**
 * @brief 获取一帧 UVC 数据，经 VDEC 解码、nonai_2d chn0 转为 RGB888 并映射给 AI 使用。
 * 任一步失败都会归还已持有的资源后返回。
 * @param dump_res 输出 RGB888 帧的虚拟地址与物理地址。
 */
int UVC_PipeLine::GetFrame(DumpRes &dump_res) {
    // 用于性能调试
    ScopedTiming st("PipeLine::GetFrame", debug_mode_);
    int ret = 0;
    const k_u32 map_len = AI_FRAME_CHANNEL * AI_FRAME_HEIGHT * AI_FRAME_WIDTH;
    void *vaddr;

    dump_res.virt_addr = 0;
    dump_res.phy_addr = 0;

    // 调用者漏了 ReleaseFrame()：先归还，否则下面会覆盖仍被持有的帧描述符
    if (held_.uvc || held_.vdec || held_.rgb || held_.rgb_vaddr) {
        printf("GetFrame: previous frame was not released, releasing it now\n");
        if (ReleaseHeld()) {
            return UVC_FRAME_FATAL;
        }
    }

    DrainStaleOutputs();

    memset(&vdec_frame_info, 0, sizeof(k_video_frame_info));
    memset(&rgb888_frame_info, 0, sizeof(k_video_frame_info));

    // 获取一帧 UVC 视频帧
    ret = uvc_host_get_frame(&cur_frame, UVC_GET_FRAME_TIMEOUT_MS);
    if (ret) {
        printf("uvc_host_get_frame fail. %d\n", ret);
        // -ENODEV：摄像头已断开，重试无用；-ENOBUFS：缓冲全被持有，说明归还逻辑出错
        return (ret == -ENODEV || ret == -ENOBUFS) ? UVC_FRAME_FATAL : UVC_FRAME_DROPPED;
    }
    held_.uvc = true;

    // 将帧数据发送给 JPEG 解码器
    ret = kd_mpi_vdec_send_stream(vdec_bind_chn_id, &cur_frame.v_stream, VDEC_TIMEOUT_MS);
    if (ret) {
        printf("kd_mpi_vdec_send_stream fail. 0x%x\n", ret);
        goto drop;
    }

    // 获取解码数据帧（超时后迟到的帧由下一次 DrainStaleOutputs() 丢弃）
    ret=kd_mpi_vdec_get_frame(vdec_bind_chn_id,&vdec_frame_info,&supplement_info,VDEC_TIMEOUT_MS);
    if (ret) {
        printf("kd_mpi_vdec_get_frame failed. 0x%x\n", ret);
        goto drop;
    }
    held_.vdec = true;

    // 发送解码数据帧到nonai2d_rgb888_chn_id通道（失败时内核不保留引用，帧仍归应用所有）
    ret = kd_mpi_nonai_2d_send_frame(nonai2d_rgb888_chn_id, &vdec_frame_info, NONAI_2D_TIMEOUT_MS);
    if (ret) {
        printf("kd_mpi_nonai_2d_send_frame failed. 0x%x\n", ret);
        goto drop;
    }

    // 从nonai2d_rgb888_chn_id通道获取处理后的RGB888数据帧
    ret = kd_mpi_nonai_2d_get_frame(nonai2d_rgb888_chn_id, &rgb888_frame_info, NONAI_2D_TIMEOUT_MS);
    if (ret) {
        printf("kd_mpi_nonai_2d_get_frame failed. 0x%x\n", ret);
        goto drop;
    }
    held_.rgb = true;

    // AI 按 AI_FRAME_* 读取该帧，尺寸不符会越界
    if (rgb888_frame_info.v_frame.width != AI_FRAME_WIDTH ||
        rgb888_frame_info.v_frame.height != AI_FRAME_HEIGHT) {
        printf("rgb888 frame is %ux%u, expected %dx%d\n",
               rgb888_frame_info.v_frame.width, rgb888_frame_info.v_frame.height,
               AI_FRAME_WIDTH, AI_FRAME_HEIGHT);
        ReleaseHeld();
        return UVC_FRAME_FATAL;
    }

    // 映射RGB888数据帧物理地址到虚拟地址
    vaddr = kd_mpi_sys_mmap(rgb888_frame_info.v_frame.phys_addr[0], map_len);
    if (vaddr == nullptr) {
        printf("kd_mpi_sys_mmap failed\n");
        ReleaseHeld();
        return UVC_FRAME_FATAL;
    }
    held_.rgb_vaddr = vaddr;
    held_.rgb_map_len = map_len;

    dump_res.virt_addr=reinterpret_cast<uintptr_t>(vaddr);
    dump_res.phy_addr=static_cast<uintptr_t>(rgb888_frame_info.v_frame.phys_addr[0]);
    return UVC_FRAME_OK;

drop:
    return ReleaseHeld() ? UVC_FRAME_FATAL : UVC_FRAME_DROPPED;
}


/**
 * @brief 把（已画框的）RGB888 帧送入 chn1 转 NV12 显示，然后归还本次迭代持有的全部资源。
 * @return UVC_FRAME_OK；送显失败为 UVC_FRAME_DROPPED；归还失败为 UVC_FRAME_FATAL。
 */
int UVC_PipeLine::ReleaseFrame(DumpRes &dump_res) {
    ScopedTiming st("PipeLine::ReleaseFrame", debug_mode_);
    int status = UVC_FRAME_OK;
    int ret = 0;

    if (held_.rgb) {
        // 发送RGB888数据帧到nonai2d_yuv420sp_chn_id通道，输出经绑定送到 VO
        ret = kd_mpi_nonai_2d_send_frame(nonai2d_yuv420sp_chn_id, &rgb888_frame_info, NONAI_2D_TIMEOUT_MS);
        if (ret) {
            // 本帧不显示；RGB/VDEC/UVC 仍归应用所有，下面照常归还
            printf("kd_mpi_nonai_2d_send_frame failed. 0x%x\n", ret);
            status = UVC_FRAME_DROPPED;
        }
    }

    if (ReleaseHeld()) {
        status = UVC_FRAME_FATAL;
    }

    dump_res.virt_addr = 0;
    dump_res.phy_addr = 0;
    return status;
}


/**
 * @brief 向视频输出插入一帧图像（预留函数，当前无操作）。
 * @param data 指向图像数据的指针。
 * @return 总是返回 0，表示成功。
 */
int UVC_PipeLine::InsertFrame(void* data) {
    ScopedTiming st("PipeLine::InsertFrame", debug_mode_);
    (void)data;
    int ret = 0;
    return ret;
}


/**
 * @brief 销毁并清理整个视频处理管线，释放资源。
 * 按 Create() 的逆序执行，每一步都会尝试，不因前面的失败而提前返回；可在 Create() 失败后调用。
 * @return 0 表示成功，否则为第一个错误码。
 */
int UVC_PipeLine::Destroy() {
    ScopedTiming st("PipeLine::Destroy", debug_mode_);
    int first_err = 0;
    int ret = 0;

    // 归还应用仍持有的帧（需要 UVC、VDEC、nonai_2d 仍然存在）
    note_error(first_err, ReleaseHeld());

    // 按数据流方向停止各生产者：VDEC -> chn0 -> chn1
    if (vdec_started_) {
        ret = kd_mpi_vdec_stop_chn(vdec_bind_chn_id);
        if (ret) {
            printf("kd_mpi_vdec_stop_chn failed. 0x%x\n", ret);
        }
        note_error(first_err, ret);
        vdec_started_ = false;
    }
    if (rgb888_started_) {
        ret = kd_mpi_nonai_2d_stop_chn(nonai2d_rgb888_chn_id);
        if (ret) {
            printf("kd_mpi_nonai_2d_stop_chn %u failed. 0x%x\n", nonai2d_rgb888_chn_id, ret);
        }
        note_error(first_err, ret);
        rgb888_started_ = false;
    }
    if (yuv420sp_started_) {
        ret = kd_mpi_nonai_2d_stop_chn(nonai2d_yuv420sp_chn_id);
        if (ret) {
            printf("kd_mpi_nonai_2d_stop_chn %u failed. 0x%x\n", nonai2d_yuv420sp_chn_id, ret);
        }
        note_error(first_err, ret);
        yuv420sp_started_ = false;
        // stop 后仍可能有在途任务完成并经绑定送到 VO，等它们结束再拆绑定和通道
        usleep(STOP_SETTLE_US);
    }

    // 禁用 VO 视频层
    if (vo_layer_enabled_) {
        ret = kd_mpi_vo_disable_layer(uvc_vo_id);
        if (ret) {
            printf("kd_mpi_vo_disable_layer failed. 0x%x\n", ret);
        }
        note_error(first_err, ret);
        vo_layer_enabled_ = false;
    }
    // 解除系统模块绑定（NONAI_2D -> VO）
    if (bound_) {
        ret = kd_mpi_sys_unbind(&nonai2d_mpp_chn, &vo_mpp_chn);
        if (ret) {
            printf("kd_mpi_sys_unbind failed:0x%x\n", ret);
        }
        note_error(first_err, ret);
        bound_ = false;
    }

    // 销毁非 AI 2D 通道（destroy 会归还通道输出队列中未取走的帧），再解除池挂接
    if (yuv420sp_created_) {
        ret = kd_mpi_nonai_2d_destroy_chn(nonai2d_yuv420sp_chn_id);
        if (ret) {
            printf("kd_mpi_nonai_2d_destroy_chn %u failed. 0x%x\n", nonai2d_yuv420sp_chn_id, ret);
        }
        note_error(first_err, ret);
        yuv420sp_created_ = false;
    }
    if (yuv420sp_attached_) {
        ret = kd_mpi_nonai_2d_detach_vb_pool(nonai2d_yuv420sp_chn_id);
        if (ret) {
            printf("kd_mpi_nonai_2d_detach_vb_pool %u failed. 0x%x\n", nonai2d_yuv420sp_chn_id, ret);
        }
        note_error(first_err, ret);
        yuv420sp_attached_ = false;
    }
    if (rgb888_created_) {
        ret = kd_mpi_nonai_2d_destroy_chn(nonai2d_rgb888_chn_id);
        if (ret) {
            printf("kd_mpi_nonai_2d_destroy_chn %u failed. 0x%x\n", nonai2d_rgb888_chn_id, ret);
        }
        note_error(first_err, ret);
        rgb888_created_ = false;
    }
    if (rgb888_attached_) {
        ret = kd_mpi_nonai_2d_detach_vb_pool(nonai2d_rgb888_chn_id);
        if (ret) {
            printf("kd_mpi_nonai_2d_detach_vb_pool %u failed. 0x%x\n", nonai2d_rgb888_chn_id, ret);
        }
        note_error(first_err, ret);
        rgb888_attached_ = false;
    }

    // 销毁解码通道
    if (vdec_created_) {
        ret = kd_mpi_vdec_destroy_chn(vdec_bind_chn_id);
        if (ret) {
            printf("kd_mpi_vdec_destroy_chn failed. 0x%x\n", ret);
        }
        note_error(first_err, ret);
        vdec_created_ = false;
    }
    if (vdec_attached_) {
        ret = kd_mpi_vdec_detach_vb_pool(vdec_bind_chn_id);
        if (ret) {
            printf("kd_mpi_vdec_detach_vb_pool failed. 0x%x\n", ret);
        }
        note_error(first_err, ret);
        vdec_attached_ = false;
    }

    // 解码通道销毁后，再关闭 UVC 采集并释放驱动缓冲。
    if (uvc_inited_) {
        uvc_host_exit();
        uvc_inited_ = false;
    }
    note_error(first_err, destroy_pool(nonai2d_yuv420sp_pool_id, "nonai2d yuv420sp"));
    note_error(first_err, destroy_pool(nonai2d_rgb888_pool_id, "nonai2d rgb888"));
    note_error(first_err, destroy_pool(vdec_pool_id, "vdec"));

    // 即使 VO 层禁用失败，也尽力给面板断电，避免半初始化状态遗留；随后关闭 fd。
    // 仅关闭 connector fd 并不会关屏。
    if (connector_on_) {
        ret = kd_mpi_connector_deinit(connector_fd_);
        if (ret) {
            printf("ERROR: kd_mpi_connector_deinit failed, ret=%d\n", ret);
        }
        note_error(first_err, ret);
        connector_on_ = false;
    }
    if (connector_fd_ >= 0) {
        ret = kd_mpi_connector_close(connector_fd_);
        if (ret) {
            printf("ERROR: kd_mpi_connector_close failed, ret=%d\n", ret);
        }
        note_error(first_err, ret);
        connector_fd_ = -1;
    }

    // 反初始化视频缓冲系统。
    if (vb_inited_) {
        ret = kd_mpi_vb_exit();
        if (ret) {
            printf("kd_mpi_vb_exit failed. 0x%x\n", ret);
        }
        note_error(first_err, ret);
        if (ret == 0) {
            vb_inited_ = false;
            vdec_pool_id = VB_INVALID_POOLID;
            nonai2d_rgb888_pool_id = VB_INVALID_POOLID;
            nonai2d_yuv420sp_pool_id = VB_INVALID_POOLID;
        }
    }

    return first_err;
}
