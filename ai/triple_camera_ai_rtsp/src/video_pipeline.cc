#include "video_pipeline.h"

/* 16字节对齐宏，用于硬件DMA/图像缓冲区对齐 */
#define ALIGN_UP_16(x)  (((x) + 15) & ~15)
#define MEM_ALIGN_4K 0x1000

/* 构造函数：初始化管线各模块的默认配置 */
PipeLine::PipeLine(int debug_mode)
{
    // ------------------------ 显示接口类型选择 ------------------------
    // 根据宏 DISPLAY_MODE 选择不同屏幕（MIPI/HDMI 等）
    if(DISPLAY_MODE==0){
        connector_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
    }
    else if(DISPLAY_MODE==1){
        connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
    }
    else if(DISPLAY_MODE==2){
        connector_type = HX8377_V2_MIPI_4LAN_1080X1920_30FPS;
    }
    else{
        // 默认回退为 1080P HDMI 输出
        connector_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
    }

    // ------------------------ VO（视频输出）相关 ID ------------------------
    vo_dev_id = K_VO_DISPLAY_DEV_ID;        // VO 设备 ID

    vi_vo_id_0  = K_VO_LAYER_VIDEO1;          // 用于显示摄像头0 视频的 VO layer
    osd_vo_id_0 = K_VO_LAYER_OSD1;            // 用于摄像头0 叠加 OSD 的 VO layer
    osd_pool_id_0 = VB_INVALID_POOLID;        // OSD 所使用的 VB 内存池，初始化为无效


    vi_vo_id_1  = K_VO_LAYER_VIDEO2;          // 用于显示摄像头1 视频的 VO layer
    osd_vo_id_1 = K_VO_LAYER_OSD2;            // 用于摄像头1 叠加 OSD 的 VO layer
    osd_pool_id_1 = VB_INVALID_POOLID;        // OSD 所使用的 VB 内存池，初始化为无效

    vi_vo_id_2  = K_VO_LAYER_VIDEO3;          // 用于显示摄像头2 视频的 VO layer
    osd_vo_id_2 = K_VO_LAYER_OSD3;            // 用于摄像头2 叠加 OSD 的 VO layer
    osd_pool_id_2 = VB_INVALID_POOLID;        // OSD 所使用的 VB 内存池，初始化为无效

    // ------------------------ Sensor / VICAP 默认配置 ------------------------
    // 默认使用 GC2093
    sensor_type_0 = GC2093_MIPI_CSI0_1920X1080_30FPS_10BIT_LINEAR;
    //sensor_type_0 = GC2093_MIPI_CSI0_1920X1080_30FPS_10BIT_LINEAR;
    // VICAP 设备 0
    vicap_dev_0 = VICAP_DEV_ID_0;
    // VICAP DEV0 CHN0 → VO 通道（视频直通显示,绑定模式）
    vicap_chn_to_vo_0 = VICAP_CHN_ID_0;
    // VICAP DEV0 CHN1 → AI 通道（用于算法推理，dump模式）
    vicap_chn_to_ai_0 = VICAP_CHN_ID_1;

    // 默认使用 GC2093
    sensor_type_1 = GC2093_MIPI_CSI1_1920X1080_30FPS_10BIT_LINEAR;
    // VICAP 设备 1
    vicap_dev_1 = VICAP_DEV_ID_1;
    // VICAP DEV1 CHN0 → VO 通道（视频直通显示，绑定模式）
    vicap_chn_to_vo_1 = VICAP_CHN_ID_0;
    // VICAP DEV1 CHN1 → AI 通道（用于算法推理，dump模式）
    vicap_chn_to_ai_1 = VICAP_CHN_ID_1;

    // 默认使用 GC2093
    sensor_type_2 = GC2093_MIPI_CSI2_1920X1080_30FPS_10BIT_LINEAR;
    // VICAP 设备 2
    vicap_dev_2 = VICAP_DEV_ID_2;
    // VICAP DEV2 CHN0 → VO 通道（视频直通显示，绑定模式）
    vicap_chn_to_vo_2 = VICAP_CHN_ID_0;
    // VICAP DEV2 CHN1 → AI 通道（用于算法推理，dump模式）
    vicap_chn_to_ai_2 = VICAP_CHN_ID_1;
    // VICAP DEV0 CHN2 → VENC 通道（用于视频编码，绑定模式）
    vicap_chn_to_venc_2 = VICAP_CHN_ID_2;

    // 调试模式开关
    debug_mode_ = debug_mode;

}

PipeLine::~PipeLine()
{
}

/* 管线创建：初始化 VB → 屏幕 → VO → OSD → VICAP → 绑定关系 */
int PipeLine::Create()
{
    ScopedTiming st("PipeLine::Create", debug_mode_);
    k_s32 ret = 0;

    // =============================================================================================
    // 1. 配置 Video Buffer（VB）系统
    // =============================================================================================
    memset(&config, 0, sizeof(k_vb_config));
    config.max_pool_cnt = 64;  // 最多支持 64 个内存池

    // 设置 VB 全局配置
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        return ret;
    }

    // 设置 VB 附加配置（JPEG、ISP 统计等）
    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("vb_set_supplement_config failed ret:%d\n", ret);
        return ret;
    }

    // 初始化 VB 子系统
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        return ret;
    }

    // =============================================================================================
    // 2. 创建 OSD 专用 VB 内存池（用于 ARGB8888 叠加图层）
    // =============================================================================================
    // 用于存放一帧 OSD 数据（如 AI 结果绘制）
    if(USE_OSD == 1){
        k_vb_pool_config pool_config;

        // 摄像头0的OSD缓冲池创建
        memset(&pool_config, 0, sizeof(pool_config));
        pool_config.blk_cnt = 3; // 3 个缓冲块，避免帧冲突
        pool_config.blk_size = VICAP_ALIGN_UP((OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL), VICAP_ALIGN_1K);
        pool_config.mode = VB_REMAP_MODE_NOCACHE; // 非 cache 映射，避免缓存一致性问题
        osd_pool_id_0 = kd_mpi_vb_create_pool(&pool_config);

        // 摄像头1的OSD缓冲池创建
        memset(&pool_config, 0, sizeof(pool_config));
        pool_config.blk_cnt = 3; // 3 个缓冲块，避免帧冲突
        pool_config.blk_size = VICAP_ALIGN_UP((OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL), VICAP_ALIGN_1K);
        pool_config.mode = VB_REMAP_MODE_NOCACHE; // 非 cache 映射，避免缓存一致性问题
        osd_pool_id_1 = kd_mpi_vb_create_pool(&pool_config);

        // 摄像头2的OSD缓冲池创建
        memset(&pool_config, 0, sizeof(pool_config));
        pool_config.blk_cnt = 3; // 3 个缓冲块，避免帧冲突
        pool_config.blk_size = VICAP_ALIGN_UP((OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL), VICAP_ALIGN_1K);
        pool_config.mode = VB_REMAP_MODE_NOCACHE; // 非 cache 映射，避免缓存一致性问题
        osd_pool_id_2 = kd_mpi_vb_create_pool(&pool_config);
    }

    // =============================================================================================
    // 3. 屏幕（Connector）配置
    // =============================================================================================
    k_connector_info connector_info;
    memset(&connector_info, 0, sizeof(k_connector_info));

    // 根据 connector 类型获取硬件参数
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("the connector type not supported!\n");
        return ret;
    }

    // 打开 connector 设备
    k_s32 connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return K_ERR_VO_NOTREADY;
    }

    // 打开电源
    ret = kd_mpi_connector_power_set(connector_fd, K_TRUE);
    if (ret) {
        printf("ERROR: kd_mpi_connector_power_set failed, ret=%d\n", ret);
        return ret;
    }

    // 初始化 connector（配置时序、分辨率等）
    ret = kd_mpi_connector_init(connector_fd, connector_info);
    if (ret) {
        printf("ERROR: kd_mpi_connector_init failed, ret=%d\n", ret);
        return ret;
    }

    // 关闭设备句柄（配置完成即可关闭）
    ret = kd_mpi_connector_close(connector_fd);
    if (ret) {
        printf("ERROR: kd_mpi_connector_close failed, ret=%d\n", ret);
        return ret;
    }

    // =============================================================================================
    // 4. 配置三个摄像头的绑定 VO层（视频输出层：用于显示摄像头画面）
    // =============================================================================================

    //******************************摄像头0的CHN0的绑定层，使用K_VO_LAYER_VIDEO1***********************
    kd_mpi_vo_disable_layer(vi_vo_id_0);  // 先关闭 layer，避免旧配置干扰
    memset(&vi_vo_attr_0, 0, sizeof(vi_vo_attr_0));
    vi_vo_attr_0.layer_id        = vi_vo_id_0;
    vi_vo_attr_0.position.x      = 0;
    vi_vo_attr_0.position.y      = 0;
    vi_vo_attr_0.img_size.width  = DISPLAY_WIDTH;
    vi_vo_attr_0.img_size.height = DISPLAY_HEIGHT;
    vi_vo_attr_0.pixel_format    = PIXEL_FORMAT_YUV_SEMIPLANAR_420; // NV12
    vi_vo_attr_0.global_alpha   = 0xFF;                            // 不透明
    vi_vo_attr_0.func            = DISPLAY_MODE? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;
    vi_vo_attr_0.rot_buf_nr      = DISPLAY_MODE? 2 : 0;
    vi_vo_attr_0.rot_buf_bpp       = 0;
    ret = kd_mpi_vo_set_layer_attr(vi_vo_id_0,&vi_vo_attr_0);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_set_layer_attr failed for camera0 video, ret=%d\n", ret);
        return ret;
    }
    ret = kd_mpi_vo_enable_layer(vi_vo_id_0);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_enable_layer failed for camera0 video, ret=%d\n", ret);
        return ret;
    }

    //******************************摄像头1的CHN0的绑定层，使用K_VO_LAYER_VIDEO1***********************
    kd_mpi_vo_disable_layer(vi_vo_id_1);  // 先关闭 layer，避免旧配置干扰
    memset(&vi_vo_attr_1, 0, sizeof(vi_vo_attr_1));
    vi_vo_attr_1.layer_id        = vi_vo_id_1;
    vi_vo_attr_1.position.x      = DISPLAY_WIDTH;
    vi_vo_attr_1.position.y      = 0;
    vi_vo_attr_1.img_size.width  = DISPLAY_WIDTH;
    vi_vo_attr_1.img_size.height = DISPLAY_HEIGHT;
    vi_vo_attr_1.pixel_format    = PIXEL_FORMAT_YUV_SEMIPLANAR_420; // NV12
    vi_vo_attr_1.global_alpha   = 0xFF;                            // 不透明
    vi_vo_attr_1.func            = DISPLAY_MODE? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;
    vi_vo_attr_1.rot_buf_nr      = DISPLAY_MODE? 2 : 0;
    vi_vo_attr_1.rot_buf_bpp       = 0;
    ret = kd_mpi_vo_set_layer_attr(vi_vo_id_1,&vi_vo_attr_1);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_set_layer_attr failed for camera1 video, ret=%d\n", ret);
        return ret;
    }
    ret = kd_mpi_vo_enable_layer(vi_vo_id_1);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_enable_layer failed for camera1 video, ret=%d\n", ret);
        return ret;
    }

    //******************************摄像头2的CHN0的绑定层，使用K_VO_LAYER_VIDEO1***********************
    kd_mpi_vo_disable_layer(vi_vo_id_2);  // 先关闭 layer，避免旧配置干扰
    memset(&vi_vo_attr_2, 0, sizeof(vi_vo_attr_2));
    vi_vo_attr_2.layer_id        = vi_vo_id_2;
    vi_vo_attr_2.position.x      = 0;
    vi_vo_attr_2.position.y      = DISPLAY_HEIGHT;
    vi_vo_attr_2.img_size.width  = DISPLAY_WIDTH;
    vi_vo_attr_2.img_size.height = DISPLAY_HEIGHT;
    vi_vo_attr_2.pixel_format    = PIXEL_FORMAT_YUV_SEMIPLANAR_420; // NV12
    vi_vo_attr_2.global_alpha   = 0xFF;                            // 不透明
    vi_vo_attr_2.func            = DISPLAY_MODE? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;
    vi_vo_attr_2.rot_buf_nr      = DISPLAY_MODE? 2 : 0;
    vi_vo_attr_2.rot_buf_bpp       = 0;

    ret = kd_mpi_vo_set_layer_attr(vi_vo_id_2,&vi_vo_attr_2);
    if (ret != K_SUCCESS) {
        printf("ERROR: kd_mpi_vo_set_layer_attr failed for camera2 video, ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vo_enable_layer(vi_vo_id_2);
    if (ret != K_SUCCESS) {
        printf("kd_mpi_vo_enable_layer failed for camera2 video, ret=%d\n", ret);
        return ret;
    }

    // =============================================================================================
    // 5. 配置 OSD 层（ARGB8888 叠加图层）
    // =============================================================================================
    if(USE_OSD == 1){
        //******************************摄像头0 AI推理效果叠加层，使用K_VO_LAYER_OSD1***********************
        kd_mpi_vo_disable_layer(osd_vo_id_0);
        memset(&osd_vo_attr_0, 0, sizeof(osd_vo_attr_0));
        osd_vo_attr_0.layer_id        = osd_vo_id_0;
        osd_vo_attr_0.position.x      = 0;
        osd_vo_attr_0.position.y      = 0;
        osd_vo_attr_0.img_size.width  = OSD_WIDTH;
        osd_vo_attr_0.img_size.height = OSD_HEIGHT;
        osd_vo_attr_0.pixel_format    = PIXEL_FORMAT_ARGB_8888;  // OSD 常用 BGRA/ARGB
        osd_vo_attr_0.global_alpha    = 0xFF;
        osd_vo_attr_0.func            = DISPLAY_MODE? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;
        osd_vo_attr_0.rot_buf_nr      = DISPLAY_MODE? 2 : 0;
        osd_vo_attr_0.rot_buf_bpp       = 0;
        ret = kd_mpi_vo_set_layer_attr(osd_vo_id_0,&osd_vo_attr_0);
        if (ret != K_SUCCESS) {
            printf("kd_mpi_vo_set_layer_attr failed for camera0 osd, ret=%d\n", ret);
            return ret;
        }
        ret = kd_mpi_vo_enable_layer(osd_vo_id_0);
        if (ret != K_SUCCESS) {
            printf("kd_mpi_vo_enable_layer failed for camera0 osd, ret=%d\n", ret);
            return ret;
        }
        // 从 OSD VB 池获取一块缓存，用于写入叠加数据
        k_s32 size = VICAP_ALIGN_UP(OSD_HEIGHT * OSD_WIDTH * OSD_CHANNEL, VICAP_ALIGN_1K);

        // 从指定内存池中申请一块缓存
        handle_0 = kd_mpi_vb_get_block(osd_pool_id_0, size, NULL);
        if (handle_0 == VB_INVALID_HANDLE)
        {
            printf("%s get vb block error for camera0 osd\n", __func__);
            return -1;
        }

        // 获取该缓存块的物理地址
        k_u64 phys_addr_0 = kd_mpi_vb_handle_to_phyaddr(handle_0);
        if (phys_addr_0 == 0)
        {
            printf("%s get phys addr error for camera0 osd\n", __func__);
            return -1;
        }

        // 映射为用户态虚拟地址（非 cache）
        k_u32* virt_addr_0 = (k_u32 *)kd_mpi_sys_mmap(phys_addr_0, size);
        if (virt_addr_0 == NULL)
        {
            printf("%s mmap error for camera0 osd\n", __func__);
            return -1;
        }

        // 初始化 OSD 帧描述结构
        memset(&osd_frame_info_0, 0, sizeof(osd_frame_info_0));
        osd_frame_info_0.v_frame.width        = OSD_WIDTH;
        osd_frame_info_0.v_frame.height       = OSD_HEIGHT;
        osd_frame_info_0.v_frame.stride[0]    = OSD_WIDTH*4;
        osd_frame_info_0.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
        osd_frame_info_0.mod_id               = K_ID_VO;
        osd_frame_info_0.pool_id              = osd_pool_id_0;
        osd_frame_info_0.v_frame.phys_addr[0] = phys_addr_0;

        // 保存虚拟地址，用于后续 memcpy 写入 OSD 数据
        insert_osd_vaddr_0 = virt_addr_0;
        printf("camera0 osd video frame: phys_addr is %lx g_pool_id is %d \n", phys_addr_0, osd_pool_id_0);

        //******************************摄像头1 AI推理效果叠加层，使用K_VO_LAYER_OSD2***********************
        kd_mpi_vo_disable_layer(osd_vo_id_1);
        memset(&osd_vo_attr_1, 0, sizeof(osd_vo_attr_1));
        osd_vo_attr_1.layer_id        = osd_vo_id_1;
        osd_vo_attr_1.position.x      = OSD_WIDTH;
        osd_vo_attr_1.position.y      = 0;
        osd_vo_attr_1.img_size.width  = OSD_WIDTH;
        osd_vo_attr_1.img_size.height = OSD_HEIGHT;
        osd_vo_attr_1.pixel_format    = PIXEL_FORMAT_ARGB_8888;  // OSD 常用 BGRA/ARGB
        osd_vo_attr_1.global_alpha    = 0xFF;
        osd_vo_attr_1.func            = DISPLAY_MODE? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;
        osd_vo_attr_1.rot_buf_nr      = DISPLAY_MODE? 2 : 0;
        osd_vo_attr_1.rot_buf_bpp       = 0;

        ret = kd_mpi_vo_set_layer_attr(osd_vo_id_1,&osd_vo_attr_1);
        if (ret != K_SUCCESS) {
            printf("kd_mpi_vo_set_layer_attr failed for camera1 osd, ret=%d\n", ret);
            return ret;
        }

        ret = kd_mpi_vo_enable_layer(osd_vo_id_1);
        if (ret != K_SUCCESS) {
            printf("kd_mpi_vo_enable_layer failed for camera1 osd, ret=%d\n", ret);
            return ret;
        }

        // 从 OSD VB 池获取一块缓存，用于写入叠加数据
        size = VICAP_ALIGN_UP(OSD_HEIGHT * OSD_WIDTH * OSD_CHANNEL, VICAP_ALIGN_1K);

        // 从指定内存池中申请一块缓存
        handle_1 = kd_mpi_vb_get_block(osd_pool_id_1, size, NULL);
        if (handle_1 == VB_INVALID_HANDLE)
        {
            printf("%s get vb block error for camera1 osd\n", __func__);
            return -1;
        }

        // 获取该缓存块的物理地址
        k_u64 phys_addr_1 = kd_mpi_vb_handle_to_phyaddr(handle_1);
        if (phys_addr_1 == 0)
        {
            printf("%s get phys addr error for camera1 osd\n", __func__);
            return -1;
        }

        // 映射为用户态虚拟地址（非 cache）
        k_u32* virt_addr_1 = (k_u32 *)kd_mpi_sys_mmap(phys_addr_1, size);
        if (virt_addr_1 == NULL)
        {
            printf("%s mmap error for camera1 osd\n", __func__);
            return -1;
        }

        // 初始化 OSD 帧描述结构
        memset(&osd_frame_info_1, 0, sizeof(osd_frame_info_1));
        osd_frame_info_1.v_frame.width        = OSD_WIDTH;
        osd_frame_info_1.v_frame.height       = OSD_HEIGHT;
        osd_frame_info_1.v_frame.stride[0]    = OSD_WIDTH*4;
        osd_frame_info_1.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
        osd_frame_info_1.mod_id               = K_ID_VO;
        osd_frame_info_1.pool_id              = osd_pool_id_1;
        osd_frame_info_1.v_frame.phys_addr[0] = phys_addr_1;

        // 保存虚拟地址，用于后续 memcpy 写入 OSD 数据
        insert_osd_vaddr_1 = virt_addr_1;
        printf("camera1 osd video frame: phys_addr is %lx g_pool_id is %d \n", phys_addr_1, osd_pool_id_1);

        //******************************摄像头2 AI推理效果叠加层，使用K_VO_LAYER_OSD3***********************
        kd_mpi_vo_disable_layer(osd_vo_id_2);
        memset(&osd_vo_attr_2, 0, sizeof(osd_vo_attr_2));
        osd_vo_attr_2.layer_id        = osd_vo_id_2;
        osd_vo_attr_2.position.x      = 0;
        osd_vo_attr_2.position.y      = OSD_HEIGHT;
        osd_vo_attr_2.img_size.width  = OSD_WIDTH;
        osd_vo_attr_2.img_size.height = OSD_HEIGHT;
        osd_vo_attr_2.pixel_format    = PIXEL_FORMAT_ARGB_8888;  // OSD 常用 BGRA/ARGB
        osd_vo_attr_2.global_alpha    = 0xFF;
        osd_vo_attr_2.func            = DISPLAY_MODE? GDMA_ROTATE_DEGREE_90 : GDMA_ROTATE_DEGREE_0;
        osd_vo_attr_2.rot_buf_nr      = DISPLAY_MODE? 2 : 0;
        osd_vo_attr_2.rot_buf_bpp       = 0;
        ret = kd_mpi_vo_set_layer_attr(osd_vo_id_2,&osd_vo_attr_2);
        if (ret != K_SUCCESS) {
            printf("kd_mpi_vo_set_layer_attr failed for camera2 osd, ret=%d\n", ret);
            return ret;
        }

        ret = kd_mpi_vo_enable_layer(osd_vo_id_2);
        if (ret != K_SUCCESS) {
            printf("kd_mpi_vo_enable_layer failed for camera2 osd, ret=%d\n", ret);
            return ret;
        }

        // 从 OSD VB 池获取一块缓存，用于写入叠加数据
        size = VICAP_ALIGN_UP(OSD_HEIGHT * OSD_WIDTH * OSD_CHANNEL, VICAP_ALIGN_1K);

        // 从指定内存池中申请一块缓存
        handle_2 = kd_mpi_vb_get_block(osd_pool_id_2, size, NULL);
        if (handle_2 == VB_INVALID_HANDLE)
        {
            printf("%s get vb block error for camera2 osd\n", __func__);
            return -1;
        }

        // 获取该缓存块的物理地址
        k_u64 phys_addr_2 = kd_mpi_vb_handle_to_phyaddr(handle_2);
        if (phys_addr_2 == 0)
        {
            printf("%s get phys addr error for camera2 osd\n", __func__);
            return -1;
        }

        // 映射为用户态虚拟地址（非 cache）
        k_u32* virt_addr_2 = (k_u32 *)kd_mpi_sys_mmap(phys_addr_2, size);
        if (virt_addr_2 == NULL)
        {
            printf("%s mmap error for camera2 osd\n", __func__);
            return -1;
        }

        // 初始化 OSD 帧描述结构
        memset(&osd_frame_info_2, 0, sizeof(osd_frame_info_2));
        osd_frame_info_2.v_frame.width        = OSD_WIDTH;
        osd_frame_info_2.v_frame.height       = OSD_HEIGHT;
        osd_frame_info_2.v_frame.stride[0]    = OSD_WIDTH*4;
        osd_frame_info_2.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
        osd_frame_info_2.mod_id               = K_ID_VO;
        osd_frame_info_2.pool_id              = osd_pool_id_2;
        osd_frame_info_2.v_frame.phys_addr[0] = phys_addr_2;

        // 保存虚拟地址，用于后续 memcpy 写入 OSD 数据
        insert_osd_vaddr_2 = virt_addr_2;
        printf("camera2 osd video frame: phys_addr is %lx g_pool_id is %d \n", phys_addr_2, osd_pool_id_2);
    }

    
    // =============================================================================================
    // 6. VICAP 设备属性配置和通道属性配置
    // =============================================================================================

    //*********************************摄像头0设备配置和通道配置**************************************
    // 自动探测 Sensor
    k_vicap_probe_config probe_cfg_0;
    k_vicap_sensor_info sensor_info_0;
    probe_cfg_0.csi_num = 0;
    probe_cfg_0.width   = ISP_WIDTH;
    probe_cfg_0.height  = ISP_HEIGHT;
    probe_cfg_0.fps     = 30;
    if(0x00 != kd_mpi_sensor_adapt_get(&probe_cfg_0, &sensor_info_0)) {
        printf("vicap, can't probe sensor on %d, output %dx%d@%d\n",
               probe_cfg_0.csi_num, probe_cfg_0.width, probe_cfg_0.height, probe_cfg_0.fps);
        return -1;
    }
    sensor_type_0 =  sensor_info_0.sensor_type;
    memset(&sensor_info_0, 0, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_get_sensor_info(sensor_type_0, &sensor_info_0);
    if (ret) {
        printf("vicap dev0, the sensor type not supported!\n");
        return ret;
    }
    // 配置 VICAP 设备属性（采集窗口、工作模式、ISP 功能等）
    k_vicap_dev_attr dev_attr_0;
    memset(&dev_attr_0, 0, sizeof(k_vicap_dev_attr));
    dev_attr_0.acq_win.h_start = 0;
    dev_attr_0.acq_win.v_start = 0;
    dev_attr_0.acq_win.width   = ISP_WIDTH;
    dev_attr_0.acq_win.height  = ISP_HEIGHT;
    dev_attr_0.mode            = VICAP_WORK_OFFLINE_MODE;  // 在线模式
    dev_attr_0.pipe_ctrl.data  = 0xFFFFFFFF;
    dev_attr_0.pipe_ctrl.bits.af_enable   = 0;
    dev_attr_0.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr_0.pipe_ctrl.bits.dnr3_enable = 1;
    dev_attr_0.cpature_frame   = 0;
    dev_attr_0.sensor_info     = sensor_info_0;
    dev_attr_0.buffer_num = SENSOR_DEV_BUFFER_NUM;
    dev_attr_0.buffer_size = VICAP_ALIGN_UP((ISP_WIDTH * ISP_HEIGHT * 2), VICAP_ALIGN_1K);
    dev_attr_0.buffer_pool_id=VB_INVALID_POOLID;

    ret = kd_mpi_vicap_set_dev_attr(vicap_dev_0, dev_attr_0);
    if (ret) {
        printf("vicap dev0, kd_mpi_vicap_set_dev_attr failed.\n");
        return ret;
    }

    // VICAP 通道 0：输出到 VO 显示
    k_vicap_chn_attr dev0_chn0_attr;
    memset(&dev0_chn0_attr, 0, sizeof(k_vicap_chn_attr));
    dev0_chn0_attr.out_win.width  = DISPLAY_WIDTH;
    dev0_chn0_attr.out_win.height = DISPLAY_HEIGHT;
    dev0_chn0_attr.crop_win       = dev_attr_0.acq_win;
    dev0_chn0_attr.scale_win      = dev0_chn0_attr.out_win;
    dev0_chn0_attr.crop_enable    = K_FALSE;
    dev0_chn0_attr.scale_enable   = K_FALSE;
    dev0_chn0_attr.chn_enable     = K_TRUE;
    dev0_chn0_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    dev0_chn0_attr.buffer_num     = VO_CHN_BUFFER_NUM;
    dev0_chn0_attr.buffer_size    = VICAP_ALIGN_UP((DISPLAY_WIDTH * DISPLAY_HEIGHT * 2), VICAP_ALIGN_1K);
    dev0_chn0_attr.buffer_pool_id = VB_INVALID_POOLID;

    printf("vicap dev0, kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", dev0_chn0_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev_0, vicap_chn_to_vo_0, dev0_chn0_attr);
    if (ret) {
        printf("vicap dev0, kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    // 绑定 VICAP → VO（视频直通显示）
    vicap_mpp_chn_0.mod_id = K_ID_VI;
    vicap_mpp_chn_0.dev_id = vicap_dev_0;
    vicap_mpp_chn_0.chn_id = vicap_chn_to_vo_0;
    vo_mpp_chn_0.mod_id    = K_ID_VO;
    vo_mpp_chn_0.dev_id    = vo_dev_id;
    vo_mpp_chn_0.chn_id    = vi_vo_id_0;
    ret = kd_mpi_sys_bind(&vicap_mpp_chn_0, &vo_mpp_chn_0);
    if (ret) {
        printf("vicap dev0 kd_mpi_sys_bind failed:0x%x\n", ret);
    }

    // VICAP 通道 1：输出给 AI 使用（RGB888 Planar）
    k_vicap_chn_attr dev0_chn1_attr;
    memset(&dev0_chn1_attr, 0, sizeof(k_vicap_chn_attr));
    dev0_chn1_attr.out_win.width  = AI_FRAME_WIDTH;
    dev0_chn1_attr.out_win.height = AI_FRAME_HEIGHT;
    dev0_chn1_attr.crop_win       = dev_attr_0.acq_win;
    dev0_chn1_attr.scale_win      = dev0_chn1_attr.out_win;
    dev0_chn1_attr.crop_enable    = K_FALSE;
    dev0_chn1_attr.scale_enable   = K_FALSE;
    dev0_chn1_attr.chn_enable     = K_TRUE;
    dev0_chn1_attr.pix_format     = PIXEL_FORMAT_RGB_888_PLANAR; // AI 常用输入格式
    dev0_chn1_attr.buffer_num     = AI_CHN_BUFFER_NUM;
    dev0_chn1_attr.buffer_size    = VICAP_ALIGN_UP((AI_FRAME_WIDTH * AI_FRAME_HEIGHT * 3), VICAP_ALIGN_1K);
    dev0_chn1_attr.buffer_pool_id = VB_INVALID_POOLID;


    printf("vicap dev0 kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", dev0_chn1_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev_0 , vicap_chn_to_ai_0, dev0_chn1_attr);
    if (ret) {
        printf("vicap dev0 kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    // set to header file database parse mode
    ret = kd_mpi_vicap_set_database_parse_mode(vicap_dev_0, VICAP_DATABASE_PARSE_XML_JSON);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_set_database_parse_mode failed.\n");
        return ret;
    }

    //*********************************摄像头1设备配置和通道配置**************************************
    k_vicap_probe_config probe_cfg_1;
    k_vicap_sensor_info sensor_info_1;
    probe_cfg_1.csi_num = 1;
    probe_cfg_1.width   = ISP_WIDTH;
    probe_cfg_1.height  = ISP_HEIGHT;
    probe_cfg_1.fps     = 30;
    if(0x00 != kd_mpi_sensor_adapt_get(&probe_cfg_1, &sensor_info_1)) {
        printf("vicap, can't probe sensor on %d, output %dx%d@%d\n",
               probe_cfg_1.csi_num, probe_cfg_1.width, probe_cfg_1.height, probe_cfg_1.fps);
        return -1;
    }
    sensor_type_1 =  sensor_info_1.sensor_type;
    memset(&sensor_info_1, 0, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_get_sensor_info(sensor_type_1, &sensor_info_1);
    if (ret) {
        printf("vicap dev1, the sensor type not supported!\n");
        return ret;
    }

    // 配置 VICAP 设备属性（采集窗口、工作模式、ISP 功能等）
    k_vicap_dev_attr dev_attr_1;
    memset(&dev_attr_1, 0, sizeof(k_vicap_dev_attr));
    dev_attr_1.acq_win.h_start = 0;
    dev_attr_1.acq_win.v_start = 0;
    dev_attr_1.acq_win.width   = ISP_WIDTH;
    dev_attr_1.acq_win.height  = ISP_HEIGHT;
    dev_attr_1.mode            = VICAP_WORK_OFFLINE_MODE;  // 在线模式
    dev_attr_1.pipe_ctrl.data  = 0xFFFFFFFF;
    dev_attr_1.pipe_ctrl.bits.af_enable   = 0;
    dev_attr_1.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr_1.pipe_ctrl.bits.dnr3_enable = 1;
    dev_attr_1.cpature_frame   = 0;
    dev_attr_1.sensor_info     = sensor_info_1;
    dev_attr_1.buffer_num = SENSOR_DEV_BUFFER_NUM;
    dev_attr_1.buffer_size = VICAP_ALIGN_UP((ISP_WIDTH * ISP_HEIGHT * 2), VICAP_ALIGN_1K);
    dev_attr_1.buffer_pool_id=VB_INVALID_POOLID;


    ret = kd_mpi_vicap_set_dev_attr(vicap_dev_1, dev_attr_1);
    if (ret) {
        printf("vicap dev1, kd_mpi_vicap_set_dev_attr failed.\n");
        return ret;
    }

    // VICAP 通道 0：输出到 VO 显示
    k_vicap_chn_attr dev1_chn0_attr;
    memset(&dev1_chn0_attr, 0, sizeof(k_vicap_chn_attr));
    dev1_chn0_attr.out_win.width  = DISPLAY_WIDTH;
    dev1_chn0_attr.out_win.height = DISPLAY_HEIGHT;
    dev1_chn0_attr.crop_win       = dev_attr_1.acq_win;
    dev1_chn0_attr.scale_win      = dev1_chn0_attr.out_win;
    dev1_chn0_attr.crop_enable    = K_FALSE;
    dev1_chn0_attr.scale_enable   = K_FALSE;
    dev1_chn0_attr.chn_enable     = K_TRUE;
    dev1_chn0_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420; // NV12
    dev1_chn0_attr.buffer_num     = VO_CHN_BUFFER_NUM;
    dev1_chn0_attr.buffer_size    = VICAP_ALIGN_UP((DISPLAY_WIDTH * DISPLAY_HEIGHT * 2), VICAP_ALIGN_1K);
    dev1_chn0_attr.buffer_pool_id = VB_INVALID_POOLID;

    printf("vicap dev1, kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", dev1_chn0_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev_1, vicap_chn_to_vo_1, dev1_chn0_attr);
    if (ret) {
        printf("vicap dev1, kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    // 绑定 VICAP → VO（视频直通显示）
    vicap_mpp_chn_1.mod_id = K_ID_VI;
    vicap_mpp_chn_1.dev_id = vicap_dev_1;
    vicap_mpp_chn_1.chn_id = vicap_chn_to_vo_1;
    vo_mpp_chn_1.mod_id    = K_ID_VO;
    vo_mpp_chn_1.dev_id    = vo_dev_id;
    vo_mpp_chn_1.chn_id    = vi_vo_id_1;
    ret = kd_mpi_sys_bind(&vicap_mpp_chn_1, &vo_mpp_chn_1);
    if (ret) {
        printf("vicap dev1 kd_mpi_sys_bind failed:0x%x\n", ret);
        return ret;
    }

    // VICAP 通道 1：输出给 AI 使用（RGB888 Planar）
    k_vicap_chn_attr dev1_chn1_attr;
    memset(&dev1_chn1_attr, 0, sizeof(k_vicap_chn_attr));
    dev1_chn1_attr.out_win.width  = AI_FRAME_WIDTH;
    dev1_chn1_attr.out_win.height = AI_FRAME_HEIGHT;
    dev1_chn1_attr.crop_win       = dev_attr_1.acq_win;
    dev1_chn1_attr.scale_win      = dev1_chn1_attr.out_win;
    dev1_chn1_attr.crop_enable    = K_FALSE;
    dev1_chn1_attr.scale_enable   = K_FALSE;
    dev1_chn1_attr.chn_enable     = K_TRUE;
    dev1_chn1_attr.pix_format     = PIXEL_FORMAT_RGB_888_PLANAR; // AI 常用输入格式
    dev1_chn1_attr.buffer_num     = AI_CHN_BUFFER_NUM;
    dev1_chn1_attr.buffer_size    = VICAP_ALIGN_UP((AI_FRAME_WIDTH * AI_FRAME_HEIGHT * 3 ), VICAP_ALIGN_1K);
    dev1_chn1_attr.buffer_pool_id = VB_INVALID_POOLID;

    printf("vicap dev1 kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", dev1_chn1_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev_1, vicap_chn_to_ai_1, dev1_chn1_attr);
    if (ret) {
        printf("vicap dev1 kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    // set to header file database parse mode
    ret = kd_mpi_vicap_set_database_parse_mode(vicap_dev_1, VICAP_DATABASE_PARSE_XML_JSON);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_set_database_parse_mode failed.\n");
        return ret;
    }


    //*********************************摄像头2设备配置和通道配置**************************************
    k_vicap_probe_config probe_cfg_2;
    k_vicap_sensor_info sensor_info_2;
    probe_cfg_2.csi_num = 2;
    probe_cfg_2.width   = ISP_WIDTH;
    probe_cfg_2.height  = ISP_HEIGHT;
    probe_cfg_2.fps     = 30;
    if(0x00 != kd_mpi_sensor_adapt_get(&probe_cfg_2, &sensor_info_2)) {
        printf("vicap, can't probe sensor on %d, output %dx%d@%d\n",
               probe_cfg_2.csi_num, probe_cfg_2.width, probe_cfg_2.height, probe_cfg_2.fps);
        return -1;
    }
    sensor_type_2 =  sensor_info_2.sensor_type;
    memset(&sensor_info_2, 0, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_get_sensor_info(sensor_type_2, &sensor_info_2);
    if (ret) {
        printf("vicap dev2, the sensor type not supported!\n");
        return ret;
    }

    // 配置 VICAP 设备属性（采集窗口、工作模式、ISP 功能等）
    k_vicap_dev_attr dev_attr_2;
    memset(&dev_attr_2, 0, sizeof(k_vicap_dev_attr));
    dev_attr_2.acq_win.h_start = 0;
    dev_attr_2.acq_win.v_start = 0;
    dev_attr_2.acq_win.width   = ISP_WIDTH;
    dev_attr_2.acq_win.height  = ISP_HEIGHT;
    dev_attr_2.mode            = VICAP_WORK_OFFLINE_MODE;  // 在线模式
    dev_attr_2.pipe_ctrl.data  = 0xFFFFFFFF;
    dev_attr_2.pipe_ctrl.bits.af_enable   = 0;
    dev_attr_2.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr_2.pipe_ctrl.bits.dnr3_enable = 1;
    dev_attr_2.cpature_frame   = 0;
    dev_attr_2.sensor_info     = sensor_info_2;
    dev_attr_2.buffer_num = SENSOR_DEV_BUFFER_NUM;
    dev_attr_2.buffer_size = VICAP_ALIGN_UP((ISP_WIDTH * ISP_HEIGHT * 2), VICAP_ALIGN_1K);
    dev_attr_2.buffer_pool_id=VB_INVALID_POOLID;


    ret = kd_mpi_vicap_set_dev_attr(vicap_dev_2, dev_attr_2);
    if (ret) {
        printf("vicap dev2, kd_mpi_vicap_set_dev_attr failed.\n");
        return ret;
    }

    // VICAP 通道 0：输出到 VO 显示
    k_vicap_chn_attr dev2_chn0_attr;
    memset(&dev2_chn0_attr, 0, sizeof(k_vicap_chn_attr));
    dev2_chn0_attr.out_win.width  = DISPLAY_WIDTH;
    dev2_chn0_attr.out_win.height = DISPLAY_HEIGHT;
    dev2_chn0_attr.crop_win       = dev_attr_2.acq_win;
    dev2_chn0_attr.scale_win      = dev2_chn0_attr.out_win;
    dev2_chn0_attr.crop_enable    = K_FALSE;
    dev2_chn0_attr.scale_enable   = K_FALSE;
    dev2_chn0_attr.chn_enable     = K_TRUE;
    dev2_chn0_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420; // NV12
    dev2_chn0_attr.buffer_num     = VO_CHN_BUFFER_NUM;
    dev2_chn0_attr.buffer_size    = VICAP_ALIGN_UP((DISPLAY_WIDTH * DISPLAY_HEIGHT * 2), VICAP_ALIGN_1K);
    dev2_chn0_attr.buffer_pool_id = VB_INVALID_POOLID;

    printf("vicap dev2, kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", dev2_chn0_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev_2, vicap_chn_to_vo_2, dev2_chn0_attr);
    if (ret) {
        printf("vicap dev2, kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    // 绑定 VICAP → VO（视频直通显示）
    vicap_mpp_chn_2.mod_id = K_ID_VI;
    vicap_mpp_chn_2.dev_id = vicap_dev_2;
    vicap_mpp_chn_2.chn_id = vicap_chn_to_vo_2;
    vo_mpp_chn_2.mod_id    = K_ID_VO;
    vo_mpp_chn_2.dev_id    = vo_dev_id;
    vo_mpp_chn_2.chn_id    = vi_vo_id_2;
    ret = kd_mpi_sys_bind(&vicap_mpp_chn_2, &vo_mpp_chn_2);
    if (ret) {
        printf("vicap dev2 kd_mpi_sys_bind failed:0x%x\n", ret);
    }


    // VICAP 通道 1：输出给 AI 使用（RGB888 Planar）
    k_vicap_chn_attr dev2_chn1_attr;
    memset(&dev2_chn1_attr, 0, sizeof(k_vicap_chn_attr));
    dev2_chn1_attr.out_win.width  = AI_FRAME_WIDTH;
    dev2_chn1_attr.out_win.height = AI_FRAME_HEIGHT;
    dev2_chn1_attr.crop_win       = dev_attr_2.acq_win;
    dev2_chn1_attr.scale_win      = dev2_chn1_attr.out_win;
    dev2_chn1_attr.crop_enable    = K_FALSE;
    dev2_chn1_attr.scale_enable   = K_FALSE;
    dev2_chn1_attr.chn_enable     = K_TRUE;
    dev2_chn1_attr.pix_format     = PIXEL_FORMAT_RGB_888_PLANAR; // AI 常用输入格式
    dev2_chn1_attr.buffer_num     = AI_CHN_BUFFER_NUM;
    dev2_chn1_attr.buffer_size    = VICAP_ALIGN_UP((AI_FRAME_WIDTH * AI_FRAME_HEIGHT * 3 ), VICAP_ALIGN_1K);
    dev2_chn1_attr.buffer_pool_id = VB_INVALID_POOLID;

    printf("vicap dev2 kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", dev2_chn1_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev_2, vicap_chn_to_ai_2, dev2_chn1_attr);
    if (ret) {
        printf("vicap dev2 kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    if (ENABLE_RTSP_SERVER){
        // VICAP 通道 2：输出给VENC使用（NV12）
        k_vicap_chn_attr dev2_chn2_attr;
        memset(&dev2_chn2_attr, 0, sizeof(k_vicap_chn_attr));
        dev2_chn2_attr.out_win.width  = VENC_FRAME_WIDTH;
        dev2_chn2_attr.out_win.height = VENC_FRAME_HEIGHT;
        dev2_chn2_attr.crop_win       = dev_attr_2.acq_win;
        dev2_chn2_attr.scale_win      = dev2_chn2_attr.out_win;
        dev2_chn2_attr.crop_enable    = K_FALSE;
        dev2_chn2_attr.scale_enable   = K_FALSE;
        dev2_chn2_attr.chn_enable     = K_TRUE;
        dev2_chn2_attr.pix_format     = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        dev2_chn2_attr.buffer_num     = VENC_CHN_BUFFER_NUM;
        dev2_chn2_attr.buffer_size    = VICAP_ALIGN_UP((VENC_FRAME_WIDTH * VENC_FRAME_HEIGHT * 2), MEM_ALIGN_4K);
        dev2_chn2_attr.buffer_pool_id = VB_INVALID_POOLID;
        dev2_chn2_attr.alignment      = 12;

        printf("vicap dev2 kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", dev2_chn2_attr.buffer_size);
        ret = kd_mpi_vicap_set_chn_attr(vicap_dev_2, vicap_chn_to_venc_2, dev2_chn2_attr);
        if (ret) {
            printf("vicap dev2 kd_mpi_vicap_set_chn_attr failed.\n");
            return ret;
        }
    }

    // set to header file database parse mode
    ret = kd_mpi_vicap_set_database_parse_mode(vicap_dev_2, VICAP_DATABASE_PARSE_XML_JSON);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_set_database_parse_mode failed.\n");
        return ret;
    }
    // =============================================================================================
    // 7. VICAP 设备初始化
    // =============================================================================================
    // 初始化 VICAP
    printf("vicap dev0 kd_mpi_vicap_init\n");
    ret = kd_mpi_vicap_init(vicap_dev_0);
    if (ret) {
        printf("vicap dev0 kd_mpi_vicap_init failed.\n");
        return ret;
    }

    // 初始化 VICAP
    printf("vicap dev1 kd_mpi_vicap_init\n");
    ret = kd_mpi_vicap_init(vicap_dev_1);
    if (ret) {
        printf("vicap dev1 kd_mpi_vicap_init failed.\n");
        return ret;
    }

    // 初始化 VICAP
    printf("vicap dev2 kd_mpi_vicap_init\n");
    ret = kd_mpi_vicap_init(vicap_dev_2);
    if (ret) {
        printf("vicap dev2 kd_mpi_vicap_init failed.\n");
        return ret;
    }

    // =============================================================================================
    // 8. VICAP 设备启动
    // =============================================================================================
    //启动数据流
    printf("vicap dev0 kd_mpi_vicap_start_stream\n");
    ret = kd_mpi_vicap_start_stream(vicap_dev_0);
    if (ret) {
        printf("vicap dev0 kd_mpi_vicap_init failed.\n");
        return ret;
    }

    printf("vicap dev1 kd_mpi_vicap_start_stream\n");
    ret = kd_mpi_vicap_start_stream(vicap_dev_1);
    if (ret) {
        printf("vicap dev1 kd_mpi_vicap_init failed.\n");
        return ret;
    }

    // 启动数据流
    printf("vicap dev2 kd_mpi_vicap_start_stream\n");
    ret = kd_mpi_vicap_start_stream(vicap_dev_2);
    if (ret) {
        printf("vicap dev2 kd_mpi_vicap_init failed.\n");
        return ret;
    }

    if (ENABLE_RTSP_SERVER)
    {
        rtsp_server_.init(VENC_FRAME_WIDTH,VENC_FRAME_HEIGHT,2000,"test0",8554);
        rtsp_server_.start();
        rtsp_server_.bind_vi_chn(vicap_dev_2, vicap_chn_to_venc_2);
    }

    return ret;
}

/* 从 VICAP 通道 1 获取一帧，用于 AI 推理 ,可以选择某一个摄像头进行设置*/
int PipeLine::GetFrame(DumpRes &dump_res,int sensor_id){
    ScopedTiming st("PipeLine::GetFrame", debug_mode_);
    int ret=0;
    if(sensor_id==0){
        memset(&dump_info_0, 0, sizeof(k_video_frame_info));
        // 从 VICAP dump 一帧（阻塞最多 1000ms）
        ret = kd_mpi_vicap_dump_frame(vicap_dev_0, vicap_chn_to_ai_0, VICAP_DUMP_RGB, &dump_info_0, 1500);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_frame failed.\n");
            return ret;
        }

        // 将物理地址映射为虚拟地址，供 CPU 访问
        dump_res.virt_addr = reinterpret_cast<uintptr_t>(
            kd_mpi_sys_mmap(dump_info_0.v_frame.phys_addr[0],
                            AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH));
        dump_res.phy_addr = reinterpret_cast<uintptr_t>(dump_info_0.v_frame.phys_addr[0]);
    }
    else if(sensor_id==1){
        memset(&dump_info_1, 0, sizeof(k_video_frame_info));
        // 从 VICAP dump 一帧（阻塞最多 1000ms）
        ret = kd_mpi_vicap_dump_frame(vicap_dev_1, vicap_chn_to_ai_1, VICAP_DUMP_RGB, &dump_info_1, 1500);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_frame failed.\n");
            return ret;
        }

        // 将物理地址映射为虚拟地址，供 CPU 访问
        dump_res.virt_addr = reinterpret_cast<uintptr_t>(
            kd_mpi_sys_mmap(dump_info_1.v_frame.phys_addr[0],
                            AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH));
        dump_res.phy_addr = reinterpret_cast<uintptr_t>(dump_info_1.v_frame.phys_addr[0]);
    }
    else if(sensor_id==2){
        memset(&dump_info_2, 0, sizeof(k_video_frame_info));
        // 从 VICAP dump 一帧（阻塞最多 1000ms）
        ret = kd_mpi_vicap_dump_frame(vicap_dev_2, vicap_chn_to_ai_2, VICAP_DUMP_RGB, &dump_info_2, 1500);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_frame failed.\n");
            return ret;
        }

        // 将物理地址映射为虚拟地址，供 CPU 访问
        dump_res.virt_addr = reinterpret_cast<uintptr_t>(
            kd_mpi_sys_mmap(dump_info_2.v_frame.phys_addr[0],
                            AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH));
        dump_res.phy_addr = reinterpret_cast<uintptr_t>(dump_info_2.v_frame.phys_addr[0]);
    }
    else{
        memset(&dump_info_2, 0, sizeof(k_video_frame_info));
        // 从 VICAP dump 一帧（阻塞最多 1000ms）
        ret = kd_mpi_vicap_dump_frame(vicap_dev_2, vicap_chn_to_ai_2, VICAP_DUMP_RGB, &dump_info_2, 1500);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_frame failed.\n");
            return ret;
        }

        // 将物理地址映射为虚拟地址，供 CPU 访问
        dump_res.virt_addr = reinterpret_cast<uintptr_t>(
            kd_mpi_sys_mmap_cached(dump_info_2.v_frame.phys_addr[0],
                            AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH));
        dump_res.phy_addr = reinterpret_cast<uintptr_t>(dump_info_2.v_frame.phys_addr[0]);
    }

    return ret;
}

/* 释放当前 dump 帧 */
int PipeLine::ReleaseFrame(DumpRes &dump_res,int sensor_id){
    ScopedTiming st("PipeLine::ReleaseFrame", debug_mode_);
    int ret=0;
    if(sensor_id==0){
        // 解除虚拟地址映射
        kd_mpi_sys_munmap(reinterpret_cast<void*>(dump_res.virt_addr),
                        AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH);

        // 释放 VICAP dump 帧
        ret = kd_mpi_vicap_dump_release(vicap_dev_0, vicap_chn_to_ai_0, &dump_info_0);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_release failed.\n");
        }
    }
    else if(sensor_id==1){
        // 解除虚拟地址映射
        kd_mpi_sys_munmap(reinterpret_cast<void*>(dump_res.virt_addr),
                        AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH);

        // 释放 VICAP dump 帧
        ret = kd_mpi_vicap_dump_release(vicap_dev_1, vicap_chn_to_ai_1, &dump_info_1);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_release failed.\n");
        }
    }
    else if(sensor_id==2){
        // 解除虚拟地址映射
        kd_mpi_sys_munmap(reinterpret_cast<void*>(dump_res.virt_addr),
                        AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH);

        // 释放 VICAP dump 帧
        ret = kd_mpi_vicap_dump_release(vicap_dev_2, vicap_chn_to_ai_2, &dump_info_2);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_release failed.\n");
        }
    }else{
        // 解除虚拟地址映射
        kd_mpi_sys_munmap(reinterpret_cast<void*>(dump_res.virt_addr),
                        AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH);

        // 释放 VICAP dump 帧
        ret = kd_mpi_vicap_dump_release(vicap_dev_2, vicap_chn_to_ai_2, &dump_info_2);
        if (ret)
        {
            printf("kd_mpi_vicap_dump_release failed.\n");
        }
    }

    return ret;
}

/* 向 OSD layer 插入一帧（用于叠加显示 AI 结果） */
int PipeLine::InsertFrame(void* osd_data,int vo_layer_id){
    ScopedTiming st("PipeLine::InsertFrame", debug_mode_);
    int ret=0;
    if(vo_layer_id==4){
        // 将外部生成的 OSD 数据拷贝到 VB 映射的内存中
        memcpy(insert_osd_vaddr_0, osd_data, OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);

        // 插入到 VO 的 OSD layer
        if (kd_mpi_vo_insert_frame(osd_vo_id_0, &osd_frame_info_0) != K_SUCCESS) {
            printf("ERROR: kd_mpi_vo_insert_frame failed for OSD\n");
            return ret;
        }
    }
    else if(vo_layer_id==5){
        // 将外部生成的 OSD 数据拷贝到 VB 映射的内存中
        memcpy(insert_osd_vaddr_1, osd_data, OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);

        // 插入到 VO 的 OSD layer
        if (kd_mpi_vo_insert_frame(osd_vo_id_1, &osd_frame_info_1) != K_SUCCESS) {
            printf("ERROR: kd_mpi_vo_insert_frame failed for OSD\n");
            return ret;
        }
    }
    else if(vo_layer_id==6){
        // 将外部生成的 OSD 数据拷贝到 VB 映射的内存中
        memcpy(insert_osd_vaddr_2, osd_data, OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);

        // 插入到 VO 的 OSD layer
        if (kd_mpi_vo_insert_frame(osd_vo_id_2, &osd_frame_info_2) != K_SUCCESS) {
            printf("ERROR: kd_mpi_vo_insert_frame failed for OSD\n");
            return ret;
        }
    }
    else{
        // 将外部生成的 OSD 数据拷贝到 VB 映射的内存中
        memcpy(insert_osd_vaddr_2, osd_data, OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);

        // 插入到 VO 的 OSD layer
        if (kd_mpi_vo_insert_frame(osd_vo_id_2, &osd_frame_info_2) != K_SUCCESS) {
            printf("ERROR: kd_mpi_vo_insert_frame failed for OSD\n");
            return ret;
        }
    }

    return ret;
}

/* 销毁管线，释放所有资源 */
int PipeLine::Destroy()
{
    ScopedTiming st("PipeLine::Destroy", debug_mode_);
    int ret=0;
    // ------------------ 停止 VICAP DEV0 并解除绑定和逆初始化 ------------------
    ret = kd_mpi_vicap_stop_stream(vicap_dev_0);
    if (ret) {
        printf("vicap dev0 kd_mpi_vicap_stop_stream failed.\n");
        return ret;
    }

    // 反初始化 VICAP
    ret = kd_mpi_vicap_deinit(vicap_dev_0);
    if (ret) {
        printf("vicap dev0 kd_mpi_vicap_deinit failed.\n");
        return ret;
    }

    //  解除 VI → VO 绑定
    ret = kd_mpi_vo_disable_layer(vi_vo_id_0);
    if (ret) {
        printf("vicap dev0 kd_mpi_vo_disable_layer failed.\n");
        return ret;
    }

    vicap_mpp_chn_0.mod_id = K_ID_VI;
    vicap_mpp_chn_0.dev_id = vicap_dev_0;
    vicap_mpp_chn_0.chn_id = vicap_chn_to_vo_0;
    vo_mpp_chn_0.mod_id    = K_ID_VO;
    vo_mpp_chn_0.dev_id    = vo_dev_id;
    vo_mpp_chn_0.chn_id    = vi_vo_id_0;
    ret = kd_mpi_sys_unbind(&vicap_mpp_chn_0, &vo_mpp_chn_0);
    if (ret) {
        printf("vicap dev0 kd_mpi_sys_unbind failed:0x%x\n", ret);
    }

    /* 等待一帧时间，确保 VO 释放 VB */
    k_u32 display_ms = 1000 / 33;
    usleep(1000 * display_ms);

    // 销毁 OSD 内存池
    if(USE_OSD == 1)
    {
        ret = kd_mpi_vo_disable_layer(osd_vo_id_0);
        if (ret) {
            printf("vicap dev0 kd_mpi_vo_disable_layer failed.\n");
            return ret;
        }
        ret = kd_mpi_vb_release_block(handle_0);
        if (ret) {
            printf("kd_mpi_vb_release_block failed.\n");
            return ret;
        }

        if (osd_pool_id_0 != VB_INVALID_POOLID){
            ret = kd_mpi_sys_munmap(reinterpret_cast<void*>(insert_osd_vaddr_0),
                                    OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);
            if (ret) {
                printf("vicap dev0 kd_mpi_sys_munmap failed.\n");
                return ret;
            }
            ret = kd_mpi_vb_destory_pool(osd_pool_id_0);
            if (ret) {
                printf("kd_mpi_vb_destory_pool failed.\n");
                return ret;
            }
            osd_pool_id_0 = VB_INVALID_POOLID;
        }
    }

    // ------------------ 停止 VICAP DEV1 并解除绑定和逆初始化 ------------------
    ret = kd_mpi_vicap_stop_stream(vicap_dev_1);
    if (ret) {
        printf("vicap dev1 kd_mpi_vicap_stop_stream failed.\n");
        return ret;
    }

    // 反初始化 VICAP
    ret = kd_mpi_vicap_deinit(vicap_dev_1);
    if (ret) {
        printf("vicap dev1 kd_mpi_vicap_deinit failed.\n");
        return ret;
    }

    // 解除 VI → VO 绑定
    ret = kd_mpi_vo_disable_layer(vi_vo_id_1);
    if (ret) {
        printf("vicap dev1 kd_mpi_vo_disable_layer failed.\n");
        return ret;
    }

    vicap_mpp_chn_1.mod_id = K_ID_VI;
    vicap_mpp_chn_1.dev_id = vicap_dev_1;
    vicap_mpp_chn_1.chn_id = vicap_chn_to_vo_1;
    vo_mpp_chn_1.mod_id    = K_ID_VO;
    vo_mpp_chn_1.dev_id    = vo_dev_id;
    vo_mpp_chn_1.chn_id    = vi_vo_id_1;
    ret = kd_mpi_sys_unbind(&vicap_mpp_chn_1, &vo_mpp_chn_1);
    if (ret) {
        printf("vicap dev1 kd_mpi_sys_unbind failed:0x%x\n", ret);
    }

    /* 等待一帧时间，确保 VO 释放 VB */
    display_ms = 1000 / 33;
    usleep(1000 * display_ms);

    if(USE_OSD == 1)
    {
        ret = kd_mpi_vo_disable_layer(osd_vo_id_1);
        if (ret) {
            printf("vicap dev1 kd_mpi_vo_disable_layer failed.\n");
            return ret;
        }
        ret = kd_mpi_vb_release_block(handle_1);
        if (ret) {
            printf("vicap dev1 kd_mpi_vb_release_block failed.\n");
            return ret;
        }

        //销毁 OSD 内存池
        if (osd_pool_id_1 != VB_INVALID_POOLID){
            ret = kd_mpi_sys_munmap(reinterpret_cast<void*>(insert_osd_vaddr_1),
                                    OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);
            if (ret) {
                printf("vicap dev1 kd_mpi_sys_munmap failed.\n");
                return ret;
            }
            ret = kd_mpi_vb_destory_pool(osd_pool_id_1);
            if (ret) {
                printf("vicap dev1 kd_mpi_vb_destory_pool failed.\n");
                return ret;
            }
            osd_pool_id_1 = VB_INVALID_POOLID;
        }
    }

    // ------------------ 停止 VICAP DEV2 并解除绑定和逆初始化 ------------------
    ret = kd_mpi_vicap_stop_stream(vicap_dev_2);
    if (ret) {
        printf("vicap dev2 kd_mpi_vicap_stop_stream failed.\n");
        return ret;
    }

    // 反初始化 VICAP
    ret = kd_mpi_vicap_deinit(vicap_dev_2);
    if (ret) {
        printf("vicap dev2 kd_mpi_vicap_deinit failed.\n");
        return ret;
    }

    //  解除 VI → VO 绑定
    ret = kd_mpi_vo_disable_layer(vi_vo_id_2);
    if (ret) {
        printf("vicap dev2 kd_mpi_vo_disable_layer failed.\n");
        return ret;
    }

    vicap_mpp_chn_2.mod_id = K_ID_VI;
    vicap_mpp_chn_2.dev_id = vicap_dev_2;
    vicap_mpp_chn_2.chn_id = vicap_chn_to_vo_2;
    vo_mpp_chn_2.mod_id    = K_ID_VO;
    vo_mpp_chn_2.dev_id    = vo_dev_id;
    vo_mpp_chn_2.chn_id    = vi_vo_id_2;
    ret = kd_mpi_sys_unbind(&vicap_mpp_chn_2, &vo_mpp_chn_2);
    if (ret) {
        printf("vicap dev2 kd_mpi_sys_unbind failed:0x%x\n", ret);
    }

    /* 等待一帧时间，确保 VO 释放 VB */
    display_ms = 1000 / 33;
    usleep(1000 * display_ms);

    if(USE_OSD == 1)
    {
        ret = kd_mpi_vo_disable_layer(osd_vo_id_2);
        if (ret) {
            printf("vicap dev2 kd_mpi_vo_disable_layer failed.\n");
            return ret;
        }
        ret = kd_mpi_vb_release_block(handle_2);
        if (ret) {
            printf("vicap dev2 kd_mpi_vb_release_block failed.\n");
            return ret;
        }

        // 销毁 OSD 内存池
        if (osd_pool_id_2 != VB_INVALID_POOLID){
            ret = kd_mpi_sys_munmap(reinterpret_cast<void*>(insert_osd_vaddr_2),
                                    OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);
            if (ret) {
                printf("vicap dev2 kd_mpi_sys_munmap failed.\n");
                return ret;
            }
            ret = kd_mpi_vb_destory_pool(osd_pool_id_2);
            if (ret) {
                printf("vicap dev2 kd_mpi_vb_destory_pool failed.\n");
                return ret;
            }
            osd_pool_id_1 = VB_INVALID_POOLID;
        }
    }

    // ------------------ 停止 RTSP 服务器 ------------------

    if (ENABLE_RTSP_SERVER)
    {
        rtsp_server_.stop();
        rtsp_server_.deinit();
    }

    // ------------------ 反初始化 VB ------------------
    ret = kd_mpi_vb_exit();
    if (ret) {
        printf("kd_mpi_vb_exit failed.\n");
        return ret;
    }
    return 0;
}