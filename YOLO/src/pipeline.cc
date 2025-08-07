#include "pipeline.h"

#define ALIGN_UP_16(x)  (((x) + 15) & ~15)

PipeLine::PipeLine(GeneralConfig &general_config,int debug_mode)
{
    general_config_ = general_config;
    //配置屏幕类型
    if(general_config_.DISPLAY_MODE==0){
        connector_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
    }
    else if(general_config_.DISPLAY_MODE==1){
        connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
    }
    else if(general_config_.DISPLAY_MODE==2){
        connector_type = HX8377_V2_MIPI_4LAN_1080X1920_30FPS;
    }
    else{
        connector_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
    }
    

    vo_chn_id = K_VO_LAYER1;

    osd_chn_id = K_VO_OSD3;
    // 在start()方法中自动适配类型
    sensor_type = SENSOR_TYPE_MAX;
    // vicap设备ID
    vicap_dev=VICAP_DEV_ID_0;
    // vicap到vo通道ID
    vicap_chn_to_vo=VICAP_CHN_ID_0;
    // vicap到AI通道ID
    vicap_chn_to_ai=VICAP_CHN_ID_1;

    vo_dev_id=K_VO_DISPLAY_DEV_ID;
    vo_bind_chn_id=K_VO_DISPLAY_CHN_ID1;

    debug_mode_=debug_mode;
}

PipeLine::~PipeLine()
{
}

int PipeLine::Create()
{
    ScopedTiming st("PipeLine::Create", debug_mode_);
    k_s32 ret = 0;
    k_u32 pool_id;
    k_vb_pool_config pool_config;

    //---------------------------- 配置video buffer------------------------------------------------
    memset(&config, 0, sizeof(k_vb_config));
    config.max_pool_cnt = 64;
    //VB for YUV420SP format, to Display；创建buffer, YUV420SP格式大小，直接绑定到Display显示
    config.comm_pool[0].blk_cnt = 4;
    config.comm_pool[0].mode = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[0].blk_size = VICAP_ALIGN_UP((general_config_.DISPLAY_WIDTH * general_config_.DISPLAY_HEIGHT * 3 / 2), VICAP_ALIGN_1K);
    //VB for RGBP888 format, to AI；创建buffer,RGBP888格式大小，用于送给AI通道进行预处理
    config.comm_pool[1].blk_cnt = 4;
    config.comm_pool[1].mode = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[1].blk_size = VICAP_ALIGN_UP((general_config_.AI_FRAME_WIDTH * general_config_.AI_FRAME_HEIGHT * 3 ), VICAP_ALIGN_1K);
    //VB for ARGB8888 format, to OSD；创建buffer,ARGB88808格式大小，用于创建一个空图，绘制AI推理的结果
    if(general_config_.USE_OSD == 1){
        config.comm_pool[2].blk_cnt = 4;
        config.comm_pool[2].mode = VB_REMAP_MODE_NOCACHE;
        config.comm_pool[2].blk_size = VICAP_ALIGN_UP((general_config_.OSD_WIDTH * general_config_.OSD_HEIGHT * general_config_.OSD_CHANNEL), VICAP_ALIGN_1K);
        osd_pool_id=2;
        if(general_config_.DISPLAY_ROTATE==1){
            config.comm_pool[3].blk_cnt = 4;
            config.comm_pool[3].mode = VB_REMAP_MODE_NOCACHE;
            config.comm_pool[3].blk_size = VICAP_ALIGN_UP((general_config_.OSD_WIDTH * general_config_.OSD_HEIGHT * general_config_.OSD_CHANNEL), VICAP_ALIGN_1K);
            gdma_pool_id=3;
        }
    }
    // 设置vb配置
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        return ret;
    }
    // 设置vb附加配置，如DCF信息/ISP统计信息/ISP实时参数等
    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("vb_set_supplement_config failed ret:%d\n", ret);
        return ret;
    }
    // vb初始化
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        return ret;
    }
    //------------------------------------------------------------------------------------------------
    
    // ---------------------------------配置屏幕-------------------------------------------------------
    // 配置connector info
    k_connector_info connector_info;
    memset(&connector_info, 0, sizeof(k_connector_info));
    //根据connector的类型获取数据结构
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("the connector type not supported!\n");
        return ret;
    }
    //打开connector设备
    k_s32 connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return K_ERR_VO_NOTREADY;
    }
    //打开电源
    kd_mpi_connector_power_set(connector_fd, K_TRUE);
    //connector初始化
    kd_mpi_connector_init(connector_fd, connector_info);
    //--------------------------------------------------------------------------------------------------

    //-----------------------------------配置vo-----------------------------------------------------------
    //初始化VO配置，包括分辨率、是否旋转、显示位置
    //Layer1设置
    k_vo_video_layer_attr vo_attr;
    memset(&vo_attr, 0, sizeof(k_vo_video_layer_attr));
    vo_attr.display_rect = {0,0};
    vo_attr.img_size = {(unsigned int)general_config_.DISPLAY_HEIGHT,(unsigned int)general_config_.DISPLAY_WIDTH};
    vo_attr.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    if (vo_attr.pixel_format != PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    {
        printf("input pix format failed \n");
        return -1;
    }
    vo_attr.stride = (general_config_.DISPLAY_WIDTH / 8 - 1) + ((general_config_.DISPLAY_HEIGHT - 1) << 16);
    if(general_config_.DISPLAY_ROTATE==0){
        vo_attr.func = K_ROTATION_0;
    }
    else if(general_config_.DISPLAY_ROTATE==1){
        vo_attr.func = K_ROTATION_90;
    }
    else{
        vo_attr.func = K_ROTATION_0;
    }
    // 配置检查，rotate只能在layer0和layer1使用
    if ((vo_chn_id >= K_MAX_VO_LAYER_NUM) || ((vo_attr.func & K_VO_SCALER_ENABLE) && (vo_chn_id != K_VO_LAYER0)) || ((vo_attr.func != K_ROTATION_0) && (vo_chn_id == K_VO_LAYER2)))
    {
        printf("input layer num failed \n");
        return -1 ;
    }
    //设置VO layer1的属性
    kd_mpi_vo_set_video_layer_attr(vo_chn_id, &vo_attr);
    //使能该layer
    kd_mpi_vo_enable_video_layer(vo_chn_id);
    //---------------------------------------------------------------------------------------------------

    //-----------------------------------配置OSD---------------------------------------------------------
    if(general_config_.USE_OSD == 1){
        if(general_config_.DISPLAY_ROTATE==1){
            //初始化OSD配置
            osd_info osd;
            osd.act_size.width = general_config_.OSD_HEIGHT ;
            osd.act_size.height = general_config_.OSD_WIDTH;
            osd.offset.x = 0;
            osd.offset.y = 0;
            osd.global_alptha = 0xff;
            osd.format = PIXEL_FORMAT_BGRA_8888;
            //配置OSD通道属性
            k_vo_video_osd_attr osd_attr;
            memset(&osd_attr, 0, sizeof(k_vo_video_osd_attr));
            osd_attr.global_alptha = 0xff;
            osd_attr.pixel_format = PIXEL_FORMAT_BGRA_8888;
            osd_attr.display_rect = {0,0};
            osd_attr.img_size = {(unsigned int)general_config_.OSD_HEIGHT,(unsigned int)general_config_.OSD_WIDTH};
            if (osd_attr.pixel_format == PIXEL_FORMAT_ABGR_8888 || osd_attr.pixel_format == PIXEL_FORMAT_ARGB_8888 || osd_attr.pixel_format == PIXEL_FORMAT_BGRA_8888)
            {
                osd_attr.stride  = general_config_.OSD_HEIGHT * 4 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_565 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_565)
            {
                osd_attr.stride  = general_config_.OSD_HEIGHT * 2 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_888 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_888)
            {
                osd_attr.stride  = general_config_.OSD_HEIGHT * 3 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_4444 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_4444)
            {
                osd_attr.stride  = general_config_.OSD_HEIGHT * 2 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_1555 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_1555)
            {
                osd_attr.stride  = general_config_.OSD_HEIGHT * 2 / 8;
            }
            else
            {
                printf("set osd pixel format failed  \n");
                return -1;
            }
            kd_mpi_vo_set_video_osd_attr(osd_chn_id, &osd_attr);
            kd_mpi_vo_osd_enable(osd_chn_id);
        }
        else{
            //初始化OSD配置
            osd_info osd;
            osd.act_size.width = general_config_.OSD_WIDTH ;
            osd.act_size.height = general_config_.OSD_HEIGHT;
            osd.offset.x = 0;
            osd.offset.y = 0;
            osd.global_alptha = 0xff;
            osd.format = PIXEL_FORMAT_BGRA_8888;
            //配置OSD通道属性
            k_vo_video_osd_attr osd_attr;
            memset(&osd_attr, 0, sizeof(k_vo_video_osd_attr));
            osd_attr.global_alptha = 0xff;
            osd_attr.pixel_format = PIXEL_FORMAT_BGRA_8888;
            osd_attr.display_rect = {0,0};
            osd_attr.img_size = {(unsigned int)general_config_.OSD_WIDTH,(unsigned int)general_config_.OSD_HEIGHT};
            if (osd_attr.pixel_format == PIXEL_FORMAT_ABGR_8888 || osd_attr.pixel_format == PIXEL_FORMAT_ARGB_8888 || osd_attr.pixel_format == PIXEL_FORMAT_BGRA_8888)
            {
                osd_attr.stride  = general_config_.OSD_WIDTH * 4 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_565 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_565)
            {
                osd_attr.stride  = general_config_.OSD_WIDTH * 2 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_888 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_888)
            {
                osd_attr.stride  = general_config_.OSD_WIDTH * 3 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_4444 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_4444)
            {
                osd_attr.stride  = general_config_.OSD_WIDTH * 2 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_1555 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_1555)
            {
                osd_attr.stride  = general_config_.OSD_WIDTH * 2 / 8;
            }
            else
            {
                printf("set osd pixel format failed  \n");
                return -1;
            }
            kd_mpi_vo_set_video_osd_attr(osd_chn_id, &osd_attr);
            kd_mpi_vo_osd_enable(osd_chn_id);
        }
        //从osd的缓冲池获取该帧的缓存块，并初始化一个OSD帧数据，并绑定指针pic_vaddr用于拷贝OSD结果数据
        k_s32 size = VICAP_ALIGN_UP(general_config_.OSD_HEIGHT * general_config_.OSD_WIDTH * general_config_.OSD_CHANNEL, VICAP_ALIGN_1K);
        //用户态获取一个缓存块，传入参数，缓存池id和缓存块大小，osd_pool_id在内存分配时确定
        handle = kd_mpi_vb_get_block(osd_pool_id, size, NULL);
        if (handle == VB_INVALID_HANDLE)
        {
            printf("%s get vb block error\n", __func__);
            return -1;
        }
        //用户态获取该缓存块的物理地址
        k_u64 phys_addr = kd_mpi_vb_handle_to_phyaddr(handle);
        if (phys_addr == 0)
        {
            printf("%s get phys addr error\n", __func__);
            return -1;
        }
        //映射为虚拟地址
        k_u32* virt_addr = (k_u32 *)kd_mpi_sys_mmap(phys_addr, size);
        //带cache的虚拟地址
        // virt_addr = (k_u32 *)kd_mpi_sys_mmap_cached(phys_addr, size);
        if (virt_addr == NULL)
        {
            printf("%s mmap error\n", __func__);
            return -1;
        }
        //创建OSD数据帧，并初始化帧信息，并将该帧的虚拟地址绑定到insert_osd_vaddr上
        memset(&osd_frame_info, 0, sizeof(osd_frame_info));
        osd_frame_info.v_frame.width = general_config_.OSD_HEIGHT;
        osd_frame_info.v_frame.height = general_config_.OSD_WIDTH;
        osd_frame_info.v_frame.stride[0] = general_config_.OSD_HEIGHT;
        osd_frame_info.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
        osd_frame_info.mod_id = K_ID_VO;
        osd_frame_info.pool_id = osd_pool_id;
        osd_frame_info.v_frame.phys_addr[0] = phys_addr;
        insert_osd_vaddr = virt_addr;
        printf("phys_addr is %lx g_pool_id is %d \n", phys_addr, osd_pool_id);
    }
    //---------------------------------------------------------------------------------------------

    //-------------------------------配置GDMA旋转--------------------------------------------------
    if(general_config_.DISPLAY_ROTATE==1){
        gdma_dev_attr.burst_len = 0;
        gdma_dev_attr.ckg_bypass = (k_bool)0xff;
        gdma_dev_attr.outstanding = 7;

        memset(&dma_attr, 0, sizeof(k_dma_chn_attr_u));
        dma_attr.gdma_attr.buffer_num = 3;
        dma_attr.gdma_attr.rotation = DEGREE_90;
        dma_attr.gdma_attr.x_mirror = K_FALSE;
        dma_attr.gdma_attr.y_mirror = K_FALSE;
        dma_attr.gdma_attr.width = general_config_.OSD_WIDTH;
        dma_attr.gdma_attr.height = general_config_.OSD_HEIGHT;
        dma_attr.gdma_attr.src_stride[0] = general_config_.OSD_WIDTH * 4;
        dma_attr.gdma_attr.dst_stride[0] = general_config_.OSD_HEIGHT * 4;
        dma_attr.gdma_attr.work_mode = DMA_UNBIND;
        dma_attr.gdma_attr.pixel_format = DMA_PIXEL_FORMAT_ABGR_8888;

        //从dma的缓冲池获取该帧的缓存块，并初始化一帧数据，并绑定指针pic_vaddr用于拷贝结果数据
        k_s32 size = VICAP_ALIGN_UP(general_config_.OSD_HEIGHT * general_config_.OSD_WIDTH * general_config_.OSD_CHANNEL, VICAP_ALIGN_1K);
        //用户态获取一个缓存块，传入参数，缓存池id和缓存块大小，osd_pool_id在内存分配时确定
        gdma_handle = kd_mpi_vb_get_block(gdma_pool_id, size, NULL);
        if (gdma_handle == VB_INVALID_HANDLE)
        {
            printf("%s get vb block error\n", __func__);
            return -1;
        }
        //用户态获取该缓存块的物理地址  
        k_u64 phys_addr = kd_mpi_vb_handle_to_phyaddr(gdma_handle);
        if (phys_addr == 0)
        {
            printf("%s get phys addr error\n", __func__);
            return -1;
        }
        //映射为虚拟地址
        k_u8* virt_addr = (k_u8 *)kd_mpi_sys_mmap(phys_addr, size);
        //带cache的虚拟地址
        // virt_addr = (k_u32 *)kd_mpi_sys_mmap_cached(phys_addr, size);
        if (virt_addr == NULL)
        {
            printf("%s mmap error\n", __func__);
            return -1;
        }
        //创建GDMA数据帧，并初始化帧信息，并将该帧的虚拟地址绑定到insert_gdma_vaddr上
        memset(&gdma_frame_info, 0, sizeof(gdma_frame_info));
        gdma_frame_info.v_frame.width = general_config_.OSD_WIDTH;
        gdma_frame_info.v_frame.height = general_config_.OSD_HEIGHT;
        gdma_frame_info.v_frame.stride[0] = general_config_.OSD_WIDTH;
        gdma_frame_info.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
        gdma_frame_info.mod_id = K_ID_DMA;
        gdma_frame_info.pool_id = gdma_pool_id;
        gdma_frame_info.v_frame.phys_addr[0] = phys_addr;
        gdma_frame_info.v_frame.virt_addr[0] = (k_u64)(intptr_t)virt_addr;
        insert_gdma_vaddr = (void*)virt_addr;
        printf("dma phys_addr is %lx  dma g_pool_id is %d \n", phys_addr, gdma_pool_id);

        ret=kd_mpi_dma_set_dev_attr(&gdma_dev_attr);
        if(ret){
            printf("gdma dma dev set failed!\n");
            return ret;
        }

        ret = kd_mpi_dma_request_chn(GDMA_TYPE);
        if(ret){
            printf("gdma dma chn request failed!\n");
            return ret;
        }

        ret = kd_mpi_dma_set_chn_attr(0, &dma_attr);
        if (ret)
        {
            printf("set dma chn attr failed\r\n");
            return ret;
        }

        ret = kd_mpi_dma_start_dev();
        if (ret)
        {
            printf("start dma dev failed.\r\n");
            return ret;
        }
        ret = kd_mpi_dma_start_chn(0);
        if (ret)
        {
            printf("start dma chn failed.\r\n");
            return ret;
        }
    }
    //--------------------------------------------------------------------------------------------


    //------------------------------- 配置Sensor & vicap-----------------------------------------------------
    //sensor类型自动探测
    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_info sensor_info;
    probe_cfg.csi_num = CONFIG_MPP_SENSOR_DEFAULT_CSI + 1;
    probe_cfg.width = general_config_.ISP_WIDTH;
    probe_cfg.height = general_config_.ISP_HEIGHT;
    probe_cfg.fps = 30;
    if(0x00 != kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info)) {
        printf("vicap, can't probe sensor on %d, output %dx%d@%d\n", probe_cfg.csi_num, probe_cfg.width, probe_cfg.height, probe_cfg.fps);
        return -1;
    }
    sensor_type =  sensor_info.sensor_type;
    memset(&sensor_info, 0, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
    if (ret) {
        printf("vicap, the sensor type not supported!\n");
        return ret;
    }

    //初始化vicap的设备
    k_vicap_dev_attr dev_attr;
    memset(&dev_attr, 0, sizeof(k_vicap_dev_attr));
    dev_attr.acq_win.h_start = 0;
    dev_attr.acq_win.v_start = 0;
    dev_attr.acq_win.width = general_config_.ISP_WIDTH;
    dev_attr.acq_win.height = general_config_.ISP_HEIGHT;
    dev_attr.mode = VICAP_WORK_ONLINE_MODE;
    dev_attr.pipe_ctrl.data = 0xFFFFFFFF;
    dev_attr.pipe_ctrl.bits.af_enable = 0;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.pipe_ctrl.bits.dnr3_enable = 0;
    dev_attr.cpature_frame = 0;
    dev_attr.sensor_info = sensor_info;
    ret = kd_mpi_vicap_set_dev_attr(vicap_dev, dev_attr);
    if (ret) {
        printf("vicap, kd_mpi_vicap_set_dev_attr failed.\n");
        return ret;
    }

    // 配置vicap的通道0，即vicap_chn_to_vo
    k_vicap_chn_attr chn0_attr;
    memset(&chn0_attr, 0, sizeof(k_vicap_chn_attr));
    chn0_attr.out_win.h_start = 0;
    chn0_attr.out_win.v_start = 0;
    chn0_attr.out_win.width = general_config_.DISPLAY_WIDTH;
    chn0_attr.out_win.height = general_config_.DISPLAY_HEIGHT;
    chn0_attr.crop_win = dev_attr.acq_win;
    chn0_attr.scale_win = chn0_attr.out_win;
    chn0_attr.crop_enable = K_FALSE;
    chn0_attr.scale_enable = K_FALSE;
    chn0_attr.chn_enable = K_TRUE;
    chn0_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn0_attr.buffer_num = VICAP_MAX_FRAME_COUNT;
    chn0_attr.buffer_size = config.comm_pool[0].blk_size;
    printf("vicap ...kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", chn0_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev, vicap_chn_to_vo, chn0_attr);
    if (ret) {
        printf("vicap, kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    //初始化绑定信息，绑定vicap的通道0到
    vicap_mpp_chn.mod_id = K_ID_VI;
    vicap_mpp_chn.dev_id = vicap_dev;
    vicap_mpp_chn.chn_id = vicap_chn_to_vo;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = vo_dev_id;
    vo_mpp_chn.chn_id = vo_bind_chn_id;
    ret = kd_mpi_sys_bind(&vicap_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("kd_mpi_sys_bind failed:0x%x\n", ret);
    }

    //配置通道1
    k_vicap_chn_attr chn1_attr;
    memset(&chn1_attr, 0, sizeof(k_vicap_chn_attr));
    chn1_attr.out_win.h_start = 0;
    chn1_attr.out_win.v_start = 0;
    chn1_attr.out_win.width = general_config_.AI_FRAME_WIDTH;
    chn1_attr.out_win.height = general_config_.AI_FRAME_HEIGHT;
    chn1_attr.crop_win = dev_attr.acq_win;
    chn1_attr.scale_win = chn1_attr.out_win;
    chn1_attr.crop_enable = K_FALSE;
    chn1_attr.scale_enable = K_FALSE;
    chn1_attr.chn_enable = K_TRUE;
    chn1_attr.pix_format = PIXEL_FORMAT_RGB_888_PLANAR;
    chn1_attr.buffer_num = VICAP_MAX_FRAME_COUNT;
    chn1_attr.buffer_size = config.comm_pool[1].blk_size;
    printf("kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", chn1_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev, vicap_chn_to_ai, chn1_attr);
    if (ret) {
        printf("kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }

    ret = kd_mpi_vicap_set_database_parse_mode(vicap_dev, VICAP_DATABASE_PARSE_XML_JSON);
    if (ret) {
        printf("kd_mpi_vicap_set_database_parse_mode failed.\n");
        return ret;
    }
    printf("kd_mpi_vicap_init\n");
    ret = kd_mpi_vicap_init(vicap_dev);
    if (ret) {
        printf("kd_mpi_vicap_init failed.\n");
    }
    printf("kd_mpi_vicap_start_stream\n");
    ret = kd_mpi_vicap_start_stream(vicap_dev);
    if (ret) {
        printf("kd_mpi_vicap_init failed.\n");
    }
    //----------------------------------------------------------------------------------------------------------
    return ret;
}

void PipeLine::GetFrame(DumpRes &dump_res){
    ScopedTiming st("PipeLine::GetFrame", debug_mode_);
    int ret=0;
    memset(&dump_info, 0, sizeof(k_video_frame_info));
    ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
    if (ret)
    {
        printf("kd_mpi_vicap_dump_frame failed.\n");
    }
    auto vbvaddr = kd_mpi_sys_mmap(dump_info.v_frame.phys_addr[0], general_config_.AI_FRAME_CHANNEL*general_config_.AI_FRAME_HEIGHT*general_config_.AI_FRAME_WIDTH);
    dump_res.virt_addr=reinterpret_cast<uintptr_t>(vbvaddr);
    dump_res.phy_addr=reinterpret_cast<uintptr_t>(dump_info.v_frame.phys_addr[0]);
}

int PipeLine::ReleaseFrame(){
    ScopedTiming st("PipeLine::ReleaseFrame", debug_mode_);
    int ret=0;
    ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
    if (ret)
    {
        printf("kd_mpi_vicap_dump_release failed.\n");
    }
    return ret;
}

int PipeLine::InsertFrame(void* osd_data){
    ScopedTiming st("PipeLine::InsertFrame", debug_mode_);
    int ret=0;
    if(general_config_.DISPLAY_ROTATE==1){
        memcpy(insert_gdma_vaddr, osd_data, general_config_.OSD_WIDTH * general_config_.OSD_HEIGHT * general_config_.OSD_CHANNEL);
        ret = kd_mpi_dma_send_frame(0, &gdma_frame_info, 100);
        if (ret)
        {
            printf("dma send frame failed.\r\n");
            return ret;
        }
        ret = kd_mpi_dma_get_frame(0, &osd_frame_info, -1);
        if (ret)
        {
            printf("dma get frame failed.\r\n");
            return ret;
        }
        kd_mpi_dma_release_frame(0,  &osd_frame_info);
    }
    else{
        memcpy(insert_osd_vaddr, osd_data, general_config_.OSD_WIDTH * general_config_.OSD_HEIGHT * general_config_.OSD_CHANNEL);
    }
    // memcpy(insert_osd_vaddr, osd_data, general_config_.OSD_WIDTH * general_config_.OSD_HEIGHT * general_config_.OSD_CHANNEL);
    ret=kd_mpi_vo_chn_insert_frame(osd_chn_id + 3, &osd_frame_info);
    if (ret)
    {
        printf("kd_mpi_vo_chn_insert_frame failed.\n");
    }
    return ret; 
}

int PipeLine::Destroy()
{
    ScopedTiming st("PipeLine::Destroy", debug_mode_);
    int ret=0;
    if(general_config_.DISPLAY_ROTATE==1){
        ret=kd_mpi_dma_stop_chn(0);
        if (ret)
        {
            printf("stop dma chn error\r\n");
            return ret;
        }
        ret = kd_mpi_dma_stop_dev();
        if (ret)
        {
            printf("stop dma dev error\r\n");
            return ret;
        }
        ret =  kd_mpi_vb_release_block(gdma_handle);
        if (ret)
        {
            printf("release dma block error\r\n");
            return ret;
        }
    }
    //OSD release
    if(general_config_.USE_OSD == 1)
    {
        kd_mpi_vo_osd_disable(osd_chn_id);
        kd_mpi_vb_release_block(handle);
    }
    printf("kd_mpi_vb_release_block\n");

    //vicap停止
    ret = kd_mpi_vicap_stop_stream(vicap_dev);
    if (ret) {
        printf("kd_mpi_vicap_stop_stream failed.\n");
        return ret;
    }
    //vicap反初始化
    ret = kd_mpi_vicap_deinit(vicap_dev);
    if (ret) {
        printf("kd_mpi_vicap_deinit failed.\n");
        return ret;
    }

    //vicap解绑定
    kd_mpi_vo_disable_video_layer(vo_chn_id);
    vicap_mpp_chn.mod_id = K_ID_VI;
    vicap_mpp_chn.dev_id = vicap_dev;
    vicap_mpp_chn.chn_id = vicap_chn_to_vo;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = vo_dev_id;
    vo_mpp_chn.chn_id = vo_bind_chn_id;
    ret = kd_mpi_sys_unbind(&vicap_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("kd_mpi_sys_unbind failed:0x%x\n", ret);
    }

    /*Allow one frame time for the VO to release the VB block*/
    k_u32 display_ms = 1000 / 33;
    usleep(1000 * display_ms);
    //vb反初始化
    ret = kd_mpi_vb_exit();
    if (ret) {
        printf("kd_mpi_vb_exit failed.\n");
        return ret;
    }

    return 0;
}