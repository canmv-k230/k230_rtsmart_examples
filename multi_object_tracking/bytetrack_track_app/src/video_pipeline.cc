#include "video_pipeline.h"

#define ALIGN_UP_16(x)  (((x) + 15) & ~15)

PipeLine::PipeLine(int debug_mode)
{
    //set vo connector type
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
        connector_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
    }
    

    vo_chn_id = K_VO_LAYER1;

    osd_chn_id = K_VO_OSD3;
    // sensor type
    sensor_type = GC2093_MIPI_CSI2_1920X1080_30FPS_10BIT_LINEAR;
    // vicap dev id
    vicap_dev=VICAP_DEV_ID_0;
    // vicap chn id to vo chn id
    vicap_chn_to_vo=VICAP_CHN_ID_0;
    // vicap chn id to kmodel chn id
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

    //---------------------------- set video buffer------------------------------------------------
    memset(&config, 0, sizeof(k_vb_config));
    config.max_pool_cnt = 64;
    //VB for YUV420SP format, to Display；
    config.comm_pool[0].blk_cnt = 4;
    config.comm_pool[0].mode = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[0].blk_size = VICAP_ALIGN_UP((DISPLAY_WIDTH * DISPLAY_HEIGHT * 3 / 2), VICAP_ALIGN_1K);
    //VB for RGBP888 format, to kmodel；
    config.comm_pool[1].blk_cnt = 4;
    config.comm_pool[1].mode = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[1].blk_size = VICAP_ALIGN_UP((AI_FRAME_WIDTH * AI_FRAME_HEIGHT * 3 ), VICAP_ALIGN_1K);
    //VB for ARGB8888 format, to OSD；
    if(USE_OSD == 1){
        config.comm_pool[2].blk_cnt = 3;
        config.comm_pool[2].mode = VB_REMAP_MODE_NOCACHE;
        config.comm_pool[2].blk_size = VICAP_ALIGN_UP((OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL), VICAP_ALIGN_1K);
        osd_pool_id=2;
        // VB for BGRA8888 format, to GDMA；
        if(DISPLAY_MODE==1){
            config.comm_pool[3].blk_cnt = 4;
            config.comm_pool[3].mode = VB_REMAP_MODE_NOCACHE;
            config.comm_pool[3].blk_size = VICAP_ALIGN_UP((OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL), VICAP_ALIGN_1K);
            gdma_pool_id=3;
        }

    }
    // set vb config
    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        return ret;
    }
    //  set vb supplement config, such as DCF info/ISP stats info/ISP real-time params etc.
    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("vb_set_supplement_config failed ret:%d\n", ret);
        return ret;
    }
    //  vb init
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        return ret;
    }
    //------------------------------------------------------------------------------------------------
    
    // ---------------------------------set display-------------------------------------------------------
    //  set connector info
    k_connector_info connector_info;
    memset(&connector_info, 0, sizeof(k_connector_info));
    //  get connector info
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("the connector type not supported!\n");
        return ret;
    }
    //  open connector device
    k_s32 connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return K_ERR_VO_NOTREADY;
    }
    //  set connector power on
    kd_mpi_connector_power_set(connector_fd, K_TRUE);
    //  connector init
    kd_mpi_connector_init(connector_fd, connector_info);
    //--------------------------------------------------------------------------------------------------

    //-----------------------------------set vo-----------------------------------------------------------
    //  set vo config
    k_vo_video_layer_attr vo_attr;
    memset(&vo_attr, 0, sizeof(k_vo_video_layer_attr));
    vo_attr.display_rect = {0,0};
    vo_attr.img_size = {(unsigned int)DISPLAY_HEIGHT,(unsigned int)DISPLAY_WIDTH};
    vo_attr.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    if (vo_attr.pixel_format != PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    {
        printf("input pix format failed \n");
        return -1;
    }
    vo_attr.stride = (DISPLAY_WIDTH / 8 - 1) + ((DISPLAY_HEIGHT - 1) << 16);
    if(DISPLAY_ROTATE==0){
        vo_attr.func = K_ROTATION_0;
    }
    else if(DISPLAY_ROTATE==1){
        vo_attr.func = K_ROTATION_90;
    }
    else if(DISPLAY_ROTATE==2){
        vo_attr.func = K_ROTATION_180;
    }
    else if(DISPLAY_ROTATE==3){
        vo_attr.func = K_ROTATION_270;
    }
    else{
        vo_attr.func = K_ROTATION_0;
    }
    //  check vo config, rotate only support layer0 and layer1
    if ((vo_chn_id >= K_MAX_VO_LAYER_NUM) || ((vo_attr.func & K_VO_SCALER_ENABLE) && (vo_chn_id != K_VO_LAYER0)) || ((vo_attr.func != K_ROTATION_0) && (vo_chn_id == K_VO_LAYER2)))
    {
        printf("input layer num failed \n");
        return -1 ;
    }
    //  set vo layer attr
    kd_mpi_vo_set_video_layer_attr(vo_chn_id, &vo_attr);
    //  enable vo layer
    kd_mpi_vo_enable_video_layer(vo_chn_id);
    //---------------------------------------------------------------------------------------------------

    //-----------------------------------set osd-----------------------------------------------------------
    if(USE_OSD == 1){
        //  set osd config
        if(DISPLAY_MODE==1){
            osd_info osd;
            osd.act_size.width = OSD_HEIGHT;
            osd.act_size.height = OSD_WIDTH;
            osd.offset.x = 0;
            osd.offset.y = 0;
            osd.global_alptha = 0xff;
            osd.format = PIXEL_FORMAT_BGRA_8888;
            //  set osd channel attr
            k_vo_video_osd_attr osd_attr;
            memset(&osd_attr, 0, sizeof(k_vo_video_osd_attr));
            osd_attr.global_alptha = 0xff;
            osd_attr.pixel_format = PIXEL_FORMAT_BGRA_8888;
            osd_attr.display_rect = {0,0};
            osd_attr.img_size = {(unsigned int)OSD_HEIGHT,(unsigned int)OSD_WIDTH};
            if (osd_attr.pixel_format == PIXEL_FORMAT_ABGR_8888 || osd_attr.pixel_format == PIXEL_FORMAT_ARGB_8888 || osd_attr.pixel_format == PIXEL_FORMAT_BGRA_8888)
            {
                osd_attr.stride  = OSD_HEIGHT * 4 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_565 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_565)
            {
                osd_attr.stride  = OSD_HEIGHT * 2 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_888 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_888)
            {
                osd_attr.stride  = OSD_HEIGHT * 3 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_4444 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_4444)
            {
                osd_attr.stride  = OSD_HEIGHT * 2 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_1555 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_1555)
            {
                osd_attr.stride  = OSD_HEIGHT * 2 / 8;
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
            osd_info osd;
            osd.act_size.width = OSD_WIDTH;
            osd.act_size.height = OSD_HEIGHT;
            osd.offset.x = 0;
            osd.offset.y = 0;
            osd.global_alptha = 0xff;
            osd.format = PIXEL_FORMAT_BGRA_8888;
            //  set osd channel attr
            k_vo_video_osd_attr osd_attr;
            memset(&osd_attr, 0, sizeof(k_vo_video_osd_attr));
            osd_attr.global_alptha = 0xff;
            osd_attr.pixel_format = PIXEL_FORMAT_BGRA_8888;
            osd_attr.display_rect = {0,0};
            osd_attr.img_size = {(unsigned int)OSD_WIDTH,(unsigned int)OSD_HEIGHT};
            if (osd_attr.pixel_format == PIXEL_FORMAT_ABGR_8888 || osd_attr.pixel_format == PIXEL_FORMAT_ARGB_8888 || osd_attr.pixel_format == PIXEL_FORMAT_BGRA_8888)
            {
                osd_attr.stride  = OSD_WIDTH * 4 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_565 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_565)
            {
                osd_attr.stride  = OSD_WIDTH * 2 / 8;
            }
            else if (osd_attr.pixel_format == PIXEL_FORMAT_RGB_888 || osd_attr.pixel_format == PIXEL_FORMAT_BGR_888)
            {
                osd_attr.stride  = OSD_WIDTH * 3 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_4444 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_4444)
            {
                osd_attr.stride  = OSD_WIDTH * 2 / 8;
            }
            else if(osd_attr.pixel_format == PIXEL_FORMAT_ARGB_1555 || osd_attr.pixel_format == PIXEL_FORMAT_ABGR_1555)
            {
                osd_attr.stride  = OSD_WIDTH * 2 / 8;
            }
            else
            {
                printf("set osd pixel format failed  \n");
                return -1;
            }
            kd_mpi_vo_set_video_osd_attr(osd_chn_id, &osd_attr);
            kd_mpi_vo_osd_enable(osd_chn_id);
        }
        //get osd block from osd pool,and init osd frame info
        k_s32 size = VICAP_ALIGN_UP(OSD_HEIGHT * OSD_WIDTH * OSD_CHANNEL, VICAP_ALIGN_1K);
        handle = kd_mpi_vb_get_block(osd_pool_id, size, NULL);
        if (handle == VB_INVALID_HANDLE)
        {
            printf("%s get vb block error\n", __func__);
            return -1;
        }
        //get phys addr
        k_u64 phys_addr = kd_mpi_vb_handle_to_phyaddr(handle);
        if (phys_addr == 0)
        {
            printf("%s get phys addr error\n", __func__);
            return -1;
        }
        //map phys addr to virt addr
        k_u32* virt_addr = (k_u32 *)kd_mpi_sys_mmap(phys_addr, size);
        if (virt_addr == NULL)
        {
            printf("%s mmap error\n", __func__);
            return -1;
        }
        //create osd data frame,and init osd frame info
        memset(&osd_frame_info, 0, sizeof(osd_frame_info));
        osd_frame_info.v_frame.width = OSD_HEIGHT;
        osd_frame_info.v_frame.height = OSD_WIDTH;
        osd_frame_info.v_frame.stride[0] = OSD_HEIGHT;
        osd_frame_info.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
        osd_frame_info.mod_id = K_ID_VO;
        osd_frame_info.pool_id = osd_pool_id;
        osd_frame_info.v_frame.phys_addr[0] = phys_addr;
        insert_osd_vaddr = virt_addr;
        printf("phys_addr is %lx g_pool_id is %d \n", phys_addr, osd_pool_id);   
    }
    //---------------------------------------------------------------------------------------------

    //-------------------------------set GDMA for rotate--------------------------------------------------
    if (DISPLAY_MODE == 1) {
        gdma_dev_attr.burst_len = 0;
        gdma_dev_attr.ckg_bypass = (k_bool)0xff;
        gdma_dev_attr.outstanding = 7;

        memset(&dma_attr, 0, sizeof(k_dma_chn_attr_u));
        dma_attr.gdma_attr.buffer_num = 3;
        dma_attr.gdma_attr.rotation = DEGREE_90;
        dma_attr.gdma_attr.x_mirror = K_FALSE;
        dma_attr.gdma_attr.y_mirror = K_FALSE;
        dma_attr.gdma_attr.width = OSD_WIDTH;
        dma_attr.gdma_attr.height = OSD_HEIGHT;
        dma_attr.gdma_attr.src_stride[0] = OSD_WIDTH * 4;
        dma_attr.gdma_attr.dst_stride[0] = OSD_HEIGHT * 4;
        dma_attr.gdma_attr.work_mode = DMA_UNBIND;
        dma_attr.gdma_attr.pixel_format = DMA_PIXEL_FORMAT_ABGR_8888;

        // Obtain the buffer block for the current frame from the DMA buffer pool,
        // initialize one frame of data, and bind the pointer pic_vaddr
        // for copying the result data
        k_s32 size = VICAP_ALIGN_UP(OSD_HEIGHT * OSD_WIDTH * OSD_CHANNEL, VICAP_ALIGN_1K);

        // Get a buffer block in user space by passing the buffer pool ID and buffer size;
        // osd_pool_id is determined during memory allocation
        gdma_handle = kd_mpi_vb_get_block(gdma_pool_id, size, NULL);
        if (gdma_handle == VB_INVALID_HANDLE)
        {
            printf("%s get vb block error\n", __func__);
            return -1;
        }

        // Get the physical address of the buffer block in user space
        k_u64 phys_addr = kd_mpi_vb_handle_to_phyaddr(gdma_handle);
        if (phys_addr == 0)
        {
            printf("%s get phys addr error\n", __func__);
            return -1;
        }

        // Map the physical address to a virtual address
        k_u8* virt_addr = (k_u8 *)kd_mpi_sys_mmap(phys_addr, size);
        if (virt_addr == NULL)
        {
            printf("%s mmap error\n", __func__);
            return -1;
        }

        // Create a GDMA frame and initialize FrameINFO,
        // bind virt_addr to insert_gdma_vaddr
        memset(&gdma_frame_info, 0, sizeof(gdma_frame_info));
        gdma_frame_info.v_frame.width = OSD_WIDTH;
        gdma_frame_info.v_frame.height = OSD_HEIGHT;
        gdma_frame_info.v_frame.stride[0] = OSD_WIDTH;
        gdma_frame_info.v_frame.pixel_format = PIXEL_FORMAT_BGRA_8888;
        gdma_frame_info.mod_id = K_ID_DMA;
        gdma_frame_info.pool_id = gdma_pool_id;
        gdma_frame_info.v_frame.phys_addr[0] = phys_addr;
        gdma_frame_info.v_frame.virt_addr[0] = (k_u64)(intptr_t)virt_addr;
        insert_gdma_vaddr = (void*)virt_addr;
        printf("dma phys_addr is %lx  dma g_pool_id is %d \n", phys_addr, gdma_pool_id);

        ret = kd_mpi_dma_set_dev_attr(&gdma_dev_attr);
        if (ret) {
            printf("gdma dma dev set failed!\n");
            return ret;
        }

        ret = kd_mpi_dma_request_chn(GDMA_TYPE);
        if (ret) {
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

    //------------------------------- set Sensor & vicap-----------------------------------------------------
    //sensor probe config
    k_vicap_probe_config probe_cfg;
    k_vicap_sensor_info sensor_info;
    probe_cfg.csi_num = CONFIG_MPP_SENSOR_DEFAULT_CSI;
    probe_cfg.width = ISP_WIDTH;
    probe_cfg.height = ISP_HEIGHT;
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

    //init vicap dev attr
    k_vicap_dev_attr dev_attr;
    memset(&dev_attr, 0, sizeof(k_vicap_dev_attr));
    dev_attr.acq_win.h_start = 0;
    dev_attr.acq_win.v_start = 0;
    dev_attr.acq_win.width = ISP_WIDTH;
    dev_attr.acq_win.height = ISP_HEIGHT;
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

    // set vicap chn 0 attr
    k_vicap_chn_attr chn0_attr;
    memset(&chn0_attr, 0, sizeof(k_vicap_chn_attr));
    chn0_attr.out_win.h_start = 0;
    chn0_attr.out_win.v_start = 0;
    chn0_attr.out_win.width = DISPLAY_WIDTH;
    chn0_attr.out_win.height = DISPLAY_HEIGHT;
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

    //init vicap chn 0 bind info
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

    //set vicap chn 1 attr
    k_vicap_chn_attr chn1_attr;
    memset(&chn1_attr, 0, sizeof(k_vicap_chn_attr));
    chn1_attr.out_win.h_start = 0;
    chn1_attr.out_win.v_start = 0;
    chn1_attr.out_win.width = AI_FRAME_WIDTH;
    chn1_attr.out_win.height = AI_FRAME_HEIGHT;
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
    dump_res.virt_addr=reinterpret_cast<uintptr_t>(kd_mpi_sys_mmap(dump_info.v_frame.phys_addr[0], AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH));
    dump_res.phy_addr=reinterpret_cast<uintptr_t>(dump_info.v_frame.phys_addr[0]);
}

int PipeLine::ReleaseFrame(DumpRes &dump_res){
    ScopedTiming st("PipeLine::ReleaseFrame", debug_mode_);
    int ret=0;
    kd_mpi_sys_munmap(reinterpret_cast<void*>(dump_res.virt_addr), AI_FRAME_CHANNEL*AI_FRAME_HEIGHT*AI_FRAME_WIDTH);
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
    if(DISPLAY_MODE==1){
        memcpy(insert_gdma_vaddr, osd_data, OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);
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
        memcpy(insert_osd_vaddr, osd_data, OSD_WIDTH * OSD_HEIGHT * OSD_CHANNEL);
    }
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
    if(DISPLAY_MODE==1){
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
    if(USE_OSD == 1)
    {
        kd_mpi_vo_osd_disable(osd_chn_id);
        kd_mpi_vb_release_block(handle);
    }
    printf("kd_mpi_vb_release_block\n");

    //vicap stop stream
    ret = kd_mpi_vicap_stop_stream(vicap_dev);
    if (ret) {
        printf("kd_mpi_vicap_stop_stream failed.\n");
        return ret;
    }
    //vicap deinit
    ret = kd_mpi_vicap_deinit(vicap_dev);
    if (ret) {
        printf("kd_mpi_vicap_deinit failed.\n");
        return ret;
    }

    //vicap unbind
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
    //vb deinit
    ret = kd_mpi_vb_exit();
    if (ret) {
        printf("kd_mpi_vb_exit failed.\n");
        return ret;
    }

    return 0;
}