
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/functional/ai2d/ai2d_builder.h>
#include "mpi_sys_api.h"

/* vicap */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <signal.h>
#include <atomic>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>
#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_isp_api.h"
#include "mpi_sys_api.h"
#include "k_vo_comm.h"
#include "mpi_vo_api.h"
#include "vo_test_case.h"
#include "mpi_vo_api.h"
#include "sys/ioctl.h"
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "k_autoconf_comm.h"

using std::string;
using std::vector;
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;
using namespace nncase::F::k230;

// ISP
#define ISP_INPUT_WIDTH (1920)
#define ISP_INPUT_HEIGHT (1080)
//to display
#define ISP_CHN0_WIDTH  (1920)
#define ISP_CHN0_HEIGHT (1080)
// to AI
#define CHANNEL 3
#define ISP_CHN1_WIDTH  (1280)
#define ISP_CHN1_HEIGHT (720)

// AI to Display
#define vicap_install_osd  (1)
#define osd_id          K_VO_OSD3
#define OSD_WIDTH       (1920)
#define OSD_HEIGHT      (1080)

k_vicap_dev vicap_dev;
k_u32 osd_pool_id;
k_vb_blk_handle handle;
k_video_frame_info vf_info;
void *pic_vaddr;

std::atomic<bool> isp_stop(false);

//定义检测框类型
typedef struct Bbox{
	cv::Rect box;
	float confidence;
	int index;
}Bbox;

//颜色盘，共80种颜色，类别大于80时取余
std::vector<cv::Scalar> color_four = {
       cv::Scalar(127, 220, 20, 60),
       cv::Scalar(127, 119, 11, 32),
       cv::Scalar(127, 0, 0, 142),
       cv::Scalar(127, 0, 0, 230),
       cv::Scalar(127, 106, 0, 228),
       cv::Scalar(127, 0, 60, 100),
       cv::Scalar(127, 0, 80, 100),
       cv::Scalar(127, 0, 0, 70),
       cv::Scalar(127, 0, 0, 192),
       cv::Scalar(127, 250, 170, 30),
       cv::Scalar(127, 100, 170, 30),
       cv::Scalar(127, 220, 220, 0),
       cv::Scalar(127, 175, 116, 175),
       cv::Scalar(127, 250, 0, 30),
       cv::Scalar(127, 165, 42, 42),
       cv::Scalar(127, 255, 77, 255),
       cv::Scalar(127, 0, 226, 252),
       cv::Scalar(127, 182, 182, 255),
       cv::Scalar(127, 0, 82, 0),
       cv::Scalar(127, 120, 166, 157),
       cv::Scalar(127, 110, 76, 0),
       cv::Scalar(127, 174, 57, 255),
       cv::Scalar(127, 199, 100, 0),
       cv::Scalar(127, 72, 0, 118),
       cv::Scalar(127, 255, 179, 240),
       cv::Scalar(127, 0, 125, 92),
       cv::Scalar(127, 209, 0, 151),
       cv::Scalar(127, 188, 208, 182),
       cv::Scalar(127, 0, 220, 176),
       cv::Scalar(127, 255, 99, 164),
       cv::Scalar(127, 92, 0, 73),
       cv::Scalar(127, 133, 129, 255),
       cv::Scalar(127, 78, 180, 255),
       cv::Scalar(127, 0, 228, 0),
       cv::Scalar(127, 174, 255, 243),
       cv::Scalar(127, 45, 89, 255),
       cv::Scalar(127, 134, 134, 103),
       cv::Scalar(127, 145, 148, 174),
       cv::Scalar(127, 255, 208, 186),
       cv::Scalar(127, 197, 226, 255),
       cv::Scalar(127, 171, 134, 1),
       cv::Scalar(127, 109, 63, 54),
       cv::Scalar(127, 207, 138, 255),
       cv::Scalar(127, 151, 0, 95),
       cv::Scalar(127, 9, 80, 61),
       cv::Scalar(127, 84, 105, 51),
       cv::Scalar(127, 74, 65, 105),
       cv::Scalar(127, 166, 196, 102),
       cv::Scalar(127, 208, 195, 210),
       cv::Scalar(127, 255, 109, 65),
       cv::Scalar(127, 0, 143, 149),
       cv::Scalar(127, 179, 0, 194),
       cv::Scalar(127, 209, 99, 106),
       cv::Scalar(127, 5, 121, 0),
       cv::Scalar(127, 227, 255, 205),
       cv::Scalar(127, 147, 186, 208),
       cv::Scalar(127, 153, 69, 1),
       cv::Scalar(127, 3, 95, 161),
       cv::Scalar(127, 163, 255, 0),
       cv::Scalar(127, 119, 0, 170),
       cv::Scalar(127, 0, 182, 199),
       cv::Scalar(127, 0, 165, 120),
       cv::Scalar(127, 183, 130, 88),
       cv::Scalar(127, 95, 32, 0),
       cv::Scalar(127, 130, 114, 135),
       cv::Scalar(127, 110, 129, 133),
       cv::Scalar(127, 166, 74, 118),
       cv::Scalar(127, 219, 142, 185),
       cv::Scalar(127, 79, 210, 114),
       cv::Scalar(127, 178, 90, 62),
       cv::Scalar(127, 65, 70, 15),
       cv::Scalar(127, 127, 167, 115),
       cv::Scalar(127, 59, 105, 106),
       cv::Scalar(127, 142, 108, 45),
       cv::Scalar(127, 196, 172, 0),
       cv::Scalar(127, 95, 54, 80),
       cv::Scalar(127, 128, 76, 255),
       cv::Scalar(127, 201, 57, 1),
       cv::Scalar(127, 246, 0, 122),
       cv::Scalar(127, 191, 162, 208)};

// 根据类别数使用模运算循环获取颜色
std::vector<cv::Scalar> getColorsForClasses(int num_classes) {
    std::vector<cv::Scalar> colors;
    int num_available_colors = color_four.size(); 
    for (int i = 0; i < num_classes; ++i) {
        colors.push_back(color_four[i % num_available_colors]);
    }
    return colors;
}

// 后处理IOU计算
float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
	int xx1, yy1, xx2, yy2;
 
	xx1 = std::max(rect1.x, rect2.x);
	yy1 = std::max(rect1.y, rect2.y);
	xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
	yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
 
	int insection_width, insection_height;
	insection_width = std::max(0, xx2 - xx1 + 1);
	insection_height = std::max(0, yy2 - yy1 + 1);
 
	float insection_area, union_area, iou;
	insection_area = float(insection_width) * insection_height;
	union_area = float(rect1.width*rect1.height + rect2.width*rect2.height - insection_area);
	iou = insection_area / union_area;

	return iou;
}

//NMS非极大值抑制，bboxes是待处理框Bbox实例的列表，indices是NMS后剩余的bboxes框索引
void nms(std::vector<Bbox> &bboxes,  float confThreshold, float nmsThreshold, std::vector<int> &indices)
{	
	sort(bboxes.begin(), bboxes.end(), [](Bbox a, Bbox b) { return a.confidence > b.confidence; });
	int updated_size = bboxes.size();
	for (int i = 0; i < updated_size; i++)
	{
		if (bboxes[i].confidence < confThreshold)
			continue;
		indices.push_back(bboxes[i].index);
		for (int j = i + 1; j < updated_size;)
		{
			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
			if (iou > nmsThreshold)
			{
				bboxes.erase(bboxes.begin() + j);
				updated_size = bboxes.size();
			}
            else
            {
                j++;    
            }
		}
	}
}

// 配置摄像头参数
int vicap_start()
{
    k_s32 ret = 0;
    k_u32 pool_id;
    k_vb_pool_config pool_config;
    printf("sample_vicap ...\n");

    // 配置video buffer
    k_vb_config config;
    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;
    //VB for YUV420SP output, to vo
    config.comm_pool[0].blk_cnt = 3;
    config.comm_pool[0].mode = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[0].blk_size = VICAP_ALIGN_UP((ISP_CHN0_WIDTH * ISP_CHN0_HEIGHT * 3 / 2), VICAP_ALIGN_1K);
    //VB for RGB888 output, to AI
    config.comm_pool[1].blk_cnt = 3;
    config.comm_pool[1].mode = VB_REMAP_MODE_NOCACHE;
    config.comm_pool[1].blk_size = VICAP_ALIGN_UP((ISP_CHN1_WIDTH * ISP_CHN1_HEIGHT * 3 ), VICAP_ALIGN_1K);

    if(vicap_install_osd == 1){
        config.comm_pool[2].blk_cnt = 2;
        config.comm_pool[2].mode = VB_REMAP_MODE_NOCACHE;
        config.comm_pool[2].blk_size = VICAP_ALIGN_UP((OSD_WIDTH * OSD_HEIGHT * 4), VICAP_ALIGN_1K);
        osd_pool_id=2;
    }

    ret = kd_mpi_vb_set_config(&config);
    if (ret) {
        printf("vb_set_config failed ret:%d\n", ret);
        return ret;
    }
    k_vb_supplement_config supplement_config;
    memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret) {
        printf("vb_set_supplement_config failed ret:%d\n", ret);
        return ret;
    }
    ret = kd_mpi_vb_init();
    if (ret) {
        printf("vb_init failed ret:%d\n", ret);
        return ret;
    }
    printf("sample_vicap ...kd_mpi_vicap_get_sensor_info\n");

    
    // 初始化并配置Display
    k_s32 connector_fd;
	k_connector_type connector_type = LT9611_MIPI_4LAN_1920X1080_30FPS;
    k_connector_info connector_info;
    memset(&connector_info, 0, sizeof(k_connector_info));
    ret = kd_mpi_get_connector_info(connector_type, &connector_info);
    if (ret) {
        printf("sample_vicap, the sensor type not supported!\n");
        return ret;
    }
    connector_fd = kd_mpi_connector_open(connector_info.connector_name);
    if (connector_fd < 0) {
        printf("%s, connector open failed.\n", __func__);
        return K_ERR_VO_NOTREADY;
    }
    kd_mpi_connector_power_set(connector_fd, K_TRUE);
    kd_mpi_connector_init(connector_fd, connector_info);

    //配置VI->VO绑定信息，包括分辨率、是否旋转、显示位置
    layer_info info;
    memset(&info, 0, sizeof(info));
    info.act_size.width = ISP_CHN0_WIDTH;
    info.act_size.height = ISP_CHN0_HEIGHT;
    info.format = PIXEL_FORMAT_YVU_PLANAR_420;
    info.func = 0;//K_ROTATION_180;////K_ROTATION_90;
    info.global_alptha = 0xff;
    info.offset.x = 0;//x
    info.offset.y = 0;//y;

    //VO层设置
    k_vo_layer chn_id = K_VO_LAYER1;
    k_vo_video_layer_attr attr;
    if ((chn_id >= K_MAX_VO_LAYER_NUM) || ((info.func & K_VO_SCALER_ENABLE) && (chn_id != K_VO_LAYER0))
            || ((info.func != 0) && (chn_id == K_VO_LAYER2)))
    {
        printf("input layer num failed \n");
        return -1 ;
    }
    attr.display_rect = info.offset;
    attr.img_size = info.act_size;
    info.size = info.act_size.height * info.act_size.width * 3 / 2;
    attr.pixel_format = info.format;
    if (info.format != PIXEL_FORMAT_YVU_PLANAR_420)
    {
        printf("input pix format failed \n");
        return -1;
    }
    attr.stride = (info.act_size.width / 8 - 1) + ((info.act_size.height - 1) << 16);
    attr.func = info.func;
    attr.scaler_attr = info.attr;
    kd_mpi_vo_set_video_layer_attr(chn_id, &attr);
    kd_mpi_vo_enable_video_layer(chn_id);

    //配置OSD部分
    if(vicap_install_osd == 1){
        osd_info osd;
        osd.act_size.width = OSD_WIDTH ;
        osd.act_size.height = OSD_HEIGHT;
        osd.offset.x = 0;
        osd.offset.y = 0;
        osd.global_alptha = 0xff;
        osd.format = PIXEL_FORMAT_ARGB_8888;

        k_vo_video_osd_attr attr;
        attr.global_alptha = osd.global_alptha;

        if (osd.format == PIXEL_FORMAT_ABGR_8888 || osd.format == PIXEL_FORMAT_ARGB_8888)
        {
            osd.size = osd.act_size.width  * osd.act_size.height * 4;
            osd.stride  = osd.act_size.width * 4 / 8;
        }
        else
        {
            printf("set osd pixel format failed  \n");
        }

        attr.stride = osd.stride;
        attr.pixel_format = osd.format;
        attr.display_rect = osd.offset;
        attr.img_size = osd.act_size;
        kd_mpi_vo_set_video_osd_attr(osd_id, &attr);
        kd_mpi_vo_osd_enable(osd_id);
    }


    k_u64 phys_addr = 0;
    k_u32 *virt_addr;
    memset(&vf_info, 0, sizeof(vf_info));
    vf_info.v_frame.width = OSD_WIDTH;
    vf_info.v_frame.height = OSD_HEIGHT;
    vf_info.v_frame.stride[0] = OSD_WIDTH;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    k_s32 size = VICAP_ALIGN_UP(vf_info.v_frame.height * vf_info.v_frame.width * 4, VICAP_ALIGN_1K);
    handle = kd_mpi_vb_get_block(osd_pool_id, size, NULL);
    if (handle == VB_INVALID_HANDLE)
    {
        printf("%s get vb block error\n", __func__);
        return K_FAILED;
    }
    phys_addr = kd_mpi_vb_handle_to_phyaddr(handle);
    if (phys_addr == 0)
    {
        printf("%s get phys addr error\n", __func__);
        return K_FAILED;
    }
    virt_addr = (k_u32 *)kd_mpi_sys_mmap(phys_addr, size);

    if (virt_addr == NULL)
    {
        printf("%s mmap error\n", __func__);
        return K_FAILED;
    }
    vf_info.mod_id = K_ID_VO;
    vf_info.pool_id = osd_pool_id;
    vf_info.v_frame.phys_addr[0] = phys_addr;
    if (vf_info.v_frame.pixel_format == PIXEL_FORMAT_YVU_PLANAR_420)
        vf_info.v_frame.phys_addr[1] = phys_addr + (vf_info.v_frame.height * vf_info.v_frame.stride[0]);
    pic_vaddr = virt_addr;
    printf("phys_addr is %lx g_pool_id is %d \n", phys_addr, osd_pool_id);

    // 配置Sensor部分
    k_vicap_sensor_type sensor_type = OV5647_MIPI_CSI0_1920X1080_30FPS_10BIT_LINEAR;
    //初始化dev_id
    vicap_dev = VICAP_DEV_ID_0;
    //配置sensor dev
    k_vicap_sensor_info sensor_info;
    memset(&sensor_info, 0, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
    if (ret) {
        printf("sample_vicap, the sensor type not supported!\n");
        return ret;
    }
    k_vicap_dev_attr dev_attr;
    memset(&dev_attr, 0, sizeof(k_vicap_dev_attr));
    dev_attr.acq_win.h_start = 0;
    dev_attr.acq_win.v_start = 0;
    dev_attr.acq_win.width = ISP_INPUT_WIDTH;
    dev_attr.acq_win.height = ISP_INPUT_HEIGHT;
    dev_attr.mode = VICAP_WORK_ONLINE_MODE;
    dev_attr.pipe_ctrl.data = 0xFFFFFFFF;
    dev_attr.pipe_ctrl.bits.af_enable = 0;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.pipe_ctrl.bits.dnr3_enable = 0;
    dev_attr.cpature_frame = 0;
    memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));
    ret = kd_mpi_vicap_set_dev_attr(vicap_dev, dev_attr);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_set_dev_attr failed.\n");
        return ret;
    }

    // 配置通道0
    k_vicap_chn_attr chn_attr;
    k_vicap_chn vicap_chn0 = VICAP_CHN_ID_0;
    memset(&chn_attr, 0, sizeof(k_vicap_chn_attr));
    chn_attr.out_win.h_start = 0;
    chn_attr.out_win.v_start = 0;
    chn_attr.out_win.width = ISP_CHN0_WIDTH;
    chn_attr.out_win.height = ISP_CHN0_HEIGHT;
    chn_attr.crop_win = dev_attr.acq_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_FALSE;
    chn_attr.scale_enable = K_FALSE;
    // chn_attr.dw_enable = K_FALSE;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_YVU_PLANAR_420;
    chn_attr.buffer_num = VICAP_MAX_FRAME_COUNT;//at least 3 buffers for isp
    chn_attr.buffer_size = config.comm_pool[0].blk_size;
    printf("sample_vicap ...kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", chn_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev, vicap_chn0, chn_attr);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }
    //bind vicap chn 0 to vo
    k_mpp_chn vicap_mpp_chn;
    k_mpp_chn vo_mpp_chn;
    vicap_mpp_chn.mod_id = K_ID_VI;
    vicap_mpp_chn.dev_id = vicap_dev;
    vicap_mpp_chn.chn_id = vicap_chn0;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = K_VO_DISPLAY_CHN_ID1;
    ret = kd_mpi_sys_bind(&vicap_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("kd_mpi_sys_unbind failed:0x%x\n", ret);
    }
    printf("sample_vicap ...dwc_dsi_init\n");

    //配置通道1
    k_vicap_chn vicap_chn1 = VICAP_CHN_ID_1;
    chn_attr.out_win.h_start = 0;
    chn_attr.out_win.v_start = 0;
    chn_attr.out_win.width = ISP_CHN1_WIDTH ;
    chn_attr.out_win.height = ISP_CHN1_HEIGHT;
    chn_attr.crop_win = dev_attr.acq_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_FALSE;
    chn_attr.scale_enable = K_FALSE;
    // chn_attr.dw_enable = K_FALSE;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_BGR_888_PLANAR;
    chn_attr.buffer_num = VICAP_MAX_FRAME_COUNT;//at least 3 buffers for isp
    chn_attr.buffer_size = config.comm_pool[1].blk_size;
    printf("sample_vicap ...kd_mpi_vicap_set_chn_attr, buffer_size[%d]\n", chn_attr.buffer_size);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev, vicap_chn1, chn_attr);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_set_chn_attr failed.\n");
        return ret;
    }
    ret = kd_mpi_vicap_set_database_parse_mode(vicap_dev, VICAP_DATABASE_PARSE_XML_JSON);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_set_database_parse_mode failed.\n");
        return ret;
    }
    printf("sample_vicap ...kd_mpi_vicap_init\n");
    ret = kd_mpi_vicap_init(vicap_dev);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_init failed.\n");
    }
    printf("sample_vicap ...kd_mpi_vicap_start_stream\n");
    ret = kd_mpi_vicap_start_stream(vicap_dev);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_init failed.\n");
    }

    return ret;
}

int vicap_stop(){
    if(vicap_install_osd == 1)
    {
        kd_mpi_vo_osd_disable(osd_id);
        kd_mpi_vb_release_block(handle);
    }
    printf("sample_vicap ...kd_mpi_vicap_stop_stream\n");
    int ret = kd_mpi_vicap_stop_stream(vicap_dev);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_init failed.\n");
        return ret;
    }
    ret = kd_mpi_vicap_deinit(vicap_dev);
    if (ret) {
        printf("sample_vicap, kd_mpi_vicap_deinit failed.\n");
        return ret;
    }
    kd_mpi_vo_disable_video_layer(K_VO_LAYER1);
    k_mpp_chn vicap_mpp_chn;
    k_mpp_chn vo_mpp_chn;
    k_vicap_chn vicap_chn0 = VICAP_CHN_ID_0;
    vicap_mpp_chn.mod_id = K_ID_VI;
    vicap_mpp_chn.dev_id = vicap_dev;
    vicap_mpp_chn.chn_id = vicap_chn0;
    vo_mpp_chn.mod_id = K_ID_VO;
    vo_mpp_chn.dev_id = K_VO_DISPLAY_DEV_ID;
    vo_mpp_chn.chn_id = K_VO_DISPLAY_CHN_ID1;

    ret = kd_mpi_sys_unbind(&vicap_mpp_chn, &vo_mpp_chn);
    if (ret) {
        printf("kd_mpi_sys_unbind failed:0x%x\n", ret);
    }

    /*Allow one frame time for the VO to release the VB block*/
    k_u32 display_ms = 1000 / 33;
    usleep(1000 * display_ms);
    ret = kd_mpi_vb_exit();
    if (ret) {
        printf("sample_vicap, kd_mpi_vb_exit failed.\n");
        return ret;
    }

    return 0;
}

int camera_inference(char *argv[]){
    int debug_mode=atoi(argv[4]);
    // 加载模型
    interpreter interp;     
    std::ifstream ifs(argv[1], std::ios::binary);
    interp.load_model(ifs).expect("Invalid kmodel");

    //初始化shape容器和输出数据指针容器，用于存储多个输入和多个输出的shape信息以及推理输出数据的指针
    vector<vector<int>> input_shapes;   
    vector<vector<int>> output_shapes;
    vector<float *> p_outputs;

    // 获取模型的输入信息，并初始化输入tensor
    for (int i = 0; i < interp.inputs_size(); i++)
    {
        auto desc = interp.input_desc(i);
        auto shape = interp.input_shape(i);
        auto tensor = host_runtime_tensor::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create input tensor");
        interp.input_tensor(i, tensor).expect("cannot set input tensor");
        vector<int> in_shape;
        if (debug_mode> 1)
            std::cout<<"input "<< std::to_string(i) <<" datatype: "<<std::to_string(desc.datatype)<<" , shape: ";
        for (int j = 0; j < shape.size(); ++j)
        {
            in_shape.push_back(shape[j]);
            if (debug_mode> 1)
                std::cout<<shape[j]<<" ";
        }
        if (debug_mode> 1)
            std::cout<<std::endl;
        input_shapes.push_back(in_shape);
    }

    // 获取模型输出的shape信息，并初始化输出的tensor
    for (size_t i = 0; i < interp.outputs_size(); i++)
    {
        auto desc = interp.output_desc(i);
        auto shape = interp.output_shape(i);
        auto tensor = host_runtime_tensor::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create output tensor");
        interp.output_tensor(i, tensor).expect("cannot set output tensor");
        vector<int> out_shape;
        if (debug_mode> 1)
            std::cout<<"output "<< std::to_string(i) <<" datatype: "<<std::to_string(desc.datatype)<<" , shape: ";
        for (int j = 0; j < shape.size(); ++j)
        {
            out_shape.push_back(shape[j]);
            if (debug_mode> 1)
                std::cout<<shape[j]<<" ";
        }
        if (debug_mode> 1)
            std::cout<<std::endl;
        output_shapes.push_back(out_shape);
    }

    // 计算预处理参数，这里计算的是短边padding的参数值
    int width = input_shapes[0][3];
    int height = input_shapes[0][2];
    float ratiow = (float)width / ISP_CHN1_WIDTH;
    float ratioh = (float)height / ISP_CHN1_HEIGHT;
    float ratio = ratiow < ratioh ? ratiow : ratioh;
    int new_w = (int)(ratio * ISP_CHN1_WIDTH);
    int new_h = (int)(ratio * ISP_CHN1_HEIGHT);
    float dw = (float)(width - new_w) / 2;
    float dh = (float)(height - new_h) / 2;
    int top = (int)(roundf(0));
    int bottom = (int)(roundf(dh * 2 + 0.1));
    int left = (int)(roundf(0));
    int right = (int)(roundf(dw * 2 - 0.1));

    // 创建AI2D输入tensor，并将CHW_RGB数据拷贝到tensor中，并回写到DDR
    dims_t ai2d_in_shape{1, 3, ISP_CHN1_HEIGHT, ISP_CHN1_WIDTH};
    runtime_tensor ai2d_in_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, ai2d_in_shape, hrt::pool_shared).expect("cannot create input tensor");
    // auto input_buf = ai2d_in_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    // memcpy(reinterpret_cast<char *>(input_buf.data()), chw_vec.data(), chw_vec.size());
    hrt::sync(ai2d_in_tensor, sync_op_t::sync_write_back, true).expect("write back input failed");

    // 创建AI2D输出tensor,因为AI2D的输出tensor给到模型的输入tensor去推理，这里为了节省内存，直接获取model的输入tensor，使得AI2D处理后的输出直接给到model输入
    runtime_tensor ai2d_out_tensor = interp.input_tensor(0).expect("cannot get input tensor");
    dims_t out_shape = ai2d_out_tensor.shape();

    // 设置AI2D参数，AI2D支持5种预处理方法，crop/shift/pad/resize/affine。这里开启pad和resize，并配置padding的大小和数值，设置resize的插值方法，如果要配置其他的预处理方法也是类似
    ai2d_datatype_t ai2d_dtype{ai2d_format::NCHW_FMT, ai2d_format::NCHW_FMT, ai2d_in_tensor.datatype(), ai2d_out_tensor.datatype()};
    ai2d_crop_param_t crop_param{false, 0, 0, 0, 0};
    ai2d_shift_param_t shift_param{false, 0};
    ai2d_pad_param_t pad_param{true, {{0, 0}, {0, 0}, {top, bottom}, {left, right}}, ai2d_pad_mode::constant, {114, 114, 114}};
    ai2d_resize_param_t resize_param{true, ai2d_interp_method::tf_bilinear, ai2d_interp_mode::half_pixel};
    ai2d_affine_param_t affine_param{false, ai2d_interp_method::cv2_bilinear, 0, 0, 127, 1, {0.5, 0.1, 0.0, 0.1, 0.5, 0.0}};

    // 构造ai2d_builder
    ai2d_builder builder(ai2d_in_shape, out_shape, ai2d_dtype, crop_param, shift_param, pad_param, resize_param, affine_param);
    builder.build_schedule();

    // 标签名称
    std::vector<std::string> classes{"apple","banana","orange"};
    // 置信度阈值
    float conf_thresh=atof(argv[2]);
    // nms阈值
    float nms_thresh=atof(argv[3]);
    //类别数
    int class_num=classes.size();
    // 根据类别数获取颜色，用于后续画图
    std::vector<cv::Scalar> class_colors = getColorsForClasses(class_num);

    // output0 [num_class+4,(w/8)*(h/8)+(w/16)*(h/16)+(w/32)*(h/32)]
    float *output0;
    // 每个框的特征长度，ckass_num个分数+4个坐标
    int f_len=class_num+4;
    // 根据模型的输入分辨率计算总输出框数
    int num_box=((input_shapes[0][2]/8)*(input_shapes[0][3]/8)+(input_shapes[0][2]/16)*(input_shapes[0][3]/16)+(input_shapes[0][2]/32)*(input_shapes[0][3]/32));
    // 申请框数据内存
    float *output_det = new float[num_box * f_len];

    // 解析每个框的信息，class_num+4为一个框，前四个数据为坐标值，后面的class_num个分数，选择分数最大的作为识别的类别，因为开始的时候做了padding+resize，所以模型推理的坐标是基于与处理后的图像的结果，要先把框的坐标使用ratio映射回原图
    std::vector<Bbox> bboxes;

    vicap_start();
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = ISP_CHN1_WIDTH * ISP_CHN1_HEIGHT * CHANNEL;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    k_video_frame_info dump_info;

    while(!isp_stop){
        // 从vivcap中读取一帧图像到dump_info
        memset(&dump_info, 0, sizeof(k_video_frame_info));
        ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
        if (ret)
        {
            printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
            continue;
        }
        auto vbvaddr = kd_mpi_sys_mmap(dump_info.v_frame.phys_addr[0], size);

        uintptr_t v_addr=reinterpret_cast<uintptr_t>(vbvaddr);
        uintptr_t p_addr=reinterpret_cast<uintptr_t>(dump_info.v_frame.phys_addr[0]);
        dims_t in_shape { 1, CHANNEL, ISP_CHN1_HEIGHT, ISP_CHN1_WIDTH };
        ai2d_in_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)v_addr, compute_size(in_shape) },false, hrt::pool_shared, p_addr).expect("cannot create input tensor");
        hrt::sync(ai2d_in_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
        // 执行ai2d，实现从ai2d_in_tensor->ai2d_out_tensor的预处理过程
        builder.invoke(ai2d_in_tensor,ai2d_out_tensor).expect("error occurred in ai2d running");

        // 执行模型推理的过程
        interp.run().expect("error occurred in running model");

        // 获取模型输出数据的指针
        p_outputs.clear();
        for (int i = 0; i < interp.outputs_size(); i++)
        {
            auto out = interp.output_tensor(i).expect("cannot get output tensor");
            auto buf = out.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();
            float *p_out = reinterpret_cast<float *>(buf.data());
            p_outputs.push_back(p_out);
        }

        // 模型推理结束后，进行后处理
        output0= p_outputs[0];
        // 将输出数据排布从[num_class+4,(w/8)*(h/8)+(w/16)*(h/16)+(w/32)*(h/32)]调整为[(w/8)*(h/8)+(w/16)*(h/16)+(w/32)*(h/32),num_class+4],方便后续处理
        for(int r = 0; r < num_box; r++)
        {
            for(int c = 0; c < f_len; c++)
            {
                output_det[r*f_len + c] = output0[c*num_box + r];
            }
        }

        bboxes.clear();
        for(int i=0;i<num_box;i++){
            float* vec=output_det+i*f_len;
            float box[4]={vec[0],vec[1],vec[2],vec[3]};
            float* class_scores=vec+4;
            float* max_class_score_ptr=std::max_element(class_scores,class_scores+class_num);
            float score=*max_class_score_ptr;
            int max_class_index = max_class_score_ptr - class_scores; // 计算索引
            if(score>conf_thresh){
                Bbox bbox;
                float x_=box[0]/ratio*1.0;
                float y_=box[1]/ratio*1.0;
                float w_=box[2]/ratio*1.0;
                float h_=box[3]/ratio*1.0;
                int x=int(MAX(x_-0.5*w_,0));
                int y=int(MAX(y_-0.5*h_,0));
                int w=int(w_);
                int h=int(h_);
                if (w <= 0 || h <= 0) { continue; }
                bbox.box=cv::Rect(x,y,w,h);
                bbox.confidence=score;
                bbox.index=max_class_index;
                bboxes.push_back(bbox);
            }

        }
        //执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
        std::vector<int> nms_result;
        nms(bboxes, conf_thresh, nms_thresh, nms_result);

        cv::Mat osd_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        // 将识别的框绘制到原图片上并保存为结果图片result,jpg
        for (int i = 0; i < nms_result.size(); i++) {
            int res=nms_result[i];
            cv::Rect box=bboxes[res].box;
            int idx=bboxes[res].index;
            float score=bboxes[res].confidence;
            int x=int(box.x*float(OSD_WIDTH)/ISP_CHN1_WIDTH);
            int y=int(box.y*float(OSD_HEIGHT)/ISP_CHN1_HEIGHT);
            int w=int(box.width*float(OSD_WIDTH)/ISP_CHN1_WIDTH);
            int h=int(box.height*float(OSD_HEIGHT)/ISP_CHN1_HEIGHT);
            cv::Rect new_box(x,y,w,h);
            cv::rectangle(osd_frame, new_box, class_colors[idx], 2, 8);
            cv::putText(osd_frame, classes[idx]+" "+std::to_string(score), cv::Point(MIN(new_box.x + 5,OSD_WIDTH), MAX(new_box.y - 10,0)), cv::FONT_HERSHEY_DUPLEX, 1, class_colors[idx], 2, 0);
        }

        memcpy(pic_vaddr, osd_frame.data, OSD_WIDTH * OSD_HEIGHT * 4);
        kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info); 

        ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
        if (ret)
        {
            printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
        }

    }
    delete[] output_det;
    vicap_stop();
    
    

    return 0;
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <kmodel> <conf_thresh> <nms_thresh> <debug_mode>" << std::endl;
        return -1;
    }
    std::thread thread_isp(camera_inference, argv);
    while (getchar() != 'q')
    {
        usleep(10000);
    }
    isp_stop = true;
    thread_isp.join();
    
    return 0;
}