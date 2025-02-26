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
#include <nncase/runtime/runtime_op_utility.h>
#include "pipeline.h"
#include "yolov5.h"
#include "yolov8.h"
#include "yolo11.h"

using std::string;
using std::vector;
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;
using namespace nncase::F::k230;

std::atomic<bool> isp_stop(false);

// 函数定义：yolo_video_inference，用于执行视频推理
int yolo_video_inference(GeneralConfig &general_config,YoloConfig &yolo_config){
    // 创建一个FrameSize对象，用于存储图像的宽度和高度
    FrameSize image_wh={general_config.AI_FRAME_WIDTH,general_config.AI_FRAME_HEIGHT};
    // 从标签文件中读取标签
    std::vector<std::string> labels=readLabelsFromTxt(yolo_config.labels_txt_filepath);
    // 创建一个DumpRes对象，用于存储帧数据
    DumpRes dump_res;
    // 创建一个空的向量，用于存储YOLO检测结果
    std::vector<YOLOBbox> yolo_results;
    // 创建一个空的Mat对象，用于存储绘制的帧
    cv::Mat draw_frame(general_config.OSD_HEIGHT, general_config.OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    // 创建一个维度向量，用于定义输入张量的形状
    dims_t in_shape { 1, general_config.AI_FRAME_CHANNEL, general_config.AI_FRAME_HEIGHT, general_config.AI_FRAME_WIDTH };
    // 创建一个运行时张量，用于存储输入数据
    runtime_tensor input_tensor;
    // 创建一个PipeLine对象，用于处理视频流
    PipeLine pl(general_config,yolo_config.debug_mode);
    // 初始化PipeLine对象
    pl.Create();
    // 如果模型类型为yolo11，则执行yolo11的推理流程
    if(strcmp(yolo_config.model_type, "yolo11") == 0){
        // 创建一个Yolo11对象，用于执行yolo11的推理流程
        Yolo11 yolo11(yolo_config.task_type,yolo_config.task_mode,yolo_config.kmodel_path,yolo_config.conf_thres,yolo_config.nms_thres,yolo_config.mask_thres,labels,image_wh,yolo_config.debug_mode);
        // 循环执行推理流程，直到isp_stop为true
        while(!isp_stop){
            // 创建一个ScopedTiming对象，用于计算总时间
            ScopedTiming st("total time", 1);
            // 从PipeLine中获取一帧数据
            pl.GetFrame(dump_res);
            // 创建一个运行时张量，用于存储输入数据
            input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
            // 将输入张量的数据同步到设备上
            hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
            // 执行预处理
            yolo11.pre_process(input_tensor);
            // 执行推理
            yolo11.inference();
            // 执行后处理
            yolo11.post_process(yolo_results);
            // 将绘制的帧设置为黑色
            draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
            if(general_config.DISPLAY_MODE==1){
                cv::rotate(draw_frame, draw_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                // 在绘制的帧上绘制检测结果
                yolo11.draw_results(draw_frame,yolo_results);
                cv::rotate(draw_frame, draw_frame, cv::ROTATE_90_CLOCKWISE);
            }else{
                // 在绘制的帧上绘制检测结果
                yolo11.draw_results(draw_frame,yolo_results);
            }
            
            // 将绘制的帧插入到PipeLine中
            pl.InsertFrame(draw_frame.data);
            // 释放帧数据
            pl.ReleaseFrame();
        }
    }
    // 如果模型类型为yolov8，则执行yolov8的推理流程
    else if(strcmp(yolo_config.model_type, "yolov8") == 0){
        // 创建一个Yolov8对象，用于执行yolov8的推理流程
        Yolov8 yolov8(yolo_config.task_type,yolo_config.task_mode,yolo_config.kmodel_path,yolo_config.conf_thres,yolo_config.nms_thres,yolo_config.mask_thres,labels,image_wh,yolo_config.debug_mode);
        // 循环执行推理流程，直到isp_stop为true
        while(!isp_stop){
            // 创建一个ScopedTiming对象，用于计算总时间
            ScopedTiming st("total time", 1);
            // 从PipeLine中获取一帧数据
            pl.GetFrame(dump_res);
            // 创建一个运行时张量，用于存储输入数据
            input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
            // 将输入张量的数据同步到设备上
            hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
            // 执行预处理
            yolov8.pre_process(input_tensor);
            // 执行推理
            yolov8.inference();
            // 执行后处理
            yolov8.post_process(yolo_results);
            // 将绘制的帧设置为黑色
            draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
            if(general_config.DISPLAY_MODE==1){
                cv::rotate(draw_frame, draw_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                // 在绘制的帧上绘制检测结果
                yolov8.draw_results(draw_frame,yolo_results);
                cv::rotate(draw_frame, draw_frame, cv::ROTATE_90_CLOCKWISE);
            }else{
                // 在绘制的帧上绘制检测结果
                yolov8.draw_results(draw_frame,yolo_results);
            }
            // 将绘制的帧插入到PipeLine中
            pl.InsertFrame(draw_frame.data);
            // 释放帧数据
            pl.ReleaseFrame();
        }
    }
    // 如果模型类型为yolov5，则执行yolov5的推理流程
    else if(strcmp(yolo_config.model_type, "yolov5") == 0){
        // 创建一个Yolov5对象，用于执行yolov5的推理流程
        Yolov5 yolov5(yolo_config.task_type,yolo_config.task_mode,yolo_config.kmodel_path,yolo_config.conf_thres,yolo_config.nms_thres,yolo_config.mask_thres,labels,image_wh,yolo_config.debug_mode);
        // 循环执行推理流程，直到isp_stop为true
        while(!isp_stop){
            // 创建一个ScopedTiming对象，用于计算总时间
            ScopedTiming st("total time", 1);
            // 从PipeLine中获取一帧数据
            pl.GetFrame(dump_res);
            // 创建一个运行时张量，用于存储输入数据
            input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)dump_res.virt_addr, compute_size(in_shape) },false, hrt::pool_shared, dump_res.phy_addr).expect("cannot create input tensor");
            // 将输入张量的数据同步到设备上
            hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
            // 执行预处理
            yolov5.pre_process(input_tensor);
            // 执行推理
            yolov5.inference();
            // 执行后处理
            yolov5.post_process(yolo_results);
            // 将绘制的帧设置为黑色
            draw_frame.setTo(cv::Scalar(0, 0, 0, 0));
            if(general_config.DISPLAY_MODE==1){
                cv::rotate(draw_frame, draw_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                // 在绘制的帧上绘制检测结果
                yolov5.draw_results(draw_frame,yolo_results);
                cv::rotate(draw_frame, draw_frame, cv::ROTATE_90_CLOCKWISE);
            }else{
                // 在绘制的帧上绘制检测结果
                yolov5.draw_results(draw_frame,yolo_results);
            }
            // 将绘制的帧插入到PipeLine中
            pl.InsertFrame(draw_frame.data);
            // 释放帧数据
            pl.ReleaseFrame();
        }
    }
    // 如果模型类型不是yolov5、yolov8或yolo11，则打印错误信息并返回-1
    else{
        std::cout << "仅支持模型: yolov5/yolov8/yolo11 " << std::endl;
        // 销毁PipeLine对象
        pl.Destroy();
        return -1;
    }
    // 销毁PipeLine对象
    pl.Destroy();
    return 0;
}

int yolo_image_inference(GeneralConfig &general_config,YoloConfig &yolo_config){    
    // 读取图像文件
    cv::Mat ori_img = cv::imread(yolo_config.image_path);
    // 获取图像的宽度和高度
    FrameSize image_wh={ori_img.cols,ori_img.rows};
    // 创建一个空的向量，用于存储图像数据
    std::vector<uint8_t> chw_vec;
    // 创建一个包含3个元素的向量，用于存储图像的BGR通道
    std::vector<cv::Mat> bgrChannels(3);
    // 将图像分割成BGR通道
    cv::split(ori_img, bgrChannels);
    // 遍历BGR通道，将每个通道的数据转换为一维向量，并将其添加到chw_vec中
    for (auto i = 2; i > -1; i--)
    {
        std::vector<uint8_t> data = std::vector<uint8_t>(bgrChannels[i].reshape(1, 1));
        chw_vec.insert(chw_vec.end(), data.begin(), data.end());
    }
    // 从标签文件中读取标签
    std::vector<std::string> labels=readLabelsFromTxt(yolo_config.labels_txt_filepath);
    // 创建一个空的向量，用于存储YOLO检测结果
    std::vector<YOLOBbox> yolo_results;
    // 创建一个维度向量，用于定义输入张量的形状
    dims_t in_shape { 1, 3, ori_img.rows, ori_img.cols };
    // 创建一个运行时张量，用于存储输入数据
    runtime_tensor input_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("cannot create input tensor");
    // 获取输入张量的主机缓冲区
    auto input_buf = input_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    // 将图像数据复制到输入张量的主机缓冲区中
    memcpy(reinterpret_cast<char *>(input_buf.data()), chw_vec.data(), chw_vec.size());
    // 将输入张量的数据同步到设备上
    hrt::sync(input_tensor, sync_op_t::sync_write_back, true).expect("write back input failed");

    // 如果模型类型为yolo11，则执行yolo11的推理流程
    if(strcmp(yolo_config.model_type, "yolo11") == 0){
        Yolo11 yolo11(yolo_config.task_type,yolo_config.task_mode,yolo_config.kmodel_path,yolo_config.conf_thres,yolo_config.nms_thres,yolo_config.mask_thres,labels,image_wh,yolo_config.debug_mode);
        yolo11.pre_process(input_tensor);
        yolo11.inference();
        yolo11.post_process(yolo_results);
        yolo11.draw_results(ori_img,yolo_results);
    }
    // 如果模型类型为yolov8，则执行yolov8的推理流程
    else if(strcmp(yolo_config.model_type, "yolov8") == 0){
        Yolov8 yolov8(yolo_config.task_type,yolo_config.task_mode,yolo_config.kmodel_path,yolo_config.conf_thres,yolo_config.nms_thres,yolo_config.mask_thres,labels,image_wh,yolo_config.debug_mode);
        yolov8.pre_process(input_tensor);
        yolov8.inference();
        yolov8.post_process(yolo_results);
        yolov8.draw_results(ori_img,yolo_results);
    }
    // 如果模型类型为yolov5，则执行yolov5的推理流程
    else if(strcmp(yolo_config.model_type, "yolov5") == 0){
        Yolov5 yolov5(yolo_config.task_type,yolo_config.task_mode,yolo_config.kmodel_path,yolo_config.conf_thres,yolo_config.nms_thres,yolo_config.mask_thres,labels,image_wh,yolo_config.debug_mode);
        yolov5.pre_process(input_tensor);
        yolov5.inference();
        yolov5.post_process(yolo_results);
        yolov5.draw_results(ori_img,yolo_results);
    }
    // 如果模型类型不是yolov5、yolov8或yolo11，则打印错误信息并返回-1
    else{
        std::cout << "仅支持模型: yolov5/yolov8/yolo11 " << std::endl;
        return -1;
    }

    // 将任务类型和模型类型转换为字符串
    std::string task_(yolo_config.task_type);
    std::string model_(yolo_config.model_type);
    // 将推理结果保存为图像文件
    cv::imwrite("result_"+model_+"_"+task_+".jpg",ori_img);
    return 0;
}

void _help(){
    printf("Please input:\n");
    printf("-ai_frame_width: default 640\n");
    printf("-ai_frame_height: default 360\n");
    printf("-display_mode: default 0,\n");
    printf("   mode 0: LT9611\n");
    printf("   mode 1: ST7701\n");
    printf("   mode 2: HX8377\n");
    printf("-model_type: default yolov8, yolov5/yolov8/yolo11\n");
    printf("-task_type: default detect, classify/detect/segment\n");
    printf("-task_mode: default video, image/video\n");
    printf("-image_path: default test.jpg, image path\n");
    printf("-kmodel_path: default yolov8n.kmodel, kmodel path\n");
    printf("-labels_txt_filepath: default coco_labels.txt, labels txt filepath\n");
    printf("-conf_thres: default 0.35\n");
    printf("-nms_thres: default 0.65\n");
    printf("-mask_thres: default 0.5\n");
    printf("-debug_mode: default 0, 0/1\n");
}

// 主函数入口，程序从这里开始执行
int main(int argc, char *argv[])
{
    // 打印程序名称、编译日期和时间
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;

    GeneralConfig general_config;
    YoloConfig yolo_config;

    // 遍历命令行参数，解析并设置配置项
    for (int i = 1; i < argc; i += 2)
    {
        if (strcmp(argv[i], "-help") == 0)
        {
            // 打印帮助信息
            _help();
            return 0;
        }
        else if (strcmp(argv[i], "-ai_frame_width") == 0)
        {
            // 设置AI帧宽度
            general_config.AI_FRAME_WIDTH = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-ai_frame_height") == 0)
        {
            // 设置AI帧高度
            general_config.AI_FRAME_HEIGHT = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-display_mode") == 0)
        {
            // 设置显示模式
            general_config.DISPLAY_MODE = atoi(argv[i + 1]);
            if(general_config.DISPLAY_MODE == 0){
                // 设置显示分辨率和旋转角度
                general_config.DISPLAY_WIDTH = 1920;
                general_config.DISPLAY_HEIGHT = 1080;
                general_config.DISPLAY_ROTATE = 0;
            }
            else if(general_config.DISPLAY_MODE == 1){
                // 设置显示分辨率和旋转角度
                general_config.DISPLAY_WIDTH = 800;
                general_config.DISPLAY_HEIGHT = 480;
                general_config.DISPLAY_ROTATE = 1;
                general_config.OSD_WIDTH = 480;
                general_config.OSD_HEIGHT = 800;
            }
            else if(general_config.DISPLAY_MODE == 2){
                // 设置显示分辨率和旋转角度
                general_config.DISPLAY_WIDTH = 1080;
                general_config.DISPLAY_HEIGHT = 1920;
                general_config.DISPLAY_ROTATE = 0;
                general_config.OSD_WIDTH = 1080;
                general_config.OSD_HEIGHT = 1920;
            }
            else{
                // 打印错误信息并退出
                printf("Error :Invalid arguments %s\n", argv[i]);
                _help();
                return -1;
            }
        }
        else if (strcmp(argv[i], "-model_type") == 0)
        {
            // 设置模型类型
            yolo_config.model_type = argv[i + 1];
        }
        else if (strcmp(argv[i], "-task_type") == 0)
        {
            // 设置任务类型
            yolo_config.task_type = argv[i + 1];
        }
        else if (strcmp(argv[i], "-task_mode") == 0)
        {
            // 设置任务模式
            yolo_config.task_mode = argv[i + 1];
        }
        else if (strcmp(argv[i], "-image_path") == 0)
        {
            // 设置图像路径
            yolo_config.image_path = argv[i + 1];
        }
        else if (strcmp(argv[i], "-kmodel_path") == 0)
        {
            // 设置模型路径
            yolo_config.kmodel_path = argv[i + 1];
        }
        else if (strcmp(argv[i], "-labels_txt_filepath") == 0)
        {
            // 设置标签文件路径
            yolo_config.labels_txt_filepath = argv[i + 1];
        }
        else if (strcmp(argv[i], "-conf_thres") == 0)
        {
            // 设置置信度阈值
            yolo_config.conf_thres = atof(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-nms_thres") == 0)
        {
            // 设置非极大值抑制阈值
            yolo_config.nms_thres = atof(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-mask_thres") == 0)
        {
            // 设置掩码阈值
            yolo_config.mask_thres = atof(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-debug_mode") == 0)
        {
            // 设置调试模式
            yolo_config.debug_mode = atoi(argv[i + 1]);
        }
        else
        {
            // 打印错误信息并退出
            printf("Error :Invalid arguments %s\n", argv[i]);
            _help();
            return -1;
        }
    }
    
    // 如果任务模式为视频，则执行视频推理
    if (strcmp(yolo_config.task_mode, "video") == 0)
    {
        // 创建一个新线程来执行视频推理
        std::thread thread_isp(yolo_video_inference,std::ref(general_config),std::ref(yolo_config));
        // 等待用户输入'q'来停止视频推理
        while (getchar() != 'q')
        {
            usleep(10000);
        }
        // 设置停止标志
        isp_stop = true;
        // 等待视频推理线程结束
        thread_isp.join();
    }
    // 如果任务模式为图像，则执行图像推理
    else if(strcmp(yolo_config.task_mode, "image") == 0){
        yolo_image_inference(general_config,yolo_config);
    }
    // 如果任务模式既不是视频也不是图像，则打印错误信息并退出
    else{
        printf("only support: video/image\n");
        return -1;
    }
    return 0;
}