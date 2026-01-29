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
#include "setting.h"
#include "video_pipeline.h"

using std::string;
using std::vector;
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;
using namespace nncase::F::k230;

/**
 * @brief ISP采集线程退出标志
 */
std::atomic<bool> isp_stop(false);

/**
 * @brief 单张/帧图片大小（CHW格式）
 */
typedef struct FrameCHWSize
{
    int channel; // 通道数
    int height;  // 高
    int width;   // 宽
} FrameCHWSize;

/**
 * @brief 目标检测框结构体
 */
typedef struct Bbox{
	cv::Rect box;   // 边界框坐标
	float confidence; // 置信度
	int index;      // 类别索引
}Bbox;

/**
 * @brief 颜色盘，共80种颜色（带Alpha通道），用于绘制检测框
 *        当类别数大于80时，使用取余方式循环取色
 */
const std::vector<cv::Scalar> color_four = {
    cv::Scalar(60, 20, 220, 127),
    cv::Scalar(32, 11, 119, 127),
    cv::Scalar(142, 0, 0, 127),
    cv::Scalar(230, 0, 0, 127),
    cv::Scalar(228, 0, 106, 127),
    cv::Scalar(100, 60, 0, 127),
    cv::Scalar(100, 80, 0, 127),
    cv::Scalar(70, 0, 0, 127),
    cv::Scalar(192, 0, 0, 127),
    cv::Scalar(30, 170, 250, 127),
    cv::Scalar(30, 170, 100, 127),
    cv::Scalar(0, 220, 220, 127),
    cv::Scalar(175, 116, 175, 127),
    cv::Scalar(30, 0, 250, 127),
    cv::Scalar(42, 42, 165, 127),
    cv::Scalar(255, 77, 255, 127),
    cv::Scalar(252, 226, 0, 127),
    cv::Scalar(255, 182, 182, 127),
    cv::Scalar(0, 82, 0, 127),
    cv::Scalar(157, 166, 120, 127),
    cv::Scalar(0, 76, 110, 127),
    cv::Scalar(255, 57, 174, 127),
    cv::Scalar(0, 100, 199, 127),
    cv::Scalar(118, 0, 72, 127),
    cv::Scalar(240, 179, 255, 127),
    cv::Scalar(92, 125, 0, 127),
    cv::Scalar(151, 0, 209, 127),
    cv::Scalar(182, 208, 188, 127),
    cv::Scalar(176, 220, 0, 127),
    cv::Scalar(164, 99, 255, 127),
    cv::Scalar(73, 0, 92, 127),
    cv::Scalar(255, 129, 133, 127),
    cv::Scalar(255, 180, 78, 127),
    cv::Scalar(0, 228, 0, 127),
    cv::Scalar(243, 255, 174, 127),
    cv::Scalar(255, 89, 45, 127),
    cv::Scalar(103, 134, 134, 127),
    cv::Scalar(174, 148, 145, 127),
    cv::Scalar(186, 208, 255, 127),
    cv::Scalar(255, 226, 197, 127),
    cv::Scalar(1, 134, 171, 127),
    cv::Scalar(54, 63, 109, 127),
    cv::Scalar(255, 138, 207, 127),
    cv::Scalar(95, 0, 151, 127),
    cv::Scalar(61, 80, 9, 127),
    cv::Scalar(51, 105, 84, 127),
    cv::Scalar(105, 65, 74, 127),
    cv::Scalar(102, 196, 166, 127),
    cv::Scalar(210, 195, 208, 127),
    cv::Scalar(65, 109, 255, 127),
    cv::Scalar(149, 143, 0, 127),
    cv::Scalar(194, 0, 179, 127),
    cv::Scalar(106, 99, 209, 127),
    cv::Scalar(0, 121, 5, 127),
    cv::Scalar(205, 255, 227, 127),
    cv::Scalar(208, 186, 147, 127),
    cv::Scalar(1, 69, 153, 127),
    cv::Scalar(161, 95, 3, 127),
    cv::Scalar(0, 255, 163, 127),
    cv::Scalar(170, 0, 119, 127),
    cv::Scalar(199, 182, 0, 127),
    cv::Scalar(120, 165, 0, 127),
    cv::Scalar(88, 130, 183, 127),
    cv::Scalar(0, 32, 95, 127),
    cv::Scalar(135, 114, 130, 127),
    cv::Scalar(133, 129, 110, 127),
    cv::Scalar(118, 74, 166, 127),
    cv::Scalar(185, 142, 219, 127),
    cv::Scalar(114, 210, 79, 127),
    cv::Scalar(62, 90, 178, 127),
    cv::Scalar(15, 70, 65, 127),
    cv::Scalar(115, 167, 127, 127),
    cv::Scalar(106, 105, 59, 127),
    cv::Scalar(45, 108, 142, 127),
    cv::Scalar(0, 172, 196, 127),
    cv::Scalar(80, 54, 95, 127),
    cv::Scalar(255, 76, 128, 127),
    cv::Scalar(1, 57, 201, 127),
    cv::Scalar(122, 0, 246, 127),
    cv::Scalar(208, 162, 191, 127)
};

/**
 * @brief 根据类别数生成颜色表
 * @param num_classes 类别数量
 * @return 颜色列表
 */
std::vector<cv::Scalar> getColorsForClasses(int num_classes) {
    std::vector<cv::Scalar> colors;
    int num_available_colors = color_four.size();
    for (int i = 0; i < num_classes; ++i) {
        colors.push_back(color_four[i % num_available_colors]);
    }
    return colors;
}

/**
 * @brief 计算两个矩形框的 IoU（交并比）
 */
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

/**
 * @brief 非极大值抑制（NMS）
 * @param bboxes 待处理的检测框集合
 * @param confThreshold 置信度阈值
 * @param nmsThreshold NMS阈值
 * @param indices 输出：保留下来的框索引
 */
void nms(std::vector<Bbox> &bboxes,  float confThreshold, float nmsThreshold, std::vector<int> &indices)
{
	sort(bboxes.begin(), bboxes.end(), [](Bbox a, Bbox b) { return a.confidence > b.confidence; });
	int updated_size = bboxes.size();
	for (int i = 0; i < updated_size; i++)
	{
		if (bboxes[i].confidence < confThreshold)
			continue;
		indices.push_back(i);
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

/**
 * @brief 摄像头采集 + AI推理主线程
 */
int camera_inference(char *argv[])
{
    /************************************************************
     * Phase 0: 参数解析与基础变量初始化
     ************************************************************/
    int debug_mode = atoi(argv[4]);

    // AI 输入图像尺寸（CHW）
    FrameCHWSize image_size = {AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH};

    // OSD 图层（RGBA），用于绘制检测框与文字
    cv::Mat draw_frame(OSD_HEIGHT, OSD_WIDTH, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    /************************************************************
     * Phase 1: 视频管线初始化（ISP → DRM → OSD）
     ************************************************************/
    PipeLine pl(debug_mode);
    pl.Create();

    // 存放 ISP 获取的一帧数据（虚拟地址 + 物理地址）
    DumpRes dump_res;

    /************************************************************
     * Phase 2: KModel 加载与 Interpreter 初始化
     ************************************************************/
    interpreter interp;
    std::ifstream ifs(argv[1], std::ios::binary);
    interp.load_model(ifs).expect("Invalid kmodel");

    /************************************************************
     * Phase 3: 输入 / 输出 Tensor 初始化与 Shape 记录
     ************************************************************/
    vector<vector<int>> input_shapes;
    vector<vector<int>> output_shapes;
    vector<float *> p_outputs;

    // ---------- 初始化模型输入 Tensor ----------
    for (int i = 0; i < interp.inputs_size(); i++)
    {
        auto desc = interp.input_desc(i);
        auto shape = interp.input_shape(i);

        auto tensor = host_runtime_tensor::create(
            desc.datatype, shape, hrt::pool_shared)
            .expect("cannot create input tensor");

        interp.input_tensor(i, tensor).expect("cannot set input tensor");

        vector<int> in_shape;
        if (debug_mode > 1)
            std::cout << "input " << i << " datatype: " << desc.datatype << " , shape: ";

        for (int j = 0; j < shape.size(); ++j)
        {
            in_shape.push_back(shape[j]);
            if (debug_mode > 1)
                std::cout << shape[j] << " ";
        }

        if (debug_mode > 1)
            std::cout << std::endl;

        input_shapes.push_back(in_shape);
    }

    // ---------- 初始化模型输出 Tensor ----------
    for (size_t i = 0; i < interp.outputs_size(); i++)
    {
        auto desc = interp.output_desc(i);
        auto shape = interp.output_shape(i);

        auto tensor = host_runtime_tensor::create(
            desc.datatype, shape, hrt::pool_shared)
            .expect("cannot create output tensor");

        interp.output_tensor(i, tensor).expect("cannot set output tensor");

        vector<int> out_shape;
        if (debug_mode > 1)
            std::cout << "output " << i << " datatype: " << desc.datatype << " , shape: ";

        for (int j = 0; j < shape.size(); ++j)
        {
            out_shape.push_back(shape[j]);
            if (debug_mode > 1)
                std::cout << shape[j] << " ";
        }

        if (debug_mode > 1)
            std::cout << std::endl;

        output_shapes.push_back(out_shape);
    }

    /************************************************************
     * Phase 4: 计算 Resize + Padding 参数（YOLO LetterBox）
     ************************************************************/
    int width  = input_shapes[0][3];
    int height = input_shapes[0][2];

    float ratiow = (float)width  / AI_FRAME_WIDTH;
    float ratioh = (float)height / AI_FRAME_HEIGHT;
    float ratio  = ratiow < ratioh ? ratiow : ratioh;

    int new_w = (int)(ratio * AI_FRAME_WIDTH);
    int new_h = (int)(ratio * AI_FRAME_HEIGHT);

    float dw = (float)(width  - new_w) / 2;
    float dh = (float)(height - new_h) / 2;

    int top    = (int)(roundf(0));
    int bottom = (int)(roundf(dh * 2 + 0.1));
    int left   = (int)(roundf(0));
    int right  = (int)(roundf(dw * 2 - 0.1));

    /************************************************************
     * Phase 5: AI2D Tensor 与 Builder 配置
     ************************************************************/
    dims_t ai2d_in_shape{1, AI_FRAME_CHANNEL, AI_FRAME_HEIGHT, AI_FRAME_WIDTH};

    runtime_tensor ai2d_in_tensor;

    // 直接复用模型输入 Tensor 作为 AI2D 输出，避免额外拷贝
    runtime_tensor ai2d_out_tensor =
        interp.input_tensor(0).expect("cannot get input tensor");

    dims_t out_shape = ai2d_out_tensor.shape();

    // AI2D 数据类型配置（NCHW → NCHW，uint8）
    ai2d_datatype_t ai2d_dtype{
        ai2d_format::NCHW_FMT,
        ai2d_format::NCHW_FMT,
        typecode_t::dt_uint8,
        typecode_t::dt_uint8};

    // 各 AI2D 功能模块参数
    ai2d_crop_param_t   crop_param{false, 0, 0, 0, 0};
    ai2d_shift_param_t  shift_param{false, 0};
    ai2d_pad_param_t    pad_param{
        true,
        {{0, 0}, {0, 0}, {top, bottom}, {left, right}},
        ai2d_pad_mode::constant,
        {114, 114, 114}};
    ai2d_resize_param_t resize_param{
        true,
        ai2d_interp_method::tf_bilinear,
        ai2d_interp_mode::half_pixel};
    ai2d_affine_param_t affine_param{
        false,
        ai2d_interp_method::cv2_bilinear,
        0, 0, 127, 1,
        {0.5, 0.1, 0.0, 0.1, 0.5, 0.0}};

    // 构建 AI2D 调度器
    ai2d_builder builder(
        ai2d_in_shape,
        out_shape,
        ai2d_dtype,
        crop_param,
        shift_param,
        pad_param,
        resize_param,
        affine_param);

    builder.build_schedule();

    /************************************************************
     * Phase 6: 后处理与绘制相关参数初始化
     ************************************************************/
    std::vector<std::string> classes{"apple", "banana", "orange"};

    float conf_thresh = atof(argv[2]);
    float nms_thresh  = atof(argv[3]);
    int class_num     = classes.size();

    std::vector<cv::Scalar> class_colors =
        getColorsForClasses(class_num);

    float *output0;
    int f_len = class_num + 4;

    int num_box =
        ((input_shapes[0][2] / 8)  * (input_shapes[0][3] / 8) +
         (input_shapes[0][2] / 16) * (input_shapes[0][3] / 16) +
         (input_shapes[0][2] / 32) * (input_shapes[0][3] / 32));

    float *output_det = new float[num_box * f_len];

    std::vector<Bbox> bboxes;

    /************************************************************
     * Phase 7: 主循环（采集 → 预处理 → 推理 → 后处理 → 显示）
     ************************************************************/
    while (!isp_stop)
    {
        // ---------- 获取一帧 ISP 图像 ----------
        pl.GetFrame(dump_res);

        // ---------- 创建 AI2D 输入 Tensor（零拷贝绑定 ISP Buffer） ----------
        ai2d_in_tensor = host_runtime_tensor::create(
            typecode_t::dt_uint8,
            ai2d_in_shape,
            {(gsl::byte *)dump_res.virt_addr,
             compute_size(ai2d_in_shape)},
            false,
            hrt::pool_shared,
            dump_res.phy_addr)
            .expect("cannot create input tensor");

        hrt::sync(ai2d_in_tensor, sync_op_t::sync_write_back, true)
            .expect("sync write_back failed");

        // ---------- 执行 AI2D 预处理 ----------
        builder.invoke(ai2d_in_tensor, ai2d_out_tensor)
            .expect("error occurred in ai2d running");

        // ---------- 执行模型推理 ----------
        interp.run().expect("error occurred in running model");

        // ---------- 获取模型输出 ----------
        p_outputs.clear();
        for (int i = 0; i < interp.outputs_size(); i++)
        {
            auto out = interp.output_tensor(i).expect("cannot get output tensor");
            auto buf = out.impl()->to_host().unwrap()
                           ->buffer().as_host().unwrap()
                           .map(map_access_::map_read).unwrap()
                           .buffer();
            p_outputs.push_back(reinterpret_cast<float *>(buf.data()));
        }

        /********************************************************
         * Phase 8: 后处理（解码 + 置信度筛选 + NMS）
         ********************************************************/
        output0 = p_outputs[0];

        // 转置输出布局（C x N → N x C）
        for (int r = 0; r < num_box; r++)
        {
            for (int c = 0; c < f_len; c++)
            {
                output_det[r * f_len + c] =
                    output0[c * num_box + r];
            }
        }

        bboxes.clear();

        for (int i = 0; i < num_box; i++)
        {
            float *vec = output_det + i * f_len;
            float box[4] = {vec[0], vec[1], vec[2], vec[3]};
            float *class_scores = vec + 4;

            auto max_class_score_ptr =
                std::max_element(class_scores,
                                 class_scores + class_num);

            float score = *max_class_score_ptr;
            int max_class_index =
                max_class_score_ptr - class_scores;

            if (score > conf_thresh)
            {
                Bbox bbox;

                float x_ = box[0] / ratio;
                float y_ = box[1] / ratio;
                float w_ = box[2] / ratio;
                float h_ = box[3] / ratio;

                int x = int(MAX(x_ - 0.5 * w_, 0));
                int y = int(MAX(y_ - 0.5 * h_, 0));
                int w = int(w_);
                int h = int(h_);

                if (w <= 0 || h <= 0)
                    continue;

                bbox.box = cv::Rect(x, y, w, h);
                bbox.confidence = score;
                bbox.index = max_class_index;
                bboxes.push_back(bbox);
            }
        }

        // ---------- 执行 NMS ----------
        std::vector<int> nms_result;
        nms(bboxes, conf_thresh, nms_thresh, nms_result);

        /********************************************************
         * Phase 9: OSD 绘制与显示
         ********************************************************/
        draw_frame.setTo(cv::Scalar(0, 0, 0, 0));

        for (int i = 0; i < nms_result.size(); i++)
        {
            int res = nms_result[i];
            cv::Rect box = bboxes[res].box;
            int idx = bboxes[res].index;
            float score = bboxes[res].confidence;

            int x = int(box.x * float(OSD_WIDTH) / AI_FRAME_WIDTH);
            int y = int(box.y * float(OSD_HEIGHT) / AI_FRAME_HEIGHT);
            int w = int(box.width  * float(OSD_WIDTH) / AI_FRAME_WIDTH);
            int h = int(box.height * float(OSD_HEIGHT) / AI_FRAME_HEIGHT);

            cv::Rect new_box(x, y, w, h);

            cv::rectangle(draw_frame, new_box, class_colors[idx], 2, 8);
            cv::putText(draw_frame,
                        classes[idx] + " " + std::to_string(score),
                        cv::Point(MIN(new_box.x + 5, OSD_HEIGHT),
                                  MAX(new_box.y - 10, 0)),
                        cv::FONT_HERSHEY_DUPLEX,
                        1,
                        class_colors[idx],
                        2,
                        0);
        }

        // ---------- OSD 合成并释放帧 ----------
        pl.InsertFrame(draw_frame.data);
        pl.ReleaseFrame(dump_res);
    }

    /************************************************************
     * Phase 10: 资源释放
     ************************************************************/
    delete[] output_det;
    pl.Destroy();
    return 0;
}


/**
 * @brief 程序入口
 */
int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <kmodel> <conf_thresh> <nms_thresh> <debug_mode>" << std::endl;
        return -1;
    }

    // 启动推理线程
    std::thread thread_isp(camera_inference, argv);

    // 主线程监听退出指令
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    // 设置退出标志并等待线程结束
    isp_stop = true;
    thread_isp.join();
    return 0;
}
