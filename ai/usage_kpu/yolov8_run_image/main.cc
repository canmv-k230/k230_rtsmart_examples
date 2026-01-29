#include <iostream>
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
// #include "utils.h"

// 使用标准命名空间
using std::string;
using std::vector;
// 使用 nncase 相关命名空间
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;
using namespace nncase::F::k230;

// 定义检测框结构体：包含矩形框、置信度和类别索引
typedef struct Bbox{
	cv::Rect box;      // 边界框
	float confidence; // 置信度
	int index;         // 类别索引
}Bbox;

// 颜色盘，共80种颜色（RGBA），当类别数大于80时取模循环使用
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

// 根据类别数获取对应数量的颜色（超出则循环取模）
std::vector<cv::Scalar> getColorsForClasses(int num_classes) {
    std::vector<cv::Scalar> colors;
    int num_available_colors = color_four.size(); 
    for (int i = 0; i < num_classes; ++i) {
        colors.push_back(color_four[i % num_available_colors]);
    }
    return colors;
}

// 计算两个矩形框的 IOU（交并比）
float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
	int xx1, yy1, xx2, yy2;
 
	// 计算交集区域左上角与右下角坐标
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

// 非极大值抑制（NMS）
// bboxes: 输入候选框
// confThreshold: 置信度阈值
// nmsThreshold: IOU 阈值
// indices: 输出保留框的索引
void nms(std::vector<Bbox> &bboxes,  float confThreshold, float nmsThreshold, std::vector<int> &indices)
{	
	// 按置信度从大到小排序
	sort(bboxes.begin(), bboxes.end(), [](Bbox a, Bbox b) { return a.confidence > b.confidence; });
	int updated_size = bboxes.size();
	for (int i = 0; i < updated_size; i++)
	{
		// 过滤低于置信度阈值的框
		if (bboxes[i].confidence < confThreshold)
			continue;
		indices.push_back(i);
		for (int j = i + 1; j < updated_size;)
		{
			// 计算 IOU
			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
			if (iou > nmsThreshold)
			{
				// 删除重叠度过高的框
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

int main(int argc, char *argv[])
{
    // 打印程序名称与编译时间
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;

    // 参数校验
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <kmodel> <image> <debug_mode>" << std::endl;
        return -1;
    }

    // 调试模式
    int debug_mode=atoi(argv[3]);

    // =========================
    // 1. 加载 KModel 模型
    // =========================
    interpreter interp;     
    std::ifstream ifs(argv[1], std::ios::binary);
    interp.load_model(ifs).expect("Invalid kmodel");

    // 初始化输入输出 shape 容器及输出指针容器
    vector<vector<int>> input_shapes;   
    vector<vector<int>> output_shapes;
    vector<float *> p_outputs;

    // =========================
    // 2. 初始化输入 Tensor
    // =========================
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

    // =========================
    // 3. 初始化输出 Tensor
    // =========================
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

    // =========================
    // 4. 读取图片并转为 CHW + RGB
    // =========================
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;
    std::vector<uint8_t> chw_vec;
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(ori_img, bgrChannels);
    for (auto i = 2; i > -1; i--)
    {
        std::vector<uint8_t> data = std::vector<uint8_t>(bgrChannels[i].reshape(1, 1));
        chw_vec.insert(chw_vec.end(), data.begin(), data.end());
    }

    // =========================
    // 5. 计算 Pad + Resize 参数
    // =========================
    int width = input_shapes[0][3];
    int height = input_shapes[0][2];
    float ratiow = (float)width / ori_w;
    float ratioh = (float)height / ori_h;
    float ratio = ratiow < ratioh ? ratiow : ratioh;
    int new_w = (int)(ratio * ori_w);
    int new_h = (int)(ratio * ori_h);
    float dw = (float)(width - new_w) / 2;
    float dh = (float)(height - new_h) / 2;
    int top = (int)(roundf(0));
    int bottom = (int)(roundf(dh * 2 + 0.1));
    int left = (int)(roundf(0));
    int right = (int)(roundf(dw * 2 - 0.1));

    // =========================
    // 6. 构造 AI2D 输入 Tensor 并写入数据
    // =========================
    dims_t ai2d_in_shape{1, 3, ori_h, ori_w};
    runtime_tensor ai2d_in_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, ai2d_in_shape, hrt::pool_shared).expect("cannot create input tensor");
    auto input_buf = ai2d_in_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    memcpy(reinterpret_cast<char *>(input_buf.data()), chw_vec.data(), chw_vec.size());
    hrt::sync(ai2d_in_tensor, sync_op_t::sync_write_back, true).expect("write back input failed");

    // =========================
    // 7. 直接使用模型输入 Tensor 作为 AI2D 输出
    // =========================
    runtime_tensor ai2d_out_tensor = interp.input_tensor(0).expect("cannot get input tensor");
    dims_t out_shape = ai2d_out_tensor.shape();

    // =========================
    // 8. 配置 AI2D 预处理参数
    // =========================
    ai2d_datatype_t ai2d_dtype{ai2d_format::NCHW_FMT, ai2d_format::NCHW_FMT, ai2d_in_tensor.datatype(), ai2d_out_tensor.datatype()};
    ai2d_crop_param_t crop_param{false, 0, 0, 0, 0};
    ai2d_shift_param_t shift_param{false, 0};
    ai2d_pad_param_t pad_param{true, {{0, 0}, {0, 0}, {top, bottom}, {left, right}}, ai2d_pad_mode::constant, {114, 114, 114}};
    ai2d_resize_param_t resize_param{true, ai2d_interp_method::tf_bilinear, ai2d_interp_mode::half_pixel};
    ai2d_affine_param_t affine_param{false, ai2d_interp_method::cv2_bilinear, 0, 0, 127, 1, {0.5, 0.1, 0.0, 0.1, 0.5, 0.0}};

    // =========================
    // 9. 构建并执行 AI2D
    // =========================
    ai2d_builder builder(ai2d_in_shape, out_shape, ai2d_dtype, crop_param, shift_param, pad_param, resize_param, affine_param);
    builder.build_schedule();
    builder.invoke(ai2d_in_tensor,ai2d_out_tensor).expect("error occurred in ai2d running");

    // =========================
    // 10. 执行模型推理
    // =========================
    interp.run().expect("error occurred in running model");

    // =========================
    // 11. 获取模型输出
    // =========================
    p_outputs.clear();
    for (int i = 0; i < interp.outputs_size(); i++)
    {
        auto out = interp.output_tensor(i).expect("cannot get output tensor");
        auto buf = out.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();
        float *p_out = reinterpret_cast<float *>(buf.data());
        p_outputs.push_back(p_out);
    }

    // =========================
    // 12. 后处理（解码 + NMS）
    // =========================
    std::vector<std::string> classes{"apple","banana","orange"};
    float conf_thresh=0.25;
    float nms_thresh=0.45;
    int class_num=classes.size();
    std::vector<cv::Scalar> class_colors = getColorsForClasses(class_num);

    float *output0 = p_outputs[0];
    int f_len=class_num+4;
    int num_box=((input_shapes[0][2]/8)*(input_shapes[0][3]/8)+(input_shapes[0][2]/16)*(input_shapes[0][3]/16)+(input_shapes[0][2]/32)*(input_shapes[0][3]/32));
    float *output_det = new float[num_box * f_len];

    for(int r = 0; r < num_box; r++)
    {
        for(int c = 0; c < f_len; c++)
        {
            output_det[r*f_len + c] = output0[c*num_box + r];
        }
    }

    // 解析检测框并映射回原图坐标
    std::vector<Bbox> bboxes;
    for(int i=0;i<num_box;i++){
        float* vec=output_det+i*f_len;
        float box[4]={vec[0],vec[1],vec[2],vec[3]};
        float* class_scores=vec+4;
        float* max_class_score_ptr=std::max_element(class_scores,class_scores+class_num);
        float score=*max_class_score_ptr;
        int max_class_index = max_class_score_ptr - class_scores;
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

	// 执行非极大值抑制
	std::vector<int> nms_result;
	nms(bboxes, conf_thresh, nms_thresh, nms_result);

    // =========================
    // 13. 绘制检测结果并保存
    // =========================
    for (int i = 0; i < nms_result.size(); i++) {
        int res=nms_result[i];
        cv::Rect box=bboxes[res].box;
        int idx=bboxes[res].index;
		cv::rectangle(ori_img, box, class_colors[idx], 2, 8);
        cv::putText(ori_img, classes[idx], cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, class_colors[idx], 2, 0);
	}
    cv::imwrite("result.jpg", ori_img); 
    
    delete[] output_det;

    return 0;
}
