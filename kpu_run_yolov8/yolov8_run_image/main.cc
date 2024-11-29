
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

using std::string;
using std::vector;
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;
using namespace nncase::F::k230;

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


int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <kmodel> <image> <debug_mode>" << std::endl;
        return -1;
    }

    int debug_mode=atoi(argv[3]);

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

    // 读入图片，并将数据处理成CHW和RGB格式
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

    // 计算预处理参数，这里计算的是短边padding的参数值
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

    // 创建AI2D输入tensor，并将CHW_RGB数据拷贝到tensor中，并回写到DDR
    dims_t ai2d_in_shape{1, 3, ori_h, ori_w};
    runtime_tensor ai2d_in_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, ai2d_in_shape, hrt::pool_shared).expect("cannot create input tensor");
    auto input_buf = ai2d_in_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    memcpy(reinterpret_cast<char *>(input_buf.data()), chw_vec.data(), chw_vec.size());
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
    // 标签名称
    std::vector<std::string> classes{"apple","banana","orange"};
    // 置信度阈值
    float conf_thresh=0.25;
    // nms阈值
    float nms_thresh=0.45;
    //类别数
    int class_num=classes.size();
    // 根据类别数获取颜色，用于后续画图
    std::vector<cv::Scalar> class_colors = getColorsForClasses(class_num);

    // output0 [num_class+4,(w/8)*(h/8)+(w/16)*(h/16)+(w/32)*(h/32)]
    float *output0 = p_outputs[0];
    // 每个框的特征长度，ckass_num个分数+4个坐标
    int f_len=class_num+4;
    // 根据模型的输入分辨率计算总输出框数
    int num_box=((input_shapes[0][2]/8)*(input_shapes[0][3]/8)+(input_shapes[0][2]/16)*(input_shapes[0][3]/16)+(input_shapes[0][2]/32)*(input_shapes[0][3]/32));
    // 申请框数据内存
    float *output_det = new float[num_box * f_len];
    // 将输出数据排布从[num_class+4,(w/8)*(h/8)+(w/16)*(h/16)+(w/32)*(h/32)]调整为[(w/8)*(h/8)+(w/16)*(h/16)+(w/32)*(h/32),num_class+4],方便后续处理
    for(int r = 0; r < num_box; r++)
    {
        for(int c = 0; c < f_len; c++)
        {
            output_det[r*f_len + c] = output0[c*num_box + r];
        }
    }

    // 解析每个框的信息，class_num+4为一个框，前四个数据为坐标值，后面的class_num个分数，选择分数最大的作为识别的类别，因为开始的时候做了padding+resize，所以模型推理的坐标是基于与处理后的图像的结果，要先把框的坐标使用ratio映射回原图
    std::vector<Bbox> bboxes;
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

    // 将识别的框绘制到原图片上并保存为结果图片result,jpg
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