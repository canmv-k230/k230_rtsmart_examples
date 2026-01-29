/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nncase/functional/ai2d/ai2d_builder.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;
using namespace nncase::F::k230;

using cv::Mat;
using std::cout;
using std::endl;
using std::ifstream;
using std::vector;

//颜色盘，共80种颜色，类别大于80时取余
const std::vector<cv::Scalar> color_four = {
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

/**
 * @brief 单张/帧图片大小
 */
typedef struct FrameCHWSize
{
    size_t channel;
    size_t height; // 高
    size_t width;  // 宽
} FrameCHWSize;


// 根据类别数使用模运算循环获取颜色
std::vector<cv::Scalar> getColorsForClasses(int num_classes);


/**
 * @brief 工具类
 * 封装了AI常用的函数，包括二进制文件读取、文件保存、图片预处理等操作
 */
class Utils
{
public:
    /**
     * @brief 读取2进制文件
     * @param file_name 文件路径
     * @return 文件对应类型的数据
     */
    template <class T>
    static vector<T> read_binary_file(const char *file_name)
    {
        ifstream ifs(file_name, std::ios::binary);
        ifs.seekg(0, ifs.end);
        size_t len = ifs.tellg();
        vector<T> vec(len / sizeof(T), 0);
        ifs.seekg(0, ifs.beg);
        ifs.read(reinterpret_cast<char *>(vec.data()), len);
        ifs.close();
        return vec;
    }

    /**
     * @brief               打印数据
     * @param data          需打印数据对应指针
     * @param size          需打印数据大小
     * @return              None
     */
    template <class T>
    static void dump(const T *data, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            cout << data[i] << " ";
        }
        cout << endl;
    }

     /**
     * @brief 读取2进制文件
     * @param file_name 文件路径
     * @param outi      读取到的数据
     * @return None
     */
    static void read_binary_file_bin(std::string file_name,unsigned char *outi);

    /**
     * @brief 将bin文件转为Mat
     * @param bin_data_path 文件路径
     * @param mat_width     Mat宽
     * @param mat_height    Mat高
     * @param image_argb    转换之后的Mat数据
     * @return None
     */
    static void bin_2_mat(std::string bin_data_path, int mat_width, int mat_height, cv::Mat &image_argb);

    // 静态成员函数不依赖于类的实例，可以直接通过类名调用
    /**
     * @brief               将数据以2进制方式保存为文件
     * @param file_name     保存文件路径+文件名
     * @param data          需要保存的数据
     * @param size          需要保存的长度
     * @return              None
     */
    static void dump_binary_file(const char *file_name, char *data, const size_t size);

    /**
     * @brief               将数据保存为灰度图片
     * @param file_name     保存图片路径+文件名
     * @param frame_size    保存图片的宽、高
     * @param data          需要保存的数据
     * @return              None
     */
    static void dump_gray_image(const char *file_name, const FrameCHWSize &frame_size, unsigned char *data);

    /**
     * @brief               将数据保存为彩色图片
     * @param file_name     保存图片路径+文件名
     * @param frame_size    保存图片的宽、高
     * @param data          需要保存的数据
     * @return              None
     */
    static void dump_color_image(const char *file_name, const FrameCHWSize &frame_size, unsigned char *data);

    //------------------------------ai2d处理配置---------------------------------------------------------------

    static void padding_resize_one_side_set(FrameCHWSize input_shape, FrameCHWSize output_shape, std::unique_ptr<ai2d_builder> &builder, const cv::Scalar padding);

    static void padding_resize_two_side_set(FrameCHWSize input_shape, FrameCHWSize output_shape, std::unique_ptr<ai2d_builder> &builder, const cv::Scalar padding);

    static void center_crop_resize_set(FrameCHWSize input_shape,FrameCHWSize output_shape,std::unique_ptr<ai2d_builder> &builder);

    static void crop_resize_set(FrameCHWSize input_shape,FrameCHWSize output_shape,int x,int y,int w,int h,std::unique_ptr<ai2d_builder> &builder);

    static void crop_resize_out2RGBP_out2HWC_set(FrameCHWSize input_shape,FrameCHWSize output_shape,int x,int y,int w,int h,std::unique_ptr<ai2d_builder> &builder);

    static void crop_set(FrameCHWSize input_shape,int x,int y,int w,int h,std::unique_ptr<ai2d_builder> &builder);

    static void resize_set(FrameCHWSize input_shape, FrameCHWSize output_shape, std::unique_ptr<ai2d_builder> &builder);

    static void ratio_resize_set(FrameCHWSize input_shape, int max_size, std::unique_ptr<ai2d_builder> &builder);

    static void affine_set(FrameCHWSize input_shape, FrameCHWSize output_shape,  std::unique_ptr<ai2d_builder> &builder,float *affine_matrix);

    //------------------------------cv::Mat处理---------------------------------------------------------------
    static cv::Mat padding_resize(cv::Mat &img, FrameCHWSize &frame_size, cv::Scalar &padding_value);

    static cv::Mat resize(cv::Mat &ori_img,FrameCHWSize &frame_size);

    static cv::Mat bgr_to_rgb(cv::Mat &ori_img);
};

#endif
