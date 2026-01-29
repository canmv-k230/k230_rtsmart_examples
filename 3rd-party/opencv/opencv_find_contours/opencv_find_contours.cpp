#include <iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <stdlib.h>
 
using namespace std;
using namespace cv;
 
/**
 * @brief 主函数，程序的入口点。该函数用于读取图像，检测图像中的轮廓，并将轮廓绘制出来保存为新图像。
 * 
 * 此函数首先读取一张图像，检查图像是否成功加载。若加载成功，将图像转换为灰度图，
 * 然后进行二值化处理。接着使用 `findContours` 函数检测图像中的轮廓，
 * 最后为每个轮廓随机分配颜色并绘制到新的图像上，将绘制好的图像保存为文件。
 * 
 * @return int 程序退出状态码，0 表示正常退出，-1 表示图像加载失败。
 */
int main()
{
    // 定义三个 Mat 对象，分别用于存储原始图像、灰度图像和最终的轮廓图像
    Mat src, grayImage, dstImage;
    // 从文件中读取图像，并将其存储在 src 对象中
    src = imread("./a.jpg");
 
    // 判断图像是否加载成功
    if (src.empty())
    {
        // 若图像为空，输出错误信息
        cout << "图像加载失败" << endl;
        // 返回 -1 表示程序异常退出
        return -1;
    }
 
    // 将原始图像转换为灰度图，结果存储在 grayImage 中
    cvtColor(src, grayImage, COLOR_BGR2GRAY);
 
    // 定义两个变量，contours 用于存储检测到的轮廓，hierarchy 用于存储轮廓的层次结构
    vector<vector<Point>>contours;
    vector<Vec4i>hierarchy;
 
    // 对灰度图像进行二值化处理，像素值大于 120 的设为 255，小于等于 120 的设为 0
    grayImage = grayImage > 120;
    // 使用 OpenCV 的 findContours 函数检测二值化图像中的轮廓
    // 参数依次为：输入图像、存储轮廓的向量、存储轮廓层次结构的向量、轮廓检索模式、轮廓逼近方法
    findContours(grayImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
 
    // 创建一个全黑的图像，用于绘制轮廓，大小与灰度图像相同，类型为 8 位无符号 3 通道
    dstImage = Mat::zeros(grayImage.size(), CV_8UC3);
    // 遍历所有检测到的轮廓
    for (long unsigned int i = 0; i < hierarchy.size(); i++)
    {
        // 为每个轮廓随机生成一个颜色
        Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
        // 在 dstImage 上绘制当前轮廓，使用随机颜色填充，线条宽度为 8，同时使用层次结构信息
        drawContours(dstImage, contours, i, color, CV_FILLED, 8, hierarchy);
    }
    // 将绘制好轮廓的图像保存为文件
    imwrite("lena_contours.jpg", dstImage);
    // 程序正常结束，返回 0
    return 0;
}

