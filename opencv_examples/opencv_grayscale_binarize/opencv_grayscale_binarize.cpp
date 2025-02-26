#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

/**
 * @brief 主函数，程序的入口点。
 * 
 * 此函数的主要功能是读取一张彩色图像，将其转换为灰度图像并保存，
 * 然后对灰度图像进行二值化处理并保存处理后的图像。
 * 
 * @return int 程序退出状态码，0 表示正常退出，-1 表示图像读取失败。
 */
int main()
{
    // 读取指定路径的彩色图像，第二个参数 1 表示读取彩色图像
    Mat lena = imread("./1.bmp", 1);
    // 检查图像是否成功读取
    if (!lena.data) {
        // 若图像读取失败，输出错误信息
        cout << "Input Image reading error !" << endl;
        // 返回 -1 表示程序异常退出
        return -1;
    }

    // 定义两个 Mat 对象，分别用于存储灰度图像和二值化后的图像
    Mat lena_gray, lena_threshold;
    // 将彩色图像转换为灰度图像，使用 COLOR_BGR2GRAY 转换标志
    cvtColor(lena, lena_gray, COLOR_BGR2GRAY);
    // 将转换后的灰度图像保存为文件
    imwrite("./lena_gray.jpg", lena_gray);

    // 对灰度图像进行二值化处理
    // 第一个参数是输入的灰度图像，第二个参数是输出的二值化图像
    // 第三个参数是阈值 170，第四个参数是最大值 255
    // 第五个参数 THRESH_BINARY 表示二值化类型
    threshold(lena_gray, lena_threshold, 170, 255, THRESH_BINARY);

    // 将二值化后的图像保存为文件
    imwrite("./lena_threshold.jpg", lena_threshold);

    // 程序正常结束，返回 0
    return 0;
}
