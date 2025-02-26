#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;

/**
 * @brief 主函数，程序的入口点
 * 
 * 此函数用于读取一张图像，计算其每个颜色通道（蓝、绿、红）的直方图，
 * 并将直方图绘制到一个画布上，最后将绘制好的直方图保存为文件。
 * 
 * @return int 程序退出状态码，0 表示正常退出，-1 表示图像读取失败
 */
int main()
{
    // 读取指定路径的图像文件，并将其存储在 Mat 对象 src 中
    Mat src = imread("./a.jpg");
    // 检查图像是否成功读取
    if (src.empty()) 
    {
        // 若图像为空，输出错误信息
        cout << "no picture" << endl;
        // 返回 -1 表示程序异常退出
        return -1;
    }
    // 定义一个向量 all_channel，用于存储分离后的各个颜色通道
    vector<Mat> all_channel;
    // 使用 split 函数将图像的三个颜色通道（蓝、绿、红）分离出来，存储到 all_channel 向量中
    split(src,all_channel); //split函数将图像的三通道分别提取出来，放到all_channel数组里面
    
    // 定义直方图的 bin 数量，即直方图的区间数量
    const int bin = 256;
    // 定义直方图的取值范围，从 0 到 255
    float bin_range[2] = { 0,255 };
    // 定义一个指针数组 ranges，方便后续 calcHist 函数的参数传递
    const float* ranges[1] = { bin_range };//这样做只是方便下面clacHist函数的传参
    
    // 定义三个 Mat 对象，分别用于存储蓝色、绿色和红色通道的直方图数据
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    
    // 计算蓝色通道的直方图数据
    calcHist(&all_channel[0], 1, 0, Mat(), b_hist, 1, &bin, ranges, true, false);
    // 计算绿色通道的直方图数据
    calcHist(&all_channel[1], 1, 0, Mat(), g_hist, 1, &bin, ranges, true, false);
    // 计算红色通道的直方图数据
    calcHist(&all_channel[2], 1, 0, Mat(), r_hist, 1, &bin, ranges, true, false);

    // 设置直方图画布的宽度
    int hist_w = 512;  
    // 设置直方图画布的高度
    int hist_h = 400;
    // 计算直方图中每个 bin 的宽度，使用 cvRound 函数进行四舍五入
    int bin_w = cvRound((double)hist_w/bin); //设置直方图中每一点的步长，通过hist_w/bin计算得出。cvRound()函数是“四舍五入”的作用。
    // 创建一个全黑的画布，用于绘制直方图
    Mat hist_canvas = Mat::zeros(hist_h, hist_w, CV_8UC3);
    // 对蓝色通道的直方图数据进行归一化处理，使其值范围在 0 到 255 之间
    normalize(b_hist, b_hist, 0, 255, NORM_MINMAX, -1, Mat());
    // 对绿色通道的直方图数据进行归一化处理，使其值范围在 0 到 255 之间
    normalize(g_hist, g_hist, 0, 255, NORM_MINMAX, -1, Mat());
    // 对红色通道的直方图数据进行归一化处理，使其值范围在 0 到 255 之间
    normalize(r_hist, r_hist, 0, 255, NORM_MINMAX, -1, Mat());
    // 定义一个 Mat 对象，用于临时存储直方图数据
    Mat hist;
    // 定义一个变量，用于存储直方图的最大值
    double max_val;
    // 计算直方图的最大值
    minMaxLoc(hist, 0, &max_val, 0, 0);//计算直方图的最大像素值
    // 遍历每个 bin，绘制直方图
    for (int i = 1; i < 256; i++)
    {
        // 绘制蓝色分量的直方图，使用蓝色线条
        line(hist_canvas, Point((i - 1)*bin_w, hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point((i)*bin_w, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0),2);
        // 绘制绿色分量的直方图，使用绿色线条
        line(hist_canvas, Point((i - 1)*bin_w, hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point((i)*bin_w, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0),2);
        // 绘制红色分量的直方图，使用红色线条
        line(hist_canvas, Point((i - 1)*bin_w, hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point((i)*bin_w, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255),2);
    }
    // 将绘制好的直方图保存为文件
    imwrite("lena_hist.jpg", hist_canvas);
    // 程序正常退出，返回 0
    return 0;
}

