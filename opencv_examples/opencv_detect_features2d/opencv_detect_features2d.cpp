#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;

/**
 * @brief 主函数，程序的入口点
 * 
 * 此函数的主要功能是读取一张图像，使用 FAST 特征检测器检测图像中的特征点，
 * 并将检测到的特征点绘制在原图像上，最后将处理后的图像保存为文件。
 * 
 * @return int 程序退出状态码，0 表示正常退出
 */
int main()
{
  // 定义一个 Mat 对象，用于存储从文件中读取的图像
  Mat image;
  // 从文件中读取图像，并将其存储在 image 中
  image = imread("./test.jpg");
  // 定义一个存储关键点的向量，用于存储检测到的特征点
  // vector of keyPoints
  std::vector<KeyPoint> keyPoints;
  // 创建一个 FAST 特征检测器对象，阈值设置为 40
  Ptr<FeatureDetector> fast=FastFeatureDetector::create(40);
  // 调用 FAST 特征检测器的 detect 方法，对图像进行特征点检测
  // 检测结果存储在 keyPoints 向量中
  // feature point detection
  fast->detect(image,keyPoints);
  // 在原图像上绘制检测到的特征点
  // 参数依次为：原图像、关键点向量、输出图像、绘制颜色（白色）、绘制标志（覆盖原图像）
  drawKeypoints(image, keyPoints, image, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
  // 将绘制好特征点的图像保存为文件
  imwrite("./demo24_fast_feature.jpg", image);
  // 程序正常结束，返回 0
  return 0;
}
