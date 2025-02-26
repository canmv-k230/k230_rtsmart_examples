#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

/**
 * @brief 检测并绘制图像中的人脸和眼睛
 * 
 * 此函数使用级联分类器检测图像中的人脸和眼睛，并在原始图像上绘制检测结果。
 * 
 * @param img 输入的图像，检测和绘制操作将在该图像上进行。
 * @param cascade 用于检测人脸的级联分类器。
 * @param nestedCascade 用于检测眼睛的级联分类器。
 * @param scale 图像缩小的比例，用于加快检测速度。
 * @param tryflip 是否翻转图像进行额外的检测。
 */
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale, bool tryflip );

/**
 * @brief 主函数，程序的入口点
 * 
 * 此函数读取图像，加载人脸和眼睛的级联分类器，调用 detectAndDraw 函数进行检测和绘制，最后返回 0 表示程序正常结束。
 * 
 * @return int 程序退出状态码，0 表示正常退出。
 */
int main()
{
    // 定义一个 Mat 对象，用于存储读取的图像帧
    Mat frame;
    // 定义一个 Mat 对象，用于存储边缘检测结果（此处未使用）
    Mat edges;

    // 定义两个级联分类器对象，分别用于人脸和眼睛的检测
    CascadeClassifier cascade, nestedCascade;
    // 加载人脸检测的级联分类器模型
    cascade.load("./haarcascade_frontalface_alt.xml");
    // 加载眼睛检测的级联分类器模型
    nestedCascade.load("./haarcascade_eye.xml");
    // 从文件中读取图像并存储到 frame 中
    frame = imread("./1.bmp");
    // 调用 detectAndDraw 函数进行人脸和眼睛的检测与绘制
    detectAndDraw( frame, cascade, nestedCascade, 2, 0);
    // 程序正常结束，返回 0
    return 0;
}

/**
 * @brief 检测并绘制图像中的人脸和眼睛
 * 
 * 此函数使用级联分类器检测图像中的人脸和眼睛，并在原始图像上绘制检测结果。
 * 
 * @param img 输入的图像，检测和绘制操作将在该图像上进行。
 * @param cascade 用于检测人脸的级联分类器。
 * @param nestedCascade 用于检测眼睛的级联分类器。
 * @param scale 图像缩小的比例，用于加快检测速度。
 * @param tryflip 是否翻转图像进行额外的检测。
 */
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale, bool tryflip)
{
    // 初始化计数器
    int i = 0;
    // 建立用于存放人脸的向量容器
    vector<Rect> faces, faces2;
    // 定义一些颜色，用来标示不同的人脸
    const static Scalar colors[] = {
        CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    // 建立缩小的图片，加快检测速度
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1);
    // 转成灰度图像，Harr特征基于灰度图
    cvtColor(img, gray, COLOR_BGR2GRAY );
    // 改变图像大小，使用双线性差值
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    // 变换后的图像进行直方图均值化处理
    equalizeHist(smallImg, smallImg);
    // 检测人脸
    // detectMultiScale函数中smallImg表示的是要检测的输入图像为smallImg，faces表示检测到的人脸目标序列，1.1表示
    // 每次图像尺寸减小的比例为1.1，2表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大
    // 小都可以检测到人脸),CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的
    // 最小最大尺寸
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        |CASCADE_SCALE_IMAGE
        ,Size(30, 30));
    // 如果使能，翻转图像继续检测
    if(tryflip)
    {
        // 水平翻转图像
        flip(smallImg, smallImg, 1);
        // 在翻转后的图像上进行人脸检测
        cascade.detectMultiScale( smallImg, faces2,
            1.1, 2, 0
            |CASCADE_SCALE_IMAGE
            ,Size(30, 30) );
        // 将翻转后检测到的人脸位置转换回原始图像的位置，并添加到 faces 向量中
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }

    // 遍历检测到的人脸
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        // 定义一个 Mat 对象，用于存储人脸区域的图像
        Mat smallImgROI;
        // 定义一个向量，用于存储检测到的眼睛区域
        vector<Rect> nestedObjects;
        // 定义一个点，用于表示圆心
        Point center;
        // 从预定义的颜色数组中选择一种颜色来绘制人脸
        Scalar color = colors[i%8];
        // 定义圆的半径
        int radius;

        // 计算人脸的宽高比
        double aspect_ratio = (double)r->width/r->height;
        // 如果宽高比在合理范围内
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            // 计算人脸中心的坐标，并根据缩放比例还原到原始图像尺寸
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            // 计算人脸的半径，并根据缩放比例还原到原始图像尺寸
            radius = cvRound((r->width + r->height)*0.25*scale);
            // 在原始图像上绘制圆形来标记人脸
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            // 如果宽高比不在合理范围内，绘制矩形来标记人脸
            rectangle( img, cv::Point(cvRound(r->x*scale), cvRound(r->y*scale)),
            Point(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
            color, 3, 8, 0);
        // 如果眼睛检测的级联分类器为空，则跳过眼睛检测
        if( nestedCascade.empty() )
            continue;
        // 提取人脸区域的图像
        smallImgROI = smallImg(*r);
        // 同样方法检测人眼
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            |CASCADE_SCALE_IMAGE
            ,Size(30, 30) );
        // 遍历检测到的眼睛区域
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            // 计算眼睛中心的坐标，并根据缩放比例还原到原始图像尺寸
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            // 计算眼睛的半径，并根据缩放比例还原到原始图像尺寸
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            // 在原始图像上绘制圆形来标记眼睛
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    imwrite("./demo26_result.jpg", img);
}

