#pragma once
#include <opencv2/opencv.hpp>

class QuickDemo {
public:
    void colorSpace_demo(cv::Mat& image);//颜色空间转换示例声明
    void mat_creation_demo(cv::Mat& image);//Mat创建示例声明
    void pixel_visit_demo(cv::Mat& image);//像素访问示例声明
    void operation_demo(cv::Mat& image);//图像运算示例声明
    void trackbar_demo(cv::Mat& image);//Trackbar示例声明
    void key_demo(cv::Mat& image);//键盘事件示例声明
    void color_style_demo(cv::Mat& image);//颜色风格转换示例声明
    void bitwise_demo(cv::Mat& image1,cv::Mat& image2);//位运算示例声明
    void channels_demo(cv::Mat& image);//通道分离与合并示例声明
    void inrange_demo(cv::Mat& image);//图像区域操作示例声明
    void pixel_statistics_demo(cv::Mat& image);//像素统计示例声明
    void drawing_demo(cv::Mat& image);//图像绘制示例声明
    void random_draw_demo(cv::Mat& image);//随机图形绘制示例声明
    void polyline_demo(cv::Mat& image);//多边形绘制示例声明
    void mouse_draw_demo(cv::Mat& image);//鼠标交互绘制示例声明
    void norm_demo(cv::Mat& image1);//转换类型事例声明
    void resize_demo(cv::Mat& image);//图像缩放示例声明
    void flip_demo(cv::Mat& image);//图像翻转示例声明
    void rotate_demo(cv::Mat& image);//图像旋转示例声明
    void video_demo();//视频处理示例声明
    void histogram_demo(cv::Mat& image);//直方图示例声明
    void histogram_2d_demo(cv::Mat& image);//二维直方图示例声明
};