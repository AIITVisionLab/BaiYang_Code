#include "quickdemo.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
int main()
{
    cv::Mat image =cv::imread("/home/jokerbai/文档/hhhhh/code/opencv/test.jpg");
    if(image.empty())
    {
        std::cout<<"Could not open or find the image"<<std::endl;
        return -1;
    }
    cv::imshow("original image",image);
    QuickDemo qd;
    qd.histogram_2d_demo(image);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
