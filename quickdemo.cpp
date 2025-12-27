#include "quickdemo.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <opencv2/videoio.hpp>
#include <vector>

namespace {
constexpr const char *kTrackbarWindow = "trackbar_demo";
constexpr const char *kTrackbarBrightness = "brightness";
constexpr const char *kTrackbarContrast = "contrast";

void on_trackbar(int, void *userdata) {
  auto *image = static_cast<cv::Mat *>(userdata);
  if (image == nullptr || image->empty()) {
    return;
  }

  try {
    int brightness = cv::getTrackbarPos(kTrackbarBrightness, kTrackbarWindow);
    int contrast = cv::getTrackbarPos(kTrackbarContrast, kTrackbarWindow);

    const double alpha = std::max(0.0, contrast / 100.0);
    const double beta = static_cast<double>(brightness - 100);

    cv::Mat dst;
    image->convertTo(dst, -1, alpha, beta);
    cv::imshow(kTrackbarWindow, dst);
  } catch (const cv::Exception &) {
    return;
  }
}
} // namespace

// 颜色空间转换示例
void QuickDemo::colorSpace_demo(cv::Mat &image) {
  cv::Mat grey, hsv;
  cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  imshow("hsv", hsv);
  imshow("grey", grey);
  imwrite("/home/jokerbai/文档/hhhhh/code/opencv/grey.jpg", grey);
  imwrite("/home/jokerbai/文档/hhhhh/code/opencv/hsv.jpg", hsv);
}

// Mat创建示例
void QuickDemo::mat_creation_demo(cv::Mat &image) {
  // cv::Mat m1,m2;
  // m1=image.clone();
  // image.copyTo(m2);

  // 创建空白
  cv::Mat m3 = cv::Mat::zeros(cv::Size(4, 4), CV_8UC3);
  m3 = cv::Scalar(255, 0, 0);
  std::cout << "m3:" << std::endl << m3 << std::endl;
  cv::imshow("m3", m3);
}

// 像素访问示例
void QuickDemo::pixel_visit_demo(cv::Mat &image) {
  int w = image.cols;
  int h = image.rows;
  int dims = image.channels();
  // for(int row=0;row<h;row++)
  // {
  //     for(int col=0;col<w;col++)
  //     {
  //         if(dims==1)//灰度图
  //         {
  //             int pv=image.at<uchar>(row,col);
  //             image.at<uchar>(row,col)=255-pv;//反色处理

  //         }
  //         if(dims==3)//彩色图
  //         {
  //             cv::Vec3b bgr=image.at<cv::Vec3b>(row,col);
  //             image.at<cv::Vec3b>(row,col)[0]=255-bgr[0];
  //             image.at<cv::Vec3b>(row,col)[1]=255-bgr[1];
  //             image.at<cv::Vec3b>(row,col)[2]=255-bgr[2];
  //         }
  //     }
  // }

  // 指针访问（避免越界：col 应该遍历 w；并且按 channel 正确索引）
  for (int row = 0; row < h; row++) {
    uchar *current_row = image.ptr<uchar>(row);
    for (int col = 0; col < w; col++) {
      if (dims == 1) {
        current_row[col] = static_cast<uchar>(255 - current_row[col]);
      } else if (dims == 3) {
        uchar *p = current_row + col * 3;
        p[0] = static_cast<uchar>(255 - p[0]);
        p[1] = static_cast<uchar>(255 - p[1]);
        p[2] = static_cast<uchar>(255 - p[2]);
      }
    }
  }
  cv::imshow("pixel_visit_demo", image);
  cv::imwrite("/home/jokerbai/文档/hhhhh/code/opencv/pixel_visit_demo.jpg",
              image);
}

// 图像运算示例
void QuickDemo::operation_demo(cv::Mat &image) {
  // cv::Mat dst;
  // dst=image+cv::Scalar(100,100,100);//图像亮度增加
  // cv::imshow("operation_demo",dst);
  // cv::imwrite("/home/jokerbai/文档/hhhhh/code/opencv/operation_demo.jpg",dst);

  // 简易操作
  cv::Mat dst = cv::Mat::zeros(image.size(), image.type());
  cv::Mat m = cv::Mat::zeros(image.size(), image.type());
  m = cv::Scalar(50, 50, 50);
  cv::add(image, m, dst);
  cv::imshow("operation_demo", dst);
  cv::imwrite("/home/jokerbai/文档/hhhhh/code/opencv/operation_demo.jpg", dst);
  // 减法类似subtract
  // 乘法类似multiply
  // 除法类似divide
}

// Trackbar示例
void QuickDemo::trackbar_demo(cv::Mat &image) {
  cv::namedWindow(kTrackbarWindow, cv::WINDOW_AUTOSIZE);
  cv::createTrackbar(kTrackbarBrightness, kTrackbarWindow, nullptr, 200,
                     on_trackbar, (void *)&image);
  cv::createTrackbar(kTrackbarContrast, kTrackbarWindow, nullptr, 200,
                     on_trackbar, (void *)&image);
  cv::setTrackbarPos(kTrackbarBrightness, kTrackbarWindow, 100); // 亮度
  cv::setTrackbarPos(kTrackbarContrast, kTrackbarWindow, 100);   // 对比度
  on_trackbar(0, (void *)&image);
}

// 键盘事件示例
void QuickDemo::key_demo(cv::Mat &image) {
  cv::namedWindow("key_demo", cv::WINDOW_AUTOSIZE);
  cv::Mat dst = image.clone();
  cv::imshow("key_demo", dst);

  while (true) {
    int c = cv::waitKey(100);
    if (c == 27) {
      std::cout << "You pressed ESC,exit" << std::endl;
      break;
    }

    if (c == 49) {
      std::cout << "You pressed 1" << std::endl;
      cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
    } else if (c == 50) {
      std::cout << "You pressed 2" << std::endl;
      cv::cvtColor(image, dst, cv::COLOR_BGR2HSV);
    } else if (c == 51) {
      std::cout << "You pressed 3" << std::endl;
      cv::Mat m(image.size(), image.type(), cv::Scalar(50, 50, 5));
      cv::add(image, m, dst);
    } else if (c > 0) {
      std::cout << "You pressed other key" << std::endl;
      dst = image;
    }

    cv::imshow("key_demo", dst);
  }
}

void QuickDemo::color_style_demo(cv::Mat &image) {
  std::vector<int> colormap = {
      cv::COLORMAP_AUTUMN, cv::COLORMAP_BONE,    cv::COLORMAP_JET,
      cv::COLORMAP_WINTER, cv::COLORMAP_RAINBOW, cv::COLORMAP_OCEAN,
      cv::COLORMAP_SUMMER, cv::COLORMAP_SPRING,  cv::COLORMAP_COOL,
      cv::COLORMAP_HSV,    cv::COLORMAP_PINK,    cv::COLORMAP_HOT,
      cv::COLORMAP_PARULA};
  cv::namedWindow("颜色风格", cv::WINDOW_AUTOSIZE);
  cv::Mat dst;
  int index = 0;
  int delay = 1000;
  while (true) {
    cv::applyColorMap(image, dst, colormap[index]);

    cv::imshow("颜色风格", dst);
    int key = cv::waitKey(delay);
    if (key == 27) {
      std::cout << "按ESC退出演示" << std::endl;
      break;
    }
    if (key == 32) {
      int pause_key = cv::waitKey(0);
      if (pause_key == 27)
        break;
    }
    index = (index + 1) % colormap.size();
  }
}

void QuickDemo::bitwise_demo(cv::Mat &image, cv::Mat &image2) {
  cv::Mat m1 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
  cv::Mat m2 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
  cv::rectangle(m1, cv::Rect(100, 100, 80, 80), cv::Scalar(255, 255, 0), -1, 8,
                0);
  cv::rectangle(m2, cv::Rect(150, 150, 80, 80), cv::Scalar(0, 255, 255), -1, 8,
                0);
  cv::imshow("m1", m1);
  cv::imshow("m2", m2);
  cv::Mat dst_and, dst_or, dst_xor, dst_not1, dst_not2;
  cv::bitwise_and(m1, m2, dst_and);
  cv::bitwise_or(m1, m2, dst_or);
  cv::bitwise_xor(m1, m2, dst_xor);
  cv::bitwise_not(m1, dst_not1);
  cv::bitwise_not(m2, dst_not2);

  cv::imshow("bitwise_and", dst_and);
  cv::imshow("bitwise_or", dst_or);
  cv::imshow("bitwise_xor", dst_xor);
  cv::imshow("bitwise_not1", dst_not1);
  cv::imshow("bitwise_not2", dst_not2);
  cv::waitKey(0);
}

void QuickDemo::channels_demo(cv::Mat &image) {
  std::vector<cv::Mat> mv;
  cv::split(image, mv); // 通道分离
  // cv::imshow("blue channel",mv[0]);
  // cv::imshow("green channel",mv[1]);
  // cv::imshow("red channel",mv[2]);

  cv::Mat dst;
  mv[0] = 0;
  cv::merge(mv, dst); // 通道合并
  cv::imshow("after merge", dst);
  int from_to[] = {0, 2, 1, 1, 2, 0}; // 交换蓝色和红色通道
  cv::mixChannels(&image, 1, &dst, 1, from_to, 3);
  cv::imshow("after mixChannels", dst);

  cv::waitKey(0);
}

void QuickDemo::inrange_demo(cv::Mat &image) {
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  // OpenCV HSV: H=[0,179], S/V=[0,255]
  // 这里做一个最小示例：提取红色区域（红色跨越 0/179，需要两个区间）
  cv::Mat mask1, mask2, mask;
  cv::Scalar lower1(0, 43, 46), upper1(10, 255, 255);
  cv::Scalar lower2(160, 43, 46), upper2(179, 255, 255);
  cv::inRange(hsv, lower1, upper1, mask1);
  cv::inRange(hsv, lower2, upper2, mask2);
  cv::bitwise_or(mask1, mask2, mask);

  cv::Mat dst = cv::Mat::zeros(image.size(), image.type());
  image.copyTo(dst, mask);
  cv::imshow("mask", mask);
  cv::imshow("inrange demo - red color", dst);
  cv::waitKey(0);
}

void QuickDemo::pixel_statistics_demo(cv::Mat &image) {
  double minv = 0.0, maxv = 0.0;
  cv::Point minLoc, maxLoc;
  std::vector<cv::Mat> mv;
  cv::split(image, mv);
  for (int i = 0; i != mv.size(); i++) {
    cv::minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, cv::Mat());
    std::cout << "channel " << i << ":" << std::endl;
    std::cout << "  min value: " << minv << " at " << minLoc << std::endl;
    std::cout << "  max value: " << maxv << " at " << maxLoc << std::endl;
    cv::Mat mean, stddev;
    cv::meanStdDev(mv[i], mean, stddev);
    std::cout << "  mean: " << mean << std::endl;
  }
  cv::Mat mean, stddev;
  cv::meanStdDev(image, mean, stddev);
  std::cout << "mean: " << mean << std::endl;
  std::cout << "stddev: " << stddev << std::endl;
}

void QuickDemo::drawing_demo(cv::Mat &image) {
  cv::Rect rect(200, 200, 200, 150);
  cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2, 8, 0);
  cv::circle(image, cv::Point(400, 400), 50, cv::Scalar(255, 0, 0), -1, 8, 0);
  cv::line(image, cv::Point(100, 100), cv::Point(500, 100),
           cv::Scalar(0, 0, 255), 3, 8, 0);
  cv::putText(image, "Hello OpenCV", cv::Point(150, 50),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2, 8,
              false);
  cv::imshow("drawing_demo", image);
  cv::imwrite("/home/jokerbai/文档/hhhhh/code/opencv/drawing_demo.jpg", image);
  cv::waitKey(0);
}

void QuickDemo::random_draw_demo(cv::Mat &image) {
  cv::Mat canvas = cv::Mat::zeros(image.size(), image.type());
  cv::RNG rng(12345);
  while (true) {
    int c = cv::waitKey(10);
    if (c == 27) {
      std::cout << "Press ESC to exit" << std::endl;
      break;
    }
    int x1 = rng.uniform(0, canvas.cols);
    int y1 = rng.uniform(0, canvas.rows);
    int x2 = rng.uniform(0, canvas.cols);
    int y2 = rng.uniform(0, canvas.rows);
    int b = rng.uniform(0, 255);
    int g = rng.uniform(0, 255);
    int r = rng.uniform(0, 255);
    canvas = cv::Scalar(0, 0, 0); // 清空画布
    cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(b, g, r),
             2, 8, 0);
    cv::imshow("random_draw_demo", canvas);
  }
}

void QuickDemo::polyline_demo(cv::Mat &image) {
  cv::Mat canvas = cv::Mat::zeros(image.size(), image.type());
  cv::Point p1(100, 100);
  cv::Point p2(350, 100);
  cv::Point p3(450, 280);
  cv::Point p4(320, 450);
  cv::Point p5(80, 400);
  ;
  std::vector<cv::Point> pts;
  pts.push_back(p1);
  pts.push_back(p2);
  pts.push_back(p3);
  pts.push_back(p4);
  pts.push_back(p5);
  // cv::polylines(canvas,pts,true,cv::Scalar(0,255,0),8,8,0);
  // cv::fillPoly(canvas,pts,cv::Scalar(255,0,0),cv::LINE_AA,0);
  std::vector<std::vector<cv::Point>> contours;
  cv::drawContours(canvas, contours, -1, cv::Scalar(255, 0, 0), -1);
  cv::imshow("polyline_demo", canvas);
  cv::waitKey(0);
}



//鼠标画框,每次画框清除前一个框
static void mouse_callback(int event, int x, int y, int flags, void *param) {
  static cv::Point pt1, pt2;
  cv::Mat &image = *(cv::Mat *)param;
  cv::Mat temp_image = image.clone();

  if (event == cv::EVENT_LBUTTONDOWN) {
    pt1 = cv::Point(x, y);
  } else if (event == cv::EVENT_LBUTTONUP) {
    pt2 = cv::Point(x, y);
    cv::rectangle(temp_image, pt1, pt2, cv::Scalar(0, 255, 0), 2, 8, 0);
    cv::imshow("mouse_draw_demo", temp_image);
  } else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
    pt2 = cv::Point(x, y);
    cv::rectangle(temp_image, pt1, pt2, cv::Scalar(0, 255, 0), 2, 8, 0);
    cv::imshow("mouse_draw_demo", temp_image);
  }
}

void QuickDemo::mouse_draw_demo(cv::Mat &image) {
  cv::namedWindow("mouse_draw_demo", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("mouse_draw_demo", mouse_callback, (void *)&image);
  cv::imshow("mouse_draw_demo", image);
  cv::waitKey(0);
}

void QuickDemo::norm_demo(cv::Mat &image1) {
  cv::Mat image2;
  image1.convertTo(image2, CV_32F, 1.0 / 255.0);

  std::cout << image1.type() << std::endl;
  cv::normalize(image2, image2, 0.0, 1.0, cv::NORM_MINMAX);
  std::cout << image2.type() << std::endl;

  cv::Mat show;
  image2.convertTo(show, CV_8U, 255.0);
  cv::imshow("norm_demo", show);
  cv::waitKey(0);
}


void QuickDemo::resize_demo(cv::Mat &image) {
  cv::Mat zoomin, zoomout;
  int h=image.rows;
  int w=image.cols;
  cv::resize(image, zoomin, cv::Size(w*0.5,h*0.5),0,0,cv::INTER_LINEAR);//缩小,
  cv::resize(image, zoomout, cv::Size(w*1.5,h*1.5),0,0,cv::INTER_LINEAR);//放大
  cv::imshow("zoomin", zoomin);
  cv::imshow("zoomout", zoomout);
}


void QuickDemo::flip_demo(cv::Mat &image) {
  cv::Mat flip_x, flip_y, flip_xy;
  cv::flip(image,flip_x,0);   // 关于x轴翻转
  cv::flip(image, flip_y, 1);  // 关于y轴翻转
  cv::flip(image, flip_xy, -1); // 关于xy轴翻转
  cv::imshow("flip_x", flip_x);
  cv::imshow("flip_y", flip_y);
  cv::imshow("flip_xy", flip_xy);
}

void QuickDemo::rotate_demo(cv::Mat&image0){
  cv::Mat dst,M;
  int w=image0.cols;
  int h=image0.rows;
  M=cv::getRotationMatrix2D(cv::Point2f(w/2,h/2),45,1.0);//旋转矩阵
  double cos=abs(M.at<double>(0,0));
  double sin=abs(M.at<double>(0,1));
  int newh=int(h*cos+w*sin);
  int neww=int(w*cos+h*sin);
  M.at<double>(0,2)+=(neww/2-w/2);
  M.at<double>(1,2)+=(newh/2-h/2);
  
  cv::warpAffine(image0,dst,M,cv::Size(neww,newh),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(255,255,0));
  cv::imshow("rotate_demo",dst);
}


void QuickDemo::video_demo(){
  cv::VideoCapture capture("/home/jokerbai/文档/hhhhh/code/opencv/1.mp4");//除了打开默认设想图像头，还可以传入视频文件路径打开视频文件
  int freame_width=capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int frame_height=capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int count =capture.get(cv::CAP_PROP_FRAME_COUNT);
  double fps=capture.get(cv::CAP_PROP_FPS);
  std::cout<<"frame width:"<<freame_width<<std::endl;
  std::cout<<"frame height:"<<frame_height<<std::endl;
  std::cout<<"frame count:"<<count<<std::endl;
  std::cout<<"fps:"<<fps<<std::endl;

  if(!capture.isOpened()){
    std::cout<<"Could not open the camera"<<std::endl;
    return;
  }
  
  cv::Mat frame;
  while(true)
  {
    capture>>frame;
    cv::flip(frame, frame, 1);
    if(frame.empty()){
      std::cout<<"Could not read a frame from the camera"<<std::endl;
      break;
    }
    cv::imshow("video_demo",frame);
    int c=cv::waitKey(30);
    if(c==27){
      std::cout<<"Press ESC to exit"<<std::endl;
      break;
    }
  }
  capture.release();
}

void QuickDemo::histogram_demo(cv::Mat&image){
  std::vector<cv::Mat> bgr_planes;
  //三通道分离
  cv::split(image,bgr_planes);
  const int channels[]={0};
  const int bin[1]={256};//256个直方图柱
  float hrange[2]={0,256};//像素值范围
  const float* ranges[1]={hrange};//像素值范围指针
  cv::Mat b_hist,g_hist,r_hist;
  //计算各通道直方图
  cv::calcHist(&bgr_planes[0],1,0,cv::Mat(),b_hist,1,bin,ranges,true,false);
  cv::calcHist(&bgr_planes[1],1,0,cv::Mat(),g_hist,1,bin,ranges,true,false);
  cv::calcHist(&bgr_planes[2],1,0,cv::Mat(),r_hist,1,bin,ranges,true,false);

  //显示直方图
  int hist_w=512;int hist_h=400;
  int bin_w=cvRound((double)hist_w/bin[0]);
  cv::Mat histImage=cv::Mat::zeros(hist_h,hist_w,CV_8UC3);
  //归一化直方图数据
  cv::normalize(b_hist,b_hist,0,histImage.rows,cv::NORM_MINMAX);
  cv::normalize(g_hist,g_hist,0,histImage.rows,cv::NORM_MINMAX);
  cv::normalize(r_hist,r_hist,0,histImage.rows,cv::NORM_MINMAX);
  //绘制直方图曲线
  for(int i=1;i<bin[0];i++)
  {
    cv::line(histImage,cv::Point(bin_w*(i-1),hist_h-cvRound(b_hist.at<float>(i-1))),
             cv::Point(bin_w*(i),hist_h-cvRound(b_hist.at<float>(i))),
             cv::Scalar(255,0,0),2,8,0);
    cv::line(histImage,cv::Point(bin_w*(i-1),hist_h-cvRound(g_hist.at<float>(i-1))),
             cv::Point(bin_w*(i),hist_h-cvRound(g_hist.at<float>(i))),
             cv::Scalar(0,255,0),2,8,0);
    cv::line(histImage,cv::Point(bin_w*(i-1),hist_h-cvRound(r_hist.at<float>(i-1))),
             cv::Point(bin_w*(i),hist_h-cvRound(r_hist.at<float>(i))),
             cv::Scalar(0,0,255),2,8,0);
  }
  cv::imshow("histogram_demo",histImage);

}


void QuickDemo::histogram_2d_demo(cv::Mat&image){
  cv::Mat hsv,hs_hist;
  cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
  int hbins=30,sbins=32;
  int hist_bins[]={hbins,sbins};
  float h_range[]={0,180};
  float s_range[]={0,256};
  const float* ranges[]={h_range,s_range};
  const float* hs_ranges[]={h_range,s_range};
  int hs_channels[]={0,1};
  cv::calcHist(&hsv,1,hs_channels,cv::Mat(),hs_hist,2,hist_bins,hs_ranges,true,false);
  //归一化
  double maxVal=0;
  cv::minMaxLoc(hs_hist,0,&maxVal,0,0);
  int scale=10;
  cv::Mat hist2d_image=cv::Mat::zeros(sbins*scale,hbins*scale,CV_8UC3);
  for(int h=0;h<hbins;h++)
  {
    for(int s=0;s<sbins;s++)
    {
      float binVal=hs_hist.at<float>(h,s);
      {
        float binVal=hs_hist.at<float>(h,s);
        int intensity=cvRound(binVal*255/maxVal);
        cv::rectangle(hist2d_image,cv::Point(h*scale,s*scale),
                      cv::Point((h+1)*scale-1,(s+1)*scale-1),
                      cv::Scalar::all(intensity),-1);

      }
    }
  }
  cv::imshow("histogram_2d_demo",hist2d_image);
  cv::imwrite("/home/jokerbai/文档/hhhhh/code/opencv/histogram_2d_demo.jpg",hist2d_image);

}