/**
 * @file DrawUtils.h
 * 绘制工具：在图像上画检测结果
 */

#ifndef DRAW_UTILS_H
#define DRAW_UTILS_H

#include "Detection.h"
#include "ClassLabels.h"
#include <QImage>
#include <QPainter>
#include <opencv2/core.hpp>

namespace yolo {

/**
 * 绘制样式配置
 */
struct DrawStyle {
    int lineWidth = 2;              ///< 线条宽度
    int fontSize = 12;              ///< 字体大小
    float labelOpacity = 0.7f;      ///< 标签背景透明度
    bool showLabel = true;          ///< 是否显示标签
    bool showConfidence = true;     ///< 是否显示置信度
    bool showClassName = true;      ///< 是否显示类别名
    bool fillBox = false;           ///< 是否填充边界框
    float fillOpacity = 0.2f;       ///< 填充透明度
    bool showKeypoints = true;      ///< 是否显示关键点
    bool showSkeleton = true;       ///< 是否显示骨架
    int keypointRadius = 5;         ///< 关键点半径
    bool showMask = true;           ///< 是否显示分割掩码
    float maskOpacity = 0.5f;       ///< 掩码透明度
    bool showOBB = true;            ///< 是否显示旋转框
    QString fontFamily = "Arial";   ///< 字体
    QColor textColor = Qt::white;   ///< 文字颜色
    bool shadowEffect = true;       ///< 是否加阴影
};

/**
 * 绘制工具类
 */
class DrawUtils {
public:
    /**
     * 在 QImage 上绘制检测结果
     */
    static void drawDetections(QImage& image, 
                               const DetectionResult& detections,
                               const ClassLabels& labels,
                               const DrawStyle& style = DrawStyle());

    /**
        * 在 cv::Mat 上绘制检测结果
     */
    static void drawDetections(cv::Mat& image,
                               const DetectionResult& detections,
                               const ClassLabels& labels,
                               const DrawStyle& style = DrawStyle());

    /**
        * 绘制单个边界框
     */
    static void drawBoundingBox(QImage& image,
                                const Detection& detection,
                                const QColor& color,
                                const DrawStyle& style = DrawStyle());

    /**
        * 绘制旋转边界框
     */
    static void drawOBB(QImage& image,
                        const OrientedBoundingBox& obb,
                        const QColor& color,
                        int lineWidth = 2);

    /**
        * 绘制关键点和骨架
     */
    static void drawKeypoints(QImage& image,
                              const QVector<Keypoint>& keypoints,
                              const QColor& color,
                              const DrawStyle& style = DrawStyle());

    /**
        * 绘制分割掩码
     */
    static void drawMask(QImage& image,
                         const SegmentMask& mask,
                         const BoundingBox& bbox,
                         const QColor& color,
                         float opacity = 0.5f);

    /**
        * 绘制标签
     */
    static void drawLabel(QPainter& painter,
                          const QString& text,
                          const QPointF& position,
                          const QColor& bgColor,
                          const DrawStyle& style = DrawStyle());

    /**
        * 绘制统计信息
     */
    static void drawStatistics(QImage& image,
                               const DetectionResult& result,
                               const ClassLabels& labels);

    /**
        * 绘制 FPS
     */
    static void drawFPS(QImage& image, double fps, const QPointF& position = QPointF(10, 30));

    /**
        * cv::Mat 转 QImage（不带检测结果）
     */
    static QImage matToQImage(const cv::Mat& mat);

    /**
        * cv::Mat 转 QImage 并绘制
     */
    static QImage matToQImageWithDetections(const cv::Mat& mat,
                                            const DetectionResult& detections,
                                            const ClassLabels& labels,
                                            const DrawStyle& style = DrawStyle());

private:
    /**
        * 获取姿态骨架颜色
     */
    static QColor getSkeletonColor(int limbIndex);

    /**
        * 获取关键点颜色
     */
    static QColor getKeypointColor(int keypointIndex);
};

} // namespace yolo

#endif // DRAW_UTILS_H
