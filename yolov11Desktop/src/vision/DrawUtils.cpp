/**
 * @file DrawUtils.cpp
 * @brief 绘制工具实现
 */

#include "DrawUtils.h"
#include <QPainter>
#include <QFontMetrics>
#include <opencv2/imgproc.hpp>

namespace yolo {

void DrawUtils::drawDetections(QImage& image, 
                               const DetectionResult& detections,
                               const ClassLabels& labels,
                               const DrawStyle& style)
{
    if (image.isNull() || detections.count() == 0) {
        return;
    }

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::TextAntialiasing);

    QFont font(style.fontFamily, style.fontSize);
    font.setBold(true);
    painter.setFont(font);

    for (const auto& det : detections.detections()) {
        QColor color = labels.getClassColor(det.classId());
        
        // 绘制掩码 (分割任务)
        if (style.showMask && det.taskType() == TaskType::Segmentation) {
            drawMask(image, det.mask(), det.bbox(), color, style.maskOpacity);
        }

        // 绘制边界框或旋转边界框
        if (det.taskType() == TaskType::OBB && style.showOBB) {
            drawOBB(image, det.obb(), color, style.lineWidth);
        } else {
            const BoundingBox& bbox = det.bbox();
            QRectF rect = bbox.toQRectF();

            // 填充
            if (style.fillBox) {
                QColor fillColor = color;
                fillColor.setAlphaF(style.fillOpacity);
                painter.fillRect(rect, fillColor);
            }

            // 边框
            QPen pen(color, style.lineWidth);
            painter.setPen(pen);
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(rect);
        }

        // 绘制关键点 (姿态估计)
        if (style.showKeypoints && det.taskType() == TaskType::Pose) {
            drawKeypoints(image, det.keypoints(), color, style);
        }

        // 绘制标签
        if (style.showLabel) {
            QString labelText;
            if (style.showClassName) {
                labelText = labels.getClassName(det.classId());
            }
            if (style.showConfidence) {
                if (!labelText.isEmpty()) labelText += ": ";
                labelText += QString::number(det.confidence() * 100, 'f', 1) + "%";
            }

            if (!labelText.isEmpty()) {
                QPointF labelPos(det.bbox().x, det.bbox().y - 5);
                drawLabel(painter, labelText, labelPos, color, style);
            }
        }
    }

    painter.end();
}

void DrawUtils::drawDetections(cv::Mat& image,
                               const DetectionResult& detections,
                               const ClassLabels& labels,
                               const DrawStyle& style)
{
    if (image.empty() || detections.count() == 0) {
        return;
    }

    for (const auto& det : detections.detections()) {
        QColor qcolor = labels.getClassColor(det.classId());
        cv::Scalar color(qcolor.blue(), qcolor.green(), qcolor.red());

        const BoundingBox& bbox = det.bbox();
        cv::Rect rect(
            static_cast<int>(bbox.x),
            static_cast<int>(bbox.y),
            static_cast<int>(bbox.width),
            static_cast<int>(bbox.height)
        );

        // 确保rect在图像范围内
        rect &= cv::Rect(0, 0, image.cols, image.rows);

        // 填充
        if (style.fillBox) {
            cv::Mat overlay = image.clone();
            cv::rectangle(overlay, rect, color, -1);
            cv::addWeighted(overlay, style.fillOpacity, image, 1 - style.fillOpacity, 0, image);
        }

        // 边框
        cv::rectangle(image, rect, color, style.lineWidth);

        // 绘制关键点
        if (style.showKeypoints && det.taskType() == TaskType::Pose) {
            const auto& keypoints = det.keypoints();
            
            // 绘制骨架
            if (style.showSkeleton) {
                for (const auto& limb : CocoLabels::SKELETON) {
                    int idx1 = limb.first;
                    int idx2 = limb.second;
                    if (idx1 < keypoints.size() && idx2 < keypoints.size()) {
                        const auto& kp1 = keypoints[idx1];
                        const auto& kp2 = keypoints[idx2];
                        if (kp1.confidence > 0.5f && kp2.confidence > 0.5f) {
                            cv::line(image,
                                    cv::Point(static_cast<int>(kp1.x), static_cast<int>(kp1.y)),
                                    cv::Point(static_cast<int>(kp2.x), static_cast<int>(kp2.y)),
                                    color, 2);
                        }
                    }
                }
            }

            // 绘制关键点
            for (const auto& kp : keypoints) {
                if (kp.confidence > 0.5f) {
                    cv::circle(image,
                              cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                              style.keypointRadius, color, -1);
                }
            }
        }

        // 标签
        if (style.showLabel) {
            QString labelText;
            if (style.showClassName) {
                labelText = labels.getClassName(det.classId());
            }
            if (style.showConfidence) {
                if (!labelText.isEmpty()) labelText += ": ";
                labelText += QString::number(det.confidence() * 100, 'f', 1) + "%";
            }

            if (!labelText.isEmpty()) {
                std::string text = labelText.toStdString();
                int baseline;
                cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 
                                                    0.5, 1, &baseline);
                
                cv::Point textOrg(rect.x, rect.y - 5);
                if (textOrg.y < textSize.height) {
                    textOrg.y = rect.y + rect.height + textSize.height + 5;
                }

                // 背景
                cv::rectangle(image,
                             cv::Point(textOrg.x, textOrg.y - textSize.height - 2),
                             cv::Point(textOrg.x + textSize.width + 4, textOrg.y + 4),
                             color, -1);
                
                // 文字
                cv::putText(image, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 
                           0.5, cv::Scalar(255, 255, 255), 1);
            }
        }
    }
}

void DrawUtils::drawBoundingBox(QImage& image,
                                const Detection& detection,
                                const QColor& color,
                                const DrawStyle& style)
{
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    const BoundingBox& bbox = detection.bbox();
    QRectF rect = bbox.toQRectF();

    if (style.fillBox) {
        QColor fillColor = color;
        fillColor.setAlphaF(style.fillOpacity);
        painter.fillRect(rect, fillColor);
    }

    QPen pen(color, style.lineWidth);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(rect);

    painter.end();
}

void DrawUtils::drawOBB(QImage& image,
                        const OrientedBoundingBox& obb,
                        const QColor& color,
                        int lineWidth)
{
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    QPen pen(color, lineWidth);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);

    QVector<QPointF> corners = obb.getCorners();
    
    QPolygonF polygon;
    for (const auto& corner : corners) {
        polygon << corner;
    }
    
    painter.drawPolygon(polygon);
    painter.end();
}

void DrawUtils::drawKeypoints(QImage& image,
                              const QVector<Keypoint>& keypoints,
                              const QColor& color,
                              const DrawStyle& style)
{
    if (keypoints.isEmpty()) return;

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    // 绘制骨架
    if (style.showSkeleton) {
        for (int i = 0; i < CocoLabels::SKELETON.size(); ++i) {
            const auto& limb = CocoLabels::SKELETON[i];
            int idx1 = limb.first;
            int idx2 = limb.second;
            
            if (idx1 < keypoints.size() && idx2 < keypoints.size()) {
                const auto& kp1 = keypoints[idx1];
                const auto& kp2 = keypoints[idx2];
                
                if (kp1.confidence > 0.5f && kp2.confidence > 0.5f) {
                    QColor limbColor = getSkeletonColor(i);
                    painter.setPen(QPen(limbColor, 2));
                    painter.drawLine(QPointF(kp1.x, kp1.y), QPointF(kp2.x, kp2.y));
                }
            }
        }
    }

    // 绘制关键点
    for (int i = 0; i < keypoints.size(); ++i) {
        const auto& kp = keypoints[i];
        if (kp.confidence > 0.5f) {
            QColor kpColor = getKeypointColor(i);
            painter.setPen(Qt::NoPen);
            painter.setBrush(kpColor);
            painter.drawEllipse(QPointF(kp.x, kp.y), 
                               style.keypointRadius, style.keypointRadius);
        }
    }

    painter.end();
}

void DrawUtils::drawMask(QImage& image,
                         const SegmentMask& mask,
                         const BoundingBox& bbox,
                         const QColor& color,
                         float opacity)
{
    if (mask.data.isEmpty()) return;

    // 创建mask图像
    QImage maskImage(mask.width, mask.height, QImage::Format_ARGB32);
    maskImage.fill(Qt::transparent);

    QColor maskColor = color;
    maskColor.setAlphaF(opacity);

    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(x, y) > 0.5f) {
                maskImage.setPixelColor(x, y, maskColor);
            }
        }
    }

    // 缩放mask到bbox大小
    QImage scaledMask = maskImage.scaled(
        static_cast<int>(bbox.width),
        static_cast<int>(bbox.height),
        Qt::IgnoreAspectRatio,
        Qt::SmoothTransformation
    );

    // 绘制到原图
    QPainter painter(&image);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    painter.drawImage(QPointF(bbox.x, bbox.y), scaledMask);
    painter.end();
}

void DrawUtils::drawLabel(QPainter& painter,
                          const QString& text,
                          const QPointF& position,
                          const QColor& bgColor,
                          const DrawStyle& style)
{
    QFontMetrics fm(painter.font());
    QRect textRect = fm.boundingRect(text);
    
    // 调整位置
    QPointF pos = position;
    if (pos.y() - textRect.height() - 4 < 0) {
        pos.setY(position.y() + textRect.height() + 10);
    }

    // 背景
    QRectF bgRect(pos.x(), pos.y() - textRect.height() - 4,
                  textRect.width() + 8, textRect.height() + 4);
    
    QColor bg = bgColor;
    bg.setAlphaF(style.labelOpacity);
    painter.fillRect(bgRect, bg);

    // 文字阴影
    if (style.shadowEffect) {
        painter.setPen(QColor(0, 0, 0, 100));
        painter.drawText(QPointF(pos.x() + 5, pos.y() - 3), text);
    }

    // 文字
    painter.setPen(style.textColor);
    painter.drawText(QPointF(pos.x() + 4, pos.y() - 4), text);
}

void DrawUtils::drawStatistics(QImage& image,
                               const DetectionResult& result,
                               const ClassLabels& labels)
{
    QPainter painter(&image);
    painter.setRenderHint(QPainter::TextAntialiasing);

    QFont font("Arial", 11);
    painter.setFont(font);

    int y = 30;
    int x = 10;
    int lineHeight = 20;

    // 背景
    QRect bgRect(x - 5, y - 15, 200, lineHeight * 5);
    painter.fillRect(bgRect, QColor(0, 0, 0, 150));

    painter.setPen(Qt::white);

    // 检测数量
    painter.drawText(x, y, QString("Detections: %1").arg(result.count()));
    y += lineHeight;

    // 推理时间
    painter.drawText(x, y, QString("Inference: %1 ms").arg(result.inferenceTime(), 0, 'f', 1));
    y += lineHeight;

    // 总时间
    painter.drawText(x, y, QString("Total: %1 ms").arg(result.totalTime(), 0, 'f', 1));
    y += lineHeight;

    // FPS
    double fps = result.totalTime() > 0 ? 1000.0 / result.totalTime() : 0;
    painter.drawText(x, y, QString("FPS: %1").arg(fps, 0, 'f', 1));

    painter.end();
}

void DrawUtils::drawFPS(QImage& image, double fps, const QPointF& position)
{
    QPainter painter(&image);
    painter.setRenderHint(QPainter::TextAntialiasing);

    QFont font("Arial", 14, QFont::Bold);
    painter.setFont(font);

    QString text = QString("FPS: %1").arg(fps, 0, 'f', 1);
    
    // 背景
    QFontMetrics fm(font);
    QRect textRect = fm.boundingRect(text);
    QRectF bgRect(position.x() - 2, position.y() - textRect.height(),
                  textRect.width() + 10, textRect.height() + 4);
    painter.fillRect(bgRect, QColor(0, 0, 0, 150));

    // 颜色根据FPS变化
    QColor color;
    if (fps >= 30) {
        color = Qt::green;
    } else if (fps >= 15) {
        color = Qt::yellow;
    } else {
        color = Qt::red;
    }

    painter.setPen(color);
    painter.drawText(position, text);
    painter.end();
}

QImage DrawUtils::matToQImage(const cv::Mat& mat)
{
    if (mat.empty()) {
        return QImage();
    }

    // 转换为RGB格式
    cv::Mat rgb;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
    } else {
        rgb = mat;
    }

    QImage image(rgb.data, rgb.cols, rgb.rows, 
                static_cast<int>(rgb.step), QImage::Format_RGB888);
    return image.copy();  // 返回深拷贝
}

QImage DrawUtils::matToQImageWithDetections(const cv::Mat& mat,
                                            const DetectionResult& detections,
                                            const ClassLabels& labels,
                                            const DrawStyle& style)
{
    if (mat.empty()) {
        return QImage();
    }

    // 转换为QImage
    cv::Mat rgb;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    } else {
        rgb = mat;
    }

    QImage image(rgb.data, rgb.cols, rgb.rows, 
                static_cast<int>(rgb.step), QImage::Format_RGB888);
    QImage result = image.copy();

    // 绘制检测结果
    drawDetections(result, detections, labels, style);

    return result;
}

QColor DrawUtils::getSkeletonColor(int limbIndex)
{
    // 根据身体部位返回不同颜色
    static const QVector<QColor> colors = {
        QColor(255, 128, 0),   // 头部
        QColor(255, 128, 0),
        QColor(255, 128, 0),
        QColor(255, 128, 0),
        QColor(255, 153, 51),  // 上肢
        QColor(255, 153, 51),
        QColor(255, 153, 51),
        QColor(255, 153, 51),
        QColor(255, 153, 51),
        QColor(51, 153, 255),  // 躯干
        QColor(51, 153, 255),
        QColor(51, 153, 255),
        QColor(0, 255, 0),     // 下肢
        QColor(0, 255, 0),
        QColor(0, 255, 0),
        QColor(0, 255, 0),
    };

    if (limbIndex >= 0 && limbIndex < colors.size()) {
        return colors[limbIndex];
    }
    return Qt::white;
}

QColor DrawUtils::getKeypointColor(int keypointIndex)
{
    // 根据关键点类型返回颜色
    if (keypointIndex <= 4) {
        return QColor(255, 128, 0);  // 头部
    } else if (keypointIndex <= 10) {
        return QColor(255, 153, 51); // 上肢
    } else {
        return QColor(0, 255, 0);    // 下肢
    }
}

} // namespace yolo
