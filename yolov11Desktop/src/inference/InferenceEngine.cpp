/**
 * @file InferenceEngine.cpp
 * 推理引擎基类实现
 */

#include "InferenceEngine.h"
#include <opencv2/imgproc.hpp>
#include <QDebug>

namespace yolo {

InferenceEngine::InferenceEngine()
    : m_lastInferenceTime(0)
    , m_totalInferenceTime(0)
    , m_inferenceCount(0)
{
}

InferenceEngine::~InferenceEngine()
{
}

DetectionResult InferenceEngine::infer(const QImage& image)
{
    cv::Mat mat = qImageToMat(image);
    return infer(mat);
}

QVector<DetectionResult> InferenceEngine::inferBatch(const QVector<cv::Mat>& images)
{
    QVector<DetectionResult> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        results.append(infer(image));
    }
    
    return results;
}

double InferenceEngine::averageInferenceTime() const
{
    if (m_inferenceCount == 0) return 0;
    return m_totalInferenceTime / m_inferenceCount;
}

void InferenceEngine::resetTimingStats()
{
    m_lastInferenceTime = 0;
    m_totalInferenceTime = 0;
    m_inferenceCount = 0;
}

void InferenceEngine::warmup(int iterations)
{
    if (!isLoaded()) return;

    cv::Mat dummyImage(m_config.inputSize.height(), m_config.inputSize.width(), CV_8UC3, cv::Scalar(128, 128, 128));
    
    for (int i = 0; i < iterations; ++i) {
        infer(dummyImage);
    }
    
    resetTimingStats();  // 重置统计，不计入预热
}

cv::Mat InferenceEngine::preprocess(const cv::Mat& image, PreprocessInfo& info)
{
    info.originalWidth = image.cols;
    info.originalHeight = image.rows;
    info.inputWidth = m_config.inputSize.width();
    info.inputHeight = m_config.inputSize.height();

    cv::Mat result;

    if (m_config.letterbox) {
        // Letterbox 预处理（保持宽高比）
        float scale = std::min(
            static_cast<float>(info.inputWidth) / image.cols,
            static_cast<float>(info.inputHeight) / image.rows
        );
        
        int newWidth = static_cast<int>(image.cols * scale);
        int newHeight = static_cast<int>(image.rows * scale);
        
        info.scaleX = 1.0f / scale;
        info.scaleY = 1.0f / scale;
        info.offsetX = (info.inputWidth - newWidth) / 2.0f;
        info.offsetY = (info.inputHeight - newHeight) / 2.0f;

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

        result = cv::Mat(info.inputHeight, info.inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(result(cv::Rect(
            static_cast<int>(info.offsetX), 
            static_cast<int>(info.offsetY), 
            newWidth, newHeight
        )));
    } else {
        // 直接缩放到目标尺寸
        cv::resize(image, result, cv::Size(info.inputWidth, info.inputHeight), 0, 0, cv::INTER_LINEAR);
        info.scaleX = static_cast<float>(image.cols) / info.inputWidth;
        info.scaleY = static_cast<float>(image.rows) / info.inputHeight;
        info.offsetX = 0;
        info.offsetY = 0;
    }

    // BGR -> RGB
    if (m_config.swapRB) {
        cv::cvtColor(result, result, cv::COLOR_BGR2RGB);
    }

    return result;
}

cv::Mat InferenceEngine::qImageToMat(const QImage& image)
{
    QImage convertedImage;
    
    switch (image.format()) {
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32:
        case QImage::Format_ARGB32_Premultiplied:
            convertedImage = image.convertToFormat(QImage::Format_RGB888);
            break;
        case QImage::Format_RGB888:
            convertedImage = image;
            break;
        default:
            convertedImage = image.convertToFormat(QImage::Format_RGB888);
            break;
    }

    cv::Mat mat(convertedImage.height(), convertedImage.width(), CV_8UC3,
                const_cast<uchar*>(convertedImage.bits()),
                static_cast<size_t>(convertedImage.bytesPerLine()));
    
    cv::Mat result;
    cv::cvtColor(mat, result, cv::COLOR_RGB2BGR);  // Qt 用 RGB，OpenCV 用 BGR
    return result.clone();
}

void InferenceEngine::reportProgress(int progress, const QString& message)
{
    if (m_progressCallback) {
        m_progressCallback(progress, message);
    }
}

} // namespace yolo
