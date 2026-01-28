/**
 * @file YoloPostProcess.cpp
 * @brief YOLO 后处理实现
 */

#include "YoloPostProcess.h"
#include <algorithm>
#include <cmath>

namespace yolo {

DetectionResult YoloPostProcess::processDetection(const float* output, 
                                                  int64_t numChannels, 
                                                  int64_t numPredictions,
                                                  const PreprocessInfo& info, 
                                                  const InferenceConfig& config,
                                                  int numClasses)
{
    DetectionResult result;
    QVector<Detection> detections;
    
    // 未指定类别数时，根据通道数推断：num_classes = channels - 4
    if (numClasses <= 0) {
        numClasses = numChannels - 4;
    }

    // 遍历所有预测
    // YOLOv8/v11 默认输出格式: [batch, channels, predictions]
    // 这里拿的是第一个 batch 的指针
    // 数据索引: data[channel_idx * numPredictions + prediction_idx]
    
    for (int64_t i = 0; i < numPredictions; ++i) {
        // 边界框 (cx, cy, w, h)
        float cx = output[0 * numPredictions + i];
        float cy = output[1 * numPredictions + i];
        float w = output[2 * numPredictions + i];
        float h = output[3 * numPredictions + i];

        // 找最高类别分数
        float maxScore = 0;
        int maxClassId = 0;
        for (int c = 0; c < numClasses; ++c) {
            float score = output[(4 + c) * numPredictions + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        // 过滤低置信度
        if (maxScore < config.confThreshold) {
            continue;
        }

        // 转成左上角坐标 (x, y, w, h)
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        // 缩放到原图坐标
        x = (x - info.offsetX) * info.scaleX;
        y = (y - info.offsetY) * info.scaleY;
        w *= info.scaleX;
        h *= info.scaleY;

        // 裁剪到图像边界
        x = std::max(0.0f, x);
        y = std::max(0.0f, y);
        w = std::min(w, static_cast<float>(info.originalWidth) - x);
        h = std::min(h, static_cast<float>(info.originalHeight) - y);

        Detection det(maxClassId, maxScore, BoundingBox(x, y, w, h));
        det.setTaskType(TaskType::Detection);
        detections.append(det);
    }

    // 执行 NMS
    QVector<Detection> nmsResults = NMS::apply(detections, config.getNMSConfig());
    
    for (auto& det : nmsResults) {
        result.addDetection(std::move(det));
    }

    return result;
}

DetectionResult YoloPostProcess::processPose(const float* output, 
                                             int64_t numChannels, 
                                             int64_t numPredictions,
                                             const PreprocessInfo& info, 
                                             const InferenceConfig& config)
{
    DetectionResult result;
    QVector<Detection> detections;

    const int numKeypoints = 17; // COCO 关键点数

    for (int64_t i = 0; i < numPredictions; ++i) {
        float cx = output[0 * numPredictions + i];
        float cy = output[1 * numPredictions + i];
        float w = output[2 * numPredictions + i];
        float h = output[3 * numPredictions + i];
        float conf = output[4 * numPredictions + i];

        if (conf < config.confThreshold) {
            continue;
        }

        // 边界框处理
        float x = (cx - w / 2.0f - info.offsetX) * info.scaleX;
        float y = (cy - h / 2.0f - info.offsetY) * info.scaleY;
        w *= info.scaleX;
        h *= info.scaleY;

        Detection det(0, conf, BoundingBox(x, y, w, h));  // 类别 0 = person
        det.setTaskType(TaskType::Pose);

        // 解析关键点
        QVector<Keypoint> keypoints;
        for (int k = 0; k < numKeypoints; ++k) {
            int baseIdx = 5 + k * 3;
            float kx = output[baseIdx * numPredictions + i];
            float ky = output[(baseIdx + 1) * numPredictions + i];
            float kconf = output[(baseIdx + 2) * numPredictions + i];

            // 缩放关键点坐标
            kx = (kx - info.offsetX) * info.scaleX;
            ky = (ky - info.offsetY) * info.scaleY;

            keypoints.append(Keypoint(kx, ky, kconf, k));
        }
        det.setKeypoints(keypoints);
        detections.append(det);
    }

    // NMS
    QVector<Detection> nmsResults = NMS::apply(detections, config.getNMSConfig());
    for (auto& det : nmsResults) {
        result.addDetection(std::move(det));
    }

    return result;
}

DetectionResult YoloPostProcess::processOBB(const float* output, 
                                            int64_t numChannels, 
                                            int64_t numPredictions,
                                            const PreprocessInfo& info, 
                                            const InferenceConfig& config)
{
    DetectionResult result;
    QVector<Detection> detections;

    int numClasses = numChannels - 5; // cx, cy, w, h, angle

    for (int64_t i = 0; i < numPredictions; ++i) {
        float cx = output[0 * numPredictions + i];
        float cy = output[1 * numPredictions + i];
        float w = output[2 * numPredictions + i];
        float h = output[3 * numPredictions + i];
        float angle = output[4 * numPredictions + i];

        // 找最高分类别
        float maxScore = 0;
        int maxClassId = 0;
        for (int c = 0; c < numClasses; ++c) {
            float score = output[(5 + c) * numPredictions + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore < config.confThreshold) {
            continue;
        }

        // 创建检测
        Detection det(maxClassId, maxScore, BoundingBox());
        det.setTaskType(TaskType::OBB);

        // 设置旋转边界框
        OrientedBoundingBox obb;
        obb.cx = (cx - info.offsetX) * info.scaleX;
        obb.cy = (cy - info.offsetY) * info.scaleY;
        obb.width = w * info.scaleX;
        obb.height = h * info.scaleY;
        obb.angle = angle;
        det.setObb(obb);

        // 生成外接矩形（AABB）
        float halfW = obb.width / 2;
        float halfH = obb.height / 2;
        float cosA = std::abs(std::cos(angle));
        float sinA = std::abs(std::sin(angle));
        float bboxW = halfW * cosA + halfH * sinA;
        float bboxH = halfW * sinA + halfH * cosA;
        det.setBbox(BoundingBox(obb.cx - bboxW, obb.cy - bboxH, bboxW * 2, bboxH * 2));

        detections.append(det);
    }

    // NMS（目前用普通 IoU，后续可换成 PolyIoU）
    QVector<Detection> nmsResults = NMS::apply(detections, config.getNMSConfig());
    for (auto& det : nmsResults) {
        result.addDetection(std::move(det));
    }

    return result;
}

DetectionResult YoloPostProcess::processClassification(const float* output, 
                                                       int64_t numClasses,
                                                       const InferenceConfig& config)
{
    DetectionResult result;

    if (numClasses <= 0) return result;

    // 取最高分（Top-1）
    float maxScore = output[0];
    int maxClassId = 0;
    for (int i = 1; i < numClasses; ++i) {
        if (output[i] > maxScore) {
            maxScore = output[i];
            maxClassId = i;
        }
    }

    // Softmax 归一化
    float sum = 0;
    for (int i = 0; i < numClasses; ++i) {
        sum += std::exp(output[i] - maxScore);
    }
    float confidence = std::exp(output[maxClassId] - maxScore) / sum;
    
    Detection det(maxClassId, confidence, BoundingBox());
    det.setTaskType(TaskType::Classification);
    result.addDetection(det);

    return result;
}

} // namespace yolo
