/**
 * @file NMS.cpp
 * @brief NMS 实现
 */

#include "NMS.h"
#include <algorithm>
#include <cmath>
#include <QMap>

namespace yolo {

QVector<Detection> NMS::apply(const QVector<Detection>& detections, const NMSConfig& config)
{
    if (detections.isEmpty()) {
        return {};
    }

    // 先按置信度过滤
    QVector<int> indices;
    indices.reserve(detections.size());
    for (int i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence() >= config.scoreThreshold) {
            indices.append(i);
        }
    }

    std::sort(indices.begin(), indices.end(), [&detections](int a, int b) {
        return detections[a].confidence() > detections[b].confidence();
    });

    QVector<Detection> result;
    QVector<bool> suppressed(indices.size(), false);

    for (int i = 0; i < indices.size() && result.size() < config.maxDetections; ++i) {
        if (suppressed[i]) continue;

        int idx = indices[i];
        result.append(detections[idx]);

        for (int j = i + 1; j < indices.size(); ++j) {
            if (suppressed[j]) continue;

            int jdx = indices[j];
            
            // 非类别无关模式下仅比较同类别
            if (!config.classAgnostic && 
                detections[idx].classId() != detections[jdx].classId()) {
                continue;
            }

            float iou = computeIoU(detections[idx].bbox(), detections[jdx].bbox());
            if (iou > config.iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

QVector<Detection> NMS::applySoft(QVector<Detection>& detections, 
                                   const NMSConfig& config, 
                                   float sigma)
{
    if (detections.isEmpty()) {
        return {};
    }

    QVector<Detection> result;
    QVector<float> scores;
    scores.reserve(detections.size());
    
    for (const auto& det : detections) {
        scores.append(det.confidence());
    }

    while (result.size() < config.maxDetections) {
        // 选择当前最高分检测
        int maxIdx = -1;
        float maxScore = config.scoreThreshold;
        for (int i = 0; i < scores.size(); ++i) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIdx = i;
            }
        }

        if (maxIdx < 0) break;

        Detection selected = detections[maxIdx];
        selected.setConfidence(scores[maxIdx]);
        result.append(selected);
        scores[maxIdx] = 0;  // 标记为已处理

        // 对其他检测做软抑制
        for (int i = 0; i < detections.size(); ++i) {
            if (scores[i] <= 0) continue;

            if (!config.classAgnostic && 
                detections[maxIdx].classId() != detections[i].classId()) {
                continue;
            }

            float iou = computeIoU(detections[maxIdx].bbox(), detections[i].bbox());
            
            // 高斯权重衰减
            float weight = std::exp(-(iou * iou) / sigma);
            scores[i] *= weight;
        }
    }

    return result;
}

QVector<Detection> NMS::applyBatched(const QVector<Detection>& detections, 
                                      const NMSConfig& config)
{
    if (detections.isEmpty()) {
        return {};
    }

    // 按类别分组
    QMap<int, QVector<int>> classIndices;
    for (int i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence() >= config.scoreThreshold) {
            classIndices[detections[i].classId()].append(i);
        }
    }

    QVector<Detection> result;

    // 对每个类别分别执行 NMS
    for (auto it = classIndices.begin(); it != classIndices.end(); ++it) {
        QVector<int>& indices = it.value();
        
        // 按置信度排序
        std::sort(indices.begin(), indices.end(), [&detections](int a, int b) {
            return detections[a].confidence() > detections[b].confidence();
        });

        QVector<int> kept = nmsForClass(detections, indices, config.iouThreshold);
        
        for (int idx : kept) {
            if (result.size() >= config.maxDetections) break;
            result.append(detections[idx]);
        }
    }

    // 最终结果再按置信度排序
    std::sort(result.begin(), result.end(), [](const Detection& a, const Detection& b) {
        return a.confidence() > b.confidence();
    });

    // 超过上限则截断
    if (result.size() > config.maxDetections) {
        result.resize(config.maxDetections);
    }

    return result;
}

QVector<int> NMS::nmsForClass(const QVector<Detection>& detections,
                              const QVector<int>& indices,
                              float iouThreshold)
{
    QVector<int> kept;
    QVector<bool> suppressed(indices.size(), false);

    for (int i = 0; i < indices.size(); ++i) {
        if (suppressed[i]) continue;

        kept.append(indices[i]);

        for (int j = i + 1; j < indices.size(); ++j) {
            if (suppressed[j]) continue;

            float iou = computeIoU(detections[indices[i]].bbox(), 
                                   detections[indices[j]].bbox());
            if (iou > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

float NMS::computeIoU(const BoundingBox& box1, const BoundingBox& box2)
{
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.right(), box2.right());
    float y2 = std::min(box1.bottom(), box2.bottom());

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    float intersectArea = (x2 - x1) * (y2 - y1);
    float unionArea = box1.area() + box2.area() - intersectArea;

    return unionArea > 0 ? intersectArea / unionArea : 0.0f;
}

float NMS::computeDIoU(const BoundingBox& box1, const BoundingBox& box2)
{
    float iou = computeIoU(box1, box2);

    // 计算中心点距离
    float cx1 = box1.centerX();
    float cy1 = box1.centerY();
    float cx2 = box2.centerX();
    float cy2 = box2.centerY();
    float centerDist = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

    // 计算外接框对角线距离
    float encloseX1 = std::min(box1.x, box2.x);
    float encloseY1 = std::min(box1.y, box2.y);
    float encloseX2 = std::max(box1.right(), box2.right());
    float encloseY2 = std::max(box1.bottom(), box2.bottom());
    float encloseDiag = (encloseX2 - encloseX1) * (encloseX2 - encloseX1) + 
                        (encloseY2 - encloseY1) * (encloseY2 - encloseY1);

    if (encloseDiag < 1e-6f) {
        return iou;
    }

    return iou - centerDist / encloseDiag;
}

float NMS::computeCIoU(const BoundingBox& box1, const BoundingBox& box2)
{
    float iou = computeIoU(box1, box2);

    // 计算中心点距离
    float cx1 = box1.centerX();
    float cy1 = box1.centerY();
    float cx2 = box2.centerX();
    float cy2 = box2.centerY();
    float centerDist = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

    // 计算外接框对角线距离
    float encloseX1 = std::min(box1.x, box2.x);
    float encloseY1 = std::min(box1.y, box2.y);
    float encloseX2 = std::max(box1.right(), box2.right());
    float encloseY2 = std::max(box1.bottom(), box2.bottom());
    float encloseDiag = (encloseX2 - encloseX1) * (encloseX2 - encloseX1) + 
                        (encloseY2 - encloseY1) * (encloseY2 - encloseY1);

    // 计算宽高比一致性
    const float PI = 3.14159265358979323846f;
    float v = (4.0f / (PI * PI)) * 
              std::pow(std::atan(box1.width / std::max(box1.height, 1e-6f)) - 
                      std::atan(box2.width / std::max(box2.height, 1e-6f)), 2);
    
    float alpha = v / (1.0f - iou + v + 1e-6f);

    if (encloseDiag < 1e-6f) {
        return iou;
    }

    return iou - centerDist / encloseDiag - alpha * v;
}

} // namespace yolo
