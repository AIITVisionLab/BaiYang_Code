/**
 * @file NMS.h
 * 非极大值抑制（NMS）接口
 */

#ifndef NMS_H
#define NMS_H

#include "Detection.h"
#include <QVector>

namespace yolo {

/**
 * NMS 配置参数
 */
struct NMSConfig {
    float iouThreshold = 0.45f;     ///< IoU 阈值
    float scoreThreshold = 0.25f;   ///< 置信度阈值
    int maxDetections = 300;        ///< 最大检测数
    bool classAgnostic = false;     ///< 是否类别无关

    NMSConfig() = default;
    NMSConfig(float iou, float score, int maxDet = 300, bool agnostic = false)
        : iouThreshold(iou), scoreThreshold(score), maxDetections(maxDet), classAgnostic(agnostic) {}
};

/**
 * NMS 工具类
 */
class NMS {
public:
    /**
     * 标准 NMS
     */
    static QVector<Detection> apply(const QVector<Detection>& detections, const NMSConfig& config);

    /**
     * 软 NMS
     */
    static QVector<Detection> applySoft(QVector<Detection>& detections, 
                                        const NMSConfig& config, 
                                        float sigma = 0.5f);

    /**
     * 按类别做批量 NMS
     */
    static QVector<Detection> applyBatched(const QVector<Detection>& detections, 
                                           const NMSConfig& config);

    /**
     * 计算 IoU
     */
    static float computeIoU(const BoundingBox& box1, const BoundingBox& box2);

    /**
     * 计算 DIoU
     */
    static float computeDIoU(const BoundingBox& box1, const BoundingBox& box2);

    /**
     * 计算 CIoU
     */
    static float computeCIoU(const BoundingBox& box1, const BoundingBox& box2);

private:
    /**
     * 对单个类别做 NMS
     */
    static QVector<int> nmsForClass(const QVector<Detection>& detections,
                                    const QVector<int>& indices,
                                    float iouThreshold);
};

} // namespace yolo

#endif // NMS_H
