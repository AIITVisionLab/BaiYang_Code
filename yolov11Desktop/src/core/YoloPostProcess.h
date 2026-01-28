/**
 * @file YoloPostProcess.h
 * @brief YOLO 后处理逻辑
 */

#ifndef YOLO_POST_PROCESS_H
#define YOLO_POST_PROCESS_H

#include "InferenceEngine.h"

namespace yolo {

/**
 * YOLO 后处理工具类
 *
 * 负责把模型输出的原始 tensor 转成 DetectionResult，
 * 不依赖具体推理后端，只处理 float 数据。
 */
class YoloPostProcess {
public:
    /**
     * 处理目标检测输出（YOLOv8/v11）
     */
    static DetectionResult processDetection(const float* data, 
                                          int64_t numChannels, 
                                          int64_t numPredictions,
                                          const PreprocessInfo& info, 
                                          const InferenceConfig& config,
                                          int numClasses = -1);

    /**
        * 处理姿态估计输出（YOLOv8/v11-Pose）
     */
    static DetectionResult processPose(const float* data, 
                                     int64_t numChannels, 
                                     int64_t numPredictions,
                                     const PreprocessInfo& info, 
                                     const InferenceConfig& config);

    /**
        * 处理旋转框（OBB）输出
     */
    static DetectionResult processOBB(const float* data, 
                                    int64_t numChannels, 
                                    int64_t numPredictions,
                                    const PreprocessInfo& info, 
                                    const InferenceConfig& config);

    /**
        * 处理图像分类输出（YOLOv8/v11-Cls）
     */
    static DetectionResult processClassification(const float* data, 
                                               int64_t numClasses,
                                               const InferenceConfig& config);
};

} // namespace yolo

#endif // YOLO_POST_PROCESS_H
