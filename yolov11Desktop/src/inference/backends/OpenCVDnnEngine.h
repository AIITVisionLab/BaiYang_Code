/**
 * @file OpenCVDnnEngine.h
 * @brief OpenCV DNN 推理引擎（备选方案，适合树莓派）
 */

#ifndef OPENCV_DNN_ENGINE_H
#define OPENCV_DNN_ENGINE_H

#include "InferenceEngine.h"
#include <opencv2/dnn.hpp>

namespace yolo {

/**
 * @brief OpenCV DNN 推理引擎
 *
 * 特点：
 * - 仅依赖 OpenCV
 * - 支持 ONNX 格式
 * - 可在树莓派上运行
 * - 支持 OpenCV 的后端与目标设备
 */
class OpenCVDnnEngine : public InferenceEngine {
public:
    OpenCVDnnEngine();
    ~OpenCVDnnEngine() override;

    // InferenceEngine 接口实现
    bool loadModel(const QString& modelPath, const InferenceConfig& config = InferenceConfig()) override;
    void unloadModel() override;
    bool isLoaded() const override;
    DetectionResult infer(const cv::Mat& image) override;
    EngineType engineType() const override { return EngineType::OpenCVDnn; }
    QString engineName() const override { return "OpenCV DNN"; }
    bool supportsGPU() const override;
    QStringList availableGPUs() const override;

    /**
     * @brief 设置后端
     * @param backend cv::dnn::DNN_BACKEND_*
     */
    void setBackend(int backend);

    /**
     * @brief 设置目标设备
     * @param target cv::dnn::DNN_TARGET_*
     */
    void setTarget(int target);

    /**
     * @brief 获取支持的后端列表
     */
    static QStringList availableBackends();

    /**
     * @brief 获取支持的目标设备列表
     */
    static QStringList availableTargets();

protected:
    DetectionResult postprocess(const std::vector<cv::Mat>& outputs, 
                                 const PreprocessInfo& info) override;

private:
    /**
     * @brief YOLOv11 检测后处理
     */
    DetectionResult postprocessYolov11(const cv::Mat& output, const PreprocessInfo& info);

    /**
     * @brief 设置最优后端
     */
    void setOptimalBackend();

private:
    cv::dnn::Net m_net;
    bool m_isLoaded;
    std::vector<std::string> m_outputLayerNames;
    int m_backend;
    int m_target;
};

} // namespace yolo

#endif // OPENCV_DNN_ENGINE_H
