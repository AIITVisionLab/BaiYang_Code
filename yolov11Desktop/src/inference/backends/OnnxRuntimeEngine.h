/**
 * @file OnnxRuntimeEngine.h
 * ONNX Runtime 推理引擎
 */

#ifndef ONNXRUNTIME_ENGINE_H
#define ONNXRUNTIME_ENGINE_H

#include "InferenceEngine.h"

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <memory>
#include <vector>

namespace yolo {

/**
 * ONNX Runtime 推理引擎实现
 *
 * 支持 YOLOv11 多种任务：
 * - Detection（目标检测）
 * - Segmentation（实例分割）
 * - Pose（姿态估计）
 * - OBB（旋转框）
 * - Classification（分类）
 */
class OnnxRuntimeEngine : public InferenceEngine {
public:
    OnnxRuntimeEngine();
    ~OnnxRuntimeEngine() override;

    // InferenceEngine 接口实现
    bool loadModel(const QString& modelPath, const InferenceConfig& config = InferenceConfig()) override;
    void unloadModel() override;
    bool isLoaded() const override;
    DetectionResult infer(const cv::Mat& image) override;
    EngineType engineType() const override { return EngineType::OnnxRuntime; }
    QString engineName() const override { return "ONNX Runtime"; }
    bool supportsGPU() const override;
    QStringList availableGPUs() const override;

protected:
    DetectionResult postprocess(const std::vector<cv::Mat>& outputs, 
                                 const PreprocessInfo& info) override;

private:
#ifdef USE_ONNXRUNTIME
    /**
     * 解析模型元数据
     */
    void parseModelMetadata();

    /**
     * 判断任务类型
     */
    TaskType detectTaskType();

    /**
     * 解析 YOLOv11 输出
     * 输出形状: [batch, 4+num_classes(+extras), num_predictions]
     * 需要转置后再处理
     */
    void parseYolov11Output(const float* data,
                            const std::vector<int64_t>& shape,
                            QVector<Detection>& detections,
                            const PreprocessInfo& info);

private:
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
    std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;
    
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<const char*> m_inputNamePtrs;
    std::vector<const char*> m_outputNamePtrs;
    std::vector<std::vector<int64_t>> m_inputShapes;
    std::vector<std::vector<int64_t>> m_outputShapes;
    
    TaskType m_taskType;
    bool m_isLoaded;
    bool m_isDynamicInput;
#else
    bool m_isLoaded = false;
#endif
};

} // namespace yolo

#endif // ONNXRUNTIME_ENGINE_H
