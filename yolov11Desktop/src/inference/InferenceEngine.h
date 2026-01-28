/**
 * @file InferenceEngine.h
 * 推理引擎抽象基类
 */

#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include "Detection.h"
#include "NMS.h"
#include <QString>
#include <QSize>
#include <QImage>
#include <opencv2/core.hpp>
#include <memory>
#include <functional>

namespace yolo {

/**
 * 推理引擎类型
 */
enum class EngineType {
    OnnxRuntime,
    OpenCVDnn,
    TensorRT,
    NCNN,
    Unknown
};

/**
 * 模型信息
 */
struct ModelInfo {
    QString name;
    QString path;
    EngineType engineType;
    QSize inputSize;
    int numClasses;
    TaskType taskType;
    bool isQuantized;
    QString version;

    ModelInfo() 
        : engineType(EngineType::Unknown)
        , inputSize(640, 640)
        , numClasses(80)
        , taskType(TaskType::Detection)
        , isQuantized(false) {}
};

/**
 * 推理配置
 */
struct InferenceConfig {
    QSize inputSize = QSize(640, 640);          ///< 输入尺寸
    float confThreshold = 0.25f;                 ///< 置信度阈值
    float iouThreshold = 0.45f;                  ///< NMS IoU 阈值
    int maxDetections = 300;                     ///< 最大检测数
    bool useGPU = false;                         ///< 是否使用 GPU
    int gpuDeviceId = 0;                         ///< GPU 设备 ID
    int numThreads = 4;                          ///< CPU 线程数
    bool enableFP16 = false;                     ///< 是否启用 FP16
    bool enableInt8 = false;                     ///< 是否启用 INT8
    bool letterbox = true;                       ///< 是否使用 letterbox 预处理
    bool swapRB = true;                          ///< 是否交换 R/B 通道
    
    // 归一化参数
    float scaleFactor = 1.0f / 255.0f;
    cv::Scalar mean = cv::Scalar(0, 0, 0);
    cv::Scalar std = cv::Scalar(1, 1, 1);

    NMSConfig getNMSConfig() const {
        return NMSConfig(iouThreshold, confThreshold, maxDetections);
    }
};

/**
 * 预处理信息（用于后处理还原坐标）
 */
struct PreprocessInfo {
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    int originalWidth = 0;
    int originalHeight = 0;
    int inputWidth = 0;
    int inputHeight = 0;
};

/**
 * 推理引擎抽象基类
 */
class InferenceEngine {
public:
    using ProgressCallback = std::function<void(int progress, const QString& message)>;

    InferenceEngine();
    virtual ~InferenceEngine();

    /**
     * 加载模型
     */
    virtual bool loadModel(const QString& modelPath, const InferenceConfig& config = InferenceConfig()) = 0;

    /**
     * 卸载模型
     */
    virtual void unloadModel() = 0;

    /**
     * 是否已加载模型
     */
    virtual bool isLoaded() const = 0;

    /**
     * 执行推理（OpenCV Mat）
     */
    virtual DetectionResult infer(const cv::Mat& image) = 0;

    /**
     * 执行推理（QImage 版本）
     */
    DetectionResult infer(const QImage& image);

    /**
     * 批量推理
     */
    virtual QVector<DetectionResult> inferBatch(const QVector<cv::Mat>& images);

    /**
     * 引擎类型
     */
    virtual EngineType engineType() const = 0;

    /**
     * 引擎名称
     */
    virtual QString engineName() const = 0;

    /**
     * 模型信息
     */
    const ModelInfo& modelInfo() const { return m_modelInfo; }

    /**
     * 推理配置
     */
    const InferenceConfig& config() const { return m_config; }

    /**
     * 设置推理配置
     */
    void setConfig(const InferenceConfig& config) { m_config = config; }

    /**
     * 设置置信度阈值
     */
    void setConfidenceThreshold(float threshold) { m_config.confThreshold = threshold; }

    /**
     * 设置 IoU 阈值
     */
    void setIoUThreshold(float threshold) { m_config.iouThreshold = threshold; }

    /**
     * 最近一次推理耗时（毫秒）
     */
    double lastInferenceTime() const { return m_lastInferenceTime; }

    /**
     * 平均推理耗时（毫秒）
     */
    double averageInferenceTime() const;

    /**
     * 重置计时统计
     */
    void resetTimingStats();

    /**
     * 获取错误信息
     */
    const QString& lastError() const { return m_lastError; }

    /**
     * 设置进度回调
     */
    void setProgressCallback(ProgressCallback callback) { m_progressCallback = callback; }

    /**
     * 是否支持 GPU
     */
    virtual bool supportsGPU() const { return false; }

    /**
     * 可用 GPU 列表
     */
    virtual QStringList availableGPUs() const { return {}; }

    /**
     * 预热模型（减少首次推理抖动）
     */
    virtual void warmup(int iterations = 3);

protected:
    /**
     * 预处理图像
     */
    virtual cv::Mat preprocess(const cv::Mat& image, PreprocessInfo& info);

    /**
     * 后处理输出
     */
    virtual DetectionResult postprocess(const std::vector<cv::Mat>& outputs, 
                                         const PreprocessInfo& info) = 0;

    /**
     * QImage 转 cv::Mat
     */
    cv::Mat qImageToMat(const QImage& image);

    /**
     * 设置错误信息
     */
    void setError(const QString& error) { m_lastError = error; }

    /**
     * 报告进度
     */
    void reportProgress(int progress, const QString& message);

protected:
    ModelInfo m_modelInfo;
    InferenceConfig m_config;
    QString m_lastError;
    double m_lastInferenceTime;
    double m_totalInferenceTime;
    int m_inferenceCount;
    ProgressCallback m_progressCallback;
};

} // namespace yolo

#endif // INFERENCE_ENGINE_H
