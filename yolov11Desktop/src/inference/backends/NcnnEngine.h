/**
 * @file NcnnEngine.h
 * @brief NCNN 推理引擎（ARM/树莓派优化）
 */

#ifndef YOLO_NCNN_ENGINE_H
#define YOLO_NCNN_ENGINE_H

#include "InferenceEngine.h"

#ifdef ENABLE_NCNN
#include <ncnn/net.h>
#include <ncnn/layer.h>
#endif

namespace yolo {

/**
 * @brief NCNN 推理引擎
 *
 * 面向嵌入式设备与 ARM 平台：
 * - 依赖轻量
 * - 支持 ARM NEON 加速
 * - 内存占用小
 * - 适合树莓派等设备
 */
class NcnnEngine : public InferenceEngine
{
public:
    /**
     * @brief 构造函数
     */
    NcnnEngine();
    
    /**
     * @brief 析构函数
     */
    ~NcnnEngine() override;
    
    /**
     * @brief 获取引擎名称
     */
    QString engineName() const override { return "NCNN"; }
    
    /**
     * @brief 获取引擎类型
     */
    EngineType engineType() const override { return EngineType::NCNN; }
    
    /**
     * @brief 加载模型
     * @param paramPath NCNN参数文件路径 (.param)
     * @param binPath NCNN二进制文件路径 (.bin)
     */
    bool loadModel(const QString& paramPath, const QString& binPath);
    
    /**
     * @brief 实现基类加载方法
     */
    bool loadModel(const QString& modelPath, const InferenceConfig& config = InferenceConfig()) override;
    
    /**
     * @brief 卸载模型
     */
    void unloadModel() override;
    
    /**
     * @brief 模型是否已加载
     */
    bool isLoaded() const override;
    
    /**
     * @brief 运行推理
     */
    DetectionResult infer(const cv::Mat& image) override;
    
    /**
     * @brief 设置Vulkan GPU加速（如果可用）
     */
    void setUseVulkan(bool use);
    
    /**
     * @brief 设置线程数
     */
    void setThreadCount(int count);
    
    /**
     * @brief 设置输入/输出层名称
     */
    void setLayerNames(const QString& inputName, const QString& outputName);
    
    /**
     * @brief 启用FP16精度
     */
    void setUseFP16(bool use);
    
    /**
     * @brief 启用int8量化
     */
    void setUseInt8(bool use);
    
    /**
     * @brief 检查Vulkan是否可用
     */
    static bool isVulkanAvailable();

protected:
    /**
     * @brief 解析输出结果
     */
    DetectionResult postprocess(const std::vector<cv::Mat>& outputs, 
                                const PreprocessInfo& info) override;

private:
#ifdef ENABLE_NCNN
    ncnn::Net m_net;
    ncnn::Option m_option;
#endif
    
    bool m_loaded;
    bool m_useVulkan;
    bool m_useFP16;
    bool m_useInt8;
    int m_threadCount;
    QString m_inputLayerName;
    QString m_outputLayerName;
    
    /// @brief YOLOv11 输出解析
    DetectionResult parseYoloOutput(const float* data, int rows, int cols, 
                                    const cv::Size& originalSize);
};

} // namespace yolo

#endif // YOLO_NCNN_ENGINE_H
