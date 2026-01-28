/**
 * @file NcnnEngine.cpp
 * @brief NCNN 推理引擎实现（ARM/树莓派优化）
 */

#include "NcnnEngine.h"
#include "../utils/Logger.h"
#include "../utils/Timer.h"
#include "../core/NMS.h"
#include <QFileInfo>
#include <opencv2/imgproc.hpp>

namespace yolo {

NcnnEngine::NcnnEngine()
    : m_loaded(false)
    , m_useVulkan(false)
    , m_useFP16(true)
    , m_useInt8(false)
    , m_threadCount(4)
    , m_inputLayerName("in0")
    , m_outputLayerName("out0")
{
#ifdef ENABLE_NCNN
    m_option.lightmode = true;
    m_option.num_threads = m_threadCount;
    m_option.use_packing_layout = true;
    LOG_DEBUG("NCNN engine created");
#else
    LOG_WARNING("NCNN is not enabled in this build");
#endif
}

NcnnEngine::~NcnnEngine()
{
    unloadModel();
}

bool NcnnEngine::loadModel(const QString& paramPath, const QString& binPath)
{
#ifdef ENABLE_NCNN
    if (m_loaded) {
        unloadModel();
    }
    
    QString actualBinPath = binPath;
    if (actualBinPath.isEmpty()) {
        QFileInfo info(paramPath);
        actualBinPath = info.absolutePath() + "/" + info.completeBaseName() + ".bin";
    }
    
    LOG_INFO(QString("Loading NCNN model: %1").arg(paramPath));
    
    m_option.num_threads = m_threadCount;
    m_option.use_fp16_packed = m_useFP16;
    m_option.use_fp16_storage = m_useFP16;
    m_option.use_fp16_arithmetic = m_useFP16;
    
    if (m_useVulkan && isVulkanAvailable()) {
        m_option.use_vulkan_compute = true;
        LOG_INFO("NCNN Vulkan compute enabled");
    } else {
        m_option.use_vulkan_compute = false;
    }
    
    m_net.opt = m_option;
    
    int ret = m_net.load_param(paramPath.toStdString().c_str());
    if (ret != 0) {
        setError(QString("Failed to load NCNN param file: %1").arg(paramPath));
        return false;
    }
    
    ret = m_net.load_model(actualBinPath.toStdString().c_str());
    if (ret != 0) {
        setError(QString("Failed to load NCNN bin file: %1").arg(actualBinPath));
        return false;
    }
    
    m_loaded = true;
    LOG_INFO("NCNN model loaded successfully");
    return true;
#else
    Q_UNUSED(paramPath);
    Q_UNUSED(binPath);
    setError("NCNN is not enabled");
    return false;
#endif
}

bool NcnnEngine::loadModel(const QString& modelPath, const InferenceConfig& config)
{
    m_config = config;
    
    if (modelPath.endsWith(".param", Qt::CaseInsensitive)) {
        return loadModel(modelPath, QString());
    }
    
    if (modelPath.endsWith(".bin", Qt::CaseInsensitive)) {
        QFileInfo info(modelPath);
        QString paramPath = info.absolutePath() + "/" + info.completeBaseName() + ".param";
        return loadModel(paramPath, modelPath);
    }
    
    return loadModel(modelPath, QString());
}

void NcnnEngine::unloadModel()
{
#ifdef ENABLE_NCNN
    if (m_loaded) {
        m_net.clear();
        m_loaded = false;
        LOG_INFO("NCNN model unloaded");
    }
#endif
}

bool NcnnEngine::isLoaded() const
{
    return m_loaded;
}

DetectionResult NcnnEngine::infer(const cv::Mat& image)
{
    DetectionResult result;
    
#ifndef ENABLE_NCNN
    Q_UNUSED(image);
    setError("NCNN is not enabled");
    return result;
#else
    if (!m_loaded || image.empty()) {
        setError("Model not loaded or empty image");
        return result;
    }
    
    Timer timer(true);
    
    // 预处理
    PreprocessInfo info;
    cv::Mat preprocessed = preprocess(image, info);
    double preprocessTime = timer.elapsedMs();
    timer.start();
    
    // 创建NCNN输入
    ncnn::Mat input = ncnn::Mat::from_pixels(
        preprocessed.data,
        ncnn::Mat::PIXEL_BGR,
        preprocessed.cols,
        preprocessed.rows
    );
    
    const float meanVals[3] = {0.0f, 0.0f, 0.0f};
    const float normVals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    input.substract_mean_normalize(meanVals, normVals);
    
    ncnn::Extractor extractor = m_net.create_extractor();
    extractor.set_light_mode(true);
    extractor.set_num_threads(m_threadCount);
    
    extractor.input(m_inputLayerName.toStdString().c_str(), input);
    
    ncnn::Mat output;
    extractor.extract(m_outputLayerName.toStdString().c_str(), output);
    
    double inferTime = timer.elapsedMs();
    timer.start();
    
    int rows = output.h;
    int cols = output.w;
    
    if (output.w > output.h && output.h <= 100) {
        rows = output.w;
        cols = output.h;
    }
    
    result = parseYoloOutput(static_cast<const float*>(output.data), rows, cols, image.size());
    
    double postprocessTime = timer.elapsedMs();
    
    result.setPreprocessTime(preprocessTime);
    result.setInferenceTime(inferTime);
    result.setPostprocessTime(postprocessTime);
    result.setOriginalSize(image.cols, image.rows);
    
    m_lastInferenceTime = inferTime;
    m_totalInferenceTime += inferTime;
    m_inferenceCount++;
    
    return result;
#endif
}

void NcnnEngine::setUseVulkan(bool use)
{
    m_useVulkan = use;
}

void NcnnEngine::setThreadCount(int count)
{
    m_threadCount = qMax(1, count);
#ifdef ENABLE_NCNN
    m_option.num_threads = m_threadCount;
#endif
}

void NcnnEngine::setLayerNames(const QString& inputName, const QString& outputName)
{
    m_inputLayerName = inputName;
    m_outputLayerName = outputName;
}

void NcnnEngine::setUseFP16(bool use)
{
    m_useFP16 = use;
}

void NcnnEngine::setUseInt8(bool use)
{
    m_useInt8 = use;
}

bool NcnnEngine::isVulkanAvailable()
{
#ifdef ENABLE_NCNN
#ifdef NCNN_VULKAN
    return ncnn::get_gpu_count() > 0;
#else
    return false;
#endif
#else
    return false;
#endif
}

DetectionResult NcnnEngine::postprocess(const std::vector<cv::Mat>& outputs, 
                                         const PreprocessInfo& info)
{
    Q_UNUSED(outputs);
    Q_UNUSED(info);
    return DetectionResult();
}

DetectionResult NcnnEngine::parseYoloOutput(const float* data, int rows, int cols,
                                            const cv::Size& originalSize)
{
    DetectionResult result;
    
    int targetWidth = m_config.inputSize.width() > 0 ? m_config.inputSize.width() : 640;
    int targetHeight = m_config.inputSize.height() > 0 ? m_config.inputSize.height() : 640;
    
    float scale = std::min(
        static_cast<float>(targetWidth) / originalSize.width,
        static_cast<float>(targetHeight) / originalSize.height
    );
    float offsetX = (targetWidth - originalSize.width * scale) / 2.0f;
    float offsetY = (targetHeight - originalSize.height * scale) / 2.0f;
    
    int numClasses = cols - 4;
    
    QVector<Detection> detections;
    
    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * cols;
        
        float maxScore = 0.0f;
        int maxClassId = 0;
        
        for (int c = 0; c < numClasses; ++c) {
            float score = row[4 + c];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }
        
        if (maxScore < m_config.confThreshold) {
            continue;
        }
        
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;
        
        x1 = (x1 - offsetX) / scale;
        y1 = (y1 - offsetY) / scale;
        x2 = (x2 - offsetX) / scale;
        y2 = (y2 - offsetY) / scale;
        
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(originalSize.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(originalSize.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(originalSize.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(originalSize.height)));
        
        Detection det(maxClassId, maxScore, BoundingBox(x1, y1, x2 - x1, y2 - y1));
        detections.push_back(det);
    }
    
    // NMS
    NMSConfig nmsConfig = m_config.getNMSConfig();
    QVector<Detection> nmsResult = NMS::apply(detections, nmsConfig);
    
    for (const auto& det : nmsResult) {
        result.addDetection(det);
    }
    
    if (result.count() > m_config.maxDetections) {
        result.sortByConfidence(true);
        while (result.count() > m_config.maxDetections) {
            result.detections().removeLast();
        }
    }
    
    return result;
}

} // namespace yolo
