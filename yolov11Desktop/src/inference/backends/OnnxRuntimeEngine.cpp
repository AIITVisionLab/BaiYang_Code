/**
 * @file OnnxRuntimeEngine.cpp
 * @brief ONNX Runtime 推理引擎实现
 */

#include "OnnxRuntimeEngine.h"
#include "YoloPostProcess.h"
#include <QDebug>
#include <QFileInfo>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo {

#ifdef USE_ONNXRUNTIME

OnnxRuntimeEngine::OnnxRuntimeEngine()
    : m_taskType(TaskType::Detection)
    , m_isLoaded(false)
    , m_isDynamicInput(false)
{
}

OnnxRuntimeEngine::~OnnxRuntimeEngine()
{
    unloadModel();
}

bool OnnxRuntimeEngine::loadModel(const QString& modelPath, const InferenceConfig& config)
{
    if (m_isLoaded) {
        unloadModel();
    }

    m_config = config;
    reportProgress(0, "初始化ONNX Runtime...");

    try {
        // 初始化环境
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOv11");
        
        // 会话选项
        m_sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_sessionOptions->SetIntraOpNumThreads(config.numThreads);
        m_sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // GPU 配置
        if (config.useGPU && supportsGPU()) {
            reportProgress(10, "配置GPU加速...");
#ifdef _WIN32
            // Windows：尝试 DirectML
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(*m_sessionOptions, config.gpuDeviceId));
                qDebug() << "Using DirectML GPU acceleration";
            } catch (...) {
                qDebug() << "DirectML not available, falling back to CPU";
            }
#else
            // Linux：尝试 CUDA
            try {
                OrtCUDAProviderOptions cudaOptions;
                cudaOptions.device_id = config.gpuDeviceId;
                m_sessionOptions->AppendExecutionProvider_CUDA(cudaOptions);
                qDebug() << "Using CUDA GPU acceleration";
            } catch (...) {
                qDebug() << "CUDA not available, falling back to CPU";
            }
#endif
        }

        reportProgress(30, "加载模型文件...");

        // 加载模型
#ifdef _WIN32
        std::wstring widePath = modelPath.toStdWString();
        m_session = std::make_unique<Ort::Session>(*m_env, widePath.c_str(), *m_sessionOptions);
#else
        std::string path = modelPath.toStdString();
        m_session = std::make_unique<Ort::Session>(*m_env, path.c_str(), *m_sessionOptions);
#endif

        // 内存信息
        m_memoryInfo = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        reportProgress(60, "解析模型结构...");

        // 读取输入信息
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputs = m_session->GetInputCount();
        m_inputNames.clear();
        m_inputNamePtrs.clear();
        m_inputShapes.clear();

        for (size_t i = 0; i < numInputs; ++i) {
            auto inputName = m_session->GetInputNameAllocated(i, allocator);
            m_inputNames.push_back(inputName.get());
            
            auto typeInfo = m_session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            m_inputShapes.push_back(tensorInfo.GetShape());
        }

        for (auto& name : m_inputNames) {
            m_inputNamePtrs.push_back(name.c_str());
        }

        // 读取输出信息
        size_t numOutputs = m_session->GetOutputCount();
        m_outputNames.clear();
        m_outputNamePtrs.clear();
        m_outputShapes.clear();

        for (size_t i = 0; i < numOutputs; ++i) {
            auto outputName = m_session->GetOutputNameAllocated(i, allocator);
            m_outputNames.push_back(outputName.get());
            
            auto typeInfo = m_session->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            m_outputShapes.push_back(tensorInfo.GetShape());
        }

        for (auto& name : m_outputNames) {
            m_outputNamePtrs.push_back(name.c_str());
        }

        // 判断是否动态输入
        if (!m_inputShapes.empty() && m_inputShapes[0].size() >= 4) {
            m_isDynamicInput = (m_inputShapes[0][2] == -1 || m_inputShapes[0][3] == -1);
            if (!m_isDynamicInput) {
                m_config.inputSize = QSize(m_inputShapes[0][3], m_inputShapes[0][2]);
            }
        }

        reportProgress(80, "解析模型元数据...");
        parseModelMetadata();
        m_taskType = detectTaskType();

        // 填充模型信息
        QFileInfo fileInfo(modelPath);
        m_modelInfo.name = fileInfo.baseName();
        m_modelInfo.path = modelPath;
        m_modelInfo.engineType = EngineType::OnnxRuntime;
        m_modelInfo.inputSize = m_config.inputSize;
        m_modelInfo.taskType = m_taskType;

        m_isLoaded = true;
        reportProgress(100, "模型加载完成");

        qDebug() << "Model loaded successfully:" << modelPath;
        qDebug() << "Input shape:" << m_inputShapes[0][0] << m_inputShapes[0][1] 
                 << m_inputShapes[0][2] << m_inputShapes[0][3];
        qDebug() << "Task type:" << static_cast<int>(m_taskType);

        return true;

    } catch (const Ort::Exception& e) {
        setError(QString("ONNX Runtime error: %1").arg(e.what()));
        qCritical() << "Failed to load model:" << e.what();
        return false;
    } catch (const std::exception& e) {
        setError(QString("Error loading model: %1").arg(e.what()));
        qCritical() << "Failed to load model:" << e.what();
        return false;
    }
}

void OnnxRuntimeEngine::unloadModel()
{
    m_session.reset();
    m_sessionOptions.reset();
    m_env.reset();
    m_memoryInfo.reset();
    m_inputNames.clear();
    m_outputNames.clear();
    m_inputNamePtrs.clear();
    m_outputNamePtrs.clear();
    m_inputShapes.clear();
    m_outputShapes.clear();
    m_isLoaded = false;
}

bool OnnxRuntimeEngine::isLoaded() const
{
    return m_isLoaded;
}

DetectionResult OnnxRuntimeEngine::infer(const cv::Mat& image)
{
    DetectionResult result;
    
    if (!m_isLoaded || image.empty()) {
        setError("Model not loaded or empty image");
        return result;
    }

    auto totalStart = std::chrono::high_resolution_clock::now();

    try {
        // 预处理
        auto preprocessStart = std::chrono::high_resolution_clock::now();
        
        PreprocessInfo info;
        info.originalWidth = image.cols;
        info.originalHeight = image.rows;
        info.inputWidth = m_config.inputSize.width();
        info.inputHeight = m_config.inputSize.height();
        
        // Letterbox 预处理
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

        cv::Mat padded(info.inputHeight, info.inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(
            static_cast<int>(info.offsetX), 
            static_cast<int>(info.offsetY), 
            newWidth, newHeight
        )));

        // 使用 blobFromImage 进行归一化和 HWC->CHW 转换
        // scaleFactor=1/255.0, swapRB=true (BGR->RGB)
        cv::Mat blob;
        cv::dnn::blobFromImage(padded, blob, 1.0/255.0, cv::Size(), cv::Scalar(), true, false);

        auto preprocessEnd = std::chrono::high_resolution_clock::now();
        double preprocessTime = std::chrono::duration<double, std::milli>(
            preprocessEnd - preprocessStart).count();

        // 创建输入tensor
        std::vector<int64_t> inputShape = {1, 3, info.inputHeight, info.inputWidth};
        size_t inputTensorSize = blob.total();
        
        std::vector<float> inputTensorValues(
            reinterpret_cast<float*>(blob.data),
            reinterpret_cast<float*>(blob.data) + inputTensorSize
        );

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *m_memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputShape.data(), inputShape.size()
        );

        // 推理
        auto inferStart = std::chrono::high_resolution_clock::now();
        
        std::vector<Ort::Value> outputTensors = m_session->Run(
            Ort::RunOptions{nullptr},
            m_inputNamePtrs.data(), &inputTensor, 1,
            m_outputNamePtrs.data(), m_outputNamePtrs.size()
        );

        auto inferEnd = std::chrono::high_resolution_clock::now();
        double inferTime = std::chrono::duration<double, std::milli>(
            inferEnd - inferStart).count();

        // 后处理
        auto postprocessStart = std::chrono::high_resolution_clock::now();
        
        // 获取输出数据
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // 输出形状: outputShape[0] x outputShape[1] x outputShape[2]

        // 确保维度符合预期 (batch=1)
        if (outputShape.size() < 2) {
             throw std::runtime_error("Invalid output shape");
        }

        int64_t numChannels = outputShape[1];
        int64_t numPredictions = outputShape.size() > 2 ? outputShape[2] : 0;

        switch (m_taskType) {
            case TaskType::Detection:
                result = YoloPostProcess::processDetection(outputData, numChannels, numPredictions, info, m_config);
                break;
            case TaskType::Segmentation:
                // 分割需要处理两个输出，这里简化为只处理检测部分
                // TODO: 完整的分割支持
                result = YoloPostProcess::processDetection(outputData, numChannels, numPredictions, info, m_config);
                for(auto& det : result.detections()) det.setTaskType(TaskType::Segmentation);
                break;
            case TaskType::Pose:
                result = YoloPostProcess::processPose(outputData, numChannels, numPredictions, info, m_config);
                break;
            case TaskType::OBB:
                result = YoloPostProcess::processOBB(outputData, numChannels, numPredictions, info, m_config);
                break;
            case TaskType::Classification:
                 result = YoloPostProcess::processClassification(outputData, numChannels, m_config);
                break;
            default:
                result = YoloPostProcess::processDetection(outputData, numChannels, numPredictions, info, m_config);
                break;
        }

        auto postprocessEnd = std::chrono::high_resolution_clock::now();
        double postprocessTime = std::chrono::duration<double, std::milli>(
            postprocessEnd - postprocessStart).count();

        // 设置时间信息
        result.setPreprocessTime(preprocessTime);
        result.setInferenceTime(inferTime);
        result.setPostprocessTime(postprocessTime);
        result.setOriginalSize(info.originalWidth, info.originalHeight);

        auto totalEnd = std::chrono::high_resolution_clock::now();
        m_lastInferenceTime = std::chrono::duration<double, std::milli>(
            totalEnd - totalStart).count();
        m_totalInferenceTime += m_lastInferenceTime;
        m_inferenceCount++;

    } catch (const Ort::Exception& e) {
        setError(QString("Inference error: %1").arg(e.what()));
        qCritical() << "Inference failed:" << e.what();
    }

    return result;
}

bool OnnxRuntimeEngine::supportsGPU() const
{
#if defined(_WIN32)
    // Windows支持DirectML
    return true;
#elif defined(__linux__)
    // Linux检查CUDA
    // 这里简化处理，实际应该检查CUDA是否可用
    return true;
#else
    return false;
#endif
}

QStringList OnnxRuntimeEngine::availableGPUs() const
{
    QStringList gpus;
    // 实际实现需要查询系统GPU
    gpus << "GPU 0 (Default)";
    return gpus;
}

DetectionResult OnnxRuntimeEngine::postprocess(const std::vector<cv::Mat>& outputs, 
                                                const PreprocessInfo& info)
{
    // 由具体的postprocess函数处理
    return DetectionResult();
}

void OnnxRuntimeEngine::parseModelMetadata()
{
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::ModelMetadata metadata = m_session->GetModelMetadata();
        
        // 获取自定义元数据
        auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
        for (const auto& key : keys) {
            auto value = metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
            QString keyStr = QString::fromStdString(key.get());
            QString valueStr = QString::fromStdString(value.get());
            
            if (keyStr == "names" || keyStr == "classes") {
                // 解析类别名称
                // 格式可能是: {0: 'person', 1: 'car', ...}
                m_modelInfo.numClasses = valueStr.count(':');
            }
            
            qDebug() << "Metadata:" << keyStr << "=" << valueStr;
        }
    } catch (...) {
        // 忽略元数据解析错误
    }
}

TaskType OnnxRuntimeEngine::detectTaskType()
{
    if (m_outputShapes.empty()) {
        return TaskType::Detection;
    }

    // 根据输出形状推断任务类型
    auto& shape = m_outputShapes[0];
    
    if (shape.size() == 2) {
        return TaskType::Classification;
    }
    
    for (const auto& name : m_outputNames) {
        if (name.find("segment") != std::string::npos || 
            name.find("mask") != std::string::npos) {
            return TaskType::Segmentation;
        }
        if (name.find("pose") != std::string::npos || 
            name.find("keypoint") != std::string::npos) {
            return TaskType::Pose;
        }
        if (name.find("obb") != std::string::npos || 
            name.find("angle") != std::string::npos) {
            return TaskType::OBB;
        }
    }

    return TaskType::Detection;
}

#else

// 无ONNX Runtime时的空实现
OnnxRuntimeEngine::OnnxRuntimeEngine() : m_isLoaded(false) {}
OnnxRuntimeEngine::~OnnxRuntimeEngine() {}

bool OnnxRuntimeEngine::loadModel(const QString& modelPath, const InferenceConfig& config) {
    setError("ONNX Runtime is not available");
    return false;
}

void OnnxRuntimeEngine::unloadModel() {}
bool OnnxRuntimeEngine::isLoaded() const { return false; }

DetectionResult OnnxRuntimeEngine::infer(const cv::Mat& image) {
    return DetectionResult();
}

bool OnnxRuntimeEngine::supportsGPU() const { return false; }
QStringList OnnxRuntimeEngine::availableGPUs() const { return {}; }

DetectionResult OnnxRuntimeEngine::postprocess(const std::vector<cv::Mat>& outputs, 
                                                const PreprocessInfo& info) {
    return DetectionResult();
}

#endif // USE_ONNXRUNTIME

} // namespace yolo
