/**
 * @file OpenCVDnnEngine.cpp
 * @brief OpenCV DNN 推理引擎实现
 */

#include "OpenCVDnnEngine.h"
#include "YoloPostProcess.h"
#include <QDebug>
#include <QFileInfo>
#include <chrono>

namespace yolo {

OpenCVDnnEngine::OpenCVDnnEngine()
    : m_isLoaded(false)
    , m_backend(cv::dnn::DNN_BACKEND_DEFAULT)
    , m_target(cv::dnn::DNN_TARGET_CPU)
{
}

OpenCVDnnEngine::~OpenCVDnnEngine()
{
    unloadModel();
}

bool OpenCVDnnEngine::loadModel(const QString& modelPath, const InferenceConfig& config)
{
    if (m_isLoaded) {
        unloadModel();
    }

    m_config = config;
    reportProgress(0, "加载模型文件...");

    try {
        std::string path = modelPath.toStdString();
        QFileInfo fileInfo(modelPath);
        QString suffix = fileInfo.suffix().toLower();

        if (suffix == "onnx") {
            m_net = cv::dnn::readNetFromONNX(path);
        } else {
            setError("Unsupported model format: " + suffix);
            return false;
        }

        if (m_net.empty()) {
            setError("Failed to load model");
            return false;
        }

        reportProgress(50, "配置推理后端...");

        // 设置后端和目标
        if (config.useGPU) {
            setOptimalBackend();
        } else {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        // 获取输出层名称
        m_outputLayerNames = m_net.getUnconnectedOutLayersNames();

        // 设置模型信息
        m_modelInfo.name = fileInfo.baseName();
        m_modelInfo.path = modelPath;
        m_modelInfo.engineType = EngineType::OpenCVDnn;
        m_modelInfo.inputSize = config.inputSize;
        m_modelInfo.taskType = TaskType::Detection;

        m_isLoaded = true;
        reportProgress(100, "模型加载完成");

        qDebug() << "Model loaded with OpenCV DNN:" << modelPath;
        qDebug() << "Backend:" << m_backend << "Target:" << m_target;

        return true;

    } catch (const cv::Exception& e) {
        setError(QString("OpenCV error: %1").arg(e.what()));
        qCritical() << "Failed to load model:" << e.what();
        return false;
    }
}

void OpenCVDnnEngine::unloadModel()
{
    m_net = cv::dnn::Net();
    m_outputLayerNames.clear();
    m_isLoaded = false;
}

bool OpenCVDnnEngine::isLoaded() const
{
    return m_isLoaded;
}

DetectionResult OpenCVDnnEngine::infer(const cv::Mat& image)
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
        
        cv::Mat blob;
        cv::Size inputSize(m_config.inputSize.width(), m_config.inputSize.height());
        
        // 使用OpenCV的blobFromImage进行预处理
        if (m_config.letterbox) {
            // Letterbox预处理
            cv::Mat preprocessed = preprocess(image, info);
            blob = cv::dnn::blobFromImage(preprocessed, m_config.scaleFactor, 
                                          inputSize, cv::Scalar(), m_config.swapRB, false);
        } else {
            blob = cv::dnn::blobFromImage(image, m_config.scaleFactor, 
                                          inputSize, cv::Scalar(), m_config.swapRB, false);
            info.originalWidth = image.cols;
            info.originalHeight = image.rows;
            info.inputWidth = inputSize.width;
            info.inputHeight = inputSize.height;
            info.scaleX = static_cast<float>(image.cols) / inputSize.width;
            info.scaleY = static_cast<float>(image.rows) / inputSize.height;
            info.offsetX = 0;
            info.offsetY = 0;
        }

        auto preprocessEnd = std::chrono::high_resolution_clock::now();
        double preprocessTime = std::chrono::duration<double, std::milli>(
            preprocessEnd - preprocessStart).count();

        // 推理
        auto inferStart = std::chrono::high_resolution_clock::now();
        
        m_net.setInput(blob);
        std::vector<cv::Mat> outputs;
        m_net.forward(outputs, m_outputLayerNames);

        auto inferEnd = std::chrono::high_resolution_clock::now();
        double inferTime = std::chrono::duration<double, std::milli>(
            inferEnd - inferStart).count();

        // 后处理
        auto postprocessStart = std::chrono::high_resolution_clock::now();
        
        result = postprocess(outputs, info);

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

    } catch (const cv::Exception& e) {
        setError(QString("Inference error: %1").arg(e.what()));
        qCritical() << "Inference failed:" << e.what();
    }

    return result;
}

bool OpenCVDnnEngine::supportsGPU() const
{
    // 检查OpenCV是否编译了GPU支持
    auto backends = cv::dnn::getAvailableBackends();
    for (const auto& backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_CUDA ||
            backend.first == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE) {
            return true;
        }
    }
    return false;
}

QStringList OpenCVDnnEngine::availableGPUs() const
{
    QStringList gpus;
    if (supportsGPU()) {
        gpus << "GPU 0 (OpenCV)";
    }
    return gpus;
}

void OpenCVDnnEngine::setBackend(int backend)
{
    m_backend = backend;
    if (m_isLoaded) {
        m_net.setPreferableBackend(backend);
    }
}

void OpenCVDnnEngine::setTarget(int target)
{
    m_target = target;
    if (m_isLoaded) {
        m_net.setPreferableTarget(target);
    }
}

QStringList OpenCVDnnEngine::availableBackends()
{
    QStringList result;
    auto backends = cv::dnn::getAvailableBackends();
    for (const auto& backend : backends) {
        switch (backend.first) {
            case cv::dnn::DNN_BACKEND_DEFAULT:
                result << "Default";
                break;
            case cv::dnn::DNN_BACKEND_OPENCV:
                result << "OpenCV";
                break;
            case cv::dnn::DNN_BACKEND_CUDA:
                result << "CUDA";
                break;
            case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE:
                result << "OpenVINO";
                break;
            default:
                result << QString("Backend %1").arg(backend.first);
                break;
        }
    }
    return result;
}

QStringList OpenCVDnnEngine::availableTargets()
{
    QStringList result;
    result << "CPU" << "FP16";
    
    auto backends = cv::dnn::getAvailableBackends();
    for (const auto& backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_CUDA) {
            result << "CUDA" << "CUDA FP16";
        }
    }
    return result;
}

DetectionResult OpenCVDnnEngine::postprocess(const std::vector<cv::Mat>& outputs, 
                                              const PreprocessInfo& info)
{
    if (outputs.empty()) {
        return DetectionResult();
    }

    // YOLOv11输出格式
    return postprocessYolov11(outputs[0], info);
}

DetectionResult OpenCVDnnEngine::postprocessYolov11(const cv::Mat& output, 
                                                     const PreprocessInfo& info)
{
    // YOLOv11输出: [1, 4+nc, num_predictions]
    cv::Mat outputMat = output;
    
    // 确保输出是3维的
    if (outputMat.dims == 3) {
        int numChannels = outputMat.size[1];
        int numPredictions = outputMat.size[2];
        float* data = reinterpret_cast<float*>(outputMat.data);
        
        // 复用 YoloPostProcess
        return YoloPostProcess::processDetection(data, numChannels, numPredictions, info, m_config);
    } 
    else if (outputMat.dims == 2) {
        // 已转置的格式 [predictions, 4+nc] - 需要手动处理
        DetectionResult result;
        QVector<Detection> detections;
        
        int numPredictions = outputMat.rows;
        int numChannels = outputMat.cols;
        int numClasses = numChannels - 4;

        for (int i = 0; i < numPredictions; ++i) {
            float cx = outputMat.at<float>(i, 0);
            float cy = outputMat.at<float>(i, 1);
            float w = outputMat.at<float>(i, 2);
            float h = outputMat.at<float>(i, 3);

            float maxScore = 0;
            int maxClassId = 0;
            for (int c = 0; c < numClasses; ++c) {
                float score = outputMat.at<float>(i, 4 + c);
                if (score > maxScore) {
                    maxScore = score;
                    maxClassId = c;
                }
            }

            if (maxScore < m_config.confThreshold) {
                continue;
            }

            float x = (cx - w / 2.0f - info.offsetX) * info.scaleX;
            float y = (cy - h / 2.0f - info.offsetY) * info.scaleY;
            w *= info.scaleX;
            h *= info.scaleY;

            x = std::max(0.0f, x);
            y = std::max(0.0f, y);
            w = std::min(w, static_cast<float>(info.originalWidth) - x);
            h = std::min(h, static_cast<float>(info.originalHeight) - y);

            Detection det(maxClassId, maxScore, BoundingBox(x, y, w, h));
            det.setTaskType(TaskType::Detection);
            detections.append(det);
        }

        QVector<Detection> nmsResults = NMS::apply(detections, m_config.getNMSConfig());
        for (auto& det : nmsResults) {
            result.addDetection(std::move(det));
        }
        return result;
    }

    return DetectionResult();
}

void OpenCVDnnEngine::setOptimalBackend()
{
    auto backends = cv::dnn::getAvailableBackends();
    
    // 优先使用CUDA
    for (const auto& backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_CUDA) {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            m_backend = cv::dnn::DNN_BACKEND_CUDA;
            m_target = cv::dnn::DNN_TARGET_CUDA;
            qDebug() << "Using CUDA backend";
            return;
        }
    }

    // 其次使用OpenVINO
    for (const auto& backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE) {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            m_backend = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
            m_target = cv::dnn::DNN_TARGET_CPU;
            qDebug() << "Using OpenVINO backend";
            return;
        }
    }

    // 默认使用CPU
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    m_backend = cv::dnn::DNN_BACKEND_OPENCV;
    m_target = cv::dnn::DNN_TARGET_CPU;
    qDebug() << "Using OpenCV CPU backend";
}

} // namespace yolo
