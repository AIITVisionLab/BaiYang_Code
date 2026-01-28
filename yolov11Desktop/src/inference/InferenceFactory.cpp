/**
 * @file InferenceFactory.cpp
 * 推理引擎工厂实现
 */

#include "InferenceFactory.h"
#include "OnnxRuntimeEngine.h"
#include "OpenCVDnnEngine.h"

#ifdef USE_NCNN
#include "NcnnEngine.h"
#endif

#ifdef USE_TENSORRT
#include "TensorRTEngine.h"
#endif

#include <QFileInfo>
#include <QDebug>

namespace yolo {

std::unique_ptr<InferenceEngine> InferenceFactory::create(EngineType type)
{
    switch (type) {
        case EngineType::OnnxRuntime:
#ifdef USE_ONNXRUNTIME
            return std::make_unique<OnnxRuntimeEngine>();
#else
            qWarning() << "ONNX Runtime not available, falling back to OpenCV DNN";
            return std::make_unique<OpenCVDnnEngine>();
#endif

        case EngineType::OpenCVDnn:
            return std::make_unique<OpenCVDnnEngine>();

        case EngineType::NCNN:
#ifdef USE_NCNN
            return std::make_unique<NcnnEngine>();
#else
            qWarning() << "NCNN not available, falling back to OpenCV DNN";
            return std::make_unique<OpenCVDnnEngine>();
#endif

        case EngineType::TensorRT:
#ifdef USE_TENSORRT
            return std::make_unique<TensorRTEngine>();
#else
            qWarning() << "TensorRT not available, falling back to ONNX Runtime";
            return create(EngineType::OnnxRuntime);
#endif

        default:
            return std::make_unique<OpenCVDnnEngine>();
    }
}

std::unique_ptr<InferenceEngine> InferenceFactory::createBest(bool useGPU)
{
    // 优先级：TensorRT > ONNX Runtime > NCNN > OpenCV DNN

    if (useGPU) {
#ifdef USE_TENSORRT
        auto engine = std::make_unique<TensorRTEngine>();
        if (engine->supportsGPU()) {
            qDebug() << "Using TensorRT engine";
            return engine;
        }
#endif

#ifdef USE_ONNXRUNTIME
        auto engine = std::make_unique<OnnxRuntimeEngine>();
        if (engine->supportsGPU()) {
            qDebug() << "Using ONNX Runtime engine with GPU";
            return engine;
        }
#endif

        auto cvEngine = std::make_unique<OpenCVDnnEngine>();
        if (cvEngine->supportsGPU()) {
            qDebug() << "Using OpenCV DNN engine with GPU";
            return cvEngine;
        }
    }

    // CPU 推理
#ifdef USE_ONNXRUNTIME
    qDebug() << "Using ONNX Runtime engine";
    return std::make_unique<OnnxRuntimeEngine>();
#endif

#ifdef USE_NCNN
    // NCNN 在 ARM 上通常更高效
#ifdef ARM_PLATFORM
    qDebug() << "Using NCNN engine (ARM optimized)";
    return std::make_unique<NcnnEngine>();
#endif
#endif

    qDebug() << "Using OpenCV DNN engine";
    return std::make_unique<OpenCVDnnEngine>();
}

std::unique_ptr<InferenceEngine> InferenceFactory::createForModel(const QString& modelPath)
{
    QFileInfo fileInfo(modelPath);
    QString suffix = fileInfo.suffix().toLower();

    if (suffix == "onnx") {
#ifdef USE_ONNXRUNTIME
        return std::make_unique<OnnxRuntimeEngine>();
#else
        return std::make_unique<OpenCVDnnEngine>();
#endif
    }
    
    if (suffix == "engine" || suffix == "trt") {
#ifdef USE_TENSORRT
        return std::make_unique<TensorRTEngine>();
#else
        qWarning() << "TensorRT engine file but TensorRT not available";
        return nullptr;
#endif
    }

    if (suffix == "param" || suffix == "bin") {
#ifdef USE_NCNN
        return std::make_unique<NcnnEngine>();
#else
        qWarning() << "NCNN model file but NCNN not available";
        return nullptr;
#endif
    }

    // 默认使用 OpenCV DNN（支持多种格式）
    return std::make_unique<OpenCVDnnEngine>();
}

QVector<EngineType> InferenceFactory::availableEngines()
{
    QVector<EngineType> engines;
    
    // OpenCV DNN 总是可用
    engines.append(EngineType::OpenCVDnn);

#ifdef USE_ONNXRUNTIME
    engines.append(EngineType::OnnxRuntime);
#endif

#ifdef USE_NCNN
    engines.append(EngineType::NCNN);
#endif

#ifdef USE_TENSORRT
    engines.append(EngineType::TensorRT);
#endif

    return engines;
}

QString InferenceFactory::engineName(EngineType type)
{
    switch (type) {
        case EngineType::OnnxRuntime:
            return "ONNX Runtime";
        case EngineType::OpenCVDnn:
            return "OpenCV DNN";
        case EngineType::TensorRT:
            return "TensorRT";
        case EngineType::NCNN:
            return "NCNN";
        default:
            return "Unknown";
    }
}

bool InferenceFactory::isEngineAvailable(EngineType type)
{
    switch (type) {
        case EngineType::OpenCVDnn:
            return true;
        case EngineType::OnnxRuntime:
#ifdef USE_ONNXRUNTIME
            return true;
#else
            return false;
#endif
        case EngineType::NCNN:
#ifdef USE_NCNN
            return true;
#else
            return false;
#endif
        case EngineType::TensorRT:
#ifdef USE_TENSORRT
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

EngineType InferenceFactory::recommendedEngine()
{
#ifdef ARM_PLATFORM
    // ARM 平台优先推荐 NCNN
#ifdef USE_NCNN
    return EngineType::NCNN;
#else
    return EngineType::OpenCVDnn;
#endif
#else
    // x86 平台优先推荐 ONNX Runtime
#ifdef USE_ONNXRUNTIME
    return EngineType::OnnxRuntime;
#else
    return EngineType::OpenCVDnn;
#endif
#endif
}

} // namespace yolo
