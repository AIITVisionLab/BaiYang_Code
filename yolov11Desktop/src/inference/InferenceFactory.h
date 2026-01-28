/**
 * @file InferenceFactory.h
 * 推理引擎工厂
 */

#ifndef INFERENCE_FACTORY_H
#define INFERENCE_FACTORY_H

#include "InferenceEngine.h"
#include <memory>

namespace yolo {

/**
 * 推理引擎工厂
 * 根据配置创建合适的引擎
 */
class InferenceFactory {
public:
    /**
     * 创建指定类型的引擎
     */
    static std::unique_ptr<InferenceEngine> create(EngineType type);

    /**
     * 自动选择合适的引擎
     */
    static std::unique_ptr<InferenceEngine> createBest(bool useGPU = false);

    /**
     * 根据模型文件选择引擎
     */
    static std::unique_ptr<InferenceEngine> createForModel(const QString& modelPath);

    /**
     * 获取可用的引擎类型
     */
    static QVector<EngineType> availableEngines();

    /**
     * 获取引擎名称
     */
    static QString engineName(EngineType type);

    /**
     * 检查引擎是否可用
     */
    static bool isEngineAvailable(EngineType type);

    /**
     * 获取推荐的引擎类型（树莓派等平台）
     */
    static EngineType recommendedEngine();
};

} // namespace yolo

#endif // INFERENCE_FACTORY_H
