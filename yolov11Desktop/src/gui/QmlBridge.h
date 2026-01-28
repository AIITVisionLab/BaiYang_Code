/**
 * @file QmlBridge.h
 * @brief QML 与 C++ 后端的桥接类
 */

#ifndef QML_BRIDGE_H
#define QML_BRIDGE_H

#include <QObject>
#include <QImage>
#include <QUrl>
#include <QThread>
#include <QMutex>
#include <QTimer>
#include <memory>
#include <atomic>

#include "Detection.h"
#include "ClassLabels.h"
#include "InferenceEngine.h"
#include "FrameProvider.h"
#include "DrawUtils.h"

namespace yolo {

/**
 * @brief 应用状态
 */
enum class AppState {
    Idle,           ///< 空闲
    Loading,        ///< 加载中
    Ready,          ///< 就绪
    Running,        ///< 运行中
    Paused,         ///< 暂停
    Error           ///< 错误
};

/**
 * @brief QML 桥接类
 *
 * 向 QML 前端提供接口，包括：
 * - 模型加载与推理控制
 * - 视频/图像源管理
 * - 检测结果访问
 * - 配置管理
 */
class QmlBridge : public QObject {
    Q_OBJECT
    
    // 属性定义
    Q_PROPERTY(bool modelLoaded READ isModelLoaded NOTIFY modelLoadedChanged)
    Q_PROPERTY(bool isRunning READ isRunning NOTIFY runningChanged)
    Q_PROPERTY(bool isPaused READ isPaused NOTIFY pausedChanged)
    Q_PROPERTY(double fps READ fps NOTIFY fpsChanged)
    Q_PROPERTY(double inferenceTime READ inferenceTime NOTIFY inferenceTimeChanged)
    Q_PROPERTY(int detectionCount READ detectionCount NOTIFY detectionCountChanged)
    Q_PROPERTY(QString currentSource READ currentSource NOTIFY sourceChanged)
    Q_PROPERTY(QString modelName READ modelName NOTIFY modelLoadedChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusChanged)
    Q_PROPERTY(float confidenceThreshold READ confidenceThreshold WRITE setConfidenceThreshold NOTIFY confidenceThresholdChanged)
    Q_PROPERTY(float iouThreshold READ iouThreshold WRITE setIoUThreshold NOTIFY iouThresholdChanged)
    Q_PROPERTY(QSize frameSize READ frameSize NOTIFY frameSizeChanged)
    Q_PROPERTY(int progress READ progress NOTIFY progressChanged)
    Q_PROPERTY(QStringList availableEngines READ availableEngines CONSTANT)
    Q_PROPERTY(QStringList availableCameras READ availableCameras NOTIFY camerasChanged)

public:
    explicit QmlBridge(QObject* parent = nullptr);
    ~QmlBridge() override;

    /// @brief 属性访问器
    bool isModelLoaded() const { return m_modelLoaded; }
    bool isRunning() const { return m_state == AppState::Running || m_state == AppState::Paused; }
    bool isPaused() const { return m_state == AppState::Paused; }
    double fps() const { return m_fps; }
    double inferenceTime() const { return m_inferenceTime; }
    int detectionCount() const { return m_detectionCount; }
    QString currentSource() const { return m_currentSource; }
    QString modelName() const { return m_modelName; }
    QString statusText() const { return m_statusText; }
    float confidenceThreshold() const { return m_confThreshold; }
    float iouThreshold() const { return m_iouThreshold; }
    QSize frameSize() const { return m_frameSize; }
    int progress() const { return m_progress; }
    QStringList availableEngines() const;
    QStringList availableCameras() const;

    /// @brief 属性设置器
    void setConfidenceThreshold(float value);
    void setIoUThreshold(float value);

public slots:
    // 模型操作
    /**
     * @brief 加载模型
     * @param modelPath 模型文件路径
     * @param engineType 推理引擎类型 (可选)
     */
    void loadModel(const QString& modelPath, const QString& engineType = "");
    
    /**
     * @brief 卸载模型
     */
    void unloadModel();

    /**
     * @brief 加载类别标签
     */
    void loadLabels(const QString& labelsPath);

    // 源操作
    /**
     * @brief 打开摄像头
     * @param cameraId 摄像头ID或设备路径
     */
    void openCamera(const QString& cameraId = "0");

    /**
     * @brief 打开视频文件
     */
    void openVideo(const QString& videoPath);

    /**
     * @brief 打开图片文件
     */
    void openImage(const QString& imagePath);

    /**
     * @brief 打开图片文件夹
     */
    void openImageFolder(const QString& folderPath);

    /**
     * @brief 打开RTSP流
     */
    void openRtspStream(const QString& url);

    /**
     * @brief 关闭当前源
     */
    void closeSource();

    // 控制操作
    /**
     * @brief 开始检测
     */
    void start();

    /**
     * @brief 停止检测
     */
    void stop();

    /**
     * @brief 暂停
     */
    void pause();

    /**
     * @brief 恢复
     */
    void resume();

    /**
     * @brief 单帧推理 (用于图片)
     */
    void inferSingle();

    /**
     * @brief 跳转到指定帧
     */
    void seekTo(int frameNumber);

    // 导出操作
    /**
     * @brief 导出当前帧
     */
    void exportCurrentFrame(const QString& path);

    /**
     * @brief 导出检测结果到JSON
     */
    void exportResults(const QString& path);

    /**
     * @brief 开始录制
     */
    void startRecording(const QString& path);

    /**
     * @brief 停止录制
     */
    void stopRecording();

    // 配置操作
    /**
     * @brief 设置输入尺寸
     */
    void setInputSize(int width, int height);

    /**
     * @brief 启用/禁用GPU
     */
    void setUseGPU(bool enabled);

    /**
     * @brief 设置目标帧率
     */
    void setTargetFps(double fps);

    /**
     * @brief 设置绘制样式
     */
    void setDrawStyle(bool showLabels, bool showConfidence, int lineWidth);

    /**
     * @brief 设置类别过滤
     */
    void setClassFilter(const QVariantList& enabledClasses);

    /**
     * @brief 获取当前检测结果
     */
    QVariantList getDetections() const;

    /**
     * @brief 获取类别列表
     */
    QVariantList getClassList() const;

    /**
     * @brief 刷新摄像头列表
     */
    void refreshCameras();

signals:
    // 状态信号
    void modelLoadedChanged();
    void runningChanged();
    void pausedChanged();
    void fpsChanged();
    void inferenceTimeChanged();
    void detectionCountChanged();
    void sourceChanged();
    void statusChanged();
    void confidenceThresholdChanged();
    void iouThresholdChanged();
    void frameSizeChanged();
    void progressChanged();
    void camerasChanged();

    // 帧和检测结果信号
    void frameReady(const QImage& frame);
    void detectionsReady(const QVariantList& detections);
    void frameWithDetectionsReady(const QImage& frame);

    // 事件信号
    void modelLoadProgress(int progress, const QString& message);
    void errorOccurred(const QString& error);
    void sourceFinished();
    void recordingStarted();
    void recordingStopped(const QString& path);

private slots:
    void onFrameReady(const Frame& frame);
    void onSourceFinished();
    void onSourceError(const QString& error);
    void updateFps();

private:
    void setState(AppState state);
    void setStatus(const QString& status);
    void processFrame(const Frame& frame);
    void setupConnections();
    void createFrameProvider(SourceType type);

private:
    // 核心组件
    std::unique_ptr<InferenceEngine> m_engine;
    std::unique_ptr<FrameProvider> m_frameProvider;
    ClassLabels m_labels;
    DrawStyle m_drawStyle;
    InferenceConfig m_inferenceConfig;

    // 状态
    AppState m_state;
    bool m_modelLoaded;
    QString m_modelName;
    QString m_currentSource;
    QString m_statusText;
    QSize m_frameSize;
    int m_progress;

    // 配置
    float m_confThreshold;
    float m_iouThreshold;
    bool m_useGPU;

    // 统计
    double m_fps;
    double m_inferenceTime;
    int m_detectionCount;
    QTimer* m_fpsTimer;
    int m_frameCount;
    qint64 m_lastFpsTime;

    // 线程安全
    QMutex m_mutex;
    DetectionResult m_lastResult;
    QImage m_lastFrame;
    std::atomic<bool> m_inferencing{false};  ///< 是否正在推理中
    std::atomic<qint64> m_lastInferTime{0};   ///< 上次推理时间戳(ms)
    int m_minInferIntervalMs{16};             ///< 最小推理间隔(ms) ~60fps目标

    // 摄像头列表
    QStringList m_cameras;
};

} // namespace yolo

#endif // QML_BRIDGE_H
