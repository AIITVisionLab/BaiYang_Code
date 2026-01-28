/**
 * @file QmlBridge.cpp
 * @brief QML 桥接类实现
 */

#include "QmlBridge.h"
#include "InferenceFactory.h"
#include <QDebug>
#include <QFileInfo>
#include <QDateTime>
#include <QJsonDocument>
#include <QFile>
#include <QUrl>
#include <QtConcurrent>

namespace yolo {

QmlBridge::QmlBridge(QObject* parent)
    : QObject(parent)
    , m_state(AppState::Idle)
    , m_modelLoaded(false)
    , m_progress(0)
    , m_confThreshold(0.15f)  // 默认置信度阈值
    , m_iouThreshold(0.45f)
    , m_useGPU(false)
    , m_fps(0)
    , m_inferenceTime(0)
    , m_detectionCount(0)
    , m_frameCount(0)
    , m_lastFpsTime(0)
{
    // 默认加载 COCO 标签
    m_labels.loadCocoLabels();

    // FPS 计时器
    m_fpsTimer = new QTimer(this);
    connect(m_fpsTimer, &QTimer::timeout, this, &QmlBridge::updateFps);
    m_fpsTimer->setInterval(1000);

    setStatus("就绪");
    refreshCameras();
}

QmlBridge::~QmlBridge()
{
    stop();
    unloadModel();
}

QStringList QmlBridge::availableEngines() const
{
    QStringList engines;
    auto available = InferenceFactory::availableEngines();
    for (auto type : available) {
        engines << InferenceFactory::engineName(type);
    }
    return engines;
}

QStringList QmlBridge::availableCameras() const
{
    return m_cameras;
}

void QmlBridge::setConfidenceThreshold(float value)
{
    if (qFuzzyCompare(m_confThreshold, value)) return;
    
    m_confThreshold = value;
    if (m_engine) {
        m_engine->setConfidenceThreshold(value);
    }
    emit confidenceThresholdChanged();
}

void QmlBridge::setIoUThreshold(float value)
{
    if (qFuzzyCompare(m_iouThreshold, value)) return;
    
    m_iouThreshold = value;
    if (m_engine) {
        m_engine->setIoUThreshold(value);
    }
    emit iouThresholdChanged();
}

void QmlBridge::loadModel(const QString& modelPath, const QString& engineType)
{
    if (modelPath.isEmpty()) {
        emit errorOccurred("模型路径不能为空");
        return;
    }

    // 将 URL 转换为本地文件路径
    QString localPath = modelPath;
    if (modelPath.startsWith("file://")) {
        localPath = QUrl(modelPath).toLocalFile();
    }

    setState(AppState::Loading);
    setStatus("正在加载模型...");

    // 卸载旧模型
    if (m_engine) {
        m_engine->unloadModel();
    }

    // 创建引擎
    if (engineType.isEmpty()) {
        m_engine = InferenceFactory::createForModel(localPath);
    } else {
        EngineType type = EngineType::OnnxRuntime;
        if (engineType == "OpenCV DNN") {
            type = EngineType::OpenCVDnn;
        } else if (engineType == "TensorRT") {
            type = EngineType::TensorRT;
        } else if (engineType == "NCNN") {
            type = EngineType::NCNN;
        }
        m_engine = InferenceFactory::create(type);
    }

    if (!m_engine) {
        setState(AppState::Error);
        setStatus("创建推理引擎失败");
        emit errorOccurred("无法创建推理引擎");
        return;
    }

    // 设置进度回调
    m_engine->setProgressCallback([this](int progress, const QString& message) {
        m_progress = progress;
        emit progressChanged();
        emit modelLoadProgress(progress, message);
    });

    // 配置引擎
    m_inferenceConfig.confThreshold = m_confThreshold;
    m_inferenceConfig.iouThreshold = m_iouThreshold;
    m_inferenceConfig.useGPU = m_useGPU;

    // 加载模型
    if (m_engine->loadModel(localPath, m_inferenceConfig)) {
        m_modelLoaded = true;
        m_modelName = QFileInfo(localPath).baseName();
        
        // 预热
        m_engine->warmup(3);
        
        setState(AppState::Ready);
        setStatus(QString("模型已加载: %1").arg(m_modelName));
        emit modelLoadedChanged();
        
        qDebug() << "Model loaded:" << m_modelName;
    } else {
        m_modelLoaded = false;
        setState(AppState::Error);
        setStatus("模型加载失败: " + m_engine->lastError());
        emit errorOccurred(m_engine->lastError());
    }
}

void QmlBridge::unloadModel()
{
    if (m_engine) {
        m_engine->unloadModel();
        m_engine.reset();
    }
    m_modelLoaded = false;
    m_modelName.clear();
    setState(AppState::Idle);
    setStatus("模型已卸载");
    emit modelLoadedChanged();
}

void QmlBridge::loadLabels(const QString& labelsPath)
{
    if (m_labels.loadFromFile(labelsPath)) {
        qDebug() << "Labels loaded:" << m_labels.count() << "classes";
    } else {
        emit errorOccurred("加载标签文件失败");
    }
}

void QmlBridge::openCamera(const QString& cameraId)
{
    closeSource();
    
    createFrameProvider(SourceType::Camera);
    auto* camera = static_cast<CameraFrameProvider*>(m_frameProvider.get());
    
    if (camera->open(cameraId)) {
        m_currentSource = QString("摄像头 %1").arg(cameraId);
        m_frameSize = camera->frameSize();
        setupConnections();
        setState(AppState::Ready);
        setStatus("摄像头已打开");
        emit sourceChanged();
        emit frameSizeChanged();
    } else {
        emit errorOccurred("打开摄像头失败");
    }
}

void QmlBridge::openVideo(const QString& videoPath)
{
    closeSource();
    
    // 将 URL 转换为本地文件路径
    QString localPath = videoPath;
    if (videoPath.startsWith("file://")) {
        localPath = QUrl(videoPath).toLocalFile();
    }
    
    createFrameProvider(SourceType::VideoFile);
    auto* video = static_cast<VideoFileProvider*>(m_frameProvider.get());
    
    if (video->open(localPath)) {
        m_currentSource = QFileInfo(localPath).fileName();
        m_frameSize = video->frameSize();
        setupConnections();
        setState(AppState::Ready);
        setStatus(QString("视频已打开: %1").arg(m_currentSource));
        emit sourceChanged();
        emit frameSizeChanged();
    } else {
        emit errorOccurred("打开视频失败");
    }
}

void QmlBridge::openImage(const QString& imagePath)
{
    closeSource();
    
    // 将 URL 转换为本地文件路径
    QString localPath = imagePath;
    if (imagePath.startsWith("file://")) {
        localPath = QUrl(imagePath).toLocalFile();
    }
    
    createFrameProvider(SourceType::ImageFile);
    auto* image = static_cast<ImageFileProvider*>(m_frameProvider.get());
    
    if (image->open(localPath)) {
        m_currentSource = QFileInfo(localPath).fileName();
        m_frameSize = image->frameSize();
        setupConnections();
        setState(AppState::Ready);
        setStatus(QString("图片已打开: %1").arg(m_currentSource));
        emit sourceChanged();
        emit frameSizeChanged();
        
        // 图片自动执行单帧推理
        inferSingle();
    } else {
        emit errorOccurred("打开图片失败");
    }
}

void QmlBridge::openImageFolder(const QString& folderPath)
{
    closeSource();
    
    // 将 URL 转换为本地文件路径
    QString localPath = folderPath;
    if (folderPath.startsWith("file://")) {
        localPath = QUrl(folderPath).toLocalFile();
    }
    
    createFrameProvider(SourceType::ImageFolder);
    auto* folder = static_cast<ImageFolderProvider*>(m_frameProvider.get());
    
    if (folder->open(localPath)) {
        m_currentSource = QFileInfo(localPath).fileName();
        m_frameSize = folder->frameSize();
        setupConnections();
        setState(AppState::Ready);
        setStatus(QString("文件夹已打开: %1 (%2张图片)")
                 .arg(m_currentSource)
                 .arg(folder->totalFrames()));
        emit sourceChanged();
        emit frameSizeChanged();
    } else {
        emit errorOccurred("打开文件夹失败");
    }
}

void QmlBridge::openRtspStream(const QString& url)
{
    closeSource();
    
    createFrameProvider(SourceType::RTSP);
    auto* rtsp = static_cast<RtspStreamProvider*>(m_frameProvider.get());
    
    if (rtsp->open(url)) {
        m_currentSource = url;
        m_frameSize = rtsp->frameSize();
        setupConnections();
        setState(AppState::Ready);
        setStatus("RTSP流已连接");
        emit sourceChanged();
        emit frameSizeChanged();
    } else {
        emit errorOccurred("连接RTSP流失败");
    }
}

void QmlBridge::closeSource()
{
    stop();
    
    if (m_frameProvider) {
        m_frameProvider->close();
        m_frameProvider.reset();
    }
    
    m_currentSource.clear();
    m_frameSize = QSize();
    setState(m_modelLoaded ? AppState::Ready : AppState::Idle);
    setStatus(m_modelLoaded ? "模型就绪" : "就绪");
    emit sourceChanged();
}

void QmlBridge::start()
{
    if (!m_frameProvider || !m_frameProvider->isOpened()) {
        emit errorOccurred("请先打开数据源");
        return;
    }

    // 如果处于暂停状态，直接恢复
    if (m_state == AppState::Paused) {
        resume();
        return;
    }
    
    // 允许在没有模型时预览，但会提示
    if (!m_modelLoaded) {
        qDebug() << "Starting preview without model loaded";
    }

    m_frameProvider->start();
    m_fpsTimer->start();
    m_frameCount = 0;
    m_lastFpsTime = QDateTime::currentMSecsSinceEpoch();
    
    setState(AppState::Running);
    setStatus(m_modelLoaded ? "正在检测..." : "预览中（未加载模型）");
    emit runningChanged();
}

void QmlBridge::stop()
{
    // 先重置推理状态，确保可以切换
    m_inferencing = false;
    
    if (m_frameProvider) {
        m_frameProvider->stop();
    }
    m_fpsTimer->stop();
    
    setState(m_modelLoaded ? AppState::Ready : AppState::Idle);
    setStatus("已停止");
    emit runningChanged();
}

void QmlBridge::pause()
{
    if (m_frameProvider) {
        m_frameProvider->pause();
    }
    // 重置推理状态
    m_inferencing = false;
    setState(AppState::Paused);
    setStatus("已暂停");
    emit pausedChanged();
}

void QmlBridge::resume()
{
    if (m_frameProvider) {
        m_frameProvider->resume();
    }
    // 重置推理状态以便继续
    m_inferencing = false;
    m_lastInferTime = 0;
    setState(AppState::Running);
    setStatus("正在检测...");
    emit pausedChanged();
}

void QmlBridge::inferSingle()
{
    if (!m_frameProvider || !m_frameProvider->isOpened() || !m_modelLoaded) {
        return;
    }

    Frame frame = m_frameProvider->getNextFrame();
    if (frame.isValid) {
        processFrame(frame);
    }
}

void QmlBridge::seekTo(int frameNumber)
{
    if (m_frameProvider) {
        m_frameProvider->seekTo(frameNumber);
    }
}

void QmlBridge::exportCurrentFrame(const QString& path)
{
    // 将 URL 转换为本地文件路径
    QString localPath = path;
    if (path.startsWith("file://")) {
        localPath = QUrl(path).toLocalFile();
    }
    
    QMutexLocker locker(&m_mutex);
    if (!m_lastFrame.isNull()) {
        m_lastFrame.save(localPath);
        setStatus(QString("帧已导出: %1").arg(localPath));
    }
}

void QmlBridge::exportResults(const QString& path)
{
    // 将 URL 转换为本地文件路径
    QString localPath = path;
    if (path.startsWith("file://")) {
        localPath = QUrl(path).toLocalFile();
    }
    
    QMutexLocker locker(&m_mutex);
    
    QJsonDocument doc(m_lastResult.toJson());
    QFile file(localPath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson(QJsonDocument::Indented));
        file.close();
        setStatus(QString("结果已导出: %1").arg(localPath));
    } else {
        emit errorOccurred("导出失败");
    }
}

void QmlBridge::startRecording(const QString& path)
{
    // TODO: 实现视频录制
    emit recordingStarted();
}

void QmlBridge::stopRecording()
{
    // TODO: 停止视频录制
    emit recordingStopped("");
}

void QmlBridge::setInputSize(int width, int height)
{
    m_inferenceConfig.inputSize = QSize(width, height);
    if (m_engine && m_modelLoaded) {
        // 需要重新加载模型
        qDebug() << "Input size changed, may need to reload model";
    }
}

void QmlBridge::setUseGPU(bool enabled)
{
    m_useGPU = enabled;
    m_inferenceConfig.useGPU = enabled;
}

void QmlBridge::setTargetFps(double fps)
{
    if (m_frameProvider) {
        m_frameProvider->setTargetFps(fps);
    }
}

void QmlBridge::setDrawStyle(bool showLabels, bool showConfidence, int lineWidth)
{
    m_drawStyle.showLabel = showLabels;
    m_drawStyle.showConfidence = showConfidence;
    m_drawStyle.lineWidth = lineWidth;
}

void QmlBridge::setClassFilter(const QVariantList& enabledClasses)
{
    // 先禁用所有类别
    for (int id : m_labels.getAllClassIds()) {
        m_labels.setClassEnabled(id, false);
    }
    
    // 启用指定类别
    for (const auto& var : enabledClasses) {
        m_labels.setClassEnabled(var.toInt(), true);
    }
}

QVariantList QmlBridge::getDetections() const
{
    QVariantList list;
    QMutexLocker locker(const_cast<QMutex*>(&m_mutex));
    
    for (const auto& det : m_lastResult.detections()) {
        QVariantMap map;
        map["classId"] = det.classId();
        map["className"] = m_labels.getClassName(det.classId());
        map["confidence"] = det.confidence();
        map["x"] = det.bbox().x;
        map["y"] = det.bbox().y;
        map["width"] = det.bbox().width;
        map["height"] = det.bbox().height;
        list.append(map);
    }
    
    return list;
}

QVariantList QmlBridge::getClassList() const
{
    QVariantList list;
    
    for (int id : m_labels.getAllClassIds()) {
        QVariantMap map;
        map["id"] = id;
        map["name"] = m_labels.getClassName(id);
        map["color"] = m_labels.getClassColor(id).name();
        map["enabled"] = m_labels.isClassEnabled(id);
        list.append(map);
    }
    
    return list;
}

void QmlBridge::refreshCameras()
{
    m_cameras = CameraFrameProvider::availableCameras();
    emit camerasChanged();
}

void QmlBridge::onFrameReady(const Frame& frame)
{
    processFrame(frame);
}

void QmlBridge::onSourceFinished()
{
    stop();
    setStatus("播放完成");
    emit sourceFinished();
}

void QmlBridge::onSourceError(const QString& error)
{
    setState(AppState::Error);
    setStatus("错误: " + error);
    emit errorOccurred(error);
}

void QmlBridge::updateFps()
{
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    qint64 elapsed = now - m_lastFpsTime;
    
    if (elapsed > 0) {
        m_fps = m_frameCount * 1000.0 / elapsed;
        emit fpsChanged();
    }
    
    m_frameCount = 0;
    m_lastFpsTime = now;
}

void QmlBridge::setState(AppState state)
{
    m_state = state;
}

void QmlBridge::setStatus(const QString& status)
{
    m_statusText = status;
    emit statusChanged();
}

void QmlBridge::processFrame(const Frame& frame)
{
    if (frame.image.empty()) {
        return;
    }

    m_frameCount++;

    // 如果没有模型，直接显示原始画面
    if (!m_engine || !m_modelLoaded) {
        QImage rawImage = DrawUtils::matToQImage(frame.image);
        if (!rawImage.isNull()) {
            m_mutex.lock();
            m_lastFrame = rawImage;
            m_mutex.unlock();
            emit frameWithDetectionsReady(rawImage);
        }
        return;
    }

    // 如果正在推理或暂停状态，跳过
    if (m_inferencing || m_state == AppState::Paused) {
        return;
    }

    // 推理节流 - 使用原子操作避免竞争
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - m_lastInferTime < m_minInferIntervalMs) {
        return;
    }
    
    m_lastInferTime = now;
    m_inferencing = true;

    // 直接使用 frame.image（避免不必要的克隆，因为 frame 是传值的）
    cv::Mat imageCopy = frame.image.clone();
    
    // 在后台线程执行推理
    QtConcurrent::run([this, imageCopy = std::move(imageCopy)]() {
        DetectionResult result;
        try {
            result = m_engine->infer(imageCopy);
        } catch (...) {
            m_inferencing = false;
            return;
        }

        // 直接在后台线程绘制到 cv::Mat（避免 QPainter 线程问题）
        cv::Mat outputMat = imageCopy.clone();
        DrawUtils::drawDetections(outputMat, result, m_labels, m_drawStyle);

        // 回到主线程更新 UI
        QMetaObject::invokeMethod(this, [this, outputMat, result]() {
            // 如果已暂停/停止，丢弃结果
            if (m_state != AppState::Running) {
                m_inferencing = false;
                return;
            }

            // 转换为 QImage
            QImage outputImage = DrawUtils::matToQImage(outputMat);
            
            if (outputImage.isNull()) {
                m_inferencing = false;
                return;
            }

            // 更新统计
            m_inferenceTime = result.totalTime();
            m_detectionCount = result.count();
            emit inferenceTimeChanged();
            emit detectionCountChanged();

            // 保存并发送结果
            m_mutex.lock();
            m_lastResult = result;
            m_lastFrame = outputImage;
            m_mutex.unlock();

            emit frameWithDetectionsReady(outputImage);
            emit detectionsReady(getDetections());

            m_inferencing = false;
        }, Qt::QueuedConnection);
    });
}

void QmlBridge::setupConnections()
{
    if (!m_frameProvider) return;

    connect(m_frameProvider.get(), &FrameProvider::frameReady,
            this, &QmlBridge::onFrameReady);
    connect(m_frameProvider.get(), &FrameProvider::finished,
            this, &QmlBridge::onSourceFinished);
    connect(m_frameProvider.get(), &FrameProvider::error,
            this, &QmlBridge::onSourceError);
    connect(m_frameProvider.get(), &FrameProvider::progressChanged,
            this, [this](int progress) {
                m_progress = progress;
                emit progressChanged();
            });
}

void QmlBridge::createFrameProvider(SourceType type)
{
    switch (type) {
        case SourceType::Camera:
            m_frameProvider = std::make_unique<CameraFrameProvider>();
            break;
        case SourceType::VideoFile:
            m_frameProvider = std::make_unique<VideoFileProvider>();
            break;
        case SourceType::ImageFile:
            m_frameProvider = std::make_unique<ImageFileProvider>();
            break;
        case SourceType::ImageFolder:
            m_frameProvider = std::make_unique<ImageFolderProvider>();
            break;
        case SourceType::RTSP:
            m_frameProvider = std::make_unique<RtspStreamProvider>();
            break;
        default:
            m_frameProvider = std::make_unique<CameraFrameProvider>();
            break;
    }
}

} // namespace yolo
