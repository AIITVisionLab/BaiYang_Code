/**
 * @file Config.cpp
 * @brief 应用程序配置管理实现
 */

#include "Config.h"
#include <QCoreApplication>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QJsonArray>
#include <QFile>
#include <QDebug>

namespace yolo {

Config& Config::instance()
{
    static Config instance;
    return instance;
}

Config::Config()
{
    QString configPath = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    m_settings = std::make_unique<QSettings>(configPath + "/config.ini", QSettings::IniFormat);
    
    setDefault();
    load();
}

Config::~Config()
{
    save();
}

void Config::setDefault()
{
    m_lastModelPath = "";
    m_preferredEngine = "ONNX Runtime";
    m_inputSize = QSize(640, 640);
    m_confThreshold = 0.25f;
    m_iouThreshold = 0.45f;
    m_maxDetections = 300;
    m_useGPU = false;
    m_gpuDeviceId = 0;
    m_showLabels = true;
    m_showConfidence = true;
    m_showBoundingBoxes = true;
    m_lineWidth = 2;
    m_fontSize = 12;
    m_targetFps = 30.0;
    m_autoStart = false;
    m_loopVideo = false;
    m_cameraWidth = 1280;
    m_cameraHeight = 720;
    m_cameraFps = 30;
    m_exportPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    m_imageFormat = "png";
    m_imageQuality = 95;
    m_darkMode = true;
    m_language = "zh_CN";
    m_windowSize = QSize(1280, 720);
    m_numThreads = 4;
    m_enableProfiling = false;
}

void Config::save()
{
    m_settings->beginGroup("Model");
    m_settings->setValue("lastPath", m_lastModelPath);
    m_settings->setValue("preferredEngine", m_preferredEngine);
    m_settings->setValue("inputWidth", m_inputSize.width());
    m_settings->setValue("inputHeight", m_inputSize.height());
    m_settings->endGroup();

    m_settings->beginGroup("Detection");
    m_settings->setValue("confidenceThreshold", m_confThreshold);
    m_settings->setValue("iouThreshold", m_iouThreshold);
    m_settings->setValue("maxDetections", m_maxDetections);
    m_settings->endGroup();

    m_settings->beginGroup("GPU");
    m_settings->setValue("useGPU", m_useGPU);
    m_settings->setValue("deviceId", m_gpuDeviceId);
    m_settings->endGroup();

    m_settings->beginGroup("Display");
    m_settings->setValue("showLabels", m_showLabels);
    m_settings->setValue("showConfidence", m_showConfidence);
    m_settings->setValue("showBoundingBoxes", m_showBoundingBoxes);
    m_settings->setValue("lineWidth", m_lineWidth);
    m_settings->setValue("fontSize", m_fontSize);
    m_settings->endGroup();

    m_settings->beginGroup("Video");
    m_settings->setValue("targetFps", m_targetFps);
    m_settings->setValue("autoStart", m_autoStart);
    m_settings->setValue("loopVideo", m_loopVideo);
    m_settings->endGroup();

    m_settings->beginGroup("Camera");
    m_settings->setValue("width", m_cameraWidth);
    m_settings->setValue("height", m_cameraHeight);
    m_settings->setValue("fps", m_cameraFps);
    m_settings->endGroup();

    m_settings->beginGroup("Export");
    m_settings->setValue("path", m_exportPath);
    m_settings->setValue("imageFormat", m_imageFormat);
    m_settings->setValue("imageQuality", m_imageQuality);
    m_settings->endGroup();

    m_settings->beginGroup("UI");
    m_settings->setValue("darkMode", m_darkMode);
    m_settings->setValue("language", m_language);
    m_settings->setValue("windowWidth", m_windowSize.width());
    m_settings->setValue("windowHeight", m_windowSize.height());
    m_settings->endGroup();

    m_settings->beginGroup("Advanced");
    m_settings->setValue("numThreads", m_numThreads);
    m_settings->setValue("enableProfiling", m_enableProfiling);
    m_settings->endGroup();

    m_settings->beginGroup("Recent");
    m_settings->setValue("models", m_recentModels);
    m_settings->setValue("sources", m_recentSources);
    m_settings->endGroup();

    m_settings->sync();
}

void Config::load()
{
    m_settings->beginGroup("Model");
    m_lastModelPath = m_settings->value("lastPath", m_lastModelPath).toString();
    m_preferredEngine = m_settings->value("preferredEngine", m_preferredEngine).toString();
    m_inputSize.setWidth(m_settings->value("inputWidth", m_inputSize.width()).toInt());
    m_inputSize.setHeight(m_settings->value("inputHeight", m_inputSize.height()).toInt());
    m_settings->endGroup();

    m_settings->beginGroup("Detection");
    m_confThreshold = m_settings->value("confidenceThreshold", m_confThreshold).toFloat();
    m_iouThreshold = m_settings->value("iouThreshold", m_iouThreshold).toFloat();
    m_maxDetections = m_settings->value("maxDetections", m_maxDetections).toInt();
    m_settings->endGroup();

    m_settings->beginGroup("GPU");
    m_useGPU = m_settings->value("useGPU", m_useGPU).toBool();
    m_gpuDeviceId = m_settings->value("deviceId", m_gpuDeviceId).toInt();
    m_settings->endGroup();

    m_settings->beginGroup("Display");
    m_showLabels = m_settings->value("showLabels", m_showLabels).toBool();
    m_showConfidence = m_settings->value("showConfidence", m_showConfidence).toBool();
    m_showBoundingBoxes = m_settings->value("showBoundingBoxes", m_showBoundingBoxes).toBool();
    m_lineWidth = m_settings->value("lineWidth", m_lineWidth).toInt();
    m_fontSize = m_settings->value("fontSize", m_fontSize).toInt();
    m_settings->endGroup();

    m_settings->beginGroup("Video");
    m_targetFps = m_settings->value("targetFps", m_targetFps).toDouble();
    m_autoStart = m_settings->value("autoStart", m_autoStart).toBool();
    m_loopVideo = m_settings->value("loopVideo", m_loopVideo).toBool();
    m_settings->endGroup();

    m_settings->beginGroup("Camera");
    m_cameraWidth = m_settings->value("width", m_cameraWidth).toInt();
    m_cameraHeight = m_settings->value("height", m_cameraHeight).toInt();
    m_cameraFps = m_settings->value("fps", m_cameraFps).toInt();
    m_settings->endGroup();

    m_settings->beginGroup("Export");
    m_exportPath = m_settings->value("path", m_exportPath).toString();
    m_imageFormat = m_settings->value("imageFormat", m_imageFormat).toString();
    m_imageQuality = m_settings->value("imageQuality", m_imageQuality).toInt();
    m_settings->endGroup();

    m_settings->beginGroup("UI");
    m_darkMode = m_settings->value("darkMode", m_darkMode).toBool();
    m_language = m_settings->value("language", m_language).toString();
    m_windowSize.setWidth(m_settings->value("windowWidth", m_windowSize.width()).toInt());
    m_windowSize.setHeight(m_settings->value("windowHeight", m_windowSize.height()).toInt());
    m_settings->endGroup();

    m_settings->beginGroup("Advanced");
    m_numThreads = m_settings->value("numThreads", m_numThreads).toInt();
    m_enableProfiling = m_settings->value("enableProfiling", m_enableProfiling).toBool();
    m_settings->endGroup();

    m_settings->beginGroup("Recent");
    m_recentModels = m_settings->value("models").toStringList();
    m_recentSources = m_settings->value("sources").toStringList();
    m_settings->endGroup();
}

void Config::reset()
{
    setDefault();
    save();
}

// Getters and Setters
QString Config::lastModelPath() const { return m_lastModelPath; }
void Config::setLastModelPath(const QString& path) { m_lastModelPath = path; }

QString Config::preferredEngine() const { return m_preferredEngine; }
void Config::setPreferredEngine(const QString& engine) { m_preferredEngine = engine; }

QSize Config::inputSize() const { return m_inputSize; }
void Config::setInputSize(const QSize& size) { m_inputSize = size; }

float Config::confidenceThreshold() const { return m_confThreshold; }
void Config::setConfidenceThreshold(float value) { m_confThreshold = value; }

float Config::iouThreshold() const { return m_iouThreshold; }
void Config::setIoUThreshold(float value) { m_iouThreshold = value; }

int Config::maxDetections() const { return m_maxDetections; }
void Config::setMaxDetections(int value) { m_maxDetections = value; }

bool Config::useGPU() const { return m_useGPU; }
void Config::setUseGPU(bool enabled) { m_useGPU = enabled; }

int Config::gpuDeviceId() const { return m_gpuDeviceId; }
void Config::setGpuDeviceId(int id) { m_gpuDeviceId = id; }

bool Config::showLabels() const { return m_showLabels; }
void Config::setShowLabels(bool show) { m_showLabels = show; }

bool Config::showConfidence() const { return m_showConfidence; }
void Config::setShowConfidence(bool show) { m_showConfidence = show; }

bool Config::showBoundingBoxes() const { return m_showBoundingBoxes; }
void Config::setShowBoundingBoxes(bool show) { m_showBoundingBoxes = show; }

int Config::lineWidth() const { return m_lineWidth; }
void Config::setLineWidth(int width) { m_lineWidth = width; }

int Config::fontSize() const { return m_fontSize; }
void Config::setFontSize(int size) { m_fontSize = size; }

double Config::targetFps() const { return m_targetFps; }
void Config::setTargetFps(double fps) { m_targetFps = fps; }

bool Config::autoStart() const { return m_autoStart; }
void Config::setAutoStart(bool enabled) { m_autoStart = enabled; }

bool Config::loopVideo() const { return m_loopVideo; }
void Config::setLoopVideo(bool enabled) { m_loopVideo = enabled; }

int Config::cameraWidth() const { return m_cameraWidth; }
void Config::setCameraWidth(int width) { m_cameraWidth = width; }

int Config::cameraHeight() const { return m_cameraHeight; }
void Config::setCameraHeight(int height) { m_cameraHeight = height; }

int Config::cameraFps() const { return m_cameraFps; }
void Config::setCameraFps(int fps) { m_cameraFps = fps; }

QString Config::exportPath() const { return m_exportPath; }
void Config::setExportPath(const QString& path) { m_exportPath = path; }

QString Config::imageFormat() const { return m_imageFormat; }
void Config::setImageFormat(const QString& format) { m_imageFormat = format; }

int Config::imageQuality() const { return m_imageQuality; }
void Config::setImageQuality(int quality) { m_imageQuality = quality; }

bool Config::darkMode() const { return m_darkMode; }
void Config::setDarkMode(bool enabled) { m_darkMode = enabled; }

QString Config::language() const { return m_language; }
void Config::setLanguage(const QString& lang) { m_language = lang; }

QSize Config::windowSize() const { return m_windowSize; }
void Config::setWindowSize(const QSize& size) { m_windowSize = size; }

int Config::numThreads() const { return m_numThreads; }
void Config::setNumThreads(int threads) { m_numThreads = threads; }

bool Config::enableProfiling() const { return m_enableProfiling; }
void Config::setEnableProfiling(bool enabled) { m_enableProfiling = enabled; }

QStringList Config::recentModels() const { return m_recentModels; }

void Config::addRecentModel(const QString& path)
{
    m_recentModels.removeAll(path);
    m_recentModels.prepend(path);
    if (m_recentModels.size() > 10) {
        m_recentModels = m_recentModels.mid(0, 10);
    }
}

QStringList Config::recentSources() const { return m_recentSources; }

void Config::addRecentSource(const QString& path)
{
    m_recentSources.removeAll(path);
    m_recentSources.prepend(path);
    if (m_recentSources.size() > 10) {
        m_recentSources = m_recentSources.mid(0, 10);
    }
}

QJsonObject Config::toJson() const
{
    QJsonObject json;
    
    QJsonObject model;
    model["lastPath"] = m_lastModelPath;
    model["preferredEngine"] = m_preferredEngine;
    model["inputWidth"] = m_inputSize.width();
    model["inputHeight"] = m_inputSize.height();
    json["model"] = model;
    
    QJsonObject detection;
    detection["confidenceThreshold"] = m_confThreshold;
    detection["iouThreshold"] = m_iouThreshold;
    detection["maxDetections"] = m_maxDetections;
    json["detection"] = detection;
    
    QJsonObject gpu;
    gpu["useGPU"] = m_useGPU;
    gpu["deviceId"] = m_gpuDeviceId;
    json["gpu"] = gpu;
    
    return json;
}

void Config::fromJson(const QJsonObject& json)
{
    if (json.contains("model")) {
        QJsonObject model = json["model"].toObject();
        m_lastModelPath = model["lastPath"].toString();
        m_preferredEngine = model["preferredEngine"].toString();
        m_inputSize.setWidth(model["inputWidth"].toInt());
        m_inputSize.setHeight(model["inputHeight"].toInt());
    }
    
    if (json.contains("detection")) {
        QJsonObject detection = json["detection"].toObject();
        m_confThreshold = detection["confidenceThreshold"].toDouble();
        m_iouThreshold = detection["iouThreshold"].toDouble();
        m_maxDetections = detection["maxDetections"].toInt();
    }
    
    if (json.contains("gpu")) {
        QJsonObject gpu = json["gpu"].toObject();
        m_useGPU = gpu["useGPU"].toBool();
        m_gpuDeviceId = gpu["deviceId"].toInt();
    }
}

bool Config::exportToFile(const QString& path) const
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    
    QJsonDocument doc(toJson());
    file.write(doc.toJson(QJsonDocument::Indented));
    return true;
}

bool Config::importFromFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (doc.isNull()) {
        return false;
    }
    
    fromJson(doc.object());
    return true;
}

} // namespace yolo
