/**
 * @file Config.h
 * @brief 应用程序配置管理
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <QString>
#include <QSettings>
#include <QSize>
#include <QColor>
#include <QJsonObject>
#include <memory>

namespace yolo {

/**
 * @brief 应用程序配置单例
 */
class Config {
public:
    static Config& instance();

    /// @brief 禁止拷贝
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    /// @name 模型设置
    /// @{
    QString lastModelPath() const;
    void setLastModelPath(const QString& path);
    
    QString preferredEngine() const;
    void setPreferredEngine(const QString& engine);
    
    QSize inputSize() const;
    void setInputSize(const QSize& size);

    /// @}

    /// @name 检测参数
    /// @{
    float confidenceThreshold() const;
    void setConfidenceThreshold(float value);
    
    float iouThreshold() const;
    void setIoUThreshold(float value);
    
    int maxDetections() const;
    void setMaxDetections(int value);

    /// @}

    /// @name GPU 设置
    /// @{
    bool useGPU() const;
    void setUseGPU(bool enabled);
    
    int gpuDeviceId() const;
    void setGpuDeviceId(int id);

    /// @}

    /// @name 显示设置
    /// @{
    bool showLabels() const;
    void setShowLabels(bool show);
    
    bool showConfidence() const;
    void setShowConfidence(bool show);
    
    bool showBoundingBoxes() const;
    void setShowBoundingBoxes(bool show);
    
    int lineWidth() const;
    void setLineWidth(int width);
    
    int fontSize() const;
    void setFontSize(int size);

    /// @}

    /// @name 视频设置
    /// @{
    double targetFps() const;
    void setTargetFps(double fps);
    
    bool autoStart() const;
    void setAutoStart(bool enabled);
    
    bool loopVideo() const;
    void setLoopVideo(bool enabled);

    /// @}

    /// @name 摄像头设置
    /// @{
    int cameraWidth() const;
    void setCameraWidth(int width);
    
    int cameraHeight() const;
    void setCameraHeight(int height);
    
    int cameraFps() const;
    void setCameraFps(int fps);

    /// @}

    /// @name 导出设置
    /// @{
    QString exportPath() const;
    void setExportPath(const QString& path);
    
    QString imageFormat() const;
    void setImageFormat(const QString& format);
    
    int imageQuality() const;
    void setImageQuality(int quality);

    /// @}

    /// @name 界面设置
    /// @{
    bool darkMode() const;
    void setDarkMode(bool enabled);
    
    QString language() const;
    void setLanguage(const QString& lang);
    
    QSize windowSize() const;
    void setWindowSize(const QSize& size);

    /// @}

    /// @name 高级设置
    /// @{
    int numThreads() const;
    void setNumThreads(int threads);
    
    bool enableProfiling() const;
    void setEnableProfiling(bool enabled);

    /// @}

    /// @name 最近文件
    /// @{
    QStringList recentModels() const;
    void addRecentModel(const QString& path);
    
    QStringList recentSources() const;
    void addRecentSource(const QString& path);

    /// @}

    /// @name 保存和加载
    /// @{
    void save();
    void load();
    void reset();

    /// @}

    /// @name JSON 导入导出
    /// @{
    QJsonObject toJson() const;
    void fromJson(const QJsonObject& json);
    bool exportToFile(const QString& path) const;
    bool importFromFile(const QString& path);

    /// @}

private:
    Config();
    ~Config();

    void setDefault();

    std::unique_ptr<QSettings> m_settings;
    
    /// @brief 缓存值（避免频繁读取）
    QString m_lastModelPath;
    QString m_preferredEngine;
    QSize m_inputSize;
    float m_confThreshold;
    float m_iouThreshold;
    int m_maxDetections;
    bool m_useGPU;
    int m_gpuDeviceId;
    bool m_showLabels;
    bool m_showConfidence;
    bool m_showBoundingBoxes;
    int m_lineWidth;
    int m_fontSize;
    double m_targetFps;
    bool m_autoStart;
    bool m_loopVideo;
    int m_cameraWidth;
    int m_cameraHeight;
    int m_cameraFps;
    QString m_exportPath;
    QString m_imageFormat;
    int m_imageQuality;
    bool m_darkMode;
    QString m_language;
    QSize m_windowSize;
    int m_numThreads;
    bool m_enableProfiling;
    QStringList m_recentModels;
    QStringList m_recentSources;
};

} // namespace yolo

#endif // CONFIG_H
