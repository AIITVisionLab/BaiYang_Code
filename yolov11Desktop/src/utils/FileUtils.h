/**
 * @file FileUtils.h
 * @brief 文件与路径工具函数
 */

#ifndef YOLO_FILEUTILS_H
#define YOLO_FILEUTILS_H

#include <QString>
#include <QStringList>
#include <QImage>
#include <QJsonObject>
#include <opencv2/core.hpp>
#include <vector>

namespace yolo {

/**
 * @brief 文件工具类
 *
 * 提供常用的文件操作与路径处理功能
 */
class FileUtils
{
public:
    /**
     * @brief 支持的图片格式
     */
    static QStringList supportedImageFormats();
    
    /**
     * @brief 支持的视频格式
     */
    static QStringList supportedVideoFormats();
    
    /**
     * @brief 支持的模型格式
     */
    static QStringList supportedModelFormats();
    
    /**
     * @brief 检查是否为图片文件
     */
    static bool isImageFile(const QString& path);
    
    /**
     * @brief 检查是否为视频文件
     */
    static bool isVideoFile(const QString& path);
    
    /**
     * @brief 检查是否为模型文件
     */
    static bool isModelFile(const QString& path);
    
    /**
     * @brief 检查是否为 RTSP 流
     */
    static bool isRtspStream(const QString& path);
    
    /**
     * @brief 检查是否为摄像头索引
     */
    static bool isCameraIndex(const QString& path);
    
    /**
     * @brief 获取文件扩展名（小写，不含点）
     */
    static QString getExtension(const QString& path);
    
    /**
     * @brief 获取文件名（不含扩展名）
     */
    static QString getBaseName(const QString& path);
    
    /**
     * @brief 获取目录路径
     */
    static QString getDirectory(const QString& path);
    
    /**
     * @brief 生成唯一文件名
     */
    static QString generateUniqueFileName(const QString& basePath, const QString& extension);
    
    /**
     * @brief 生成带时间戳的文件名
     */
    static QString generateTimestampFileName(const QString& prefix, const QString& extension);
    
    /**
     * @brief 确保目录存在
     */
    static bool ensureDirectoryExists(const QString& path);
    
    /**
     * @brief 获取文件大小（可读格式）
     */
    static QString formatFileSize(qint64 bytes);
    
    /**
     * @brief 扫描目录中的图片文件
     */
    static QStringList scanImagesInDirectory(const QString& path, bool recursive = false);
    
    /**
     * @brief 扫描目录中的视频文件
     */
    static QStringList scanVideosInDirectory(const QString& path, bool recursive = false);
    
    /**
     * @brief 读取图片为 OpenCV Mat
     */
    static cv::Mat readImage(const QString& path);
    
    /**
     * @brief 保存 OpenCV Mat 为图片
     */
    static bool saveImage(const cv::Mat& image, const QString& path, int quality = 95);
    
    /**
     * @brief QImage 转 OpenCV Mat
     */
    static cv::Mat qImageToMat(const QImage& image);
    
    /**
     * @brief OpenCV Mat 转 QImage
     */
    static QImage matToQImage(const cv::Mat& mat);
    
    /**
     * @brief 读取文本文件
     */
    static QString readTextFile(const QString& path);
    
    /**
     * @brief 写入文本文件
     */
    static bool writeTextFile(const QString& path, const QString& content);
    
    /**
     * @brief 读取类别标签文件
     */
    static QStringList readLabelsFile(const QString& path);
    
    /**
     * @brief 读取 JSON 文件
     */
    static QJsonObject readJsonFile(const QString& path);
    
    /**
     * @brief 写入 JSON 文件
     */
    static bool writeJsonFile(const QString& path, const QJsonObject& json, bool indented = true);
    
    /**
     * @brief 复制文件
     */
    static bool copyFile(const QString& src, const QString& dst, bool overwrite = false);
    
    /**
     * @brief 移动文件
     */
    static bool moveFile(const QString& src, const QString& dst, bool overwrite = false);
    
    /**
     * @brief 删除文件
     */
    static bool removeFile(const QString& path);
    
    /**
     * @brief 获取临时文件路径
     */
    static QString getTempFilePath(const QString& prefix = "yolo", const QString& extension = "tmp");
    
    /**
     * @brief 获取应用数据目录
     */
    static QString getAppDataPath();
    
    /**
     * @brief 获取应用缓存目录
     */
    static QString getCachePath();
    
    /**
     * @brief 清理缓存目录
     */
    static void clearCache(qint64 maxAgeMs = -1);

private:
    FileUtils() = default;  // 禁止实例化
};

/**
 * @brief 图片批量处理器
 */
class ImageBatchProcessor
{
public:
    /**
     * @brief 处理回调
     */
    using ProcessCallback = std::function<bool(const QString& path, const cv::Mat& image, int index, int total)>;
    
    /**
     * @brief 处理目录中的所有图片
     */
    static int processDirectory(const QString& inputDir, 
                                const QString& outputDir,
                                ProcessCallback callback,
                                bool recursive = false);
    
    /**
     * @brief 处理图片列表
     */
    static int processFiles(const QStringList& files,
                            const QString& outputDir,
                            ProcessCallback callback);
    
    /**
     * @brief 调整图片大小
     */
    static cv::Mat resize(const cv::Mat& image, int width, int height, bool keepAspectRatio = true);
    
    /**
     * @brief 裁剪图片
     */
    static cv::Mat crop(const cv::Mat& image, int x, int y, int width, int height);
    
    /**
     * @brief 添加填充（letterbox）
     */
    static cv::Mat letterbox(const cv::Mat& image, int targetWidth, int targetHeight,
                             const cv::Scalar& color = cv::Scalar(114, 114, 114));
};

/**
 * @brief 视频信息结构
 */
struct VideoInfo {
    QString path;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    int frameCount = 0;
    double durationSec = 0.0;
    QString codec;
    bool isValid = false;
    
    QString formatDuration() const;
};

/**
 * @brief 获取视频信息
 */
VideoInfo getVideoInfo(const QString& path);

} // namespace yolo

#endif // YOLO_FILEUTILS_H
