/**
 * @file FileUtils.cpp
 * @brief 文件与路径工具函数实现
 */

#include "FileUtils.h"
#include "Logger.h"
#include <QFile>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QTemporaryFile>
#include <QImageReader>
#include <QDirIterator>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace yolo {

QStringList FileUtils::supportedImageFormats()
{
    return {"jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff", "tif"};
}

QStringList FileUtils::supportedVideoFormats()
{
    return {"mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "mpeg", "mpg"};
}

QStringList FileUtils::supportedModelFormats()
{
    return {"onnx", "pt", "pth", "engine", "trt", "param", "bin", "ncnn"};
}

bool FileUtils::isImageFile(const QString& path)
{
    QString ext = getExtension(path);
    return supportedImageFormats().contains(ext, Qt::CaseInsensitive);
}

bool FileUtils::isVideoFile(const QString& path)
{
    QString ext = getExtension(path);
    return supportedVideoFormats().contains(ext, Qt::CaseInsensitive);
}

bool FileUtils::isModelFile(const QString& path)
{
    QString ext = getExtension(path);
    return supportedModelFormats().contains(ext, Qt::CaseInsensitive);
}

bool FileUtils::isRtspStream(const QString& path)
{
    return path.startsWith("rtsp://", Qt::CaseInsensitive) ||
           path.startsWith("rtmp://", Qt::CaseInsensitive) ||
           path.startsWith("http://", Qt::CaseInsensitive) ||
           path.startsWith("https://", Qt::CaseInsensitive);
}

bool FileUtils::isCameraIndex(const QString& path)
{
    bool ok;
    path.toInt(&ok);
    return ok;
}

QString FileUtils::getExtension(const QString& path)
{
    QFileInfo info(path);
    return info.suffix().toLower();
}

QString FileUtils::getBaseName(const QString& path)
{
    QFileInfo info(path);
    return info.completeBaseName();
}

QString FileUtils::getDirectory(const QString& path)
{
    QFileInfo info(path);
    return info.absolutePath();
}

QString FileUtils::generateUniqueFileName(const QString& basePath, const QString& extension)
{
    QString dir = getDirectory(basePath);
    QString base = getBaseName(basePath);
    QString ext = extension.startsWith('.') ? extension : '.' + extension;
    
    QString result = dir + '/' + base + ext;
    int counter = 1;
    
    while (QFile::exists(result)) {
        result = QString("%1/%2_%3%4").arg(dir, base).arg(counter).arg(ext);
        counter++;
    }
    
    return result;
}

QString FileUtils::generateTimestampFileName(const QString& prefix, const QString& extension)
{
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss_zzz");
    QString ext = extension.startsWith('.') ? extension : '.' + extension;
    return prefix + '_' + timestamp + ext;
}

bool FileUtils::ensureDirectoryExists(const QString& path)
{
    QDir dir(path);
    if (dir.exists()) return true;
    return dir.mkpath(".");
}

QString FileUtils::formatFileSize(qint64 bytes)
{
    const qint64 KB = 1024;
    const qint64 MB = 1024 * KB;
    const qint64 GB = 1024 * MB;
    
    if (bytes >= GB) {
        return QString("%1 GB").arg(bytes / static_cast<double>(GB), 0, 'f', 2);
    } else if (bytes >= MB) {
        return QString("%1 MB").arg(bytes / static_cast<double>(MB), 0, 'f', 2);
    } else if (bytes >= KB) {
        return QString("%1 KB").arg(bytes / static_cast<double>(KB), 0, 'f', 2);
    } else {
        return QString("%1 B").arg(bytes);
    }
}

QStringList FileUtils::scanImagesInDirectory(const QString& path, bool recursive)
{
    QStringList result;
    QStringList filters;
    
    for (const QString& ext : supportedImageFormats()) {
        filters << "*." + ext;
    }
    
    QDirIterator::IteratorFlags flags = QDirIterator::NoIteratorFlags;
    if (recursive) {
        flags = QDirIterator::Subdirectories;
    }
    
    QDirIterator it(path, filters, QDir::Files, flags);
    while (it.hasNext()) {
        result << it.next();
    }
    
    result.sort();
    return result;
}

QStringList FileUtils::scanVideosInDirectory(const QString& path, bool recursive)
{
    QStringList result;
    QStringList filters;
    
    for (const QString& ext : supportedVideoFormats()) {
        filters << "*." + ext;
    }
    
    QDirIterator::IteratorFlags flags = QDirIterator::NoIteratorFlags;
    if (recursive) {
        flags = QDirIterator::Subdirectories;
    }
    
    QDirIterator it(path, filters, QDir::Files, flags);
    while (it.hasNext()) {
        result << it.next();
    }
    
    result.sort();
    return result;
}

cv::Mat FileUtils::readImage(const QString& path)
{
    cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
    if (image.empty()) {
        LOG_WARNING(QString("Failed to read image: %1").arg(path));
    }
    return image;
}

bool FileUtils::saveImage(const cv::Mat& image, const QString& path, int quality)
{
    if (image.empty()) {
        LOG_ERROR("Cannot save empty image");
        return false;
    }
    
    ensureDirectoryExists(getDirectory(path));
    
    std::vector<int> params;
    QString ext = getExtension(path);
    
    if (ext == "jpg" || ext == "jpeg") {
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(quality);
    } else if (ext == "png") {
        params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        params.push_back(9 - quality / 11);  // 0-9, lower is better quality
    } else if (ext == "webp") {
        params.push_back(cv::IMWRITE_WEBP_QUALITY);
        params.push_back(quality);
    }
    
    bool success = cv::imwrite(path.toStdString(), image, params);
    if (!success) {
        LOG_ERROR(QString("Failed to save image: %1").arg(path));
    }
    return success;
}

cv::Mat FileUtils::qImageToMat(const QImage& image)
{
    QImage converted = image;
    
    switch (image.format()) {
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32:
        case QImage::Format_ARGB32_Premultiplied:
            converted = image.convertToFormat(QImage::Format_BGR888);
            break;
        case QImage::Format_RGB888:
            converted = image.rgbSwapped();
            break;
        case QImage::Format_Grayscale8:
            // 保持灰度
            break;
        default:
            converted = image.convertToFormat(QImage::Format_BGR888);
            break;
    }
    
    int type = converted.format() == QImage::Format_Grayscale8 ? CV_8UC1 : CV_8UC3;
    
    cv::Mat mat(converted.height(), converted.width(), type,
                const_cast<uchar*>(converted.bits()),
                static_cast<size_t>(converted.bytesPerLine()));
    
    return mat.clone();  // 必须克隆，因为QImage数据可能被释放
}

QImage FileUtils::matToQImage(const cv::Mat& mat)
{
    if (mat.empty()) {
        return QImage();
    }
    
    cv::Mat rgb;
    
    switch (mat.type()) {
        case CV_8UC1:
            return QImage(mat.data, mat.cols, mat.rows, 
                          static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
        case CV_8UC3:
            cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
            return QImage(rgb.data, rgb.cols, rgb.rows, 
                          static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
        case CV_8UC4:
            cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGBA);
            return QImage(rgb.data, rgb.cols, rgb.rows, 
                          static_cast<int>(rgb.step), QImage::Format_RGBA8888).copy();
        default:
            LOG_WARNING(QString("Unsupported Mat type: %1").arg(mat.type()));
            return QImage();
    }
}

QString FileUtils::readTextFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        LOG_ERROR(QString("Cannot open file for reading: %1").arg(path));
        return QString();
    }
    
    return QString::fromUtf8(file.readAll());
}

bool FileUtils::writeTextFile(const QString& path, const QString& content)
{
    ensureDirectoryExists(getDirectory(path));
    
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        LOG_ERROR(QString("Cannot open file for writing: %1").arg(path));
        return false;
    }
    
    file.write(content.toUtf8());
    return true;
}

QStringList FileUtils::readLabelsFile(const QString& path)
{
    QStringList labels;
    QString content = readTextFile(path);
    
    if (!content.isEmpty()) {
        QStringList lines = content.split('\n', Qt::SkipEmptyParts);
        for (const QString& line : lines) {
            labels << line.trimmed();
        }
    }
    
    return labels;
}

QJsonObject FileUtils::readJsonFile(const QString& path)
{
    QString content = readTextFile(path);
    if (content.isEmpty()) {
        return QJsonObject();
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(content.toUtf8());
    if (doc.isNull() || !doc.isObject()) {
        LOG_ERROR(QString("Invalid JSON file: %1").arg(path));
        return QJsonObject();
    }
    
    return doc.object();
}

bool FileUtils::writeJsonFile(const QString& path, const QJsonObject& json, bool indented)
{
    QJsonDocument doc(json);
    QString content = QString::fromUtf8(
        doc.toJson(indented ? QJsonDocument::Indented : QJsonDocument::Compact)
    );
    return writeTextFile(path, content);
}

bool FileUtils::copyFile(const QString& src, const QString& dst, bool overwrite)
{
    if (!QFile::exists(src)) {
        LOG_ERROR(QString("Source file does not exist: %1").arg(src));
        return false;
    }
    
    if (QFile::exists(dst)) {
        if (!overwrite) {
            LOG_ERROR(QString("Destination file already exists: %1").arg(dst));
            return false;
        }
        QFile::remove(dst);
    }
    
    ensureDirectoryExists(getDirectory(dst));
    return QFile::copy(src, dst);
}

bool FileUtils::moveFile(const QString& src, const QString& dst, bool overwrite)
{
    if (!QFile::exists(src)) {
        LOG_ERROR(QString("Source file does not exist: %1").arg(src));
        return false;
    }
    
    if (QFile::exists(dst)) {
        if (!overwrite) {
            LOG_ERROR(QString("Destination file already exists: %1").arg(dst));
            return false;
        }
        QFile::remove(dst);
    }
    
    ensureDirectoryExists(getDirectory(dst));
    return QFile::rename(src, dst);
}

bool FileUtils::removeFile(const QString& path)
{
    if (!QFile::exists(path)) {
        return true;  // 已经不存在
    }
    return QFile::remove(path);
}

QString FileUtils::getTempFilePath(const QString& prefix, const QString& extension)
{
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    return generateUniqueFileName(tempDir + '/' + prefix, extension);
}

QString FileUtils::getAppDataPath()
{
    QString path = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    ensureDirectoryExists(path);
    return path;
}

QString FileUtils::getCachePath()
{
    QString path = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    ensureDirectoryExists(path);
    return path;
}

void FileUtils::clearCache(qint64 maxAgeMs)
{
    QString cachePath = getCachePath();
    QDir cacheDir(cachePath);
    
    if (!cacheDir.exists()) return;
    
    QDateTime now = QDateTime::currentDateTime();
    
    QDirIterator it(cachePath, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QString filePath = it.next();
        QFileInfo info(filePath);
        
        if (maxAgeMs < 0 || info.lastModified().msecsTo(now) > maxAgeMs) {
            QFile::remove(filePath);
        }
    }
}

// ==================== ImageBatchProcessor ====================

int ImageBatchProcessor::processDirectory(const QString& inputDir, 
                                          const QString& outputDir,
                                          ProcessCallback callback,
                                          bool recursive)
{
    QStringList files = FileUtils::scanImagesInDirectory(inputDir, recursive);
    return processFiles(files, outputDir, callback);
}

int ImageBatchProcessor::processFiles(const QStringList& files,
                                       const QString& outputDir,
                                       ProcessCallback callback)
{
    FileUtils::ensureDirectoryExists(outputDir);
    
    int processed = 0;
    int total = files.size();
    
    for (int i = 0; i < total; ++i) {
        const QString& path = files[i];
        cv::Mat image = FileUtils::readImage(path);
        
        if (!image.empty()) {
            if (callback(path, image, i, total)) {
                processed++;
            }
        }
    }
    
    return processed;
}

cv::Mat ImageBatchProcessor::resize(const cv::Mat& image, int width, int height, bool keepAspectRatio)
{
    if (image.empty()) return cv::Mat();
    
    cv::Mat result;
    
    if (keepAspectRatio) {
        double scale = std::min(
            static_cast<double>(width) / image.cols,
            static_cast<double>(height) / image.rows
        );
        int newWidth = static_cast<int>(image.cols * scale);
        int newHeight = static_cast<int>(image.rows * scale);
        cv::resize(image, result, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    } else {
        cv::resize(image, result, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    }
    
    return result;
}

cv::Mat ImageBatchProcessor::crop(const cv::Mat& image, int x, int y, int width, int height)
{
    if (image.empty()) return cv::Mat();
    
    // 边界检查
    x = std::max(0, std::min(x, image.cols - 1));
    y = std::max(0, std::min(y, image.rows - 1));
    width = std::min(width, image.cols - x);
    height = std::min(height, image.rows - y);
    
    cv::Rect roi(x, y, width, height);
    return image(roi).clone();
}

cv::Mat ImageBatchProcessor::letterbox(const cv::Mat& image, int targetWidth, int targetHeight,
                                       const cv::Scalar& color)
{
    if (image.empty()) return cv::Mat();
    
    double scale = std::min(
        static_cast<double>(targetWidth) / image.cols,
        static_cast<double>(targetHeight) / image.rows
    );
    
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat result(targetHeight, targetWidth, image.type(), color);
    
    int offsetX = (targetWidth - newWidth) / 2;
    int offsetY = (targetHeight - newHeight) / 2;
    
    resized.copyTo(result(cv::Rect(offsetX, offsetY, newWidth, newHeight)));
    
    return result;
}

// ==================== VideoInfo ====================

QString VideoInfo::formatDuration() const
{
    int totalSeconds = static_cast<int>(durationSec);
    int hours = totalSeconds / 3600;
    int minutes = (totalSeconds % 3600) / 60;
    int seconds = totalSeconds % 60;
    
    if (hours > 0) {
        return QString("%1:%2:%3")
            .arg(hours)
            .arg(minutes, 2, 10, QChar('0'))
            .arg(seconds, 2, 10, QChar('0'));
    } else {
        return QString("%1:%2")
            .arg(minutes)
            .arg(seconds, 2, 10, QChar('0'));
    }
}

VideoInfo getVideoInfo(const QString& path)
{
    VideoInfo info;
    info.path = path;
    
    cv::VideoCapture cap(path.toStdString());
    if (!cap.isOpened()) {
        LOG_WARNING(QString("Cannot open video: %1").arg(path));
        return info;
    }
    
    info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    info.fps = cap.get(cv::CAP_PROP_FPS);
    info.frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    if (info.fps > 0) {
        info.durationSec = info.frameCount / info.fps;
    }
    
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    char codec[5] = {
        static_cast<char>(fourcc & 0xFF),
        static_cast<char>((fourcc >> 8) & 0xFF),
        static_cast<char>((fourcc >> 16) & 0xFF),
        static_cast<char>((fourcc >> 24) & 0xFF),
        '\0'
    };
    info.codec = QString(codec);
    
    info.isValid = true;
    cap.release();
    
    return info;
}

} // namespace yolo
