/**
 * @file FrameProvider.cpp
 * @brief 帧数据提供者实现
 */

#include "FrameProvider.h"
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <QDateTime>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace yolo {

// ===== FrameProvider =====

FrameProvider::FrameProvider(QObject* parent)
    : QObject(parent)
    , m_sourceType(SourceType::None)
    , m_fps(30.0)
    , m_targetFps(0)
    , m_currentFrame(0)
    , m_running(false)
    , m_paused(false)
    , m_frameTimer(nullptr)
{
}

FrameProvider::~FrameProvider()
{
    stop();
}

void FrameProvider::start()
{
    if (m_running || !isOpened()) {
        return;
    }

    m_running = true;
    m_paused = false;
    emit stateChanged(true, false);

    // 创建帧定时器
    double targetFps = m_targetFps > 0 ? m_targetFps : m_fps;
    int interval = static_cast<int>(1000.0 / targetFps);

    m_frameTimer = new QTimer(this);
    connect(m_frameTimer, &QTimer::timeout, this, &FrameProvider::frameLoop);
    m_frameTimer->start(interval);
}

void FrameProvider::stop()
{
    m_running = false;
    m_paused = false;

    if (m_frameTimer) {
        m_frameTimer->stop();
        m_frameTimer->deleteLater();
        m_frameTimer = nullptr;
    }

    emit stateChanged(false, false);
}

void FrameProvider::pause()
{
    m_paused = true;
    emit stateChanged(m_running, true);
}

void FrameProvider::resume()
{
    m_paused = false;
    emit stateChanged(m_running, false);
}

QImage FrameProvider::matToQImage(const cv::Mat& mat)
{
    if (mat.empty()) {
        return QImage();
    }

    switch (mat.type()) {
        case CV_8UC4: {
            return QImage(mat.data, mat.cols, mat.rows, 
                         static_cast<int>(mat.step), QImage::Format_ARGB32).copy();
        }
        case CV_8UC3: {
            cv::Mat rgb;
            cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
            return QImage(rgb.data, rgb.cols, rgb.rows, 
                         static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
        }
        case CV_8UC1: {
            return QImage(mat.data, mat.cols, mat.rows, 
                         static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
        }
        default:
            qWarning() << "Unsupported cv::Mat type:" << mat.type();
            return QImage();
    }
}

void FrameProvider::frameLoop()
{
    if (!m_running || m_paused) {
        return;
    }

    Frame frame = getNextFrame();
    
    if (frame.isValid) {
        frame.qImage = matToQImage(frame.image);
        emit frameReady(frame);

        // 更新进度
        int64_t total = totalFrames();
        if (total > 0) {
            int progress = static_cast<int>(m_currentFrame * 100 / total);
            emit progressChanged(progress);
        }
    } else {
        // 没有更多帧时结束
        stop();
        emit finished();
    }
}

// ===== CameraFrameProvider =====

CameraFrameProvider::CameraFrameProvider(QObject* parent)
    : FrameProvider(parent)
    , m_cameraIndex(0)
{
    m_sourceType = SourceType::Camera;
}

CameraFrameProvider::~CameraFrameProvider()
{
    close();
}

bool CameraFrameProvider::open(const QString& source)
{
    close();

    bool ok;
    m_cameraIndex = source.toInt(&ok);
    if (!ok) {
        // 当作设备路径打开（Linux 用 V4L2）
#ifdef __linux__
        m_capture.open(source.toStdString(), cv::CAP_V4L2);
#else
        m_capture.open(source.toStdString());
#endif
    } else {
        // 用设备索引打开（Linux 用 V4L2）
#ifdef __linux__
        m_capture.open(m_cameraIndex, cv::CAP_V4L2);
#else
        m_capture.open(m_cameraIndex);
#endif
    }

    if (!m_capture.isOpened()) {
        m_lastError = "Failed to open camera: " + source;
        emit error(m_lastError);
        qWarning() << "Camera open failed:" << source;
        return false;
    }

    m_fps = m_capture.get(cv::CAP_PROP_FPS);
    if (m_fps <= 0) m_fps = 30.0;
    
    m_frameSize = QSize(
        static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    
    m_source = source;
    m_currentFrame = 0;
    
    qDebug() << "Camera opened:" << source << m_frameSize << "fps:" << m_fps;
    return true;
}

void CameraFrameProvider::close()
{
    stop();
    if (m_capture.isOpened()) {
        m_capture.release();
    }
}

bool CameraFrameProvider::isOpened() const
{
    return m_capture.isOpened();
}

Frame CameraFrameProvider::getNextFrame()
{
    Frame frame;
    
    if (!m_capture.isOpened()) {
        return frame;
    }

    QMutexLocker locker(&m_mutex);
    
    if (m_capture.read(frame.image)) {
        frame.frameNumber = m_currentFrame++;
        frame.timestamp = QDateTime::currentMSecsSinceEpoch();
        frame.isValid = true;
    }

    return frame;
}

void CameraFrameProvider::setResolution(int width, int height)
{
    if (m_capture.isOpened()) {
        m_capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
        m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        m_frameSize = QSize(width, height);
    }
}

QStringList CameraFrameProvider::availableCameras()
{
    QStringList cameras;
    
    // 尝试打开摄像头0-9（Linux 上使用 V4L2 后端）
    for (int i = 0; i < 10; ++i) {
#ifdef __linux__
        cv::VideoCapture cap(i, cv::CAP_V4L2);
#else
        cv::VideoCapture cap(i);
#endif
        if (cap.isOpened()) {
            cameras << QString::number(i);
            cap.release();
        }
    }
    
    qDebug() << "Available cameras:" << cameras;
    return cameras;
}

// ==================== VideoFileProvider ====================

VideoFileProvider::VideoFileProvider(QObject* parent)
    : FrameProvider(parent)
    , m_totalFrames(0)
    , m_loop(false)
{
    m_sourceType = SourceType::VideoFile;
}

VideoFileProvider::~VideoFileProvider()
{
    close();
}

bool VideoFileProvider::open(const QString& source)
{
    close();

    m_capture.open(source.toStdString());
    
    if (!m_capture.isOpened()) {
        m_lastError = "Failed to open video: " + source;
        emit error(m_lastError);
        return false;
    }

    m_fps = m_capture.get(cv::CAP_PROP_FPS);
    if (m_fps <= 0) m_fps = 30.0;
    
    m_totalFrames = static_cast<int64_t>(m_capture.get(cv::CAP_PROP_FRAME_COUNT));
    m_frameSize = QSize(
        static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    
    m_source = source;
    m_currentFrame = 0;
    
    qDebug() << "Video opened:" << source << m_frameSize 
             << "fps:" << m_fps << "frames:" << m_totalFrames;
    return true;
}

void VideoFileProvider::close()
{
    stop();
    if (m_capture.isOpened()) {
        m_capture.release();
    }
    m_totalFrames = 0;
}

bool VideoFileProvider::isOpened() const
{
    return m_capture.isOpened();
}

Frame VideoFileProvider::getNextFrame()
{
    Frame frame;
    
    if (!m_capture.isOpened()) {
        return frame;
    }

    QMutexLocker locker(&m_mutex);
    
    if (m_capture.read(frame.image)) {
        frame.frameNumber = m_currentFrame++;
        frame.timestamp = static_cast<int64_t>(m_capture.get(cv::CAP_PROP_POS_MSEC));
        frame.isValid = true;
    } else if (m_loop && m_totalFrames > 0) {
        // 循环播放
        m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
        m_currentFrame = 0;
        if (m_capture.read(frame.image)) {
            frame.frameNumber = m_currentFrame++;
            frame.timestamp = 0;
            frame.isValid = true;
        }
    }

    return frame;
}

bool VideoFileProvider::seekTo(int64_t frameNumber)
{
    if (!m_capture.isOpened() || frameNumber < 0 || frameNumber >= m_totalFrames) {
        return false;
    }

    QMutexLocker locker(&m_mutex);
    m_capture.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(frameNumber));
    m_currentFrame = frameNumber;
    return true;
}

int64_t VideoFileProvider::totalFrames() const
{
    return m_totalFrames;
}

double VideoFileProvider::duration() const
{
    if (m_fps <= 0) return 0;
    return m_totalFrames / m_fps;
}

// ==================== ImageFileProvider ====================

ImageFileProvider::ImageFileProvider(QObject* parent)
    : FrameProvider(parent)
    , m_delivered(false)
{
    m_sourceType = SourceType::ImageFile;
    m_fps = 1.0;
}

ImageFileProvider::~ImageFileProvider()
{
    close();
}

bool ImageFileProvider::open(const QString& source)
{
    close();

    m_image = cv::imread(source.toStdString());
    
    if (m_image.empty()) {
        m_lastError = "Failed to open image: " + source;
        emit error(m_lastError);
        return false;
    }

    m_frameSize = QSize(m_image.cols, m_image.rows);
    m_source = source;
    m_currentFrame = 0;
    m_delivered = false;
    
    qDebug() << "Image opened:" << source << m_frameSize;
    return true;
}

void ImageFileProvider::close()
{
    stop();
    m_image.release();
    m_delivered = false;
}

bool ImageFileProvider::isOpened() const
{
    return !m_image.empty();
}

Frame ImageFileProvider::getNextFrame()
{
    Frame frame;
    
    if (m_image.empty() || m_delivered) {
        return frame;
    }

    frame.image = m_image.clone();
    frame.frameNumber = 0;
    frame.timestamp = QDateTime::currentMSecsSinceEpoch();
    frame.isValid = true;
    m_delivered = true;
    m_currentFrame = 1;

    return frame;
}

// ==================== ImageFolderProvider ====================

ImageFolderProvider::ImageFolderProvider(QObject* parent)
    : FrameProvider(parent)
    , m_currentIndex(0)
{
    m_sourceType = SourceType::ImageFolder;
    m_fps = 1.0;
    m_extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"};
}

ImageFolderProvider::~ImageFolderProvider()
{
    close();
}

bool ImageFolderProvider::open(const QString& source)
{
    close();

    QDir dir(source);
    if (!dir.exists()) {
        m_lastError = "Directory not found: " + source;
        emit error(m_lastError);
        return false;
    }

    m_imageFiles = dir.entryList(m_extensions, QDir::Files, QDir::Name);
    
    if (m_imageFiles.isEmpty()) {
        m_lastError = "No images found in: " + source;
        emit error(m_lastError);
        return false;
    }

    // 转换为完整路径
    for (int i = 0; i < m_imageFiles.size(); ++i) {
        m_imageFiles[i] = dir.absoluteFilePath(m_imageFiles[i]);
    }

    // 读取第一张图片获取尺寸
    cv::Mat firstImage = cv::imread(m_imageFiles.first().toStdString());
    if (!firstImage.empty()) {
        m_frameSize = QSize(firstImage.cols, firstImage.rows);
    }

    m_source = source;
    m_currentIndex = 0;
    m_currentFrame = 0;
    
    qDebug() << "Image folder opened:" << source << "images:" << m_imageFiles.size();
    return true;
}

void ImageFolderProvider::close()
{
    stop();
    m_imageFiles.clear();
    m_currentIndex = 0;
}

bool ImageFolderProvider::isOpened() const
{
    return !m_imageFiles.isEmpty();
}

Frame ImageFolderProvider::getNextFrame()
{
    Frame frame;
    
    if (m_imageFiles.isEmpty() || m_currentIndex >= m_imageFiles.size()) {
        return frame;
    }

    frame.image = cv::imread(m_imageFiles[m_currentIndex].toStdString());
    
    if (!frame.image.empty()) {
        frame.frameNumber = m_currentIndex;
        frame.timestamp = QDateTime::currentMSecsSinceEpoch();
        frame.isValid = true;
        m_currentIndex++;
        m_currentFrame = m_currentIndex;
    }

    return frame;
}

bool ImageFolderProvider::seekTo(int64_t frameNumber)
{
    if (frameNumber < 0 || frameNumber >= m_imageFiles.size()) {
        return false;
    }
    m_currentIndex = static_cast<int>(frameNumber);
    m_currentFrame = frameNumber;
    return true;
}

int64_t ImageFolderProvider::totalFrames() const
{
    return m_imageFiles.size();
}

void ImageFolderProvider::setFilter(const QStringList& extensions)
{
    m_extensions = extensions;
}

// ==================== RtspStreamProvider ====================

RtspStreamProvider::RtspStreamProvider(QObject* parent)
    : FrameProvider(parent)
    , m_bufferSize(1)
    , m_timeout(5000)
{
    m_sourceType = SourceType::RTSP;
}

RtspStreamProvider::~RtspStreamProvider()
{
    close();
}

bool RtspStreamProvider::open(const QString& source)
{
    close();

    // 设置RTSP选项
    m_capture.set(cv::CAP_PROP_BUFFERSIZE, m_bufferSize);
    
    // 对于RTSP流，使用FFmpeg后端
    m_capture.open(source.toStdString(), cv::CAP_FFMPEG);
    
    if (!m_capture.isOpened()) {
        // 尝试其他后端
        m_capture.open(source.toStdString(), cv::CAP_GSTREAMER);
    }
    
    if (!m_capture.isOpened()) {
        m_lastError = "Failed to open RTSP stream: " + source;
        emit error(m_lastError);
        return false;
    }

    m_fps = m_capture.get(cv::CAP_PROP_FPS);
    if (m_fps <= 0) m_fps = 25.0;
    
    m_frameSize = QSize(
        static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    
    m_source = source;
    m_currentFrame = 0;
    
    qDebug() << "RTSP stream opened:" << source << m_frameSize << "fps:" << m_fps;
    return true;
}

void RtspStreamProvider::close()
{
    stop();
    if (m_capture.isOpened()) {
        m_capture.release();
    }
}

bool RtspStreamProvider::isOpened() const
{
    return m_capture.isOpened();
}

Frame RtspStreamProvider::getNextFrame()
{
    Frame frame;
    
    if (!m_capture.isOpened()) {
        return frame;
    }

    QMutexLocker locker(&m_mutex);
    
    // 对于RTSP，需要grab后再retrieve以减少延迟
    if (m_capture.grab()) {
        if (m_capture.retrieve(frame.image)) {
            frame.frameNumber = m_currentFrame++;
            frame.timestamp = QDateTime::currentMSecsSinceEpoch();
            frame.isValid = true;
        }
    }

    return frame;
}

void RtspStreamProvider::setBufferSize(int size)
{
    m_bufferSize = size;
    if (m_capture.isOpened()) {
        m_capture.set(cv::CAP_PROP_BUFFERSIZE, size);
    }
}

void RtspStreamProvider::setTimeout(int ms)
{
    m_timeout = ms;
}

} // namespace yolo
