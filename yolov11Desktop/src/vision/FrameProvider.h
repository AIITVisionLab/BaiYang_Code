/**
 * @file FrameProvider.h
 * 帧数据提供者：统一的图像/视频源接口
 */

#ifndef FRAME_PROVIDER_H
#define FRAME_PROVIDER_H

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QThread>
#include <QTimer>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <memory>
#include <atomic>

namespace yolo {

/**
 * 数据源类型
 */
enum class SourceType {
    None,
    Camera,
    VideoFile,
    ImageFile,
    ImageFolder,
    RTSP,
    HttpStream
};

/**
 * 帧数据
 */
struct Frame {
    cv::Mat image;              ///< OpenCV 图像
    QImage qImage;              ///< Qt 图像（用于显示）
    int64_t frameNumber;        ///< 帧编号
    int64_t timestamp;          ///< 时间戳（毫秒）
    bool isValid;               ///< 是否有效

    Frame() : frameNumber(0), timestamp(0), isValid(false) {}
    
    bool isEmpty() const { return image.empty(); }
};

/**
 * 帧提供者抽象基类
 */
class FrameProvider : public QObject {
    Q_OBJECT

public:
    explicit FrameProvider(QObject* parent = nullptr);
    virtual ~FrameProvider();

    /**
     * 打开数据源
     */
    virtual bool open(const QString& source) = 0;

    /**
     * 关闭数据源
     */
    virtual void close() = 0;

    /**
     * 是否已打开
     */
    virtual bool isOpened() const = 0;

    /**
     * 开始提供帧
     */
    virtual void start();

    /**
     * 停止提供帧
     */
    virtual void stop();

    /**
     * 暂停
     */
    virtual void pause();

    /**
     * 继续
     */
    virtual void resume();

    /**
     * 获取下一帧（同步）
     */
    virtual Frame getNextFrame() = 0;

    /**
     * 跳到指定帧（仅视频）
     */
    virtual bool seekTo(int64_t frameNumber) { return false; }

    /**
     * 数据源类型
     */
    SourceType sourceType() const { return m_sourceType; }

    /**
     * 总帧数（仅视频/图像文件夹）
     */
    virtual int64_t totalFrames() const { return -1; }

    /**
     * 当前帧号
     */
    int64_t currentFrame() const { return m_currentFrame; }

    /**
     * 帧率
     */
    double fps() const { return m_fps; }

    /**
     * 设置目标帧率（用于限速）
     */
    void setTargetFps(double fps) { m_targetFps = fps; }

    /**
     * 图像尺寸
     */
    QSize frameSize() const { return m_frameSize; }

    /**
     * 是否在运行
     */
    bool isRunning() const { return m_running; }

    /**
     * 是否暂停
     */
    bool isPaused() const { return m_paused; }

    /**
     * 错误信息
     */
    QString lastError() const { return m_lastError; }

signals:
    /**
     * 新帧信号
     */
    void frameReady(const Frame& frame);

    /**
     * 播放结束
     */
    void finished();

    /**
     * 错误信号
     */
    void error(const QString& message);

    /**
     * 状态改变
     */
    void stateChanged(bool running, bool paused);

    /**
     * 进度更新（0-100）
     */
    void progressChanged(int progress);

protected:
    /**
     * cv::Mat 转 QImage
     */
    QImage matToQImage(const cv::Mat& mat);

    /**
     * 帧获取循环（在工作线程中跑）
     */
    virtual void frameLoop();

protected:
    SourceType m_sourceType;
    QString m_source;
    QString m_lastError;
    QSize m_frameSize;
    double m_fps;
    double m_targetFps;
    int64_t m_currentFrame;
    
    std::atomic<bool> m_running;
    std::atomic<bool> m_paused;
    
    QMutex m_mutex;
    std::unique_ptr<QThread> m_workerThread;
    QTimer* m_frameTimer;
};

/**
 * 摄像头帧提供者
 */
class CameraFrameProvider : public FrameProvider {
    Q_OBJECT

public:
    explicit CameraFrameProvider(QObject* parent = nullptr);
    ~CameraFrameProvider() override;

    bool open(const QString& source) override;
    void close() override;
    bool isOpened() const override;
    Frame getNextFrame() override;

    /**
     * 设置分辨率
     */
    void setResolution(int width, int height);

    /**
     * 获取可用摄像头列表
     */
    static QStringList availableCameras();

private:
    cv::VideoCapture m_capture;
    int m_cameraIndex;
};

/**
 * 视频文件帧提供者
 */
class VideoFileProvider : public FrameProvider {
    Q_OBJECT

public:
    explicit VideoFileProvider(QObject* parent = nullptr);
    ~VideoFileProvider() override;

    bool open(const QString& source) override;
    void close() override;
    bool isOpened() const override;
    Frame getNextFrame() override;
    bool seekTo(int64_t frameNumber) override;
    int64_t totalFrames() const override;

    /**
     * 视频时长（秒）
     */
    double duration() const;

    /**
     * 是否循环播放
     */
    void setLoop(bool loop) { m_loop = loop; }

private:
    cv::VideoCapture m_capture;
    int64_t m_totalFrames;
    bool m_loop;
};

/**
 * 图片文件帧提供者
 */
class ImageFileProvider : public FrameProvider {
    Q_OBJECT

public:
    explicit ImageFileProvider(QObject* parent = nullptr);
    ~ImageFileProvider() override;

    bool open(const QString& source) override;
    void close() override;
    bool isOpened() const override;
    Frame getNextFrame() override;

private:
    cv::Mat m_image;
    bool m_delivered;
};

/**
 * 图片文件夹帧提供者
 */
class ImageFolderProvider : public FrameProvider {
    Q_OBJECT

public:
    explicit ImageFolderProvider(QObject* parent = nullptr);
    ~ImageFolderProvider() override;

    bool open(const QString& source) override;
    void close() override;
    bool isOpened() const override;
    Frame getNextFrame() override;
    bool seekTo(int64_t frameNumber) override;
    int64_t totalFrames() const override;

    /**
     * 设置文件过滤器
     */
    void setFilter(const QStringList& extensions);

private:
    QStringList m_imageFiles;
    int m_currentIndex;
    QStringList m_extensions;
};

/**
 * RTSP 流帧提供者
 */
class RtspStreamProvider : public FrameProvider {
    Q_OBJECT

public:
    explicit RtspStreamProvider(QObject* parent = nullptr);
    ~RtspStreamProvider() override;

    bool open(const QString& source) override;
    void close() override;
    bool isOpened() const override;
    Frame getNextFrame() override;

    /**
     * 设置缓冲区大小
     */
    void setBufferSize(int size);

    /**
     * 设置超时（毫秒）
     */
    void setTimeout(int ms);

private:
    cv::VideoCapture m_capture;
    int m_bufferSize;
    int m_timeout;
};

} // namespace yolo

#endif // FRAME_PROVIDER_H
