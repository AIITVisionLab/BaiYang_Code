/**
 * @file VideoFrameProvider.h
 * @brief QML 视频帧提供者（向 QML 传递视频帧）
 */

#ifndef VIDEO_FRAME_PROVIDER_H
#define VIDEO_FRAME_PROVIDER_H

#include <QObject>
#include <QQuickImageProvider>
#include <QImage>
#include <QMutex>

namespace yolo {

/**
 * @brief QML 图像提供者
 *
 * 通过 "image://video/frame" 在 QML 中使用
 */
class VideoFrameProvider : public QQuickImageProvider {
public:
    VideoFrameProvider();
    ~VideoFrameProvider() override = default;

    /**
     * @brief 实现 QQuickImageProvider 接口
     */
    QImage requestImage(const QString& id, QSize* size, const QSize& requestedSize) override;

    /**
     * @brief 更新帧
     */
    void updateFrame(const QImage& frame);

    /**
     * @brief 清除帧
     */
    void clear();

    /**
     * @brief 获取当前帧
     */
    QImage currentFrame() const;

private:
    QImage m_frame;
    mutable QMutex m_mutex;
};

/**
 * @brief QML 帧信号发射器
 *
 * 由于 QQuickImageProvider 不继承 QObject，
 * 用该类发射帧更新信号。
 */
class FrameEmitter : public QObject {
    Q_OBJECT

public:
    explicit FrameEmitter(QObject* parent = nullptr);

    /**
     * @brief 发射帧更新信号
     */
    void emitFrameUpdate();

signals:
    /**
     * @brief 帧更新信号
     * QML 可监听该信号更新 Image 的 source。
     */
    void frameUpdated();
};

} // namespace yolo

#endif // VIDEO_FRAME_PROVIDER_H
