/**
 * @file VideoFrameProvider.cpp
 * @brief QML 视频帧提供者实现
 */

#include "VideoFrameProvider.h"
#include <QMutexLocker>

namespace yolo {

VideoFrameProvider::VideoFrameProvider()
    : QQuickImageProvider(QQuickImageProvider::Image)
{
}

QImage VideoFrameProvider::requestImage(const QString& id, QSize* size, const QSize& requestedSize)
{
    Q_UNUSED(id)
    
    QMutexLocker locker(&m_mutex);
    
    if (m_frame.isNull()) {
          // 返回空白图像
        QImage blank(640, 480, QImage::Format_RGB888);
        blank.fill(Qt::black);
        if (size) {
            *size = blank.size();
        }
        return blank;
    }
    
    if (size) {
        *size = m_frame.size();
    }
    
    if (requestedSize.isValid() && requestedSize != m_frame.size()) {
        return m_frame.scaled(requestedSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    
    return m_frame;
}

void VideoFrameProvider::updateFrame(const QImage& frame)
{
    QMutexLocker locker(&m_mutex);
    m_frame = frame;
}

void VideoFrameProvider::clear()
{
    QMutexLocker locker(&m_mutex);
    m_frame = QImage();
}

QImage VideoFrameProvider::currentFrame() const
{
    QMutexLocker locker(&m_mutex);
    return m_frame;
}

// FrameEmitter

FrameEmitter::FrameEmitter(QObject* parent)
    : QObject(parent)
{
}

void FrameEmitter::emitFrameUpdate()
{
    emit frameUpdated();
}

} // namespace yolo
