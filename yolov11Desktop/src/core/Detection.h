/**
 * @file Detection.h
 * @brief 检测结果相关的数据结构
 */

#ifndef DETECTION_H
#define DETECTION_H

#include <QString>
#include <QRectF>
#include <QColor>
#include <QVector>
#include <QJsonObject>
#include <QJsonArray>
#include <memory>

namespace yolo {

/**
 * 边界框数据
 */
struct BoundingBox {
    float x;        ///< 左上角 x
    float y;        ///< 左上角 y
    float width;    ///< 宽度
    float height;   ///< 高度

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(float x, float y, float w, float h) : x(x), y(y), width(w), height(h) {}

    /// 中心点 X
    float centerX() const { return x + width / 2.0f; }
    
    /// 中心点 Y
    float centerY() const { return y + height / 2.0f; }
    
    /// 面积
    float area() const { return width * height; }
    
    /// 右下角 X
    float right() const { return x + width; }
    
    /// 右下角 Y
    float bottom() const { return y + height; }
    
    /// 转成 QRectF
    QRectF toQRectF() const { return QRectF(x, y, width, height); }
    
    /// 从 QRectF 还原
    static BoundingBox fromQRectF(const QRectF& rect) {
        return BoundingBox(rect.x(), rect.y(), rect.width(), rect.height());
    }

    /// 计算与另一个框的 IoU
    float iou(const BoundingBox& other) const {
        float intersectX1 = std::max(x, other.x);
        float intersectY1 = std::max(y, other.y);
        float intersectX2 = std::min(right(), other.right());
        float intersectY2 = std::min(bottom(), other.bottom());
        
        if (intersectX2 <= intersectX1 || intersectY2 <= intersectY1) {
            return 0.0f;
        }
        
        float intersectArea = (intersectX2 - intersectX1) * (intersectY2 - intersectY1);
        float unionArea = area() + other.area() - intersectArea;
        
        return unionArea > 0 ? intersectArea / unionArea : 0.0f;
    }

    /// 按比例缩放
    BoundingBox scale(float scaleX, float scaleY) const {
        return BoundingBox(x * scaleX, y * scaleY, width * scaleX, height * scaleY);
    }

    /// 转成 JSON
    QJsonObject toJson() const {
        QJsonObject obj;
        obj["x"] = x;
        obj["y"] = y;
        obj["width"] = width;
        obj["height"] = height;
        return obj;
    }

    /// 从 JSON 读取
    static BoundingBox fromJson(const QJsonObject& obj) {
        return BoundingBox(
            obj["x"].toDouble(),
            obj["y"].toDouble(),
            obj["width"].toDouble(),
            obj["height"].toDouble()
        );
    }
};

/**
 * 关键点数据（姿态估计使用）
 */
struct Keypoint {
    float x;            ///< X 坐标
    float y;            ///< Y 坐标
    float confidence;   ///< 置信度
    int id;             ///< 关键点 ID

    Keypoint() : x(0), y(0), confidence(0), id(-1) {}
    Keypoint(float x, float y, float conf, int id = -1) 
        : x(x), y(y), confidence(conf), id(id) {}

    bool isValid() const { return confidence > 0.0f; }
};

/**
 * 分割掩码数据
 */
struct SegmentMask {
    QVector<float> data;    ///< 掩码数据
    int width;              ///< 掩码宽度
    int height;             ///< 掩码高度

    SegmentMask() : width(0), height(0) {}
    SegmentMask(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}

    float at(int x, int y) const {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            return data[y * width + x];
        }
        return 0.0f;
    }

    void set(int x, int y, float value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[y * width + x] = value;
        }
    }
};

/**
 * 旋转边界框（OBB）
 */
struct OrientedBoundingBox {
    float cx;       ///< 中心 X
    float cy;       ///< 中心 Y
    float width;    ///< 宽度
    float height;   ///< 高度
    float angle;    ///< 旋转角（弧度）

    OrientedBoundingBox() : cx(0), cy(0), width(0), height(0), angle(0) {}
    OrientedBoundingBox(float cx, float cy, float w, float h, float a)
        : cx(cx), cy(cy), width(w), height(h), angle(a) {}

    /// 计算四个角点
    QVector<QPointF> getCorners() const {
        QVector<QPointF> corners(4);
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        float hw = width / 2.0f;
        float hh = height / 2.0f;
        
        corners[0] = QPointF(cx + cosA * (-hw) - sinA * (-hh), 
                            cy + sinA * (-hw) + cosA * (-hh));
        corners[1] = QPointF(cx + cosA * (hw) - sinA * (-hh), 
                            cy + sinA * (hw) + cosA * (-hh));
        corners[2] = QPointF(cx + cosA * (hw) - sinA * (hh), 
                            cy + sinA * (hw) + cosA * (hh));
        corners[3] = QPointF(cx + cosA * (-hw) - sinA * (hh), 
                            cy + sinA * (-hw) + cosA * (hh));
        return corners;
    }
};

/**
 * 任务类型
 */
enum class TaskType {
    Detection,      ///< 目标检测
    Segmentation,   ///< 实例分割
    Pose,           ///< 姿态估计
    OBB,            ///< 旋转边界框
    Classification  ///< 分类
};

/**
 * 单条检测结果
 */
class Detection {
public:
    Detection();
    Detection(int classId, float confidence, const BoundingBox& bbox);
    ~Detection() = default;

    // 访问器
    int classId() const { return m_classId; }
    float confidence() const { return m_confidence; }
    const BoundingBox& bbox() const { return m_bbox; }
    const QString& className() const { return m_className; }
    const QColor& color() const { return m_color; }
    int trackId() const { return m_trackId; }
    const QVector<Keypoint>& keypoints() const { return m_keypoints; }
    const SegmentMask& mask() const { return m_mask; }
    const OrientedBoundingBox& obb() const { return m_obb; }
    TaskType taskType() const { return m_taskType; }

    // 设置器
    void setClassId(int id) { m_classId = id; }
    void setConfidence(float conf) { m_confidence = conf; }
    void setBbox(const BoundingBox& bbox) { m_bbox = bbox; }
    void setClassName(const QString& name) { m_className = name; }
    void setColor(const QColor& color) { m_color = color; }
    void setTrackId(int id) { m_trackId = id; }
    void setKeypoints(const QVector<Keypoint>& kps) { m_keypoints = kps; }
    void setMask(const SegmentMask& mask) { m_mask = mask; }
    void setObb(const OrientedBoundingBox& obb) { m_obb = obb; }
    void setTaskType(TaskType type) { m_taskType = type; }

    /// 按原图比例缩放坐标
    void scaleToOriginal(float scaleX, float scaleY, float offsetX = 0, float offsetY = 0);

    /// 转成 JSON
    QJsonObject toJson() const;

    /// 从 JSON 读取
    static Detection fromJson(const QJsonObject& obj);

    /// 返回显示用标签
    QString getLabel() const;

private:
    int m_classId;
    float m_confidence;
    BoundingBox m_bbox;
    QString m_className;
    QColor m_color;
    int m_trackId;
    QVector<Keypoint> m_keypoints;
    SegmentMask m_mask;
    OrientedBoundingBox m_obb;
    TaskType m_taskType;
};

/**
 * 检测结果集合
 */
class DetectionResult {
public:
    DetectionResult();
    ~DetectionResult() = default;

    /// 添加一条结果
    void addDetection(const Detection& detection);
    void addDetection(Detection&& detection);

    /// 获取所有结果
    const QVector<Detection>& detections() const { return m_detections; }
    QVector<Detection>& detections() { return m_detections; }

    /// 结果数量
    int count() const { return m_detections.size(); }

    /// 清空结果
    void clear() { m_detections.clear(); }

    /// 按置信度排序
    void sortByConfidence(bool descending = true);

    /// 按类别筛选
    DetectionResult filterByClass(int classId) const;
    DetectionResult filterByClasses(const QVector<int>& classIds) const;

    /// 按置信度筛选
    DetectionResult filterByConfidence(float minConfidence) const;

    /// 设置/获取推理耗时
    void setInferenceTime(double ms) { m_inferenceTime = ms; }
    double inferenceTime() const { return m_inferenceTime; }

    /// 设置/获取预处理耗时
    void setPreprocessTime(double ms) { m_preprocessTime = ms; }
    double preprocessTime() const { return m_preprocessTime; }

    /// 设置/获取后处理耗时
    void setPostprocessTime(double ms) { m_postprocessTime = ms; }
    double postprocessTime() const { return m_postprocessTime; }

    /// 统计总耗时
    double totalTime() const { return m_preprocessTime + m_inferenceTime + m_postprocessTime; }

    /// 设置/获取原始图像尺寸
    void setOriginalSize(int width, int height) { m_originalWidth = width; m_originalHeight = height; }
    int originalWidth() const { return m_originalWidth; }
    int originalHeight() const { return m_originalHeight; }

    /// 设置/获取帧编号
    void setFrameNumber(int64_t frame) { m_frameNumber = frame; }
    int64_t frameNumber() const { return m_frameNumber; }

    /// 设置/获取时间戳
    void setTimestamp(int64_t ts) { m_timestamp = ts; }
    int64_t timestamp() const { return m_timestamp; }

    /// 转成 JSON
    QJsonObject toJson() const;
    QJsonArray toJsonArray() const;

    /// 从 JSON 读取
    static DetectionResult fromJson(const QJsonObject& obj);

    /// 统计各类别数量
    QMap<int, int> getClassCounts() const;

private:
    QVector<Detection> m_detections;
    double m_inferenceTime;
    double m_preprocessTime;
    double m_postprocessTime;
    int m_originalWidth;
    int m_originalHeight;
    int64_t m_frameNumber;
    int64_t m_timestamp;
};

} // namespace yolo

// 注册元类型
Q_DECLARE_METATYPE(yolo::Detection)
Q_DECLARE_METATYPE(yolo::DetectionResult)
Q_DECLARE_METATYPE(yolo::BoundingBox)

#endif // DETECTION_H
