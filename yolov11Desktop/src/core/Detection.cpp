/**
 * @file Detection.cpp
 * @brief 检测结果相关实现
 */

#include "Detection.h"
#include <QJsonDocument>
#include <algorithm>

namespace yolo {

// Detection 的实现
Detection::Detection()
    : m_classId(-1)
    , m_confidence(0.0f)
    , m_trackId(-1)
    , m_taskType(TaskType::Detection)
{
}

Detection::Detection(int classId, float confidence, const BoundingBox& bbox)
    : m_classId(classId)
    , m_confidence(confidence)
    , m_bbox(bbox)
    , m_trackId(-1)
    , m_taskType(TaskType::Detection)
{
}

void Detection::scaleToOriginal(float scaleX, float scaleY, float offsetX, float offsetY)
{
    m_bbox.x = (m_bbox.x - offsetX) * scaleX;
    m_bbox.y = (m_bbox.y - offsetY) * scaleY;
    m_bbox.width *= scaleX;
    m_bbox.height *= scaleY;

    // 同步缩放关键点
    for (auto& kp : m_keypoints) {
        kp.x = (kp.x - offsetX) * scaleX;
        kp.y = (kp.y - offsetY) * scaleY;
    }

    // 同步缩放 OBB
    if (m_taskType == TaskType::OBB) {
        m_obb.cx = (m_obb.cx - offsetX) * scaleX;
        m_obb.cy = (m_obb.cy - offsetY) * scaleY;
        m_obb.width *= scaleX;
        m_obb.height *= scaleY;
    }
}

QJsonObject Detection::toJson() const
{
    QJsonObject obj;
    obj["classId"] = m_classId;
    obj["confidence"] = m_confidence;
    obj["className"] = m_className;
    obj["bbox"] = m_bbox.toJson();
    obj["trackId"] = m_trackId;
    obj["taskType"] = static_cast<int>(m_taskType);

    if (!m_keypoints.isEmpty()) {
        QJsonArray kpArray;
        for (const auto& kp : m_keypoints) {
            QJsonObject kpObj;
            kpObj["x"] = kp.x;
            kpObj["y"] = kp.y;
            kpObj["confidence"] = kp.confidence;
            kpObj["id"] = kp.id;
            kpArray.append(kpObj);
        }
        obj["keypoints"] = kpArray;
    }

    return obj;
}

Detection Detection::fromJson(const QJsonObject& obj)
{
    Detection det;
    det.m_classId = obj["classId"].toInt();
    det.m_confidence = obj["confidence"].toDouble();
    det.m_className = obj["className"].toString();
    det.m_bbox = BoundingBox::fromJson(obj["bbox"].toObject());
    det.m_trackId = obj["trackId"].toInt(-1);
    det.m_taskType = static_cast<TaskType>(obj["taskType"].toInt());

    if (obj.contains("keypoints")) {
        QJsonArray kpArray = obj["keypoints"].toArray();
        for (const auto& kpVal : kpArray) {
            QJsonObject kpObj = kpVal.toObject();
            Keypoint kp(
                kpObj["x"].toDouble(),
                kpObj["y"].toDouble(),
                kpObj["confidence"].toDouble(),
                kpObj["id"].toInt()
            );
            det.m_keypoints.append(kp);
        }
    }

    return det;
}

QString Detection::getLabel() const
{
    if (m_className.isEmpty()) {
        return QString("Class %1: %2%").arg(m_classId).arg(m_confidence * 100, 0, 'f', 1);
    }
    return QString("%1: %2%").arg(m_className).arg(m_confidence * 100, 0, 'f', 1);
}

// DetectionResult 的实现
DetectionResult::DetectionResult()
    : m_inferenceTime(0)
    , m_preprocessTime(0)
    , m_postprocessTime(0)
    , m_originalWidth(0)
    , m_originalHeight(0)
    , m_frameNumber(0)
    , m_timestamp(0)
{
}

void DetectionResult::addDetection(const Detection& detection)
{
    m_detections.append(detection);
}

void DetectionResult::addDetection(Detection&& detection)
{
    m_detections.append(std::move(detection));
}

void DetectionResult::sortByConfidence(bool descending)
{
    std::sort(m_detections.begin(), m_detections.end(),
        [descending](const Detection& a, const Detection& b) {
            return descending ? a.confidence() > b.confidence() 
                             : a.confidence() < b.confidence();
        });
}

DetectionResult DetectionResult::filterByClass(int classId) const
{
    DetectionResult result;
    result.setOriginalSize(m_originalWidth, m_originalHeight);
    result.setFrameNumber(m_frameNumber);
    result.setTimestamp(m_timestamp);
    
    for (const auto& det : m_detections) {
        if (det.classId() == classId) {
            result.addDetection(det);
        }
    }
    return result;
}

DetectionResult DetectionResult::filterByClasses(const QVector<int>& classIds) const
{
    DetectionResult result;
    result.setOriginalSize(m_originalWidth, m_originalHeight);
    result.setFrameNumber(m_frameNumber);
    result.setTimestamp(m_timestamp);
    
    for (const auto& det : m_detections) {
        if (classIds.contains(det.classId())) {
            result.addDetection(det);
        }
    }
    return result;
}

DetectionResult DetectionResult::filterByConfidence(float minConfidence) const
{
    DetectionResult result;
    result.setOriginalSize(m_originalWidth, m_originalHeight);
    result.setFrameNumber(m_frameNumber);
    result.setTimestamp(m_timestamp);
    
    for (const auto& det : m_detections) {
        if (det.confidence() >= minConfidence) {
            result.addDetection(det);
        }
    }
    return result;
}

QJsonObject DetectionResult::toJson() const
{
    QJsonObject obj;
    obj["inferenceTime"] = m_inferenceTime;
    obj["preprocessTime"] = m_preprocessTime;
    obj["postprocessTime"] = m_postprocessTime;
    obj["originalWidth"] = m_originalWidth;
    obj["originalHeight"] = m_originalHeight;
    obj["frameNumber"] = static_cast<qint64>(m_frameNumber);
    obj["timestamp"] = static_cast<qint64>(m_timestamp);
    obj["detections"] = toJsonArray();
    return obj;
}

QJsonArray DetectionResult::toJsonArray() const
{
    QJsonArray array;
    for (const auto& det : m_detections) {
        array.append(det.toJson());
    }
    return array;
}

DetectionResult DetectionResult::fromJson(const QJsonObject& obj)
{
    DetectionResult result;
    result.m_inferenceTime = obj["inferenceTime"].toDouble();
    result.m_preprocessTime = obj["preprocessTime"].toDouble();
    result.m_postprocessTime = obj["postprocessTime"].toDouble();
    result.m_originalWidth = obj["originalWidth"].toInt();
    result.m_originalHeight = obj["originalHeight"].toInt();
    result.m_frameNumber = obj["frameNumber"].toVariant().toLongLong();
    result.m_timestamp = obj["timestamp"].toVariant().toLongLong();

    QJsonArray detArray = obj["detections"].toArray();
    for (const auto& detVal : detArray) {
        result.addDetection(Detection::fromJson(detVal.toObject()));
    }

    return result;
}

QMap<int, int> DetectionResult::getClassCounts() const
{
    QMap<int, int> counts;
    for (const auto& det : m_detections) {
        counts[det.classId()]++;
    }
    return counts;
}

} // namespace yolo
