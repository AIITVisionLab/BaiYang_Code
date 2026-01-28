/**
 * @file ClassLabels.cpp
 * @brief 类别标签管理实现
 */

#include "ClassLabels.h"
#include <QFile>
#include <QTextStream>
#include <QJsonDocument>
#include <QRandomGenerator>
#include <cmath>

namespace yolo {

QVector<QColor> ClassLabels::s_defaultColors;

ClassLabels::ClassLabels()
{
    if (s_defaultColors.isEmpty()) {
        s_defaultColors = getDefaultColors();
    }
}

void ClassLabels::loadCocoLabels()
{
    clear();
    for (int i = 0; i < CocoLabels::LABELS.size(); ++i) {
        addClass(i, CocoLabels::LABELS[i]);
    }
}

bool ClassLabels::loadFromFile(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }

    clear();
    QTextStream stream(&file);
    int id = 0;

    // 先判断文件格式
    QString firstLine = stream.readLine();
    stream.seek(0);

    // JSON 格式
    if (firstLine.trimmed().startsWith('[') || firstLine.trimmed().startsWith('{')) {
        QByteArray data = file.readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        if (doc.isArray()) {
            fromJson(doc.array());
        } else if (doc.isObject()) {
            QJsonObject obj = doc.object();
            if (obj.contains("names")) {
                // YOLOv8/v11 的 names 格式
                QJsonObject names = obj["names"].toObject();
                for (auto it = names.begin(); it != names.end(); ++it) {
                    int classId = it.key().toInt();
                    QString name = it.value().toString();
                    addClass(classId, name);
                }
            }
        }
        return true;
    }

    // 纯文本格式（每行一个类别）
    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if (!line.isEmpty() && !line.startsWith('#')) {
            // 支持带 ID 的行（如: id: name 或 id name）
            if (line.contains(':')) {
                QStringList parts = line.split(':');
                if (parts.size() >= 2) {
                    id = parts[0].trimmed().toInt();
                    addClass(id, parts[1].trimmed());
                    continue;
                }
            }
            addClass(id++, line);
        }
    }

    return true;
}

void ClassLabels::loadFromList(const QStringList& labels)
{
    clear();
    for (int i = 0; i < labels.size(); ++i) {
        addClass(i, labels[i]);
    }
}

bool ClassLabels::saveToFile(const QString& filePath) const
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QJsonDocument doc(toJson());
    file.write(doc.toJson(QJsonDocument::Indented));
    return true;
}

void ClassLabels::addClass(int id, const QString& name, const QColor& color)
{
    ClassInfo info;
    info.id = id;
    info.name = name;
    info.color = color.isValid() ? color : generateColor(id);
    info.enabled = true;
    m_classes[id] = info;
}

QString ClassLabels::getClassName(int classId) const
{
    if (m_classes.contains(classId)) {
        return m_classes[classId].name;
    }
    return QString("class_%1").arg(classId);
}

QColor ClassLabels::getClassColor(int classId) const
{
    if (m_classes.contains(classId)) {
        return m_classes[classId].color;
    }
    return generateColor(classId);
}

ClassInfo ClassLabels::getClassInfo(int classId) const
{
    if (m_classes.contains(classId)) {
        return m_classes[classId];
    }
    return ClassInfo(classId, QString("class_%1").arg(classId), generateColor(classId));
}

void ClassLabels::setClassColor(int classId, const QColor& color)
{
    if (m_classes.contains(classId)) {
        m_classes[classId].color = color;
    }
}

void ClassLabels::setClassEnabled(int classId, bool enabled)
{
    if (m_classes.contains(classId)) {
        m_classes[classId].enabled = enabled;
    }
}

bool ClassLabels::isClassEnabled(int classId) const
{
    if (m_classes.contains(classId)) {
        return m_classes[classId].enabled;
    }
    return true;
}

QVector<int> ClassLabels::getAllClassIds() const
{
    return m_classes.keys().toVector();
}

QVector<int> ClassLabels::getEnabledClassIds() const
{
    QVector<int> result;
    for (auto it = m_classes.begin(); it != m_classes.end(); ++it) {
        if (it.value().enabled) {
            result.append(it.key());
        }
    }
    return result;
}

QColor ClassLabels::generateColor(int index)
{
    // 用 HSV 生成区分度更高的颜色
    float hue = std::fmod(index * 0.618033988749895f, 1.0f);  // 黄金分割
    float saturation = 0.7f + 0.3f * std::fmod(index * 0.381966011250105f, 1.0f);
    float value = 0.8f + 0.2f * std::fmod(index * 0.236067977499790f, 1.0f);
    
    return QColor::fromHsvF(hue, saturation, value);
}

QVector<QColor> ClassLabels::getDefaultColors()
{
    // 预设 80 种区分度高的颜色
    QVector<QColor> colors;
    colors.reserve(80);
    
    // 基础颜色
    const QVector<QColor> baseColors = {
        QColor(255, 0, 0),      // 红
        QColor(0, 255, 0),      // 绿
        QColor(0, 0, 255),      // 蓝
        QColor(255, 255, 0),    // 黄
        QColor(255, 0, 255),    // 品红
        QColor(0, 255, 255),    // 青
        QColor(255, 128, 0),    // 橙
        QColor(128, 0, 255),    // 紫
        QColor(0, 255, 128),    // 春绿
        QColor(255, 0, 128),    // 玫红
    };

    for (int i = 0; i < 80; ++i) {
        if (i < baseColors.size()) {
            colors.append(baseColors[i]);
        } else {
            colors.append(generateColor(i));
        }
    }
    
    return colors;
}

QJsonArray ClassLabels::toJson() const
{
    QJsonArray array;
    for (auto it = m_classes.begin(); it != m_classes.end(); ++it) {
        array.append(it.value().toJson());
    }
    return array;
}

void ClassLabels::fromJson(const QJsonArray& array)
{
    clear();
    for (const auto& val : array) {
        ClassInfo info = ClassInfo::fromJson(val.toObject());
        m_classes[info.id] = info;
    }
}

} // namespace yolo
