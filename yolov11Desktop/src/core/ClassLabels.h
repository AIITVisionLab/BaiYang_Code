/**
 * @file ClassLabels.h
 * @brief 类别标签管理接口
 */

#ifndef CLASS_LABELS_H
#define CLASS_LABELS_H

#include <QString>
#include <QStringList>
#include <QVector>
#include <QColor>
#include <QMap>
#include <QJsonObject>
#include <QJsonArray>
#include <memory>

namespace yolo {

/**
 * @brief 单个类别信息
 */
struct ClassInfo {
    int id;
    QString name;
    QColor color;
    bool enabled;       ///< 是否启用该类别显示

    ClassInfo() : id(-1), enabled(true) {}
    ClassInfo(int id, const QString& name, const QColor& color = Qt::green)
        : id(id), name(name), color(color), enabled(true) {}

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["id"] = id;
        obj["name"] = name;
        obj["color"] = color.name();
        obj["enabled"] = enabled;
        return obj;
    }

    static ClassInfo fromJson(const QJsonObject& obj) {
        ClassInfo info;
        info.id = obj["id"].toInt();
        info.name = obj["name"].toString();
        info.color = QColor(obj["color"].toString());
        info.enabled = obj["enabled"].toBool(true);
        return info;
    }
};

/**
 * @brief 类别标签管理器
 */
class ClassLabels {
public:
    ClassLabels();
    ~ClassLabels() = default;

    /// @brief 加载 COCO 80 类标签
    void loadCocoLabels();

    /// @brief 加载自定义标签文件
    bool loadFromFile(const QString& filePath);

    /// @brief 从字符串列表加载
    void loadFromList(const QStringList& labels);

    /// @brief 保存到文件
    bool saveToFile(const QString& filePath) const;

    /// @brief 添加类别
    void addClass(int id, const QString& name, const QColor& color = QColor());

    /// @brief 获取类别名称
    QString getClassName(int classId) const;

    /// @brief 获取类别颜色
    QColor getClassColor(int classId) const;

    /// @brief 获取类别信息
    ClassInfo getClassInfo(int classId) const;

    /// @brief 设置类别颜色
    void setClassColor(int classId, const QColor& color);

    /// @brief 设置类别启用状态
    void setClassEnabled(int classId, bool enabled);

    /// @brief 检查类别是否启用
    bool isClassEnabled(int classId) const;

    /// @brief 获取所有类别 ID
    QVector<int> getAllClassIds() const;

    /// @brief 获取启用的类别 ID
    QVector<int> getEnabledClassIds() const;

    /// @brief 类别数量
    int count() const { return m_classes.size(); }

    /// @brief 清空
    void clear() { m_classes.clear(); }

    /// @brief 生成随机颜色
    static QColor generateColor(int index);

    /// @brief 获取预定义颜色
    static QVector<QColor> getDefaultColors();

    /// @brief 转成 JSON
    QJsonArray toJson() const;

    /// @brief 从 JSON 读取
    void fromJson(const QJsonArray& array);

private:
    QMap<int, ClassInfo> m_classes;
    static QVector<QColor> s_defaultColors;
};

/**
 * @brief COCO 数据集 80 类标签
 */
namespace CocoLabels {
    const QStringList LABELS = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };

    /// @brief 人体姿态关键点名称（17 个点）
    const QStringList KEYPOINT_NAMES = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    };

    /// @brief 姿态骨架连接
    const QVector<QPair<int, int>> SKELETON = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4},      // 头部
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},  // 上肢
        {5, 11}, {6, 12}, {11, 12},          // 躯干
        {11, 13}, {13, 15}, {12, 14}, {14, 16}   // 下肢
    };
}

} // namespace yolo

#endif // CLASS_LABELS_H
