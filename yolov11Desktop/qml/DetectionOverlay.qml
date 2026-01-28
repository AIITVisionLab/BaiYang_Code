/**
 * @file DetectionOverlay.qml
 * @brief 检测结果覆盖层
 */

import QtQuick
import QtQuick.Controls

Item {
    id: overlay
    
    // 检测结果列表
    property var detections: []
    
    // 显示选项
    property bool showBoxes: true
    property bool showLabels: true
    property bool showConfidence: true
    property int lineWidth: 2
    property int fontSize: 12
    
    // 原图尺寸（用于坐标换算）
    property real imageWidth: 1
    property real imageHeight: 1
    
    // 颜色表
    property var colors: [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700",
        "#FF4500", "#1E90FF", "#00FA9A", "#FF1493", "#00BFFF"
    ]
    
    // 计算缩放比例
    readonly property real scaleX: width / imageWidth
    readonly property real scaleY: height / imageHeight
    
    // 根据类别获取颜色
    function getColor(classId) {
        return colors[classId % colors.length]
    }
    
    // 检测框绘制
    Repeater {
        model: showBoxes ? detections : []
        
        delegate: Item {
            id: boxItem
            
            // 坐标缩放
            x: modelData.x * overlay.scaleX
            y: modelData.y * overlay.scaleY
            width: modelData.width * overlay.scaleX
            height: modelData.height * overlay.scaleY
            
            property color boxColor: getColor(modelData.classId)
            
            // 边界框
            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.color: boxItem.boxColor
                border.width: lineWidth
                radius: 2
            }
            
            // 标签背景
            Rectangle {
                id: labelBg
                visible: showLabels || showConfidence
                
                anchors.bottom: parent.top
                anchors.left: parent.left
                anchors.bottomMargin: -1
                
                width: labelText.width + 8
                height: labelText.height + 4
                
                color: boxItem.boxColor
                radius: 2
                
                // 标签文字
                Text {
                    id: labelText
                    anchors.centerIn: parent
                    
                    text: {
                        var parts = []
                        if (showLabels && modelData.label) {
                            parts.push(modelData.label)
                        }
                        if (showConfidence) {
                            parts.push((modelData.confidence * 100).toFixed(1) + "%")
                        }
                        return parts.join(" ")
                    }
                    
                    font.pixelSize: fontSize
                    font.bold: true
                    color: "white"
                }
            }
            
            // 悬停效果
            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                
                onEntered: {
                    parent.scale = 1.02
                }
                onExited: {
                    parent.scale = 1.0
                }
                
                ToolTip.visible: containsMouse
                ToolTip.text: modelData.label + "\n置信度: " + (modelData.confidence * 100).toFixed(2) + "%\n" +
                              "位置: (" + modelData.x.toFixed(0) + ", " + modelData.y.toFixed(0) + ")\n" +
                              "尺寸: " + modelData.width.toFixed(0) + " × " + modelData.height.toFixed(0)
            }
            
            Behavior on scale {
                NumberAnimation { duration: 100 }
            }
        }
    }
    
    // 关键点绘制（用于姿态估计）
    Canvas {
        id: keypointCanvas
        anchors.fill: parent
        visible: detections.some(d => d.keypoints && d.keypoints.length > 0)
        
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            
            for (var i = 0; i < detections.length; i++) {
                var det = detections[i]
                if (!det.keypoints || det.keypoints.length === 0) continue
                
                var color = getColor(det.classId)
                
                // 绘制骨架连接
                var skeleton = [
                    [0, 1], [0, 2], [1, 3], [2, 4],  // 头部
                    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  // 上身
                    [5, 11], [6, 12], [11, 12],  // 躯干
                    [11, 13], [13, 15], [12, 14], [14, 16]  // 下身
                ]
                
                ctx.strokeStyle = color
                ctx.lineWidth = 2
                
                for (var j = 0; j < skeleton.length; j++) {
                    var s = skeleton[j]
                    if (s[0] < det.keypoints.length && s[1] < det.keypoints.length) {
                        var kp1 = det.keypoints[s[0]]
                        var kp2 = det.keypoints[s[1]]
                        
                        if (kp1.confidence > 0.5 && kp2.confidence > 0.5) {
                            ctx.beginPath()
                            ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY)
                            ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY)
                            ctx.stroke()
                        }
                    }
                }
                
                // 绘制关键点
                ctx.fillStyle = color
                for (var k = 0; k < det.keypoints.length; k++) {
                    var kp = det.keypoints[k]
                    if (kp.confidence > 0.5) {
                        ctx.beginPath()
                        ctx.arc(kp.x * scaleX, kp.y * scaleY, 4, 0, 2 * Math.PI)
                        ctx.fill()
                    }
                }
            }
        }
    }
    
    // 检测结果变更时重绘
    onDetectionsChanged: {
        keypointCanvas.requestPaint()
    }
}
