/**
 * @file StatsPanel.qml
 * @brief 统计信息面板
 */

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: statsPanel
    
    // 统计数据
    property real fps: 0
    property real inferenceTime: 0
    property real preprocessTime: 0
    property real postprocessTime: 0
    property int detectionCount: 0
    property string modelName: ""
    property string inputSize: ""
    property bool isRunning: false
    
    // 外观
    color: "#E61E1E1E" // 深灰背景
    radius: 6 // 圆角
    border.width: 1
    border.color: "#454545" // 边框颜色
    
    implicitWidth: 200
    implicitHeight: contentColumn.height + 20
    
    ColumnLayout {
        id: contentColumn
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8
        
        // 标题
        Label {
            text: "📊 性能统计"
            font.bold: true
            font.pixelSize: 14
            color: "#FFFFFF" // 标题颜色
        }
        
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#505050"
        }
        
        // FPS 显示
        RowLayout {
            Layout.fillWidth: true
            
            Label {
                text: "FPS:"
                color: "#AAAAAA"
            }
            Item { Layout.fillWidth: true }
            Label {
                text: fps.toFixed(1)
                font.bold: true
                color: fps >= 25 ? "#4CAF50" : (fps >= 15 ? "#FFC107" : "#F44336")
            }
        }
        
        // 推理时间
        RowLayout {
            Layout.fillWidth: true
            
            Label {
                text: "推理时间:"
                color: "#AAAAAA"
            }
            Item { Layout.fillWidth: true }
            Label {
                text: inferenceTime.toFixed(1) + " ms"
                font.bold: true
                color: "#FFFFFF"
            }
        }
        
        // 详细耗时（可选显示）
        ColumnLayout {
            visible: preprocessTime > 0 || postprocessTime > 0
            Layout.leftMargin: 10
            spacing: 4
            
            RowLayout {
                Label {
                    text: "预处理:"
                    font.pixelSize: 11
                    color: "#888888"
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: preprocessTime.toFixed(1) + " ms"
                    font.pixelSize: 11
                    color: "#AAAAAA"
                }
            }
            
            RowLayout {
                Label {
                    text: "后处理:"
                    font.pixelSize: 11
                    color: "#888888"
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: postprocessTime.toFixed(1) + " ms"
                    font.pixelSize: 11
                    color: "#AAAAAA"
                }
            }
        }
        
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#404040"
        }
        
        // 检测数量
        RowLayout {
            Layout.fillWidth: true
            
            Label {
                text: "检测数:"
                color: "#AAAAAA"
            }
            Item { Layout.fillWidth: true }
            Label {
                text: detectionCount.toString()
                font.bold: true
                color: "#2196F3"
            }
        }
        
        // 模型信息
        ColumnLayout {
            visible: modelName.length > 0
            spacing: 4
            
            RowLayout {
                Layout.fillWidth: true
                
                Label {
                    text: "模型:"
                    color: "#AAAAAA"
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: modelName
                    font.pixelSize: 11
                    color: "white"
                    elide: Text.ElideMiddle
                    Layout.maximumWidth: 120
                }
            }
            
            RowLayout {
                visible: inputSize.length > 0
                Layout.fillWidth: true
                
                Label {
                    text: "输入尺寸:"
                    font.pixelSize: 11
                    color: "#888888"
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: inputSize
                    font.pixelSize: 11
                    color: "#AAAAAA"
                }
            }
        }
        
        // 状态指示
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#404040"
        }
        
        RowLayout {
            Layout.fillWidth: true
            
            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: isRunning ? "#4CAF50" : "#F44336"
                
                SequentialAnimation on opacity {
                    running: isRunning
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.3; duration: 500 }
                    NumberAnimation { to: 1.0; duration: 500 }
                }
            }
            
            Label {
                text: isRunning ? "运行中" : "已停止"
                color: isRunning ? "#4CAF50" : "#F44336"
                font.pixelSize: 11
            }
        }
    }
    
    // FPS历史图表
    Rectangle {
        id: fpsChart
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 10
        height: 40
        color: "#20FFFFFF"
        radius: 4
        visible: false  // 可以通过属性控制显示
        
        property var fpsHistory: []
        property int maxHistory: 60
        
        function addFps(value) {
            fpsHistory.push(value)
            if (fpsHistory.length > maxHistory) {
                fpsHistory.shift()
            }
            canvas.requestPaint()
        }
        
        Canvas {
            id: canvas
            anchors.fill: parent
            anchors.margins: 2
            
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                
                if (fpsChart.fpsHistory.length < 2) return
                
                var maxFps = Math.max(30, Math.max(...fpsChart.fpsHistory))
                var stepX = width / (fpsChart.maxHistory - 1)
                
                ctx.strokeStyle = "#4CAF50"
                ctx.lineWidth = 1.5
                ctx.beginPath()
                
                for (var i = 0; i < fpsChart.fpsHistory.length; i++) {
                    var x = i * stepX
                    var y = height - (fpsChart.fpsHistory[i] / maxFps) * height
                    
                    if (i === 0) {
                        ctx.moveTo(x, y)
                    } else {
                        ctx.lineTo(x, y)
                    }
                }
                
                ctx.stroke()
            }
        }
    }
    
    // FPS变化时更新图表
    onFpsChanged: {
        if (fpsChart.visible) {
            fpsChart.addFps(fps)
        }
    }
}
