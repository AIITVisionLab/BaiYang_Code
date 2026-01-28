/**
 * @file main.qml
 * @brief 应用主界面
 */

import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt.labs.platform as Platform

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1280
    height: 720
    minimumWidth: 800
    minimumHeight: 600
    title: qsTr("YOLOv11 目标检测 - ") + backend.statusText

    // Material 主题与配色
    Material.theme: Material.Dark
    Material.accent: "#409EFF" // 强调色
    Material.primary: "#303133" // 主色
    Material.background: "#1E1E1E" // 背景色

    // 主布局
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // 左侧控制面板
        Rectangle {
            id: controlPanel
            Layout.preferredWidth: 280
            Layout.fillHeight: true
            color: "#252526"
            
            // 右侧分隔线
            Rectangle {
                width: 1
                height: parent.height
                anchors.right: parent.right
                color: "#3E3E42" // 分隔线颜色
                opacity: 1.0
            }

            ScrollView {
                anchors.fill: parent
                anchors.margins: 10
                contentWidth: availableWidth

                ColumnLayout {
                    width: parent.width
                    spacing: 15

                    // 标题区域
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 5

                        Label {
                            text: "YOLOv11"
                            font.pixelSize: 28
                            font.family: "Segoe UI"
                            font.bold: true
                            color: "#FFFFFF" // 标题颜色
                        }
                        Label {
                            text: "目标检测系统"
                            font.pixelSize: 14
                            color: "#aaaaaa"
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: "#333"
                    }

                    // 模型设置
                    GroupBox {
                        Layout.fillWidth: true
                        title: "模型设置"

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 10

                            Button {
                                Layout.fillWidth: true
                                text: backend.modelLoaded ? "更换模型" : "加载模型"
                                icon.source: "qrc:/resources/icons/model.svg"
                                onClicked: modelDialog.open()
                            }

                            Label {
                                text: backend.modelLoaded ? "已加载: " + backend.modelName : "未加载模型"
                                font.pixelSize: 12
                                color: backend.modelLoaded ? "#4caf50" : "#ff9800"
                                elide: Text.ElideMiddle
                                Layout.fillWidth: true
                            }

                            ComboBox {
                                id: engineCombo
                                Layout.fillWidth: true
                                model: backend.availableEngines
                                enabled: !backend.modelLoaded
                            }
                        }
                    }

                    // 数据源选择
                    GroupBox {
                        Layout.fillWidth: true
                        title: "数据源"

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 8

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 5

                                Button {
                                    Layout.fillWidth: true
                                    text: "摄像头"
                                    icon.source: "qrc:/resources/icons/camera.svg"
                                    onClicked: cameraDialog.open()
                                }

                                Button {
                                    Layout.fillWidth: true
                                    text: "视频"
                                    icon.source: "qrc:/resources/icons/video.svg"
                                    onClicked: videoDialog.open()
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 5

                                Button {
                                    Layout.fillWidth: true
                                    text: "图片"
                                    icon.source: "qrc:/resources/icons/image.svg"
                                    onClicked: imageDialog.open()
                                }

                                Button {
                                    Layout.fillWidth: true
                                    text: "文件夹"
                                    icon.source: "qrc:/resources/icons/folder.svg"
                                    onClicked: folderDialog.open()
                                }
                            }

                            TextField {
                                id: rtspInput
                                Layout.fillWidth: true
                                placeholderText: "RTSP地址..."
                                onAccepted: backend.openRtspStream(text)
                            }

                            Label {
                                text: backend.currentSource ? "当前: " + backend.currentSource : "未选择数据源"
                                font.pixelSize: 11
                                color: "#888"
                                elide: Text.ElideMiddle
                                Layout.fillWidth: true
                            }
                        }
                    }

                    // 检测参数
                    GroupBox {
                        Layout.fillWidth: true
                        title: "检测参数"

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 10

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2

                                RowLayout {
                                    Label { text: "置信度阈值" }
                                    Item { Layout.fillWidth: true }
                                    Label { text: confSlider.value.toFixed(2); color: "#4fc3f7" }
                                }
                                Slider {
                                    id: confSlider
                                    Layout.fillWidth: true
                                    from: 0.1
                                    to: 1.0
                                    value: backend.confidenceThreshold
                                    onValueChanged: backend.confidenceThreshold = value
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2

                                RowLayout {
                                    Label { text: "IoU阈值" }
                                    Item { Layout.fillWidth: true }
                                    Label { text: iouSlider.value.toFixed(2); color: "#4fc3f7" }
                                }
                                Slider {
                                    id: iouSlider
                                    Layout.fillWidth: true
                                    from: 0.1
                                    to: 1.0
                                    value: backend.iouThreshold
                                    onValueChanged: backend.iouThreshold = value
                                }
                            }

                            CheckBox {
                                id: gpuCheck
                                text: "使用GPU加速"
                                checked: false
                                onCheckedChanged: backend.setUseGPU(checked)
                            }
                        }
                    }

                    // 显示选项
                    GroupBox {
                        Layout.fillWidth: true
                        title: "显示选项"

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 5

                            CheckBox {
                                id: showLabelsCheck
                                text: "显示标签"
                                checked: true
                            }

                            CheckBox {
                                id: showConfCheck
                                text: "显示置信度"
                                checked: true
                            }

                            CheckBox {
                                id: showBoxesCheck
                                text: "显示边界框"
                                checked: true
                            }
                        }
                    }

                    // 导出功能
                    GroupBox {
                        Layout.fillWidth: true
                        title: "导出"

                        RowLayout {
                            anchors.fill: parent
                            spacing: 5

                            Button {
                                Layout.fillWidth: true
                                text: "截图"
                                onClicked: exportImageDialog.open()
                            }

                            Button {
                                Layout.fillWidth: true
                                text: "结果"
                                onClicked: exportJsonDialog.open()
                            }
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }

        // 主视图区域
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#121212"

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                // 视频/图像显示区
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#000"

                    Image {
                        id: videoView
                        anchors.fill: parent
                        anchors.margins: 10
                        fillMode: Image.PreserveAspectFit
                        cache: false
                        asynchronous: false
                        
                        // 通过图像提供器刷新画面
                        source: "image://video/frame?" + frameUpdateCounter
                        
                        property int frameUpdateCounter: 0

                        // 无画面时的提示
                        Rectangle {
                            anchors.centerIn: parent
                            width: 300
                            height: 150
                            color: "#1e1e1e"
                            radius: 10
                            visible: !backend.isRunning && !backend.currentSource

                            ColumnLayout {
                                anchors.centerIn: parent
                                spacing: 10

                                Label {
                                    text: "📷"
                                    font.pixelSize: 48
                                    Layout.alignment: Qt.AlignHCenter
                                }
                                Label {
                                    text: "选择数据源开始检测"
                                    font.pixelSize: 14
                                    color: "#888"
                                    Layout.alignment: Qt.AlignHCenter
                                }
                            }
                        }
                    }

                    // 帧刷新信号
                    Connections {
                        target: frameEmitter
                        function onFrameUpdated() {
                            videoView.frameUpdateCounter++
                        }
                    }
                }

                // 底部控制栏
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 60
                    color: "#1e1e1e"

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 15

                        // 播放控制
                        RowLayout {
                            spacing: 5

                            Button {
                                id: playButton
                                icon.source: backend.isRunning ? "qrc:/resources/icons/pause.svg" : "qrc:/resources/icons/play.svg"
                                icon.width: 24
                                icon.height: 24
                                enabled: backend.modelLoaded && backend.currentSource
                                onClicked: {
                                    if (backend.isRunning) {
                                        if (backend.isPaused) {
                                            backend.resume()
                                        } else {
                                            backend.pause()
                                        }
                                    } else {
                                        backend.start()
                                    }
                                }

                                ToolTip.visible: hovered
                                ToolTip.text: backend.isRunning ? (backend.isPaused ? "继续" : "暂停") : "开始"
                            }

                            Button {
                                icon.source: "qrc:/resources/icons/stop.svg"
                                icon.width: 24
                                icon.height: 24
                                enabled: backend.isRunning
                                onClicked: backend.stop()

                                ToolTip.visible: hovered
                                ToolTip.text: "停止"
                            }
                        }

                        // 进度条
                        ProgressBar {
                            id: progressBar
                            Layout.fillWidth: true
                            from: 0
                            to: 100
                            value: backend.progress
                            visible: backend.progress > 0 && backend.progress < 100
                        }

                        // 分隔线
                        Rectangle {
                            width: 1
                            height: 30
                            color: "#333"
                        }

                        // 统计信息
                        RowLayout {
                            spacing: 20

                            // FPS
                            RowLayout {
                                spacing: 5
                                Label { text: "FPS:"; color: "#888" }
                                Label {
                                    text: backend.fps.toFixed(1)
                                    color: backend.fps > 20 ? "#4caf50" : (backend.fps > 10 ? "#ff9800" : "#f44336")
                                    font.bold: true
                                }
                            }

                            // 推理耗时
                            RowLayout {
                                spacing: 5
                                Label { text: "推理:"; color: "#888" }
                                Label {
                                    text: backend.inferenceTime.toFixed(1) + " ms"
                                    color: "#4fc3f7"
                                }
                            }

                            // 检测数量
                            RowLayout {
                                spacing: 5
                                Label { text: "检测:"; color: "#888" }
                                Label {
                                    text: backend.detectionCount
                                    color: "#fff"
                                    font.bold: true
                                }
                            }
                        }
                    }
                }
            }
        }

        // 结果面板
        Rectangle {
            id: resultsPanel
            Layout.preferredWidth: 250
            Layout.fillHeight: true
            color: "#1e1e1e"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10

                Label {
                    text: "检测结果"
                    font.pixelSize: 16
                    font.bold: true
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 1
                    color: "#333"
                }

                // 结果列表
                ListView {
                    id: detectionList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 5

                    model: ListModel { id: detectionModel }

                    delegate: Rectangle {
                        width: detectionList.width
                        height: 50
                        color: mouseArea.containsMouse ? "#2a2a2a" : "#1e1e1e"
                        radius: 5

                        MouseArea {
                            id: mouseArea
                            anchors.fill: parent
                            hoverEnabled: true
                        }

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 10

                            Rectangle {
                                width: 4
                                height: parent.height
                                color: model.color || "#4fc3f7"
                                radius: 2
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2

                                Label {
                                    text: model.className || "Unknown"
                                    font.bold: true
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }

                                Label {
                                    text: (model.confidence * 100).toFixed(1) + "%"
                                    font.pixelSize: 11
                                    color: "#888"
                                }
                            }
                        }
                    }

                    // 空状态提示
                    Label {
                        anchors.centerIn: parent
                        text: "暂无检测结果"
                        color: "#555"
                        visible: detectionModel.count === 0
                    }
                }

                // 类别筛选
                Button {
                    Layout.fillWidth: true
                    text: "类别筛选"
                    onClicked: classFilterDialog.open()
                }
            }
        }
    }

    // 同步检测结果到列表
    Connections {
        target: backend
        function onDetectionsReady(detections) {
            detectionModel.clear()
            for (var i = 0; i < detections.length; i++) {
                detectionModel.append(detections[i])
            }
        }
    }

    // 错误提示
    Connections {
        target: backend
        function onErrorOccurred(error) {
            errorDialog.text = error
            errorDialog.open()
        }
    }

    // 文件选择对话框
    FileDialog {
        id: modelDialog
        title: "选择模型文件"
        nameFilters: ["ONNX模型 (*.onnx)", "所有文件 (*)"]
        onAccepted: backend.loadModel(selectedFile, engineCombo.currentText)
    }

    FileDialog {
        id: videoDialog
        title: "选择视频文件"
        nameFilters: ["视频文件 (*.mp4 *.avi *.mkv *.mov)", "所有文件 (*)"]
        onAccepted: backend.openVideo(selectedFile)
    }

    FileDialog {
        id: imageDialog
        title: "选择图片文件"
        nameFilters: ["图片文件 (*.jpg *.jpeg *.png *.bmp)", "所有文件 (*)"]
        onAccepted: backend.openImage(selectedFile)
    }

    FolderDialog {
        id: folderDialog
        title: "选择图片文件夹"
        onAccepted: backend.openImageFolder(selectedFolder)
    }

    FileDialog {
        id: exportImageDialog
        title: "保存截图"
        nameFilters: ["PNG图片 (*.png)", "JPEG图片 (*.jpg)"]
        fileMode: FileDialog.SaveFile
        onAccepted: backend.exportCurrentFrame(selectedFile)
    }

    FileDialog {
        id: exportJsonDialog
        title: "保存检测结果"
        nameFilters: ["JSON文件 (*.json)"]
        fileMode: FileDialog.SaveFile
        onAccepted: backend.exportResults(selectedFile)
    }

    // 摄像头选择对话框
    Dialog {
        id: cameraDialog
        title: "选择摄像头"
        standardButtons: Dialog.Ok | Dialog.Cancel
        anchors.centerIn: parent

        ColumnLayout {
            spacing: 10

            ComboBox {
                id: cameraCombo
                Layout.fillWidth: true
                model: backend.availableCameras
            }

            TextField {
                id: customCameraInput
                Layout.fillWidth: true
                placeholderText: "或输入设备路径..."
            }
        }

        onAccepted: {
            var cam = customCameraInput.text || cameraCombo.currentText
            if (cam) backend.openCamera(cam)
        }
    }

    // 类别筛选对话框
    Dialog {
        id: classFilterDialog
        title: "类别筛选"
        standardButtons: Dialog.Ok | Dialog.Cancel
        anchors.centerIn: parent
        width: 300
        height: 400

        ListView {
            anchors.fill: parent
            model: backend.getClassList()
            delegate: CheckDelegate {
                width: parent.width
                text: modelData.name
                checked: modelData.enabled
            }
        }
    }

    // 错误提示对话框
    Dialog {
        id: errorDialog
        title: "错误"
        standardButtons: Dialog.Ok
        anchors.centerIn: parent

        property alias text: errorLabel.text

        Label {
            id: errorLabel
            color: "#f44336"
        }
    }

    // 快捷键
    Shortcut {
        sequence: "Space"
        onActivated: playButton.clicked()
    }

    Shortcut {
        sequence: "Escape"
        onActivated: backend.stop()
    }

    Shortcut {
        sequence: "Ctrl+O"
        onActivated: modelDialog.open()
    }

    Shortcut {
        sequence: "Ctrl+S"
        onActivated: exportImageDialog.open()
    }
}
