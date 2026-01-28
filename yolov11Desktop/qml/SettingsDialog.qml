/**
 * @file SettingsDialog.qml
 * @brief 设置对话框
 */

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: settingsDialog
    title: "设置"
    modal: true
    standardButtons: Dialog.Ok | Dialog.Cancel | Dialog.RestoreDefaults
    
    width: 500
    height: 600
    
    property alias confidenceThreshold: confSlider.value
    property alias iouThreshold: iouSlider.value
    property alias showLabels: showLabelsSwitch.checked
    property alias showConfidence: showConfSwitch.checked
    property alias showBoxes: showBoxesSwitch.checked
    property alias darkMode: darkModeSwitch.checked
    property alias targetFps: fpsSpinBox.value
    property alias useGpu: gpuSwitch.checked
    
    signal settingsChanged()
    signal resetToDefaults()
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 10
        
        TabBar {
            id: tabBar
            Layout.fillWidth: true
            
            TabButton {
                text: "检测"
                width: implicitWidth
            }
            TabButton {
                text: "显示"
                width: implicitWidth
            }
            TabButton {
                text: "性能"
                width: implicitWidth
            }
            TabButton {
                text: "高级"
                width: implicitWidth
            }
        }
        
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabBar.currentIndex
            
            // 检测设置
            ScrollView {
                ColumnLayout {
                    width: parent.width
                    spacing: 15
                    
                    GroupBox {
                        title: "置信度阈值"
                        Layout.fillWidth: true
                        
                        ColumnLayout {
                            anchors.fill: parent
                            
                            RowLayout {
                                Slider {
                                    id: confSlider
                                    Layout.fillWidth: true
                                    from: 0.0
                                    to: 1.0
                                    value: 0.25
                                    stepSize: 0.01
                                }
                                Label {
                                    text: confSlider.value.toFixed(2)
                                    Layout.preferredWidth: 40
                                }
                            }
                            Label {
                                text: "低于此置信度的检测结果将被过滤"
                                font.pixelSize: 11
                                opacity: 0.7
                            }
                        }
                    }
                    
                    GroupBox {
                        title: "IoU阈值 (NMS)"
                        Layout.fillWidth: true
                        
                        ColumnLayout {
                            anchors.fill: parent
                            
                            RowLayout {
                                Slider {
                                    id: iouSlider
                                    Layout.fillWidth: true
                                    from: 0.0
                                    to: 1.0
                                    value: 0.45
                                    stepSize: 0.01
                                }
                                Label {
                                    text: iouSlider.value.toFixed(2)
                                    Layout.preferredWidth: 40
                                }
                            }
                            Label {
                                text: "用于非极大值抑制的交并比阈值"
                                font.pixelSize: 11
                                opacity: 0.7
                            }
                        }
                    }
                    
                    GroupBox {
                        title: "最大检测数"
                        Layout.fillWidth: true
                        
                        SpinBox {
                            id: maxDetSpinBox
                            from: 1
                            to: 1000
                            value: 300
                            editable: true
                        }
                    }
                    
                    Item { Layout.fillHeight: true }
                }
            }
            
            // 显示设置
            ScrollView {
                ColumnLayout {
                    width: parent.width
                    spacing: 15
                    
                    GroupBox {
                        title: "标注显示"
                        Layout.fillWidth: true
                        
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 10
                            
                            Switch {
                                id: showBoxesSwitch
                                text: "显示边界框"
                                checked: true
                            }
                            
                            Switch {
                                id: showLabelsSwitch
                                text: "显示类别标签"
                                checked: true
                            }
                            
                            Switch {
                                id: showConfSwitch
                                text: "显示置信度"
                                checked: true
                            }
                        }
                    }
                    
                    GroupBox {
                        title: "外观"
                        Layout.fillWidth: true
                        
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 10
                            
                            RowLayout {
                                Label { text: "线条宽度:" }
                                SpinBox {
                                    id: lineWidthSpinBox
                                    from: 1
                                    to: 10
                                    value: 2
                                }
                            }
                            
                            RowLayout {
                                Label { text: "字体大小:" }
                                SpinBox {
                                    id: fontSizeSpinBox
                                    from: 8
                                    to: 24
                                    value: 12
                                }
                            }
                            
                            Switch {
                                id: darkModeSwitch
                                text: "深色模式"
                                checked: true
                            }
                        }
                    }
                    
                    Item { Layout.fillHeight: true }
                }
            }
            
            // 性能设置
            ScrollView {
                ColumnLayout {
                    width: parent.width
                    spacing: 15
                    
                    GroupBox {
                        title: "GPU加速"
                        Layout.fillWidth: true
                        
                        ColumnLayout {
                            anchors.fill: parent
                            
                            Switch {
                                id: gpuSwitch
                                text: "启用GPU加速"
                                checked: false
                            }
                            
                            ComboBox {
                                id: gpuDeviceCombo
                                Layout.fillWidth: true
                                enabled: gpuSwitch.checked
                                model: ["GPU 0", "GPU 1"]
                            }
                            
                            Label {
                                text: "需要CUDA或Vulkan支持"
                                font.pixelSize: 11
                                opacity: 0.7
                            }
                        }
                    }
                    
                    GroupBox {
                        title: "帧率控制"
                        Layout.fillWidth: true
                        
                        RowLayout {
                            Label { text: "目标FPS:" }
                            SpinBox {
                                id: fpsSpinBox
                                from: 1
                                to: 120
                                value: 30
                            }
                        }
                    }
                    
                    GroupBox {
                        title: "线程"
                        Layout.fillWidth: true
                        
                        RowLayout {
                            Label { text: "推理线程数:" }
                            SpinBox {
                                id: threadsSpinBox
                                from: 1
                                to: 16
                                value: 4
                            }
                        }
                    }
                    
                    Item { Layout.fillHeight: true }
                }
            }
            
            // 高级设置
            ScrollView {
                ColumnLayout {
                    width: parent.width
                    spacing: 15
                    
                    GroupBox {
                        title: "推理引擎"
                        Layout.fillWidth: true
                        
                        ComboBox {
                            id: engineCombo
                            Layout.fillWidth: true
                            model: ["ONNX Runtime", "OpenCV DNN", "NCNN", "TensorRT"]
                        }
                    }
                    
                    GroupBox {
                        title: "输入尺寸"
                        Layout.fillWidth: true
                        
                        ColumnLayout {
                            RowLayout {
                                Label { text: "宽度:" }
                                SpinBox {
                                    id: inputWidthSpinBox
                                    from: 320
                                    to: 1280
                                    value: 640
                                    stepSize: 32
                                }
                            }
                            RowLayout {
                                Label { text: "高度:" }
                                SpinBox {
                                    id: inputHeightSpinBox
                                    from: 320
                                    to: 1280
                                    value: 640
                                    stepSize: 32
                                }
                            }
                            Label {
                                text: "建议使用32的倍数"
                                font.pixelSize: 11
                                opacity: 0.7
                            }
                        }
                    }
                    
                    GroupBox {
                        title: "调试"
                        Layout.fillWidth: true
                        
                        ColumnLayout {
                            Switch {
                                id: profilingSwitch
                                text: "启用性能分析"
                            }
                            Switch {
                                id: verboseSwitch
                                text: "详细日志"
                            }
                        }
                    }
                    
                    Item { Layout.fillHeight: true }
                }
            }
        }
    }
    
    onAccepted: {
        settingsChanged()
    }
    
    onReset: {
        resetToDefaults()
    }
}
