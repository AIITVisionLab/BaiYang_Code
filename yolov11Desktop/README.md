# YOLOv11 Qt 跨平台目标检测系统

基于 C++ 和 Qt6 开发的高性能 YOLOv11 目标检测应用，专为 Windows、Linux 及嵌入式设备（如 Raspberry Pi）的跨平台部署而设计。

---

## 核心特性

*   **跨平台架构**：
    *   **Windows**: 支持通过 ONNX Runtime 调用 DirectML 进行硬件加速。
    *   **Linux**: 支持完整的 CUDA 加速，适用于生产环境。
    *   **嵌入式 (Raspberry Pi)**: 针对 ARM 架构深度优化，集成腾讯 NCNN 框架并启用 NEON 加速。
*   **灵活的模型部署**：支持动态加载自定义训练的 `.onnx` 模型和类别定义文件，无需重新编译代码。
*   **现代化用户界面**：基于 Qt6 QML 构建的响应式界面，针对触摸屏交互进行了优化。
*   **模块化设计**：UI 层、视觉处理层与推理层职责分离，易于维护与扩展。

---

## 跨平台支持

本项目在不同平台上采用统一的 C++/Qt6 + CMake 结构，关键差异主要体现在**推理后端选择与硬件加速**：

* **Windows**
    * 推荐引擎：ONNX Runtime（DirectML）。
    * 优势：对 NVIDIA/AMD/Intel GPU 均有较好兼容性，部署成本低。
* **Linux**
    * 推荐引擎：ONNX Runtime（CUDA）。
    * 优势：CUDA 性能最好，适合生产环境与高吞吐场景。
* **ARM / Raspberry Pi**
    * 推荐引擎：NCNN。
    * 优势：轻量、NEON 优化明显，更适合嵌入式设备。

平台差异已在 `InferenceFactory` 中抽象处理，应用层无需改动即可在不同系统上切换后端。

---

## 跨模型支持

项目支持**多模型、多任务类型**，通过自动解析模型输出与元数据完成适配：

* **模型格式**
    * `.onnx`（推荐）
    * `.param/.bin`（NCNN）
* **任务类型**
    * Detection（目标检测）
    * Segmentation（实例分割）
    * Pose（姿态估计）
    * OBB（旋转框）
    * Classification（分类）

**跨模型的关键机制**：

1. **自动任务识别**：加载模型后根据输出层名称与形状推断任务类型（Detection/Segmentation/Pose/OBB/Cls）。
2. **统一后处理入口**：`YoloPostProcess` 封装了各任务输出解析逻辑，后端只需提供原始张量。
3. **动态输入支持**：自动读取模型输入尺寸，支持动态分辨率模型。

这使得同一套 UI 与推理管线可以无缝切换不同 YOLOv11 变体和自定义训练模型。

---

##  构建依赖

### 基础组件
*   **Qt 6.2+**: 必需，用于 QML 界面渲染及多媒体模块。
*   **CMake 3.16+**: 项目构建系统生成器。
*   **OpenCV 4.5+**: 用于图像预处理及计算机视觉基础功能。

### 推理后端
本项目采用工厂模式支持多种推理引擎：
*   **ONNX Runtime** (PC端推荐): 具备最佳的兼容性，支持 CUDA 和 DirectML。
*   **NCNN** (嵌入式推荐): 在 ARM 架构上拥有最优的性能表现。
*   **OpenCV DNN**: 作为备用后端，在其他引擎不可用时提供基础支持。

---

## 快速开始

### 获取代码

```bash
git clone <https://github.com/AIITVisionLab/yolov11Desktop.git>
cd yolov11qt
```

### 一键构建（Linux）

```bash
chmod +x scripts/build.sh
./scripts/build.sh all
```

构建产物位于 `build/` 目录下，默认生成 `yolov11qt` 可执行文件。

---

##  构建指南

### Linux (Ubuntu/Debian/Arch)

使用提供的脚本进行标准化编译：

```bash
chmod +x scripts/build.sh
./scripts/build.sh all
```

请确保使用 **Release** 模式进行构建以获得最佳推理性能。

如需切换构建类型：

```bash
BUILD_TYPE=Release ./scripts/build.sh all
```

### Windows (Qt Creator / Visual Studio)

1.  打开 `CMakeLists.txt` 加载项目。
2.  配置项目。如果未能自动定位 OpenCV，请手动设置 `OpenCV_DIR`。
3.  执行构建。
    *   *注意*: 确保运行时所需的动态库（如 `onnxruntime.dll`）位于可执行文件目录或系统 PATH 中。

### Raspberry Pi (嵌入式)

针对 ARM/树莓派的专用编译脚本：

```bash
chmod +x scripts/build_rpi.sh
./scripts/build_rpi.sh
```

*建议*: 在设备上通过 `apt` 安装依赖库 (`libopencv-dev`, `qt6-base-dev`) 而非从源码编译，以节省构建时间。

---

##  运行方式

### 启动应用

```bash
./build/yolov11qt
```

### 常用操作

*   **加载模型**：左侧“模型设置”中选择 `.onnx` 模型文件。
*   **切换引擎**：在模型加载前选择推理引擎。
*   **选择数据源**：支持摄像头、视频、图片、文件夹与 RTSP。
*   **开始/暂停/停止**：底部控制栏按钮或快捷键触发。

---

---

##  模型部署指南

### 1. 导出模型

将训练好的 PyTorch 模型导出为支持动态维度的 ONNX 格式：

```bash
yolo export model=best.pt format=onnx opset=12 dynamic=True
```

### 2. 准备类别文件

创建一个文本文件 (例如 `classes.txt`)，按顺序列出类别名称，每行一个：

```text
person
bicycle
car
```

### 3. 在应用中加载

1.  启动应用程序。
2.  进入 **Settings (设置)** 页面。
3.  选择对应的 `.onnx` 模型文件和 `classes.txt` 类别文件。
4.  如有需要，调整 **Input Size (输入尺寸)** (默认为 640x640)。
5.  点击 **Apply (应用)**。

---

##  配置与参数说明

应用内关键参数说明：

*   **Confidence Threshold**：置信度阈值，过低会导致误检增加。
*   **IoU Threshold**：NMS 阈值，过低会抑制过多目标。
*   **Input Size**：模型输入尺寸（通常 640×640），与导出模型一致时效果最佳。
*   **Use GPU**：在支持的引擎下启用 GPU（ONNX Runtime / OpenCV DNN）。

---

##  引擎选择建议

*   **桌面端（Windows/Linux）**：优先 ONNX Runtime + GPU。
*   **嵌入式（ARM）**：优先 NCNN。
*   **兼容性需求**：使用 OpenCV DNN 作为兜底。

---

##  性能优化建议

*   使用 **Release** 构建。
*   合理设置 `numThreads`（CPU）与 `useGPU`（GPU）。
*   避免过高分辨率输入；推荐 640×640 或 1280×1280。
*   对嵌入式平台优先选择轻量模型（如 `yolo11n`）。

---

---

##  项目架构

代码库结构旨在确保关注点分离，提升可维护性：

*   `src/app/`
    *   **应用层**: 入口文件 (`main.cpp`)，负责初始化 Qt 应用程序及 QML 引擎。
*   `src/gui/`
    *   **UI/桥接层**: 处理 C++ 后端与 QML 前端的交互。`QmlBridge` 负责状态管理和配置同步。
*   `src/vision/`
    *   **视觉层**: 管理视频帧采集 (`FrameProvider`) 及推理结果的绘制 (`DrawUtils`)。
*   `src/inference/`
    *   **推理层**: 目标检测的核心逻辑。
    *   `backends/`: 包含各推理引擎的具体实现 (ONNX, NCNN 等)，与主应用逻辑完全解耦。
*   `src/core/`
    *   **核心组件**: 定义基础数据结构、NMS 算法及通用工具类。
*   `qml/`
    *   **表现层**: 定义用户界面的布局与样式。

---

##  项目结构

> 说明：`*.{h,cpp}` 表示同名的头/源文件对。

```
yolov11qt/
├─ CMakeLists.txt
├─ README.md
├─ models/                 # 示例模型（.onnx/.pt）
├─ qml/                    # QML 界面
│  ├─ main.qml
│  ├─ DetectionOverlay.qml
│  ├─ SettingsDialog.qml
│  └─ StatsPanel.qml
├─ resources/              # 资源文件
│  ├─ resources.qrc
│  └─ icons/
├─ scripts/                # 构建/转换脚本
│  ├─ build.sh
│  ├─ build_rpi.sh
│  └─ convert_model.sh
└─ src/                    # C++ 源码
   ├─ app/                 # 应用入口
   │  └─ main.cpp
   ├─ core/                # 核心数据结构与算法
   │  ├─ Detection.*
   │  ├─ ClassLabels.*
   │  ├─ NMS.*
   │  └─ YoloPostProcess.*
   ├─ gui/                 # QML 桥接与图像提供器
   │  ├─ QmlBridge.*
   │  └─ VideoFrameProvider.*
   ├─ vision/              # 视觉处理与绘制
   │  ├─ FrameProvider.*
   │  └─ DrawUtils.*
   ├─ inference/           # 推理接口与工厂
   │  ├─ InferenceEngine.*
   │  ├─ InferenceFactory.*
   │  └─ backends/
   │     ├─ OnnxRuntimeEngine.*
   │     ├─ OpenCVDnnEngine.*
   │     └─ NcnnEngine.*
   └─ utils/               # 通用工具
      ├─ Config.*
      ├─ FileUtils.*
      ├─ Logger.*
      ├─ ThreadPool.*
      └─ Timer.*
```

---

##  目录速览

*   `models/`：默认模型示例。
*   `resources/`：图标与资源文件。
*   `scripts/`：构建与模型转换脚本。
*   `build/`：本地构建产物（由 CMake 生成）。


