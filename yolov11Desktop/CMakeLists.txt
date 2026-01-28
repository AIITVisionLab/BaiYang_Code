cmake_minimum_required(VERSION 3.16)
project(yolov11qt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 判断平台类型
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(ARM_PLATFORM TRUE)
    message(STATUS "Detected ARM platform")
    add_compile_definitions(ARM_PLATFORM)
else()
    set(ARM_PLATFORM FALSE)
    message(STATUS "Detected x86/x64 platform")
endif()

# 构建选项
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(ENABLE_OPENCV_DNN "Enable OpenCV DNN backend" ON)
option(ENABLE_NCNN "Enable NCNN backend (for ARM)" OFF)
option(ENABLE_TENSORRT "Enable TensorRT backend" OFF)

# 查找 Qt6 组件
find_package(Qt6 REQUIRED COMPONENTS 
    Core 
    Quick 
    Gui 
    Widgets
    Multimedia
    Concurrent
)

# 查找 OpenCV
find_package(OpenCV REQUIRED COMPONENTS 
    core 
    imgproc 
    imgcodecs 
    videoio 
    highgui
    dnn
)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# 查找 ONNX Runtime
set(USE_ONNXRUNTIME OFF)
if(ENABLE_ONNXRUNTIME)
    # 先看环境变量，再用默认路径
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
    elseif(NOT DEFINED ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_ROOT "/usr" "/usr/local" "/opt/onnxruntime")
    endif()
    
    find_library(ONNXRUNTIME_LIB 
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_ROOT}
        PATH_SUFFIXES lib lib64
    )
    
    find_path(ONNXRUNTIME_INCLUDE_DIR 
        NAMES onnxruntime_cxx_api.h
        PATHS ${ONNXRUNTIME_ROOT}
        PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
    )
    
    if(ONNXRUNTIME_LIB)
        set(USE_ONNXRUNTIME ON)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
        add_compile_definitions(ENABLE_ONNXRUNTIME)
        add_compile_definitions(USE_ONNXRUNTIME)
    else()
        message(WARNING "ONNX Runtime not found")
    endif()
endif()

# 查找 NCNN
set(USE_NCNN OFF)
if(ENABLE_NCNN OR ARM_PLATFORM)
    find_package(ncnn QUIET)
    if(ncnn_FOUND)
        set(USE_NCNN ON)
        message(STATUS "Found NCNN")
        add_compile_definitions(ENABLE_NCNN)
    else()
        message(WARNING "NCNN not found")
    endif()
endif()

# 源码文件列表
set(CORE_SOURCES
    src/core/Detection.cpp
    src/core/Detection.h
    src/core/ClassLabels.cpp
    src/core/ClassLabels.h
    src/core/NMS.cpp
    src/core/NMS.h
    src/core/YoloPostProcess.cpp
    src/core/YoloPostProcess.h
)

set(INFERENCE_SOURCES
    src/inference/InferenceEngine.cpp
    src/inference/InferenceEngine.h
    src/inference/InferenceFactory.cpp
    src/inference/InferenceFactory.h
    src/inference/backends/OnnxRuntimeEngine.cpp
    src/inference/backends/OnnxRuntimeEngine.h
    src/inference/backends/OpenCVDnnEngine.cpp
    src/inference/backends/OpenCVDnnEngine.h
    src/inference/backends/NcnnEngine.cpp
    src/inference/backends/NcnnEngine.h
)

set(VISION_SOURCES
    src/vision/FrameProvider.cpp
    src/vision/FrameProvider.h
    src/vision/DrawUtils.cpp
    src/vision/DrawUtils.h
)

set(GUI_SOURCES
    src/gui/QmlBridge.cpp
    src/gui/QmlBridge.h
    src/gui/VideoFrameProvider.cpp
    src/gui/VideoFrameProvider.h
)

set(UTILS_SOURCES
    src/utils/Logger.cpp
    src/utils/Logger.h
    src/utils/Config.cpp
    src/utils/Config.h
    src/utils/Timer.cpp
    src/utils/Timer.h
    src/utils/ThreadPool.cpp
    src/utils/ThreadPool.h
    src/utils/FileUtils.cpp
    src/utils/FileUtils.h
)

set(MAIN_SOURCES
    src/app/main.cpp
)

# QML 文件列表
set(QML_SOURCES
    qml/main.qml
    qml/SettingsDialog.qml
    qml/DetectionOverlay.qml
    qml/StatsPanel.qml
)

# 生成可执行文件
qt_add_executable(${PROJECT_NAME}
    ${MAIN_SOURCES}
    ${CORE_SOURCES}
    ${INFERENCE_SOURCES}
    ${VISION_SOURCES}
    ${GUI_SOURCES}
    ${UTILS_SOURCES}
)

# QML 模块配置
qt_add_qml_module(${PROJECT_NAME}
    URI YOLOv11App
    VERSION 1.0
    QML_FILES ${QML_SOURCES}
    RESOURCE_PREFIX /
    RESOURCES
        resources/icons/camera.svg
        resources/icons/video.svg
        resources/icons/image.svg
        resources/icons/folder.svg
        resources/icons/settings.svg
        resources/icons/play.svg
        resources/icons/pause.svg
        resources/icons/stop.svg
        resources/icons/export.svg
        resources/icons/model.svg
)

# 头文件搜索路径
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vision
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${OpenCV_INCLUDE_DIRS}
)

if(USE_ONNXRUNTIME AND ONNXRUNTIME_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()

# 链接依赖库
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Quick
    Qt6::Gui
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::Concurrent
    ${OpenCV_LIBS}
)

if(USE_ONNXRUNTIME)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIB})
endif()

if(USE_NCNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)
endif()

# 编译优化选项
if(ARM_PLATFORM)
    target_compile_options(${PROJECT_NAME} PRIVATE
        -march=armv8-a
        -mtune=cortex-a72
        -mfpu=neon-fp-armv8
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()

# 安装规则
install(TARGETS ${PROJECT_NAME}
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 打印配置摘要
message(STATUS "")
message(STATUS "========== YOLOv11Qt Configuration ==========")
message(STATUS "Platform: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Qt version: ${Qt6_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${USE_ONNXRUNTIME}")
message(STATUS "NCNN: ${USE_NCNN}")
message(STATUS "ARM Platform: ${ARM_PLATFORM}")
message(STATUS "=============================================")
message(STATUS "")
