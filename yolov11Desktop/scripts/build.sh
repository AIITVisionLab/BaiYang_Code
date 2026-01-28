#!/bin/bash
#
# YOLOv11 Qt 本地构建脚本
# 方便在开发机上编译和测试
#

set -e

# 终端颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 基本路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
INSTALL_DIR="${PROJECT_ROOT}/install"

# CMake 选项
CMAKE_OPTIONS=""

# 检测系统和架构
detect_system() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        echo_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    # 检测架构
    ARCH=$(uname -m)
    
    echo_info "系统: ${OS}, 架构: ${ARCH}"
}

# 检查依赖环境
check_dependencies() {
    echo_step "检查依赖..."
    
    # CMake
    if ! command -v cmake &> /dev/null; then
        echo_error "CMake未安装"
        echo_info "请安装: sudo apt install cmake (Ubuntu) / brew install cmake (macOS)"
        exit 1
    fi
    CMAKE_VERSION=$(cmake --version | head -1 | cut -d' ' -f3)
    echo_info "CMake版本: ${CMAKE_VERSION}"
    
    # Qt6
    if command -v qmake6 &> /dev/null; then
        QT_VERSION=$(qmake6 --version | grep -oP 'Qt version \K[0-9.]+')
        echo_info "Qt版本: ${QT_VERSION}"
    elif command -v qmake &> /dev/null; then
        QT_VERSION=$(qmake --version | grep -oP 'Qt version \K[0-9.]+')
        echo_info "Qt版本: ${QT_VERSION}"
    else
        echo_warn "Qt未找到，请确保已安装Qt6"
    fi
    
    # OpenCV
    if pkg-config --exists opencv4 2>/dev/null; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        echo_info "OpenCV版本: ${OPENCV_VERSION}"
    else
        echo_warn "OpenCV未找到 (将在CMake中检测)"
    fi
    
    # 编译器
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -1)
        echo_info "编译器: ${GCC_VERSION}"
    elif command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -1)
        echo_info "编译器: ${CLANG_VERSION}"
    fi
    
    echo_info "依赖检查完成"
}

# 安装系统依赖（Ubuntu/Debian）
install_deps_ubuntu() {
    echo_step "安装系统依赖 (Ubuntu/Debian)..."
    
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        libopencv-dev \
        qt6-base-dev \
        qt6-declarative-dev \
        qt6-multimedia-dev \
        qml6-module-qtquick \
        qml6-module-qtquick-controls \
        qml6-module-qtquick-layouts \
        qml6-module-qtquick-window \
        qml6-module-qtmultimedia
    
    echo_info "系统依赖安装完成"
}

# 下载 ONNX Runtime
download_onnxruntime() {
    echo_step "下载ONNX Runtime..."
    
    ORT_VERSION="1.16.3"
    ORT_DIR="${PROJECT_ROOT}/deps/onnxruntime"
    
    if [ -d "${ORT_DIR}" ]; then
        echo_info "ONNX Runtime已存在"
        return
    fi
    
    mkdir -p "${PROJECT_ROOT}/deps"
    cd "${PROJECT_ROOT}/deps"
    
    if [ "${OS}" == "linux" ]; then
        if [ "${ARCH}" == "x86_64" ]; then
            ORT_FILE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
            ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"
        elif [ "${ARCH}" == "aarch64" ]; then
            ORT_FILE="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
            ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"
        fi
    elif [ "${OS}" == "macos" ]; then
        ORT_FILE="onnxruntime-osx-universal2-${ORT_VERSION}.tgz"
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"
    fi
    
    if [ -n "${ORT_URL}" ]; then
        echo_info "下载: ${ORT_URL}"
        wget -q "${ORT_URL}" -O onnxruntime.tgz
        tar xf onnxruntime.tgz
        mv "onnxruntime-"* onnxruntime
        rm onnxruntime.tgz
        echo_info "ONNX Runtime下载完成"
    else
        echo_warn "不支持的平台，跳过ONNX Runtime下载"
    fi
}

# 配置CMake
configure() {
    echo_step "配置CMake..."
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    # 检测ONNX Runtime
    ORT_DIR="${PROJECT_ROOT}/deps/onnxruntime"
    if [ -d "${ORT_DIR}" ]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DONNXRUNTIME_ROOT=${ORT_DIR}"
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DENABLE_ONNXRUNTIME=ON"
    fi
    
    # 运行 CMake
    cmake "${PROJECT_ROOT}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        ${CMAKE_OPTIONS}
    
    echo_info "CMake配置完成"
}

# 开始构建
build() {
    echo_step "构建项目..."
    
    cd "${BUILD_DIR}"
    
    # 获取 CPU 核心数
    if [ "${OS}" == "macos" ]; then
        NPROC=$(sysctl -n hw.ncpu)
    else
        NPROC=$(nproc)
    fi
    
    cmake --build . --parallel ${NPROC}
    
    echo_info "构建完成"
}

# 安装
install_project() {
    echo_step "安装..."
    
    cd "${BUILD_DIR}"
    cmake --install .
    
    echo_info "安装完成: ${INSTALL_DIR}"
}

# 运行
run() {
    echo_step "运行应用..."
    
    if [ -f "${INSTALL_DIR}/bin/yolov11qt" ]; then
        "${INSTALL_DIR}/bin/yolov11qt" "$@"
    elif [ -f "${BUILD_DIR}/yolov11qt" ]; then
        "${BUILD_DIR}/yolov11qt" "$@"
    else
        echo_error "可执行文件未找到"
        exit 1
    fi
}

# 清理
clean() {
    echo_step "清理..."
    rm -rf "${BUILD_DIR}"
    rm -rf "${INSTALL_DIR}"
    echo_info "清理完成"
}

# 完整构建
all() {
    detect_system
    check_dependencies
    download_onnxruntime
    configure
    build
    install_project
}

# 显示帮助
show_help() {
    echo "YOLOv11 Qt 本地构建脚本"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  check         检查依赖"
    echo "  install-deps  安装系统依赖 (Ubuntu/Debian)"
    echo "  download-ort  下载ONNX Runtime"
    echo "  configure     配置CMake"
    echo "  build         构建项目"
    echo "  install       安装到install目录"
    echo "  all           完整构建 (configure + build + install)"
    echo "  run           运行应用"
    echo "  clean         清理构建目录"
    echo "  help          显示此帮助"
    echo ""
    echo "环境变量:"
    echo "  BUILD_TYPE    构建类型 (Debug/Release/RelWithDebInfo)"
    echo ""
    echo "示例:"
    echo "  $0 all                    # 完整构建"
    echo "  BUILD_TYPE=Debug $0 all   # Debug构建"
    echo "  $0 run --model my.onnx    # 运行应用"
}

# 主函数
main() {
    case "${1:-help}" in
        check)
            detect_system
            check_dependencies
            ;;
        install-deps)
            install_deps_ubuntu
            ;;
        download-ort)
            detect_system
            download_onnxruntime
            ;;
        configure)
            detect_system
            configure
            ;;
        build)
            build
            ;;
        install)
            install_project
            ;;
        all)
            all
            ;;
        run)
            shift
            run "$@"
            ;;
        clean)
            clean
            ;;
        help|*)
            show_help
            ;;
    esac
}

main "$@"
