#!/bin/bash
#
# YOLOv11 Qt 树莓派交叉编译脚本
# 支持 Raspberry Pi 3/4/5
#

set -e

# 终端颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 基本路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build-rpi"
INSTALL_DIR="${PROJECT_ROOT}/install-rpi"

# 树莓派交叉编译工具链
# 需要事先安装：sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
CROSS_COMPILE_PREFIX="aarch64-linux-gnu"
CC="${CROSS_COMPILE_PREFIX}-gcc"
CXX="${CROSS_COMPILE_PREFIX}-g++"

# Qt for Raspberry Pi 路径（需要自己配置）
QT_RPI_PATH="${QT_RPI_PATH:-/opt/qt6-rpi}"

# Sysroot 路径（树莓派系统镜像挂载点）
SYSROOT="${SYSROOT:-/mnt/rpi-sysroot}"

# 依赖库安装目录
DEPS_PREFIX="${PROJECT_ROOT}/deps-rpi"

# 检查环境
check_environment() {
    echo_info "检查编译环境..."
    
    # 检查交叉编译器
    if ! command -v ${CC} &> /dev/null; then
        echo_error "交叉编译器未找到: ${CC}"
        echo_info "请安装: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
        exit 1
    fi
    
    # 检查 CMake
    if ! command -v cmake &> /dev/null; then
        echo_error "CMake未找到"
        exit 1
    fi
    
    # 检查 Qt
    if [ ! -d "${QT_RPI_PATH}" ]; then
        echo_warn "Qt for Raspberry Pi 未找到: ${QT_RPI_PATH}"
        echo_info "请设置 QT_RPI_PATH 环境变量或下载Qt for Raspberry Pi"
        echo_info "可以使用: ./scripts/build_qt_for_rpi.sh 来编译Qt"
    fi
    
    echo_info "环境检查完成"
}

# 构建 OpenCV（ARM）
build_opencv() {
    echo_info "构建 OpenCV for ARM..."
    
    OPENCV_VERSION="4.8.0"
    OPENCV_SRC="${DEPS_PREFIX}/opencv-${OPENCV_VERSION}"
    OPENCV_BUILD="${DEPS_PREFIX}/opencv-build"
    OPENCV_INSTALL="${DEPS_PREFIX}/opencv"
    
    if [ -d "${OPENCV_INSTALL}" ]; then
        echo_info "OpenCV已存在，跳过构建"
        return
    fi
    
    # 下载 OpenCV
    if [ ! -d "${OPENCV_SRC}" ]; then
        mkdir -p "${DEPS_PREFIX}"
        cd "${DEPS_PREFIX}"
        wget -q "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz" -O opencv.tar.gz
        tar xf opencv.tar.gz
        rm opencv.tar.gz
    fi
    
    # 开始构建
    mkdir -p "${OPENCV_BUILD}"
    cd "${OPENCV_BUILD}"
    
    cmake "${OPENCV_SRC}" \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_INSTALL_PREFIX="${OPENCV_INSTALL}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_TBB=OFF \
        -DWITH_OPENMP=ON \
        -DWITH_IPP=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DWITH_V4L=ON \
        -DWITH_FFMPEG=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DENABLE_NEON=ON \
        -DENABLE_VFPV3=ON
    
    make -j$(nproc)
    make install
    
    echo_info "OpenCV构建完成"
}

# 构建 NCNN（ARM）
build_ncnn() {
    echo_info "构建 NCNN for ARM..."
    
    NCNN_VERSION="20231027"
    NCNN_SRC="${DEPS_PREFIX}/ncnn-${NCNN_VERSION}"
    NCNN_BUILD="${DEPS_PREFIX}/ncnn-build"
    NCNN_INSTALL="${DEPS_PREFIX}/ncnn"
    
    if [ -d "${NCNN_INSTALL}" ]; then
        echo_info "NCNN已存在，跳过构建"
        return
    fi
    
    # 下载 NCNN
    if [ ! -d "${NCNN_SRC}" ]; then
        mkdir -p "${DEPS_PREFIX}"
        cd "${DEPS_PREFIX}"
        wget -q "https://github.com/Tencent/ncnn/archive/${NCNN_VERSION}.tar.gz" -O ncnn.tar.gz
        tar xf ncnn.tar.gz
        rm ncnn.tar.gz
    fi
    
    # 开始构建
    mkdir -p "${NCNN_BUILD}"
    cd "${NCNN_BUILD}"
    
    cmake "${NCNN_SRC}" \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_INSTALL_PREFIX="${NCNN_INSTALL}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_OPENMP=ON \
        -DNCNN_VULKAN=OFF \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_TOOLS=ON \
        -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_BUILD_TESTS=OFF
    
    make -j$(nproc)
    make install
    
    echo_info "NCNN构建完成"
}

# 构建主项目
build_project() {
    echo_info "构建 YOLOv11Qt for ARM..."
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    # CMake 配置
    cmake "${PROJECT_ROOT}" \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${QT_RPI_PATH}" \
        -DOpenCV_DIR="${DEPS_PREFIX}/opencv/lib/cmake/opencv4" \
        -DNCNN_DIR="${DEPS_PREFIX}/ncnn/lib/cmake/ncnn" \
        -DENABLE_NCNN=ON \
        -DENABLE_ONNXRUNTIME=OFF \
        -DENABLE_TENSORRT=OFF \
        -DCMAKE_CXX_FLAGS="-march=armv8-a -mtune=cortex-a72"
    
    # 构建
    make -j$(nproc)
    
    # 安装
    make install
    
    echo_info "构建完成！安装目录: ${INSTALL_DIR}"
}

# 打包部署
package() {
    echo_info "打包部署文件..."
    
    PACKAGE_DIR="${PROJECT_ROOT}/yolov11qt-rpi-$(date +%Y%m%d)"
    
    mkdir -p "${PACKAGE_DIR}/lib"
    mkdir -p "${PACKAGE_DIR}/bin"
    mkdir -p "${PACKAGE_DIR}/models"
    mkdir -p "${PACKAGE_DIR}/config"
    
    # 复制可执行文件
    cp "${INSTALL_DIR}/bin/yolov11qt" "${PACKAGE_DIR}/bin/"
    
    # 复制依赖库
    cp -r "${DEPS_PREFIX}/opencv/lib/"*.so* "${PACKAGE_DIR}/lib/" 2>/dev/null || true
    cp -r "${DEPS_PREFIX}/ncnn/lib/"*.so* "${PACKAGE_DIR}/lib/" 2>/dev/null || true
    
    # 复制Qt库
    if [ -d "${QT_RPI_PATH}/lib" ]; then
        cp -r "${QT_RPI_PATH}/lib/"*.so* "${PACKAGE_DIR}/lib/" 2>/dev/null || true
    fi
    
    # 创建启动脚本
    cat > "${PACKAGE_DIR}/run.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_PHYSICAL_WIDTH=800
export QT_QPA_EGLFS_PHYSICAL_HEIGHT=480
exec "${SCRIPT_DIR}/bin/yolov11qt" "$@"
EOF
    chmod +x "${PACKAGE_DIR}/run.sh"
    
    # 创建systemd服务文件
    cat > "${PACKAGE_DIR}/yolov11qt.service" << EOF
[Unit]
Description=YOLOv11 Qt Detection Service
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=${PACKAGE_DIR}
ExecStart=${PACKAGE_DIR}/run.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    
    # 压缩
    cd "${PROJECT_ROOT}"
    tar czvf "${PACKAGE_DIR}.tar.gz" "$(basename ${PACKAGE_DIR})"
    
    echo_info "打包完成: ${PACKAGE_DIR}.tar.gz"
}

# 部署到树莓派
deploy() {
    RPI_HOST="${1:-pi@raspberrypi.local}"
    RPI_PATH="${2:-/home/pi/yolov11qt}"
    
    echo_info "部署到树莓派: ${RPI_HOST}:${RPI_PATH}"
    
    # 查找最新的包
    PACKAGE=$(ls -t ${PROJECT_ROOT}/yolov11qt-rpi-*.tar.gz 2>/dev/null | head -1)
    
    if [ -z "${PACKAGE}" ]; then
        echo_error "未找到部署包，请先运行 package"
        exit 1
    fi
    
    # 上传
    scp "${PACKAGE}" "${RPI_HOST}:/tmp/"
    
    # 解压并安装
    PACKAGE_NAME=$(basename "${PACKAGE}")
    ssh "${RPI_HOST}" << EOF
        mkdir -p ${RPI_PATH}
        cd ${RPI_PATH}
        tar xzf /tmp/${PACKAGE_NAME} --strip-components=1
        rm /tmp/${PACKAGE_NAME}
        chmod +x run.sh
        echo "部署完成！运行: cd ${RPI_PATH} && ./run.sh"
EOF
    
    echo_info "部署完成"
}

# 清理
clean() {
    echo_info "清理构建目录..."
    rm -rf "${BUILD_DIR}"
    rm -rf "${INSTALL_DIR}"
    echo_info "清理完成"
}

# 显示帮助
show_help() {
    echo "YOLOv11 Qt 树莓派交叉编译脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  check       检查编译环境"
    echo "  deps        构建依赖库 (OpenCV, NCNN)"
    echo "  build       构建主项目"
    echo "  all         构建所有 (deps + build)"
    echo "  package     打包部署文件"
    echo "  deploy      部署到树莓派 [host] [path]"
    echo "  clean       清理构建目录"
    echo "  help        显示此帮助"
    echo ""
    echo "环境变量:"
    echo "  QT_RPI_PATH   Qt for Raspberry Pi 安装路径"
    echo "  SYSROOT       树莓派系统镜像挂载点"
    echo ""
    echo "示例:"
    echo "  $0 all                           # 构建所有"
    echo "  $0 deploy pi@192.168.1.100       # 部署到指定IP"
}

# 主函数
main() {
    case "${1:-help}" in
        check)
            check_environment
            ;;
        deps)
            check_environment
            build_opencv
            build_ncnn
            ;;
        build)
            check_environment
            build_project
            ;;
        all)
            check_environment
            build_opencv
            build_ncnn
            build_project
            ;;
        package)
            package
            ;;
        deploy)
            deploy "$2" "$3"
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
