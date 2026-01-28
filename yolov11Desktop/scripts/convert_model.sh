#!/bin/bash
#
# YOLOv11 Qt 模型转换脚本
# 把 ONNX 模型转成 NCNN 格式（用于树莓派部署）
#

set -e

# 终端颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 基本路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="${PROJECT_ROOT}/tools"
NCNN_TOOLS="${TOOLS_DIR}/ncnn"

# 检查 NCNN 工具
check_ncnn_tools() {
    if [ -f "${NCNN_TOOLS}/onnx2ncnn" ]; then
        echo_info "使用本地NCNN工具"
        ONNX2NCNN="${NCNN_TOOLS}/onnx2ncnn"
        NCNNOPTIMIZE="${NCNN_TOOLS}/ncnnoptimize"
    elif command -v onnx2ncnn &> /dev/null; then
        echo_info "使用系统NCNN工具"
        ONNX2NCNN="onnx2ncnn"
        NCNNOPTIMIZE="ncnnoptimize"
    else
        echo_error "NCNN工具未找到"
        echo_info "请安装NCNN或下载预编译工具"
        exit 1
    fi
}

# 下载 NCNN 工具
download_ncnn_tools() {
    echo_info "下载NCNN工具..."
    
    mkdir -p "${NCNN_TOOLS}"
    cd "${TOOLS_DIR}"
    
    # 下载预编译的 NCNN 工具
    NCNN_VERSION="20231027"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        TOOLS_URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/ncnn-${NCNN_VERSION}-ubuntu-2204.zip"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        TOOLS_URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/ncnn-${NCNN_VERSION}-macos.zip"
    else
        echo_error "不支持的系统"
        exit 1
    fi
    
    wget -q "${TOOLS_URL}" -O ncnn.zip
    unzip -o ncnn.zip -d ncnn_extract
    
    # 复制工具到本地目录
    find ncnn_extract -name "onnx2ncnn" -exec cp {} "${NCNN_TOOLS}/" \;
    find ncnn_extract -name "ncnnoptimize" -exec cp {} "${NCNN_TOOLS}/" \;
    chmod +x "${NCNN_TOOLS}"/*
    
    rm -rf ncnn.zip ncnn_extract
    
    echo_info "NCNN工具下载完成"
}

# ONNX 转 NCNN
convert_onnx_to_ncnn() {
    INPUT_ONNX="$1"
    OUTPUT_DIR="${2:-$(dirname "$INPUT_ONNX")}"
    
    if [ ! -f "${INPUT_ONNX}" ]; then
        echo_error "ONNX模型不存在: ${INPUT_ONNX}"
        exit 1
    fi
    
    check_ncnn_tools
    
    # 生成输出文件名
    BASENAME=$(basename "${INPUT_ONNX}" .onnx)
    OUTPUT_PARAM="${OUTPUT_DIR}/${BASENAME}.param"
    OUTPUT_BIN="${OUTPUT_DIR}/${BASENAME}.bin"
    OUTPUT_PARAM_OPT="${OUTPUT_DIR}/${BASENAME}_opt.param"
    OUTPUT_BIN_OPT="${OUTPUT_DIR}/${BASENAME}_opt.bin"
    
    echo_info "转换模型: ${INPUT_ONNX}"
    
    # 执行转换
    "${ONNX2NCNN}" "${INPUT_ONNX}" "${OUTPUT_PARAM}" "${OUTPUT_BIN}"
    
    if [ $? -ne 0 ]; then
        echo_error "ONNX转NCNN失败"
        exit 1
    fi
    
    echo_info "生成: ${OUTPUT_PARAM}, ${OUTPUT_BIN}"
    
    # 再做一次优化
    echo_info "优化模型..."
    "${NCNNOPTIMIZE}" "${OUTPUT_PARAM}" "${OUTPUT_BIN}" "${OUTPUT_PARAM_OPT}" "${OUTPUT_BIN_OPT}" 0
    
    if [ $? -eq 0 ]; then
        echo_info "生成优化版本: ${OUTPUT_PARAM_OPT}, ${OUTPUT_BIN_OPT}"
    else
        echo_warn "优化失败，使用未优化版本"
    fi
    
    # 输出文件大小
    echo_info "模型大小:"
    ls -lh "${OUTPUT_DIR}/${BASENAME}"* 2>/dev/null | awk '{print "  "$NF": "$5}'
    
    echo_info "转换完成！"
}

# 量化模型（INT8）
quantize_model() {
    INPUT_PARAM="$1"
    INPUT_BIN="$2"
    CALIBRATION_IMAGES="$3"
    
    if [ -z "${CALIBRATION_IMAGES}" ]; then
        echo_error "需要提供校准图片目录"
        echo_info "用法: $0 quantize model.param model.bin /path/to/calibration/images"
        exit 1
    fi
    
    check_ncnn_tools
    
    BASENAME=$(basename "${INPUT_PARAM}" .param)
    OUTPUT_DIR=$(dirname "${INPUT_PARAM}")
    OUTPUT_TABLE="${OUTPUT_DIR}/${BASENAME}.table"
    OUTPUT_PARAM="${OUTPUT_DIR}/${BASENAME}_int8.param"
    OUTPUT_BIN="${OUTPUT_DIR}/${BASENAME}_int8.bin"
    
    echo_info "生成量化表..."
    
    # 用 ncnn2table 生成量化表（如果可用）
    if [ -f "${NCNN_TOOLS}/ncnn2table" ]; then
        "${NCNN_TOOLS}/ncnn2table" \
            "${INPUT_PARAM}" "${INPUT_BIN}" \
            "${CALIBRATION_IMAGES}" "${OUTPUT_TABLE}" \
            mean=[0,0,0] norm=[0.003921,0.003921,0.003921] shape=[640,640,3] pixel=BGR thread=4 method=kl
    else
        echo_error "ncnn2table工具不存在，无法量化"
        exit 1
    fi
    
    echo_info "量化模型..."
    "${NCNNOPTIMIZE}" \
        "${INPUT_PARAM}" "${INPUT_BIN}" \
        "${OUTPUT_PARAM}" "${OUTPUT_BIN}" \
        1 "${OUTPUT_TABLE}"
    
    echo_info "量化完成: ${OUTPUT_PARAM}, ${OUTPUT_BIN}"
}

# 验证模型
validate_model() {
    MODEL_PATH="$1"
    
    if [ -z "${MODEL_PATH}" ]; then
        echo_error "请提供模型路径"
        exit 1
    fi
    
    echo_info "验证模型: ${MODEL_PATH}"
    
    # 先检查文件是否存在
    if [[ "${MODEL_PATH}" == *.onnx ]]; then
        if [ ! -f "${MODEL_PATH}" ]; then
            echo_error "ONNX模型不存在"
            exit 1
        fi
        
        # 用 Python 验证 ONNX 模型
        python3 << EOF
import onnx
import sys

try:
    model = onnx.load("${MODEL_PATH}")
    onnx.checker.check_model(model)
    
    print("模型信息:")
    print(f"  IR版本: {model.ir_version}")
    print(f"  生产者: {model.producer_name} {model.producer_version}")
    print(f"  算子集: {[opset.version for opset in model.opset_import]}")
    
    print("\n输入:")
    for input in model.graph.input:
        shape = [dim.dim_value if dim.dim_value else dim.dim_param for dim in input.type.tensor_type.shape.dim]
        print(f"  {input.name}: {shape}")
    
    print("\n输出:")
    for output in model.graph.output:
        shape = [dim.dim_value if dim.dim_value else dim.dim_param for dim in output.type.tensor_type.shape.dim]
        print(f"  {output.name}: {shape}")
    
    print("\n✓ 模型验证通过")
except Exception as e:
    print(f"✗ 模型验证失败: {e}")
    sys.exit(1)
EOF
        
    elif [[ "${MODEL_PATH}" == *.param ]]; then
        BIN_PATH="${MODEL_PATH%.param}.bin"
        if [ ! -f "${MODEL_PATH}" ] || [ ! -f "${BIN_PATH}" ]; then
            echo_error "NCNN模型文件不完整"
            exit 1
        fi
        
        echo_info "NCNN模型文件存在"
        echo "  Param: $(ls -lh "${MODEL_PATH}" | awk '{print $5}')"
        echo "  Bin: $(ls -lh "${BIN_PATH}" | awk '{print $5}')"
        
        # 检查param文件内容
        echo_info "模型结构:"
        head -20 "${MODEL_PATH}"
    else
        echo_error "不支持的模型格式"
        exit 1
    fi
}

# 显示帮助
show_help() {
    echo "YOLOv11 模型转换工具"
    echo ""
    echo "用法: $0 [命令] [参数]"
    echo ""
    echo "命令:"
    echo "  download-tools          下载NCNN工具"
    echo "  convert <onnx> [outdir] 转换ONNX到NCNN"
    echo "  quantize <param> <bin> <images>  INT8量化"
    echo "  validate <model>        验证模型"
    echo "  help                    显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 convert yolov11n.onnx"
    echo "  $0 convert yolov11n.onnx ./models"
    echo "  $0 quantize model.param model.bin ./calibration_images"
    echo "  $0 validate yolov11n.onnx"
}

# 主函数
main() {
    case "${1:-help}" in
        download-tools)
            download_ncnn_tools
            ;;
        convert)
            convert_onnx_to_ncnn "$2" "$3"
            ;;
        quantize)
            quantize_model "$2" "$3" "$4"
            ;;
        validate)
            validate_model "$2"
            ;;
        help|*)
            show_help
            ;;
    esac
}

main "$@"
