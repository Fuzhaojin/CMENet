# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project - Prediction Script

#!/bin/bash
# =============================================================================
# CMENet 单文件预测脚本
# 用法: bash predict.sh <T1路径> <T1C路径> <TOF路径> [输出目录] [权重路径]
#
# 示例:
#   bash predict.sh ./data/T1.nii ./data/T1C.nii ./data/TOF.nii
#   bash predict.sh ./data/T1.nii ./data/T1C.nii ./data/TOF.nii ./my_output
#   bash predict.sh ./data/T1.nii ./data/T1C.nii ./data/TOF.nii ./my_output ./my_weight.pth
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 默认值
T1=""
T1C=""
TOF=""
OUTPUT_DIR="./pred_output"
WEIGHT="results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth"
DEVICE="cuda:0"
NORM_DATASET="caro"
THRESHOLD="0.5"

# 解析参数
if [ $# -ge 3 ]; then
    T1="$1"
    T1C="$2"
    TOF="$3"
fi
if [ $# -ge 4 ]; then
    OUTPUT_DIR="$4"
fi
if [ $# -ge 5 ]; then
    WEIGHT="$5"
fi

# 如果前三个参数未提供，交互式询问
if [ -z "$T1" ]; then
    echo "========================================"
    echo "  CMENet 单文件预测"
    echo "========================================"
    read -p "请输入 T1 文件路径   : " T1
    read -p "请输入 T1C 文件路径  : " T1C
    read -p "请输入 TOF 文件路径  : " TOF
    read -p "输出目录 (默认: $OUTPUT_DIR) : " input_output
    [ -n "$input_output" ] && OUTPUT_DIR="$input_output"
    read -p "权重文件路径 (默认: $WEIGHT) : " input_weight
    [ -n "$input_weight" ] && WEIGHT="$input_weight"
    echo ""
fi

echo "========================================"
echo "  开始预测"
echo "========================================"
echo "  T1    : $T1"
echo "  T1C   : $T1C"
echo "  TOF   : $TOF"
echo "  输出  : $OUTPUT_DIR"
echo "  权重  : $WEIGHT"
echo "========================================"

python predict.py \
    --t1 "$T1" \
    --t1c "$T1C" \
    --tof "$TOF" \
    --output "$OUTPUT_DIR" \
    --weight "$WEIGHT" \
    --device "$DEVICE" \
    --norm_dataset "$NORM_DATASET" \
    --threshold "$THRESHOLD"

echo ""
echo "[DONE] 预测完成，结果保存在: $OUTPUT_DIR"
