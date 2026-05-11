# Author: Zhaojjin Fu
# Date: 2026-04-30
# CMENet Project - Prediction Script

#!/bin/bash
# =============================================================================
# CMENet 数据集批量预测脚本
# 用法: bash predict_dataset.sh <数据集目录> <命名模式> [输出目录] [权重路径]
#
# 命名模式说明:
#   image   : T1/xxx_image.nii   T1C/xxx_image.nii   TOF/xxx_image.nii   (caro 风格)
#   T1      : T1/xxx_T1.nii      T1C/xxx_T1C.nii     TOF/xxx_TOF.nii     (crop30 等)
#   direct  : T1/xxxT1.nii       T1C/xxxT1C.nii      TOF/xxxTOF.nii      (final 等)
#
# 示例:
#   bash predict_dataset.sh ./data/images image
#   bash predict_dataset.sh ./data/images T1 ./output
#   bash predict_dataset.sh ./data/images direct ./output ./my_weight.pth
#
# 注意: 数据集目录必须包含 T1/ T1C/ TOF/ 三个子目录
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 默认值
DATASET_DIR=""
NAME_PATTERN=""
OUTPUT_DIR="./batch_output"
WEIGHT="results/A compare_result/CMENet_NoK_caro_network4_NK_seg/checkpoints/best-epoch1319-loss0.8286.pth"
DEVICE="cuda:0"
NORM_DATASET="caro"
THRESHOLD="0.5"

# 解析参数
if [ $# -ge 1 ]; then
    DATASET_DIR="$1"
fi
if [ $# -ge 2 ]; then
    NAME_PATTERN="$2"
fi
if [ $# -ge 3 ]; then
    OUTPUT_DIR="$3"
fi
if [ $# -ge 4 ]; then
    WEIGHT="$4"
fi

# 如果必要参数未提供，交互式询问
if [ -z "$DATASET_DIR" ] || [ -z "$NAME_PATTERN" ]; then
    echo "========================================"
    echo "  CMENet 数据集批量预测"
    echo "========================================"
    echo ""
    echo " 数据集目录结构示例:"
    echo "   dataset_dir/"
    echo "     ├── T1/"
    echo "     │   ├── patient01_T1.nii"
    echo "     │   └── patient02_T1.nii"
    echo "     ├── T1C/"
    echo "     │   ├── patient01_T1C.nii"
    echo "     │   └── patient02_T1C.nii"
    echo "     └── TOF/"
    echo "         ├── patient01_TOF.nii"
    echo "         └── patient02_TOF.nii"
    echo ""
    echo " 支持三种命名模式 (--name_pattern):"
    echo "   image  : xxx_image.nii             (caro 数据集)"
    echo "   T1     : xxx_T1.nii / xxx_T1C.nii  (crop30/QZ/SZ 等)"
    echo "   direct : xxxT1.nii / xxxT1C.nii    (final/new_caro 等)"
    echo ""

    if [ -z "$DATASET_DIR" ]; then
        read -p "请输入数据集根目录 : " DATASET_DIR
    fi
    if [ -z "$NAME_PATTERN" ]; then
        read -p "请输入命名模式 (image / T1 / direct) : " NAME_PATTERN
    fi
    read -p "输出目录 (默认: $OUTPUT_DIR) : " input_output
    [ -n "$input_output" ] && OUTPUT_DIR="$input_output"
    read -p "权重文件路径 (默认: $WEIGHT) : " input_weight
    [ -n "$input_weight" ] && WEIGHT="$input_weight"
    echo ""
fi

echo "========================================"
echo "  开始批量预测"
echo "========================================"
echo "  数据集目录 : $DATASET_DIR"
echo "  命名模式   : $NAME_PATTERN"
echo "  输出目录   : $OUTPUT_DIR"
echo "  权重文件   : $WEIGHT"
echo "========================================"

python predict_dataset.py \
    --dataset_dir "$DATASET_DIR" \
    --name_pattern "$NAME_PATTERN" \
    --output "$OUTPUT_DIR" \
    --weight "$WEIGHT" \
    --device "$DEVICE" \
    --norm_dataset "$NORM_DATASET" \
    --threshold "$THRESHOLD"

echo ""
echo "[DONE] 批量预测完成，结果保存在: $OUTPUT_DIR"
