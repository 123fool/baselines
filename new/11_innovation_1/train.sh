#!/bin/bash
# Innovation 1: MCI 转化动态条件引导 - 训练脚本
# 使用方式:
#   bash train.sh prepare    # 数据预处理 (计算 atrophy rates)
#   bash train.sh train      # ControlNet 训练
#   bash train.sh all        # 全流程 (默认)

set -e

# ==============================
# 路径配置
# ==============================
export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/innovation_1"
BRLP_SRC="${BASE_DIR}/brlp-code/src"
OUTPUT_DIR="${BASE_DIR}/output/innovation_1"
CACHE_DIR="${BASE_DIR}/cache/innovation_1"

# 预训练模型 (baseline)
AE_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/autoencoder.pth"
UNET_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/latentdiffusion.pth"
CNET_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/controlnet.pth"

# 数据 (来自 Innovation 5 的准备好的 MCI 配对 CSV)
MCI_CSV_B="${BASE_DIR}/output/innovation_5/prepared/B_mci.csv"
# Innovation 1 增强版
MCI_CSV_INN1="${OUTPUT_DIR}/prepared/B_mci_inn1.csv"

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

STEP=${1:-all}

echo "============================================"
echo "Innovation 1: MCI 转化动态条件引导"
echo "============================================"
echo "时间: $(date)"
echo "设备: GPU ${CUDA_VISIBLE_DEVICES}"
echo "步骤: ${STEP}"

# 激活环境
source activate fwz 2>/dev/null || conda activate fwz 2>/dev/null || true

# 创建目录
mkdir -p "${OUTPUT_DIR}/prepared"
mkdir -p "${OUTPUT_DIR}/controlnet"
mkdir -p "${CACHE_DIR}"

# ==============================
# Step 1: 数据预处理
# ==============================
run_prepare() {
    echo ""
    echo ">>> Step 1: 数据预处理 - 计算萎缩/扩张速率"
    echo "    输入: ${MCI_CSV_B}"
    echo "    输出: ${MCI_CSV_INN1}"
    
    ${PYTHON} "${CODE_DIR}/scripts/prepare_mci_conditions.py" \
        --input_csv "${MCI_CSV_B}" \
        --output_csv "${MCI_CSV_INN1}"
    
    echo ">>> 数据预处理完成"
}

# ==============================
# Step 2: ControlNet 训练
# ==============================
run_train() {
    echo ""
    echo ">>> Step 2: ControlNet 训练 (6-channel conditioning)"
    echo "    数据: ${MCI_CSV_INN1}"
    echo "    AE:   ${AE_PRETRAINED}"
    echo "    UNet: ${UNET_PRETRAINED}"
    
    ${PYTHON} "${CODE_DIR}/scripts/train_controlnet_mci.py" \
        --dataset_csv "${MCI_CSV_INN1}" \
        --cache_dir   "${CACHE_DIR}" \
        --output_dir  "${OUTPUT_DIR}/controlnet" \
        --aekl_ckpt   "${AE_PRETRAINED}" \
        --diff_ckpt   "${UNET_PRETRAINED}" \
        --pretrained_cnet "${CNET_PRETRAINED}" \
        --n_epochs    5 \
        --batch_size  16 \
        --lr          2.5e-5
    
    echo ">>> ControlNet 训练完成"
}

# ==============================
# 执行
# ==============================
case "${STEP}" in
    prepare)
        run_prepare
        ;;
    train)
        run_train
        ;;
    all)
        run_prepare
        run_train
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: bash train.sh [prepare|train|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Innovation 1 完成: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================"
