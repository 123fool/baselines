#!/bin/bash
# Innovation 1: MCI 转化动态条件引导 - 评估脚本
# 使用方式: bash eval.sh [epoch]
#   bash eval.sh 4    # 评估 epoch 4 的 checkpoint

set -e

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/innovation_1"
OUTPUT_DIR="${BASE_DIR}/output/innovation_1"

AE_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/autoencoder.pth"
UNET_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/latentdiffusion.pth"
MCI_CSV_INN1="${OUTPUT_DIR}/prepared/B_mci_inn1.csv"

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

EPOCH=${1:-4}
CNET_CKPT="${OUTPUT_DIR}/controlnet/cnet-ep-${EPOCH}.pth"

echo "============================================"
echo "Innovation 1 评估 - Epoch ${EPOCH}"
echo "============================================"
echo "ControlNet: ${CNET_CKPT}"

source activate fwz 2>/dev/null || conda activate fwz 2>/dev/null || true

mkdir -p "${OUTPUT_DIR}/eval"

if [ ! -f "${CNET_CKPT}" ]; then
    echo "Error: Checkpoint not found: ${CNET_CKPT}"
    echo "Available: $(ls ${OUTPUT_DIR}/controlnet/cnet-ep-*.pth 2>/dev/null || echo 'none')"
    exit 1
fi

${PYTHON} "${CODE_DIR}/scripts/evaluate_mci.py" \
    --dataset_csv "${MCI_CSV_INN1}" \
    --aekl_ckpt   "${AE_PRETRAINED}" \
    --diff_ckpt   "${UNET_PRETRAINED}" \
    --cnet_ckpt   "${CNET_CKPT}" \
    --output_dir  "${OUTPUT_DIR}/eval" \
    --max_pairs   50

echo ""
echo "评估完成: $(date)"
echo "结果: ${OUTPUT_DIR}/eval/"
