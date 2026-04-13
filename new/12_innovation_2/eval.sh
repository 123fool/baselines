#!/bin/bash
# Innovation 2: 双向时间正则化 — 评估脚本
set -e

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/innovation_2"
OUTPUT_DIR="${BASE_DIR}/output/innovation_2"

AE_CKPT="${BASE_DIR}/output/innovation_5/ae/autoencoder-ep-2.pth"
UNET_CKPT="${BASE_DIR}/brlp-train/pretrained/latentdiffusion.pth"
MCI_CSV="${BASE_DIR}/output/innovation_5/prepared/B_mci.csv"

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

EPOCH=${1:-4}
CNET_CKPT="${OUTPUT_DIR}/controlnet/cnet-btr-ep-${EPOCH}.pth"

echo "============================================"
echo "Innovation 2 评估 - Epoch ${EPOCH}"
echo "============================================"
echo "ControlNet: ${CNET_CKPT}"
echo "AE: ${AE_CKPT}"

mkdir -p "${OUTPUT_DIR}/eval"

if [ ! -f "${CNET_CKPT}" ]; then
    echo "Error: Checkpoint not found: ${CNET_CKPT}"
    echo "Available: $(ls ${OUTPUT_DIR}/controlnet/cnet-btr-ep-*.pth 2>/dev/null || echo 'none')"
    exit 1
fi

${PYTHON} "${CODE_DIR}/scripts/evaluate_btr.py" \
    --dataset_csv "${MCI_CSV}" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${UNET_CKPT}" \
    --cnet_ckpt   "${CNET_CKPT}" \
    --output_dir  "${OUTPUT_DIR}/eval" \
    --max_pairs   50 \
    --model_name  "innovation_2_btr"

echo ""
echo "评估完成: $(date)"
echo "结果: ${OUTPUT_DIR}/eval/"
