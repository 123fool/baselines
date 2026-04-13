#!/bin/bash
# Innovation 2: 双向时间正则化 — 训练脚本
set -e

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/innovation_2"
OUTPUT_DIR="${BASE_DIR}/output/innovation_2"
CACHE_DIR="${BASE_DIR}/cache/innovation_2"

# 使用改进 AE (Innovation 4/5 的 AE)
AE_CKPT="${BASE_DIR}/output/innovation_5/ae/autoencoder-ep-2.pth"
UNET_CKPT="${BASE_DIR}/brlp-train/pretrained/latentdiffusion.pth"
CNET_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/controlnet.pth"

# 数据（与 Innovation 5 / 1 相同的 MCI CSV）
MCI_CSV="${BASE_DIR}/output/innovation_5/prepared/B_mci.csv"

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

echo "============================================"
echo "Innovation 2: 双向时间正则化 (BTR)"
echo "============================================"
echo "时间: $(date)"
echo "设备: GPU ${CUDA_VISIBLE_DEVICES}"
echo "AE: ${AE_CKPT}"
echo "λ_btc: 0.5"

mkdir -p "${OUTPUT_DIR}/controlnet"
mkdir -p "${CACHE_DIR}"

${PYTHON} "${CODE_DIR}/scripts/train_controlnet_btr.py" \
    --dataset_csv "${MCI_CSV}" \
    --cache_dir   "${CACHE_DIR}" \
    --output_dir  "${OUTPUT_DIR}/controlnet" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${UNET_CKPT}" \
    --cnet_ckpt   "${CNET_PRETRAINED}" \
    --n_epochs    5 \
    --batch_size  8 \
    --lr          2.5e-5 \
    --btc_weight  0.5

echo ""
echo "训练完成: $(date)"
echo "输出: ${OUTPUT_DIR}/controlnet/"
