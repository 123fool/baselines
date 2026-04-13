#!/bin/bash
# Priority 2: 残差潜码预测 (RLP) — 训练与评估脚本
# 包含三个实验：RLP-only, BTR+RLP, 评估
set -e

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/priority_2_rlp"
OUTPUT_DIR="${BASE_DIR}/output/priority_2_rlp"
CACHE_DIR="${BASE_DIR}/cache/priority_2_rlp"

# 模型路径（与 Innovation 2 相同）
AE_CKPT="${BASE_DIR}/output/innovation_5/ae/autoencoder-ep-2.pth"
UNET_CKPT="${BASE_DIR}/brlp-train/pretrained/latentdiffusion.pth"
CNET_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/controlnet.pth"

# 数据
MCI_CSV="${BASE_DIR}/output/innovation_5/prepared/B_mci.csv"

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

echo "============================================"
echo "Priority 2: 残差潜码预测 (RLP)"
echo "============================================"
echo "时间: $(date)"
echo "设备: GPU ${CUDA_VISIBLE_DEVICES}"
echo "AE: ${AE_CKPT}"

# ─── 实验 1: RLP Only ───
echo ""
echo "==== 实验 1: RLP Only ===="
mkdir -p "${OUTPUT_DIR}/rlp_only/controlnet"
mkdir -p "${CACHE_DIR}/rlp_only"

${PYTHON} "${CODE_DIR}/scripts/train_controlnet_rlp.py" \
    --dataset_csv "${MCI_CSV}" \
    --cache_dir   "${CACHE_DIR}/rlp_only" \
    --output_dir  "${OUTPUT_DIR}/rlp_only/controlnet" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${UNET_CKPT}" \
    --cnet_ckpt   "${CNET_PRETRAINED}" \
    --n_epochs    5 \
    --batch_size  8 \
    --lr          2.5e-5

echo "RLP Only 训练完成: $(date)"

# ─── 实验 2: BTR + RLP ───
echo ""
echo "==== 实验 2: BTR + RLP ===="
mkdir -p "${OUTPUT_DIR}/btr_rlp/controlnet"
mkdir -p "${CACHE_DIR}/btr_rlp"

${PYTHON} "${CODE_DIR}/scripts/train_controlnet_btr_rlp.py" \
    --dataset_csv "${MCI_CSV}" \
    --cache_dir   "${CACHE_DIR}/btr_rlp" \
    --output_dir  "${OUTPUT_DIR}/btr_rlp/controlnet" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${UNET_CKPT}" \
    --cnet_ckpt   "${CNET_PRETRAINED}" \
    --n_epochs    5 \
    --batch_size  8 \
    --lr          2.5e-5 \
    --btc_weight  0.5

echo "BTR + RLP 训练完成: $(date)"

# ─── 评估：RLP Only ───
echo ""
echo "==== 评估: RLP Only ===="
mkdir -p "${OUTPUT_DIR}/rlp_only/eval"

# 使用最后一个 epoch 的 checkpoint
RLP_CKPT="${OUTPUT_DIR}/rlp_only/controlnet/cnet-rlp-ep-4.pth"
if [ ! -f "$RLP_CKPT" ]; then
    RLP_CKPT="${OUTPUT_DIR}/rlp_only/controlnet/cnet-rlp-ep-3.pth"
fi

${PYTHON} "${CODE_DIR}/scripts/evaluate_rlp.py" \
    --dataset_csv "${MCI_CSV}" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${UNET_CKPT}" \
    --cnet_ckpt   "${RLP_CKPT}" \
    --output_dir  "${OUTPUT_DIR}/rlp_only/eval" \
    --model_name  "rlp_only" \
    --max_pairs   50

# ─── 评估：BTR + RLP ───
echo ""
echo "==== 评估: BTR + RLP ===="
mkdir -p "${OUTPUT_DIR}/btr_rlp/eval"

BTR_RLP_CKPT="${OUTPUT_DIR}/btr_rlp/controlnet/cnet-btr-rlp-ep-4.pth"
if [ ! -f "$BTR_RLP_CKPT" ]; then
    BTR_RLP_CKPT="${OUTPUT_DIR}/btr_rlp/controlnet/cnet-btr-rlp-ep-3.pth"
fi

${PYTHON} "${CODE_DIR}/scripts/evaluate_rlp.py" \
    --dataset_csv "${MCI_CSV}" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${UNET_CKPT}" \
    --cnet_ckpt   "${BTR_RLP_CKPT}" \
    --output_dir  "${OUTPUT_DIR}/btr_rlp/eval" \
    --model_name  "btr_rlp" \
    --max_pairs   50

echo ""
echo "============================================"
echo "所有实验完成: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================"
