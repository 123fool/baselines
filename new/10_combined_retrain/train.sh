#!/bin/bash
# 方案 A: 在 Inn4 AE 基础上重训 Inn5 ControlNet
#
# 与 Innovation 5 原训练的唯一区别:
#   --aekl_ckpt 改为 Inn4 微调后的 AE (ep-4)
#
# 用法: bash train.sh

set -e

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"
BASE="/home/wangchong/data/fwz"
BRLP_SRC="${BASE}/brlp-code/src"
INN5_CODE="${BASE}/code/innovation_5"

export PYTHONPATH="${BRLP_SRC}:${INN5_CODE}/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1   # GPU 1 (~18 GiB free); GPU 0 被 eye-env 占用

# ── Checkpoint 路径 ──────────────────────────────────────────────
# Inn4 微调后的 AE（仅 Decoder 更新，Encoder 冻结 → 潜空间不变）
AE_CKPT="${BASE}/output/innovation_4/ae_training/autoencoder-ep-4.pth"
# 基线 UNet（ControlNet 从此初始化权重）
DIFF_CKPT="${BASE}/brlp-train/pretrained/latentdiffusion.pth"

# ── 数据路径 ─────────────────────────────────────────────────────
CSV="${BASE}/output/innovation_5/prepared/B_mci.csv"
CACHE_DIR="${BASE}/cache/combined_retrain"
OUTPUT_DIR="${BASE}/output/combined_retrain"

# ── 训练超参（与 Inn5 v2 完全相同，只换 AE） ─────────────────────
N_EPOCHS=5
LR="2.5e-5"
BATCH_SIZE=8
ROI_WEIGHT=3.0
REGION_ALPHA=0.5

# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "方案 A: Inn4 AE + 重训 Inn5 ControlNet"
echo "========================================================"
echo "时间:        $(date)"
echo "AE ckpt:     ${AE_CKPT}"
echo "Diff ckpt:   ${DIFF_CKPT}"
echo "CSV:         ${CSV}"
echo "Output:      ${OUTPUT_DIR}/controlnet"
echo "Epochs:      ${N_EPOCHS} | LR: ${LR} | roi_weight: ${ROI_WEIGHT}"
echo ""

# 检查 checkpoint 是否存在
for f in "$AE_CKPT" "$DIFF_CKPT" "$CSV"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: 必要文件不存在: $f"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}/controlnet"
mkdir -p "${CACHE_DIR}"

echo "开始训练 ControlNet..."
echo ""

$PYTHON "${INN5_CODE}/scripts/train_controlnet_regional.py" \
    --dataset_csv "${CSV}" \
    --cache_dir   "${CACHE_DIR}" \
    --output_dir  "${OUTPUT_DIR}/controlnet" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${DIFF_CKPT}" \
    --n_epochs    "${N_EPOCHS}" \
    --lr          "${LR}" \
    --batch_size  "${BATCH_SIZE}" \
    --roi_weight  "${ROI_WEIGHT}" \
    --region_alpha "${REGION_ALPHA}"

echo ""
echo "========================================================"
echo "训练完成: $(date)"
echo "输出: ${OUTPUT_DIR}/controlnet/cnet-ep-*.pth"
echo "========================================================"
