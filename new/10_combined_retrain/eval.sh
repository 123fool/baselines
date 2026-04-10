#!/bin/bash
# 方案 A: 评估联合重训后的模型
#
# 使用 Inn4 AE + 重训后的 ControlNet (最佳 epoch)
# 与 Baseline / Inn4 v1 / Inn5 v2 在相同 50 对测试集上对比
#
# 用法: bash eval.sh
#        bash eval.sh 3    # 指定用 epoch3 的 ControlNet

set -e

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"
BASE="/home/wangchong/data/fwz"
BRLP_SRC="${BASE}/brlp-code/src"
INN5_CODE="${BASE}/code/innovation_5"

export PYTHONPATH="${BRLP_SRC}:${INN5_CODE}/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1   # 与训练保持同一 GPU
CNET_EPOCH="${1:-}"
CNET_DIR="${BASE}/output/combined_retrain/controlnet"

if [ -n "${CNET_EPOCH}" ]; then
    CNET_CKPT="${CNET_DIR}/cnet-ep-${CNET_EPOCH}.pth"
else
    # 自动选最后一个 epoch
    CNET_CKPT=$(ls -t "${CNET_DIR}/cnet-ep-*.pth" 2>/dev/null | head -1)
fi

if [ -z "${CNET_CKPT}" ] || [ ! -f "${CNET_CKPT}" ]; then
    echo "ERROR: 找不到 ControlNet checkpoint。请先运行 train.sh"
    ls "${CNET_DIR}/" 2>/dev/null || echo "(目录不存在)"
    exit 1
fi

AE_CKPT="${BASE}/output/innovation_4/ae_training/autoencoder-ep-4.pth"
DIFF_CKPT="${BASE}/brlp-train/pretrained/latentdiffusion.pth"
CSV="${BASE}/output/innovation_5/prepared/B_mci.csv"
OUTPUT="${BASE}/output/combined_retrain/eval"
LOG="${BASE}/output/combined_retrain/eval.log"

echo "========================================================"
echo "方案 A 评估: Inn4 AE + 重训 ControlNet"
echo "========================================================"
echo "时间:        $(date)"
echo "AE:          ${AE_CKPT}"
echo "ControlNet:  ${CNET_CKPT}"
echo "Diff:        ${DIFF_CKPT}"
echo "CSV:         ${CSV}"
echo "Output:      ${OUTPUT}"
echo ""

mkdir -p "${OUTPUT}"

$PYTHON "${INN5_CODE}/scripts/evaluate_regional.py" \
    --dataset_csv "${CSV}" \
    --output_dir  "${OUTPUT}" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${DIFF_CKPT}" \
    --cnet_ckpt   "${CNET_CKPT}" \
    --max_samples 50 \
    --model_name  combined_retrain \
    2>&1 | tee "${LOG}"

echo ""
echo "========================================================"
echo "评估完成: $(date)"
echo "结果: ${OUTPUT}"
echo "========================================================"
