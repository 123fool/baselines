#!/bin/bash
# 优先级1: TPN 评估 — 对比 TPN vs Leaspy
set -e

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/tpn"
OUTPUT_DIR="${BASE_DIR}/output/tpn"

MCI_CSV="${BASE_DIR}/output/innovation_5/prepared/B_mci.csv"
if [ ! -f "${MCI_CSV}" ]; then
    MCI_CSV="${BASE_DIR}/brlp-data/B.csv"
fi

# Leaspy 模型路径 (如果已训练过)
LEASPY_DIR="${BASE_DIR}/brlp-train/pretrained"

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

echo "============================================"
echo "TPN 评估: TPN vs Leaspy"
echo "============================================"
echo "时间: $(date)"

${PYTHON} "${CODE_DIR}/scripts/evaluate_tpn.py" \
    --dataset_csv "${MCI_CSV}" \
    --tpn_ckpt    "${OUTPUT_DIR}/tpn_best.pth" \
    --output_dir  "${OUTPUT_DIR}" \
    --leaspy_dir  "${LEASPY_DIR}"

echo ""
echo "评估完成: $(date)"
