#!/bin/bash
# 优先级1: TPN 替换 Leaspy — 训练脚本
set -e

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/tpn"
OUTPUT_DIR="${BASE_DIR}/output/tpn"

# 数据 (与 Innovation 1/2/5 相同的 MCI CSV)
MCI_CSV="${BASE_DIR}/output/innovation_5/prepared/B_mci.csv"

# 如果 B_mci.csv 不存在，尝试完整 B.csv
if [ ! -f "${MCI_CSV}" ]; then
    MCI_CSV="${BASE_DIR}/brlp-data/B.csv"
fi

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

echo "============================================"
echo "优先级1: TPN (Temporal Progression Network)"
echo "============================================"
echo "时间: $(date)"
echo "CSV:  ${MCI_CSV}"
echo "输出: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

${PYTHON} "${CODE_DIR}/scripts/train_tpn.py" \
    --dataset_csv "${MCI_CSV}" \
    --output_dir  "${OUTPUT_DIR}" \
    --n_epochs    200 \
    --batch_size  64 \
    --lr          1e-3 \
    --hidden_dim  128 \
    --n_layers    3 \
    --dropout     0.1

echo ""
echo "TPN 训练完成: $(date)"
echo "输出: ${OUTPUT_DIR}/"
