#!/bin/bash
# 创新点 4+5 联合评估脚本
# 
# 原理：
#   创新点4：仅优化AE Decoder（Encoder冻结），不改变潜空间
#   创新点5：ControlNet在潜空间做区域加权噪声预测
#   由于Inn4冻结Encoder → 潜空间与基线完全一致
#   因此Inn5的ControlNet与Inn4的AE Decoder在推理阶段完全兼容
#
# 使用预训练checkpoint直接组合评估，无需重新训练
#
# 用法: bash run.sh

set -e

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"
BASE="/home/wangchong/data/fwz"
BRLP_SRC="${BASE}/brlp-code/src"
INN5_CODE="${BASE}/code/innovation_5"

export PYTHONPATH="${BRLP_SRC}:${INN5_CODE}/src:${PYTHONPATH}"

# ---- 模型 Checkpoint ----
# Inn4 v1: AE Decoder精调（最后epoch ep-4）
AE_CKPT="${BASE}/output/innovation_4/ae_training/autoencoder-ep-4.pth"
# 基线UNet扩散模型
DIFF_CKPT="${BASE}/brlp-train/pretrained/latentdiffusion.pth"
# Inn5 v2: ControlNet区域加权（ep-3最优）
CNET_CKPT="${BASE}/output/innovation_5/controlnet/cnet-ep-3.pth"

# ---- 数据 & 输出 ----
CSV="${BASE}/output/innovation_5/prepared/B_mci.csv"
OUTPUT="${BASE}/output/combined_4_5/eval"
LOG="${BASE}/output/combined_4_5/eval.log"

mkdir -p "${BASE}/output/combined_4_5"
mkdir -p "${OUTPUT}"

echo "========================================================"
echo "创新点 4+5 联合推理评估"
echo "========================================================"
echo "时间: $(date)"
echo "AE checkpoint  : ${AE_CKPT}"
echo "Diff checkpoint: ${DIFF_CKPT}"
echo "CNet checkpoint: ${CNET_CKPT}"
echo "CSV            : ${CSV}"
echo "Output dir     : ${OUTPUT}"
echo ""

# 检查 checkpoint 是否存在
for f in "$AE_CKPT" "$DIFF_CKPT" "$CNET_CKPT" "$CSV"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

echo "所有checkpoint已确认，开始评估..."
echo ""

$PYTHON "${INN5_CODE}/scripts/evaluate_regional.py" \
    --dataset_csv "$CSV" \
    --output_dir "$OUTPUT" \
    --aekl_ckpt "$AE_CKPT" \
    --diff_ckpt "$DIFF_CKPT" \
    --cnet_ckpt "$CNET_CKPT" \
    --max_samples 50 \
    --model_name combined_4_5 \
    2>&1 | tee "$LOG"

echo ""
echo "========================================================"
echo "评估完成: $(date)"
echo "结果目录: ${OUTPUT}"
echo "日志文件: ${LOG}"
echo "========================================================"
