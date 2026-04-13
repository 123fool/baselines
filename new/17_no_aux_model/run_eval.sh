#!/bin/bash
# 去辅助模型端到端验证
# 对比 GT/TPN/Skip/Linear 四种 context 来源
set -e

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/no_aux_model"
OUTPUT_DIR="${BASE_DIR}/output/no_aux_model"

AE_CKPT="${BASE_DIR}/output/innovation_5/ae/autoencoder-ep-2.pth"
UNET_CKPT="${BASE_DIR}/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT="${BASE_DIR}/output/innovation_2/controlnet/cnet-btr-ep-1.pth"
TPN_CKPT="${BASE_DIR}/output/tpn_v3b/tpn_best.pth"
MCI_CSV="${BASE_DIR}/output/innovation_5/prepared/B_mci.csv"

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

echo "============================================"
echo "去辅助模型端到端验证"
echo "============================================"
echo "AE:         ${AE_CKPT}"
echo "UNet:       ${UNET_CKPT}"
echo "ControlNet: ${CNET_CKPT}"
echo "TPN:        ${TPN_CKPT}"
echo "CSV:        ${MCI_CSV}"
echo "Output:     ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

# Check all files exist
for f in "${AE_CKPT}" "${UNET_CKPT}" "${CNET_CKPT}" "${TPN_CKPT}" "${MCI_CSV}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

${PYTHON} "${CODE_DIR}/evaluate_no_aux.py" \
    --dataset_csv "${MCI_CSV}" \
    --aekl_ckpt   "${AE_CKPT}" \
    --diff_ckpt   "${UNET_CKPT}" \
    --cnet_ckpt   "${CNET_CKPT}" \
    --tpn_ckpt    "${TPN_CKPT}" \
    --output_dir  "${OUTPUT_DIR}" \
    --max_pairs   50 \
    --methods     "GT,TPN,Skip,Linear" \
    2>&1 | tee "${OUTPUT_DIR}/eval_no_aux.log"

echo ""
echo "评估完成: $(date)"
echo "结果目录: ${OUTPUT_DIR}"
