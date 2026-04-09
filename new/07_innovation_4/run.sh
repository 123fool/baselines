#!/bin/bash
# Innovation 4: 3D Perceptual Loss + Frequency Domain Constraint
# Run AE training on server
#
# Usage: bash run.sh [train|eval|all]

set -e

# ---- Paths ----
CODE_DIR=/home/wangchong/data/fwz/code/innovation_4
BRLP_SRC=/home/wangchong/data/fwz/brlp-code/src
OUTPUT_DIR=/home/wangchong/data/fwz/output/innovation_4
PRETRAINED=/home/wangchong/data/fwz/brlp-train/pretrained
DATASET_CSV=/home/wangchong/data/fwz/data/diagnosis_categorized/mci_brlp_innovation.csv
EVAL_CSV=/home/wangchong/data/fwz/data/diagnosis_categorized/B_mci.csv
CACHE_DIR=/home/wangchong/data/fwz/cache/innovation_4
MEDNET_CKPT=${CODE_DIR}/pretrained/resnet_10_23dataset.pth
CHANGELOG=${CODE_DIR}/changelog.json

export PYTHONPATH=${BRLP_SRC}:${CODE_DIR}/src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

mkdir -p ${OUTPUT_DIR}/ae_training
mkdir -p ${OUTPUT_DIR}/eval
mkdir -p ${CACHE_DIR}

MODE=${1:-train}

# ---- Step 1: AE Training ----
if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
    echo "============================================"
    echo "Innovation 4: AE Training with 3D Perceptual + Frequency"
    echo "============================================"
    
    python ${CODE_DIR}/scripts/train_autoencoder_3d_perceptual.py \
        --dataset_csv ${DATASET_CSV} \
        --cache_dir ${CACHE_DIR} \
        --output_dir ${OUTPUT_DIR}/ae_training \
        --aekl_ckpt ${PRETRAINED}/autoencoder.pth \
        --mednet_ckpt ${MEDNET_CKPT} \
        --n_epochs 5 \
        --max_batch_size 2 \
        --batch_size 16 \
        --lr 1e-4 \
        --perc_weight 0.001 \
        --freq_weight 0.01 \
        --fft_weight 0.0 \
        --lap_levels 3 \
        --changelog ${CHANGELOG}
    
    echo "AE Training complete."
fi

# ---- Step 2: Evaluation ----
if [ "$MODE" = "eval" ] || [ "$MODE" = "all" ]; then
    echo "============================================"
    echo "Innovation 4: Evaluation"
    echo "============================================"

    # Find latest AE checkpoint
    LATEST_AE=$(ls -t ${OUTPUT_DIR}/ae_training/autoencoder-ep-*.pth 2>/dev/null | head -1)
    if [ -z "$LATEST_AE" ]; then
        echo "No AE checkpoint found. Run training first."
        exit 1
    fi
    echo "Using AE checkpoint: ${LATEST_AE}"

    python ${CODE_DIR}/scripts/evaluate_innovation4.py \
        --dataset_csv ${EVAL_CSV} \
        --output_dir ${OUTPUT_DIR}/eval \
        --aekl_ckpt ${LATEST_AE} \
        --diff_ckpt ${PRETRAINED}/latentdiffusion.pth \
        --cnet_ckpt ${PRETRAINED}/controlnet.pth \
        --model_name innovation_4

    echo "Evaluation complete."
fi

echo "Done."
