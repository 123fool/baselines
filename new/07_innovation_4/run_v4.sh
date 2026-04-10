#!/bin/bash
# Innovation 4 v4: Training + Evaluation pipeline
# Run on server: 10.96.27.109:2638, conda env: fwz

set -e

export PYTHONPATH="/home/wangchong/data/fwz/brlp-train/src:/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH"
PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"

# ========== Paths ==========
DATASET_CSV="/home/wangchong/data/fwz/output/innovation_5/prepared/A_mci.csv"
CACHE_DIR="/home/wangchong/data/fwz/cache/innovation_4_v4"
OUTPUT_DIR="/home/wangchong/data/fwz/output/innovation_4_v4/ae_training"
EVAL_OUTPUT_DIR="/home/wangchong/data/fwz/output/innovation_4_v4/eval"

# Pretrained checkpoints
AEKL_CKPT="/home/wangchong/data/fwz/brlp-train/pretrained/autoencoder.pth"
DIFF_CKPT="/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT="/home/wangchong/data/fwz/brlp-train/pretrained/controlnet.pth"
MEDNET_CKPT="/home/wangchong/data/fwz/code/innovation_4/pretrained/resnet_10_23dataset.pth"

# B_mci.csv for evaluation
EVAL_CSV="/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"

SCRIPT_DIR="/home/wangchong/data/fwz/code/innovation_4_v4/scripts"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$EVAL_OUTPUT_DIR"

echo "=============================================="
echo "Innovation 4 v4: Improved Decoder Fine-tuning"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# ========== Phase 1: Training ==========
echo "[Phase 1] Training autoencoder (decoder-only)..."
$PYTHON $SCRIPT_DIR/train_ae_v4.py \
    --dataset_csv "$DATASET_CSV" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --aekl_ckpt "$AEKL_CKPT" \
    --mednet_ckpt "$MEDNET_CKPT" \
    --n_epochs 10 \
    --max_batch_size 1 \
    --batch_size 16 \
    --lr 2e-5 \
    --perc3d_weight 0.0005 \
    --freq_weight 0.001 \
    --ssim_weight 0.5 \
    --l1_weight 1.5 \
    --warmup_start 3 \
    --warmup_end 6 \
    --latent_noise_std 0.01 \
    --latent_noise_prob 0.5

echo ""
echo "[Phase 1] Training complete!"
echo ""

# ========== Phase 2: Evaluation ==========
echo "[Phase 2] Evaluating with best checkpoint..."

# Use best checkpoint if available, otherwise use last epoch
if [ -f "$OUTPUT_DIR/autoencoder-best.pth" ]; then
    AE_EVAL_CKPT="$OUTPUT_DIR/autoencoder-best.pth"
    echo "  Using best checkpoint: $AE_EVAL_CKPT"
else
    AE_EVAL_CKPT="$OUTPUT_DIR/autoencoder-ep-9.pth"
    echo "  Using last epoch checkpoint: $AE_EVAL_CKPT"
fi

$PYTHON $SCRIPT_DIR/evaluate_innovation4.py \
    --dataset_csv "$EVAL_CSV" \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --aekl_ckpt "$AE_EVAL_CKPT" \
    --diff_ckpt "$DIFF_CKPT" \
    --cnet_ckpt "$CNET_CKPT" \
    --max_samples 50 \
    --model_name "innovation_4_v4"

echo ""
echo "=============================================="
echo "Innovation 4 v4: Pipeline complete!"
echo "End time: $(date)"
echo "=============================================="
