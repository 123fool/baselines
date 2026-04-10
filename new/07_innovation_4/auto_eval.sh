#!/bin/bash
# Auto-evaluation script: monitors training and runs eval when done
export PYTHONPATH="/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH"
PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"
LOG="/home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log"
EVAL_LOG="/home/wangchong/data/fwz/output/innovation_4_v4/eval_v4.log"

echo "Waiting for training to complete..."
while true; do
    if grep -q "Training complete" "$LOG" 2>/dev/null; then
        echo "Training complete detected! Starting evaluation..."
        break
    fi
    sleep 30
done

# Determine best checkpoint
if [ -f "/home/wangchong/data/fwz/output/innovation_4_v4/ae_training/autoencoder-best.pth" ]; then
    AE_CKPT="/home/wangchong/data/fwz/output/innovation_4_v4/ae_training/autoencoder-best.pth"
else
    AE_CKPT="/home/wangchong/data/fwz/output/innovation_4_v4/ae_training/autoencoder-ep-9.pth"
fi

echo "Using checkpoint: $AE_CKPT"
echo "Starting evaluation at $(date)..."

$PYTHON /home/wangchong/data/fwz/code/innovation_4_v4/scripts/evaluate_innovation4.py \
    --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv \
    --output_dir /home/wangchong/data/fwz/output/innovation_4_v4/eval \
    --aekl_ckpt "$AE_CKPT" \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/controlnet.pth \
    --max_samples 50 \
    --model_name innovation_4_v4 \
2>&1 | tee "$EVAL_LOG"

echo ""
echo "Evaluation complete at $(date)!"
echo "Results in /home/wangchong/data/fwz/output/innovation_4_v4/eval/"
