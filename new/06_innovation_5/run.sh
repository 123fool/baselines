#!/bin/bash
# Innovation 5: 海马体区域注意力加权 - 一键执行脚本
# 使用方式: bash run.sh [step]
#   bash run.sh dashboard    # 仅启动监控面板
#   bash run.sh ae           # 仅训练 AutoEncoder
#   bash run.sh controlnet   # 仅训练 ControlNet  
#   bash run.sh evaluate     # 仅评估
#   bash run.sh all          # 全流程执行 (默认)

set -e

# ==============================
# 路径配置
# ==============================
BASE_DIR="/home/wangchong/data/fwz"
CODE_DIR="${BASE_DIR}/code/innovation_5"
BRLP_SRC="${BASE_DIR}/brlp-code/src"  # BrLP 源码
OUTPUT_DIR="${BASE_DIR}/output/innovation_5"
CACHE_DIR="${BASE_DIR}/cache/innovation_5"
CHANGELOG="${CODE_DIR}/changelog.json"

# 预训练模型
AE_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/autoencoder.pth"
UNET_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/latentdiffusion.pth"
CNET_PRETRAINED="${BASE_DIR}/brlp-train/pretrained/controlnet.pth"

# 数据
MCI_CSV="${BASE_DIR}/data/diagnosis_categorized/mci_brlp_innovation.csv"
MCI_CSV_B="${OUTPUT_DIR}/prepared/B_mci.csv"  # 配对格式，用于 ControlNet 训练

# Innovation 5 参数
ROI_WEIGHT=3.0
REGION_ALPHA=0.5

# ==============================
# 环境准备
# ==============================
echo "============================================"
echo "Innovation 5: 海马体区域注意力加权"
echo "============================================"
echo "时间: $(date)"
echo "BASE_DIR: ${BASE_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

# 激活 conda 环境
source activate fwz 2>/dev/null || conda activate fwz 2>/dev/null || echo "Warning: conda env 'fwz' not found"

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}/ae"
mkdir -p "${OUTPUT_DIR}/controlnet"
mkdir -p "${OUTPUT_DIR}/eval_baseline"
mkdir -p "${OUTPUT_DIR}/eval_innovation5"
mkdir -p "${OUTPUT_DIR}/prepared"
mkdir -p "${CACHE_DIR}"

# 设置 Python 路径
export PYTHONPATH="${BRLP_SRC}:${CODE_DIR}/src:${PYTHONPATH}"

STEP=${1:-all}

# ==============================
# Step 0.5: 数据准备 (CSV A → CSV B)
# ==============================
prepare_data() {
    echo ""
    echo "[Step 0.5] 准备配对数据 (CSV A → CSV B)..."
    if [ -f "${MCI_CSV_B}" ]; then
        echo "  CSV B already exists, skipping."
    else
        python "${CODE_DIR}/scripts/prepare_data.py" \
            --input_csv "${MCI_CSV}" \
            --output_dir "${OUTPUT_DIR}/prepared" \
            --verify
    fi
}

# ==============================
# Step 0: 启动监控面板 (后台)
# ==============================
start_dashboard() {
    echo ""
    echo "[Step 0] 启动监控面板..."
    pip install flask psutil -q 2>/dev/null || true

    # Use the single unified dashboard entry.
    DASHBOARD_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/dashboard/server_monitor.py"
    if [ ! -f "${DASHBOARD_SCRIPT}" ]; then
        echo "  Dashboard script not found, skipping dashboard startup."
        return 0
    fi

    nohup python "${DASHBOARD_SCRIPT}" \
        --port 8501 \
        --changelog "${CHANGELOG}" \
        --results_dir "${OUTPUT_DIR}" \
        > "${OUTPUT_DIR}/dashboard.log" 2>&1 &
    DASHBOARD_PID=$!
    echo "  Dashboard PID: ${DASHBOARD_PID}"
    echo "  访问地址: http://$(hostname -I | awk '{print $1}'):8501"
    echo "  日志: ${OUTPUT_DIR}/dashboard.log"
}

# ==============================
# Step 1: AE 微调训练
# ==============================
train_ae() {
    echo ""
    echo "[Step 1] AutoEncoder 微调训练 (区域加权损失)..."
    python "${CODE_DIR}/scripts/train_autoencoder_regional.py" \
        --dataset_csv "${MCI_CSV}" \
        --cache_dir "${CACHE_DIR}/ae" \
        --output_dir "${OUTPUT_DIR}/ae" \
        --aekl_ckpt "${AE_PRETRAINED}" \
        --n_epochs 3 \
        --lr 5e-5 \
        --max_batch_size 2 \
        --batch_size 16 \
        --roi_weight ${ROI_WEIGHT} \
        --region_alpha ${REGION_ALPHA} \
        --changelog "${CHANGELOG}"
    echo "  AE 训练完成!"
}

# ==============================
# Step 1.5: 提取潜空间表征
# ==============================
extract_latents() {
    echo ""
    echo "[Step 1.5] 提取潜空间表征 (latent extraction)..."
    
    # 使用微调后的 AE (如果存在)，否则用预训练
    AE_CKPT="${OUTPUT_DIR}/ae/autoencoder-ep-2.pth"
    if [ ! -f "${AE_CKPT}" ]; then
        AE_CKPT="${AE_PRETRAINED}"
        echo "  Note: 使用预训练 AE (微调版本不存在)"
    fi
    
    python "${BRLP_SRC}/brlp/../../../scripts/prepare/extract_latents.py" \
        --dataset_csv "${MCI_CSV}" \
        --aekl_ckpt "${AE_CKPT}" \
        2>&1 || {
        # 如果 BrLP 原始脚本不可用，使用内嵌提取
        echo "  Using inline latent extraction..."
        python -c "
import os, sys, numpy as np, pandas as pd, torch
from monai import transforms
sys.path.insert(0, '${BRLP_SRC}')
from brlp import init_autoencoder, const
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
autoencoder = init_autoencoder('${AE_CKPT}').to(DEVICE).eval()
transforms_fn = transforms.Compose([
    transforms.CopyItemsD(keys={'image_path'}, names=['image']),
    transforms.LoadImageD(image_only=True, keys=['image']),
    transforms.EnsureChannelFirstD(keys=['image']),
    transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
    transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
])
df = pd.read_csv('${MCI_CSV}')
with torch.no_grad():
    for i, row in df.iterrows():
        ip = row['image_path']
        dp = ip.replace('.nii.gz', '_latent.npz').replace('.nii', '_latent.npz')
        if os.path.exists(dp): continue
        try:
            t = transforms_fn({'image_path': ip})['image'].to(DEVICE)
            z, _ = autoencoder.encode(t.unsqueeze(0))
            np.savez_compressed(dp, data=z.cpu().squeeze(0).numpy())
            if i % 50 == 0: print(f'  Extracted {i}/{len(df)}')
        except Exception as e:
            print(f'  Skip {ip}: {e}')
print(f'  Done! Extracted latents for {len(df)} images.')
"
    }
    echo "  Latent 提取完成!"
}

# ==============================
# Step 2: ControlNet 训练
# ==============================
train_controlnet() {
    echo ""
    echo "[Step 2] ControlNet 训练 (潜空间区域加权)..."
    
    # 使用微调后的 AE (如果存在)，否则用预训练
    AE_CKPT="${OUTPUT_DIR}/ae/autoencoder-ep-2.pth"
    if [ ! -f "${AE_CKPT}" ]; then
        AE_CKPT="${AE_PRETRAINED}"
        echo "  Note: 使用预训练 AE (微调版本不存在)"
    fi
    
    python "${CODE_DIR}/scripts/train_controlnet_regional.py" \
        --dataset_csv "${MCI_CSV_B}" \
        --cache_dir "${CACHE_DIR}/controlnet" \
        --output_dir "${OUTPUT_DIR}/controlnet" \
        --aekl_ckpt "${AE_CKPT}" \
        --diff_ckpt "${UNET_PRETRAINED}" \
        --n_epochs 5 \
        --lr 2.5e-5 \
        --batch_size 8 \
        --roi_weight ${ROI_WEIGHT} \
        --region_alpha ${REGION_ALPHA} \
        --changelog "${CHANGELOG}"
    echo "  ControlNet 训练完成!"
}

# ==============================
# Step 3: 评估 (基线 + Innovation 5)
# ==============================
evaluate() {
    echo ""
    echo "[Step 3a] 评估基线模型..."
    python "${CODE_DIR}/scripts/evaluate_regional.py" \
        --dataset_csv "${MCI_CSV}" \
        --confs "${CODE_DIR}/configs/train_mci.yaml" \
        --output_dir "${OUTPUT_DIR}/eval_baseline" \
        --aekl_ckpt "${AE_PRETRAINED}" \
        --diff_ckpt "${UNET_PRETRAINED}" \
        --cnet_ckpt "${CNET_PRETRAINED}" \
        --las_m 5 \
        --max_samples 20 \
        --model_name "baseline" \
        --changelog "${CHANGELOG}"

    echo ""
    echo "[Step 3b] 评估 Innovation 5 模型..."
    
    AE_CKPT="${OUTPUT_DIR}/ae/autoencoder-ep-2.pth"
    CNET_CKPT="${OUTPUT_DIR}/controlnet/cnet-ep-4.pth"
    
    if [ ! -f "${AE_CKPT}" ]; then
        AE_CKPT="${AE_PRETRAINED}"
    fi
    if [ ! -f "${CNET_CKPT}" ]; then
        CNET_CKPT="${OUTPUT_DIR}/controlnet/cnet-ep-3.pth"
    fi
    
    python "${CODE_DIR}/scripts/evaluate_regional.py" \
        --dataset_csv "${MCI_CSV}" \
        --confs "${CODE_DIR}/configs/train_mci.yaml" \
        --output_dir "${OUTPUT_DIR}/eval_innovation5" \
        --aekl_ckpt "${AE_CKPT}" \
        --diff_ckpt "${UNET_PRETRAINED}" \
        --cnet_ckpt "${CNET_CKPT}" \
        --las_m 5 \
        --max_samples 20 \
        --model_name "innovation_5" \
        --changelog "${CHANGELOG}"
    
    echo ""
    echo "============================================"
    echo "评估完成! 结果:"
    echo "  基线:        ${OUTPUT_DIR}/eval_baseline/"
    echo "  Innovation5: ${OUTPUT_DIR}/eval_innovation5/"
    echo "============================================"
}

# ==============================
# 执行
# ==============================
case ${STEP} in
    dashboard)
        start_dashboard
        ;;
    ae)
        start_dashboard
        train_ae
        ;;
    controlnet)
        start_dashboard
        prepare_data
        extract_latents
        train_controlnet
        ;;
    evaluate)
        start_dashboard
        evaluate
        ;;
    all)
        start_dashboard
        prepare_data
        train_ae
        extract_latents
        train_controlnet
        evaluate
        echo ""
        echo "全流程执行完毕!"
        ;;
    *)
        echo "Usage: bash run.sh [dashboard|ae|controlnet|evaluate|all]"
        exit 1
        ;;
esac
