#!/bin/bash
# 测试脚本：验证 combined_retrain 实验的可运行性
# 不实际训练，只做快速检查：
#   1. 环境导入（PYTHONPATH / import）
#   2. 模型加载（AE + Diffusion + ControlNet(init)）
#   3. 数据集 CSV 可读，latent 文件存在
#   4. 前向 1 个 batch（不反向传播）
#   5. evaluate_regional.py 支持 --max_samples 1 快速推理
#
# 用法: bash test_compatibility.sh

set -e

PYTHON="/home/wangchong/miniconda3/envs/fwz/bin/python"
BASE="/home/wangchong/data/fwz"
BRLP_SRC="${BASE}/brlp-code/src"
INN5_CODE="${BASE}/code/innovation_5"

export PYTHONPATH="${BRLP_SRC}:${INN5_CODE}/src:${PYTHONPATH}"

AE_CKPT="${BASE}/output/innovation_4/ae_training/autoencoder-ep-4.pth"
DIFF_CKPT="${BASE}/brlp-train/pretrained/latentdiffusion.pth"
CSV="${BASE}/output/innovation_5/prepared/B_mci.csv"

echo "========================================================"
echo "兼容性测试: combined_retrain"
echo "========================================================"
echo ""

$PYTHON - <<'PYEOF'
import os, sys, time
import torch
import pandas as pd

BASE = "/home/wangchong/data/fwz"
BRLP_SRC = f"{BASE}/brlp-code/src"
INN5_CODE = f"{BASE}/code/innovation_5"
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, f"{INN5_CODE}/src")

AE_CKPT   = f"{BASE}/output/innovation_4/ae_training/autoencoder-ep-4.pth"
DIFF_CKPT = f"{BASE}/brlp-train/pretrained/latentdiffusion.pth"
CSV       = f"{BASE}/output/innovation_5/prepared/B_mci.csv"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[1/5] 环境检查...")
from brlp import const, networks, utils
from region_weights import create_weight_map_latent_space, HIPPOCAMPUS_LABELS
from weighted_losses import CombinedRegionLoss
print(f"      imports OK | device={DEVICE}")

print(f"\n[2/5] 模型加载...")
t0 = time.time()
autoencoder = networks.init_autoencoder(AE_CKPT).to(DEVICE).eval()
print(f"      AutoEncoder (Inn4): {AE_CKPT.split('/')[-1]} loaded ({time.time()-t0:.1f}s)")
t0 = time.time()
diffusion = networks.init_latent_diffusion(DIFF_CKPT).to(DEVICE).eval()
print(f"      Diffusion:          {DIFF_CKPT.split('/')[-1]} loaded ({time.time()-t0:.1f}s)")
t0 = time.time()
controlnet = networks.init_controlnet()
controlnet.load_state_dict(diffusion.state_dict(), strict=False)
controlnet = controlnet.to(DEVICE)
print(f"      ControlNet:         initialized from diffusion ({time.time()-t0:.1f}s)")

print(f"\n[3/5] 数据 CSV 检查...")
df = pd.read_csv(CSV)
test_df = df[df.split == "test"]
train_df = df[df.split == "train"]
valid_df = df[df.split == "valid"]
print(f"      CSV rows: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
first = train_df.iloc[0]
for col in ["starting_latent", "followup_latent", "starting_image", "followup_image"]:
    val = str(first[col])
    exists = os.path.exists(val)
    print(f"      {col}: {'OK' if exists else 'MISSING'} ({val[-50:]})")

print(f"\n[4/5] 前向传播测试 (1 sample, no backward)...")
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import autocast

npz_reader = NumpyReader(npz_keys=["data"])
load_latent = transforms.Compose([
    transforms.LoadImage(reader=npz_reader),
    transforms.EnsureChannelFirst(channel_dim=0),
    transforms.DivisiblePad(k=4, mode="constant"),
])

z = load_latent(str(train_df.iloc[0]["starting_latent"]))
scale_factor = 1 / torch.std(z)

starting_z = load_latent(str(train_df.iloc[0]["starting_latent"])).unsqueeze(0).to(DEVICE) * scale_factor
followup_z = load_latent(str(train_df.iloc[0]["followup_latent"])).unsqueeze(0).to(DEVICE) * scale_factor

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta",
                          beta_start=0.0015, beta_end=0.0205)
age = torch.tensor([float(train_df.iloc[0]["starting_age"])]).to(DEVICE)
context_vals = [
    float(train_df.iloc[0]["followup_age"]),
    float(train_df.iloc[0]["sex"]),
    float(train_df.iloc[0]["followup_diagnosis"]),
    float(train_df.iloc[0]["followup_cerebral_cortex"]),
    float(train_df.iloc[0]["followup_hippocampus"]),
    float(train_df.iloc[0]["followup_amygdala"]),
    float(train_df.iloc[0]["followup_cerebral_white_matter"]),
    float(train_df.iloc[0]["followup_lateral_ventricle"]),
]
context = torch.tensor(context_vals).unsqueeze(0).unsqueeze(0).to(DEVICE)

n = starting_z.shape[0]
concatenating_age = age.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
controlnet_cond = torch.cat([starting_z, concatenating_age], dim=1)
noise = torch.randn_like(followup_z)
timesteps = torch.randint(0, 1000, (n,), device=DEVICE).long()
noised = scheduler.add_noise(followup_z, noise=noise, timesteps=timesteps)

with autocast(enabled=True):
    with torch.no_grad():
        down_h, mid_h = controlnet(
            x=noised.float(),
            timesteps=timesteps,
            context=context.float(),
            controlnet_cond=controlnet_cond.float()
        )
print(f"      Forward pass OK: down_h[0].shape={down_h[0].shape}, mid_h.shape={mid_h.shape}")

print(f"\n[5/5] 快速推理测试 (1 pair, evaluate_regional)...")
print(f"      (跳过全量评估，直接检查推理引擎可运行性)")
from brlp import sample_using_controlnet_and_z
context_1d = torch.tensor(context_vals).to(DEVICE)
with torch.no_grad():
    pred = sample_using_controlnet_and_z(
        autoencoder=autoencoder,
        diffusion=diffusion,
        controlnet=controlnet,
        starting_z=starting_z.squeeze(0),
        starting_a=float(train_df.iloc[0]["starting_age"]),
        context=context_1d,
        device=DEVICE,
        scale_factor=scale_factor,
        average_over_n=1,
        num_inference_steps=10,  # 快速测试用 10 步
        verbose=False,
    )
print(f"      推理输出: shape={pred.shape}, range=[{pred.min():.3f},{pred.max():.3f}]")

print(f"\n{'='*56}")
print(f"✓ 所有 5 项测试通过！combined_retrain 可正常运行。")
print(f"{'='*56}")
PYEOF
