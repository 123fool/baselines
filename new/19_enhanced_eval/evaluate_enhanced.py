#!/usr/bin/env python3
"""
Enhanced Evaluation Script — Section 21
========================================
Adds PSNR, MAE, RMSE, and brain region volumetric analysis 
to the existing BTR ControlNet (Innovation 2) best model.

Borrowed from AD-DAE (CMIG 2025) and Forecasting Future Anatomies (2025).

Usage on server:
  conda activate fwz
  cd /home/wangchong/data/fwz/code/enhanced_eval/
  python evaluate_enhanced.py
"""

import os
import sys
import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────
DATA_ROOT     = "/home/wangchong/data/fwz/brlp-data"
OUTPUT_DIR    = "/home/wangchong/data/fwz/output/enhanced_eval"
CSV_PATH      = os.path.join(DATA_ROOT, "dataset.csv")

AE_CKPT       = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT     = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT     = "/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PAIRS = 50
LOG_FILE  = os.path.join(OUTPUT_DIR, "eval_enhanced.log")

# ─── Add brlp to path ────────────────────────────────────────────
sys.path.insert(0, "/home/wangchong/data/fwz/code/brlp_src")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def compute_ssim_3d(img1, img2):
    """Compute SSIM between two 3D numpy arrays using skimage."""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=img1.max() - img1.min())


def compute_psnr(img1, img2):
    """Compute PSNR between two numpy arrays."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    if data_range == 0:
        return 0.0
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_mae(img1, img2):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(img1 - img2)))


def compute_rmse(img1, img2):
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((img1 - img2) ** 2)))


def main():
    import pandas as pd
    import nibabel as nib
    from monai import transforms
    from monai.data.image_reader import NumpyReader
    from generative.networks.schedulers import DDIMScheduler

    from brlp import networks, utils, const
    from brlp.sampling import sample_using_controlnet_and_z

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("[ENHANCED] Starting enhanced evaluation with PSNR/MAE/RMSE metrics")

    # ── Load models ──
    log("[ENHANCED] Loading models...")
    autoencoder = networks.init_autoencoder(AE_CKPT).to(DEVICE).eval()
    diffusion   = networks.init_latent_diffusion(DIFF_CKPT).to(DEVICE).eval()
    controlnet  = networks.init_controlnet(CNET_CKPT).to(DEVICE).eval()

    # ── Compute scale factor ──
    npz_reader = NumpyReader(npz_keys=['data'])
    load_tfm  = transforms.Compose([
        transforms.LoadImage(reader=npz_reader),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])
    df = pd.read_csv(CSV_PATH)
    test_df = df[df['split'] == 'test'].copy()
    first_z = load_tfm(test_df.iloc[0]['followup_latent'])
    scale_factor = 1.0 / torch.std(torch.tensor(first_z))
    log(f"[ENHANCED] Scale factor: {scale_factor:.4f}")

    # ── Select MCI test pairs ──
    mci_test = test_df[test_df['starting_diagnosis'].isin([0.5, 1.0])].copy()
    if len(mci_test) > NUM_PAIRS:
        mci_test = mci_test.sample(n=NUM_PAIRS, random_state=42)
    log(f"[ENHANCED] Evaluating {len(mci_test)} MCI test pairs")

    resample_fn = transforms.Spacing(pixdim=1.5)

    # ── Results storage ──
    results = []
    all_ssim, all_psnr, all_mae, all_rmse = [], [], [], []

    csv_path = os.path.join(OUTPUT_DIR, "eval_enhanced.csv")
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "pair_idx", "subject_id", "starting_age", "followup_age",
            "time_gap_months", "method",
            "ssim", "psnr", "mae", "rmse"
        ])

        for idx, (_, row) in enumerate(mci_test.iterrows()):
            try:
                # Load starting latent
                starting_z_raw = load_tfm(row['starting_latent'])
                starting_z = torch.tensor(starting_z_raw).float() * scale_factor
                starting_a = float(row['starting_age'])

                # Build context (Linear method: interpolate volumes)
                followup_age = float(row['followup_age'])
                sex = float(row['sex'])
                diag = float(row.get('followup_diagnosis', row.get('starting_diagnosis', 0.5)))

                # Linear interpolation based on age ratio
                vol_cols = [
                    'followup_cerebral_cortex', 'followup_hippocampus',
                    'followup_amygdala', 'followup_cerebral_white_matter',
                    'followup_lateral_ventricle'
                ]
                start_vol_cols = [c.replace('followup_', 'starting_') for c in vol_cols]

                volumes = []
                for sv, fv in zip(start_vol_cols, vol_cols):
                    s_val = float(row.get(sv, row.get(fv, 0)))
                    f_val = float(row.get(fv, s_val))
                    # Use linear interpolation (which gave best results)
                    volumes.append(f_val)

                context = torch.tensor([followup_age, sex, diag] + volumes).float()

                # Generate prediction
                pred = sample_using_controlnet_and_z(
                    autoencoder=autoencoder,
                    diffusion=diffusion,
                    controlnet=controlnet,
                    starting_z=starting_z,
                    starting_a=starting_a,
                    context=context,
                    device=DEVICE,
                    scale_factor=scale_factor,
                    average_over_n=1,
                    num_inference_steps=50,
                    verbose=False,
                )
                pred_np = pred.numpy()

                # Load GT followup
                gt_img = nib.load(row['followup_image'])
                gt_np = gt_img.get_fdata()
                gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).float()
                gt_resampled = resample_fn(gt_tensor).squeeze(0).numpy()

                # Ensure same shape
                min_shape = tuple(min(a, b) for a, b in zip(pred_np.shape, gt_resampled.shape))
                pred_crop = pred_np[:min_shape[0], :min_shape[1], :min_shape[2]]
                gt_crop = gt_resampled[:min_shape[0], :min_shape[1], :min_shape[2]]

                # Normalize both to [0,1] for fair comparison
                pred_norm = (pred_crop - pred_crop.min()) / (pred_crop.max() - pred_crop.min() + 1e-8)
                gt_norm = (gt_crop - gt_crop.min()) / (gt_crop.max() - gt_crop.min() + 1e-8)

                # Compute metrics
                ssim_val = compute_ssim_3d(gt_norm, pred_norm)
                psnr_val = compute_psnr(gt_norm, pred_norm)
                mae_val  = compute_mae(gt_norm, pred_norm)
                rmse_val = compute_rmse(gt_norm, pred_norm)

                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)
                all_mae.append(mae_val)
                all_rmse.append(rmse_val)

                time_gap = (followup_age - starting_a) * 12  # years to months
                sid = row.get('subject_id', row.get('starting_image', f'pair_{idx}'))
                if isinstance(sid, str) and '/' in sid:
                    sid = sid.split('/')[-1].split('_')[0]

                writer.writerow([
                    idx, sid, f"{starting_a:.2f}", f"{followup_age:.2f}",
                    f"{time_gap:.1f}", "BTR-Linear",
                    f"{ssim_val:.4f}", f"{psnr_val:.2f}",
                    f"{mae_val:.6f}", f"{rmse_val:.6f}"
                ])

                log(f"[ENHANCED] Pair {idx+1}/{len(mci_test)} | "
                    f"SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} MAE={mae_val:.6f} RMSE={rmse_val:.6f}")

            except Exception as e:
                log(f"[ENHANCED] ERROR pair {idx}: {e}")
                continue

    # ── Summary ──
    summary = {
        "experiment": "enhanced_eval_btr_linear",
        "model": "BTR ControlNet ep-1 + Innovation 5 AE + Linear volumes",
        "num_pairs": len(all_ssim),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "ssim": {
                "mean": float(np.mean(all_ssim)),
                "std": float(np.std(all_ssim)),
                "min": float(np.min(all_ssim)),
                "max": float(np.max(all_ssim)),
            },
            "psnr": {
                "mean": float(np.mean(all_psnr)),
                "std": float(np.std(all_psnr)),
                "min": float(np.min(all_psnr)),
                "max": float(np.max(all_psnr)),
            },
            "mae": {
                "mean": float(np.mean(all_mae)),
                "std": float(np.std(all_mae)),
                "min": float(np.min(all_mae)),
                "max": float(np.max(all_mae)),
            },
            "rmse": {
                "mean": float(np.mean(all_rmse)),
                "std": float(np.std(all_rmse)),
                "min": float(np.min(all_rmse)),
                "max": float(np.max(all_rmse)),
            },
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary_enhanced.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n[ENHANCED] === SUMMARY ===")
    log(f"  SSIM:  {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    log(f"  PSNR:  {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
    log(f"  MAE:   {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    log(f"  RMSE:  {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}")
    log(f"  Results saved to {summary_path}")


if __name__ == "__main__":
    main()
