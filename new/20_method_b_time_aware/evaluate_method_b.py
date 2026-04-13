#!/usr/bin/env python3
"""
Method B Evaluation: Time-Aware Context ControlNet
===================================================
Evaluates the Time-Aware Context ControlNet using the new 8-dim
temporal context vector (no brain volumes needed).

Usage on server:
  conda activate fwz
  cd /home/wangchong/data/fwz/code/brlp_src
  python -m scripts.evaluate_method_b \
    --dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv \
    --output_dir /home/wangchong/data/fwz/output/method_b_time_aware/eval \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt /home/wangchong/data/fwz/output/method_b_time_aware/controlnet/cnet-time-aware-best.pth
"""

import os
import sys
import json
import csv
import argparse
import warnings
import numpy as np
import torch
from torch.cuda.amp import autocast
from datetime import datetime

import pandas as pd
import nibabel as nib
from monai import transforms
from monai.data.image_reader import NumpyReader
from skimage.metrics import structural_similarity

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/wangchong/data/fwz/code/brlp_src")
from brlp import networks, utils, const
from brlp.sampling import sample_using_controlnet_and_z

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_FILE = None


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def compute_ssim_3d(img1, img2):
    return structural_similarity(img1, img2, data_range=img1.max() - img1.min())


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_mae(img1, img2):
    return float(np.mean(np.abs(img1 - img2)))


def compute_rmse(img1, img2):
    return float(np.sqrt(np.mean((img1 - img2) ** 2)))


def build_time_aware_context(row):
    """Build the 8-dim time-aware context vector from metadata only."""
    followup_age = float(row['followup_age'])
    starting_age = float(row['starting_age'])
    sex = float(row['sex'])
    followup_diag = float(row.get('followup_diagnosis', row.get('starting_diagnosis', 0.5)))
    starting_diag = float(row.get('starting_diagnosis', followup_diag))

    time_delta = followup_age - starting_age
    age_ratio = followup_age / max(starting_age, 50.0)
    norm_time_delta = time_delta / 10.0
    diag_change = followup_diag - starting_diag

    return torch.tensor([
        followup_age, sex, followup_diag,
        time_delta, age_ratio, starting_age, diag_change, norm_time_delta
    ]).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str)
    parser.add_argument('--num_pairs', default=50, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    global LOG_FILE
    LOG_FILE = os.path.join(args.output_dir, "eval_method_b.log")

    log("[EVAL-B] Starting Method B (Time-Aware Context) evaluation")

    # Load models
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion   = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    controlnet  = networks.init_controlnet(args.cnet_ckpt).to(DEVICE).eval()

    # Scale factor
    npz_reader = NumpyReader(npz_keys=['data'])
    load_tfm = transforms.Compose([
        transforms.LoadImage(reader=npz_reader),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])
    df = pd.read_csv(args.dataset_csv)
    test_df = df[df['split'] == 'test'].copy()
    first_z = load_tfm(test_df.iloc[0]['followup_latent'])
    scale_factor = 1.0 / torch.std(torch.tensor(first_z))
    log(f"[EVAL-B] Scale factor: {scale_factor:.4f}")

    # Select test pairs
    mci_test = test_df[test_df['starting_diagnosis'].isin([0.5, 1.0])].copy()
    if len(mci_test) > args.num_pairs:
        mci_test = mci_test.sample(n=args.num_pairs, random_state=42)
    log(f"[EVAL-B] Evaluating {len(mci_test)} MCI test pairs")

    resample_fn = transforms.Spacing(pixdim=1.5)
    all_ssim, all_psnr, all_mae, all_rmse = [], [], [], []

    csv_path = os.path.join(args.output_dir, "eval_method_b.csv")
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "pair_idx", "starting_age", "followup_age", "time_gap_months",
            "method", "ssim", "psnr", "mae", "rmse"
        ])

        for idx, (_, row) in enumerate(mci_test.iterrows()):
            try:
                starting_z_raw = load_tfm(row['starting_latent'])
                starting_z = torch.tensor(starting_z_raw).float() * scale_factor
                starting_a = float(row['starting_age'])

                # Build time-aware context (NO brain volumes needed!)
                context = build_time_aware_context(row)

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

                gt_img = nib.load(row['followup_image'])
                gt_np = gt_img.get_fdata()
                gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).float()
                gt_resampled = resample_fn(gt_tensor).squeeze(0).numpy()

                min_shape = tuple(min(a, b) for a, b in zip(pred_np.shape, gt_resampled.shape))
                pred_crop = pred_np[:min_shape[0], :min_shape[1], :min_shape[2]]
                gt_crop = gt_resampled[:min_shape[0], :min_shape[1], :min_shape[2]]

                pred_norm = (pred_crop - pred_crop.min()) / (pred_crop.max() - pred_crop.min() + 1e-8)
                gt_norm = (gt_crop - gt_crop.min()) / (gt_crop.max() - gt_crop.min() + 1e-8)

                ssim_val = compute_ssim_3d(gt_norm, pred_norm)
                psnr_val = compute_psnr(gt_norm, pred_norm)
                mae_val  = compute_mae(gt_norm, pred_norm)
                rmse_val = compute_rmse(gt_norm, pred_norm)

                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)
                all_mae.append(mae_val)
                all_rmse.append(rmse_val)

                time_gap = (float(row['followup_age']) - starting_a) * 12

                writer.writerow([
                    idx, f"{starting_a:.2f}", f"{float(row['followup_age']):.2f}",
                    f"{time_gap:.1f}", "TimeAware",
                    f"{ssim_val:.4f}", f"{psnr_val:.2f}",
                    f"{mae_val:.6f}", f"{rmse_val:.6f}"
                ])

                log(f"[EVAL-B] Pair {idx+1}/{len(mci_test)} | "
                    f"SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} MAE={mae_val:.6f}")

            except Exception as e:
                log(f"[EVAL-B] ERROR pair {idx}: {e}")
                continue

    if not all_ssim:
        log("[EVAL-B] No successful evaluations!")
        return

    summary = {
        "experiment": "method_b_time_aware_context",
        "description": "ControlNet trained with temporal context (no brain volumes)",
        "context_vector": "[followup_age, sex, diagnosis, time_delta, age_ratio, baseline_age, diag_change, norm_time]",
        "num_pairs": len(all_ssim),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "ssim": {"mean": float(np.mean(all_ssim)), "std": float(np.std(all_ssim))},
            "psnr": {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr))},
            "mae":  {"mean": float(np.mean(all_mae)),  "std": float(np.std(all_mae))},
            "rmse": {"mean": float(np.mean(all_rmse)), "std": float(np.std(all_rmse))},
        },
    }

    with open(os.path.join(args.output_dir, "summary_method_b.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n[EVAL-B] === SUMMARY ===")
    log(f"  SSIM:  {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    log(f"  PSNR:  {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
    log(f"  MAE:   {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    log(f"  RMSE:  {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}")


if __name__ == "__main__":
    main()
