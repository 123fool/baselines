#!/usr/bin/env python3
"""
Unified Evaluation Script for All Methods
==========================================
Evaluates any ControlNet checkpoint with configurable context mode.
Reports SSIM, PSNR, MAE, RMSE.

Usage on server:
  conda activate fwz
  cd /home/wangchong/data/fwz/code/brlp_src

  # Method B (time-aware context):
  python -m scripts.evaluate_all_methods \
    --dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv \
    --output_dir /home/wangchong/data/fwz/output/method_b_time_aware/eval \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt /home/wangchong/data/fwz/output/method_b_time_aware/controlnet/cnet-time-aware-best.pth \
    --context_mode time_aware --method_name "Method-B-TimeAware"

  # Method C (identity + time-aware):
  python -m scripts.evaluate_all_methods \
    --dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv \
    --output_dir /home/wangchong/data/fwz/output/method_c_identity/eval \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt /home/wangchong/data/fwz/output/method_c_identity/controlnet/cnet-identity-best.pth \
    --context_mode time_aware --method_name "Method-C-Identity"

  # Method D (frequency loss + time-aware):
  python -m scripts.evaluate_all_methods \
    --dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv \
    --output_dir /home/wangchong/data/fwz/output/method_d_freq/eval \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt /home/wangchong/data/fwz/output/method_d_freq/controlnet/cnet-freq-best.pth \
    --context_mode time_aware --method_name "Method-D-Frequency"
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


def build_context_original(row):
    """Original 8-dim context with GT brain volumes."""
    return torch.tensor([
        float(row['followup_age']),
        float(row['sex']),
        float(row.get('followup_diagnosis', row.get('starting_diagnosis', 0.5))),
        float(row.get('followup_cerebral_cortex', 0)),
        float(row.get('followup_hippocampus', 0)),
        float(row.get('followup_amygdala', 0)),
        float(row.get('followup_cerebral_white_matter', 0)),
        float(row.get('followup_lateral_ventricle', 0)),
    ]).float()


def build_context_linear(row):
    """Original 8-dim with linear-interpolated volumes (no-aux best)."""
    vol_cols = [
        'followup_cerebral_cortex', 'followup_hippocampus',
        'followup_amygdala', 'followup_cerebral_white_matter',
        'followup_lateral_ventricle'
    ]
    volumes = [float(row.get(c, 0)) for c in vol_cols]
    return torch.tensor([
        float(row['followup_age']),
        float(row['sex']),
        float(row.get('followup_diagnosis', row.get('starting_diagnosis', 0.5))),
    ] + volumes).float()


def build_context_time_aware(row):
    """Time-aware 8-dim context (no brain volumes)."""
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


CONTEXT_BUILDERS = {
    'original': build_context_original,
    'linear': build_context_linear,
    'time_aware': build_context_time_aware,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str)
    parser.add_argument('--context_mode', default='time_aware', type=str,
                        choices=['original', 'linear', 'time_aware'])
    parser.add_argument('--method_name', default='UnknownMethod', type=str)
    parser.add_argument('--num_pairs', default=50, type=int)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--average_over_n', default=1, type=int)
    args = parser.parse_args()

    # Lazy import to allow sys.path modification
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from brlp import networks, utils, const
    from brlp.sampling import sample_using_controlnet_and_z

    os.makedirs(args.output_dir, exist_ok=True)
    global LOG_FILE
    LOG_FILE = os.path.join(args.output_dir, f"eval_{args.method_name.lower().replace(' ', '_')}.log")

    log(f"[EVAL] Starting evaluation: {args.method_name}")
    log(f"[EVAL] Context mode: {args.context_mode}")
    log(f"[EVAL] Checkpoint: {args.cnet_ckpt}")

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
    log(f"[EVAL] Scale factor: {scale_factor:.4f}")

    # Select test pairs
    mci_test = test_df[test_df['starting_diagnosis'].isin([0.5, 1.0])].copy()
    if len(mci_test) > args.num_pairs:
        mci_test = mci_test.sample(n=args.num_pairs, random_state=42)
    log(f"[EVAL] Evaluating {len(mci_test)} MCI test pairs")

    context_builder = CONTEXT_BUILDERS[args.context_mode]
    resample_fn = transforms.Spacing(pixdim=1.5)

    all_ssim, all_psnr, all_mae, all_rmse = [], [], [], []

    csv_path = os.path.join(args.output_dir, f"eval_{args.method_name.lower()}.csv")
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
                context = context_builder(row)

                pred = sample_using_controlnet_and_z(
                    autoencoder=autoencoder,
                    diffusion=diffusion,
                    controlnet=controlnet,
                    starting_z=starting_z,
                    starting_a=starting_a,
                    context=context,
                    device=DEVICE,
                    scale_factor=scale_factor,
                    average_over_n=args.average_over_n,
                    num_inference_steps=args.num_inference_steps,
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
                    f"{time_gap:.1f}", args.method_name,
                    f"{ssim_val:.4f}", f"{psnr_val:.2f}",
                    f"{mae_val:.6f}", f"{rmse_val:.6f}"
                ])

                log(f"[EVAL] {args.method_name} Pair {idx+1}/{len(mci_test)} | "
                    f"SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}")

            except Exception as e:
                log(f"[EVAL] ERROR pair {idx}: {e}")
                continue

    if not all_ssim:
        log("[EVAL] No successful evaluations!")
        return

    summary = {
        "experiment": args.method_name,
        "context_mode": args.context_mode,
        "checkpoint": args.cnet_ckpt,
        "num_pairs": len(all_ssim),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "ssim": {"mean": float(np.mean(all_ssim)), "std": float(np.std(all_ssim)),
                     "min": float(np.min(all_ssim)), "max": float(np.max(all_ssim))},
            "psnr": {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr)),
                     "min": float(np.min(all_psnr)), "max": float(np.max(all_psnr))},
            "mae":  {"mean": float(np.mean(all_mae)),  "std": float(np.std(all_mae))},
            "rmse": {"mean": float(np.mean(all_rmse)), "std": float(np.std(all_rmse))},
        },
    }

    summary_path = os.path.join(args.output_dir, f"summary_{args.method_name.lower()}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n[EVAL] === {args.method_name} SUMMARY ===")
    log(f"  SSIM:  {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    log(f"  PSNR:  {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
    log(f"  MAE:   {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    log(f"  RMSE:  {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}")
    log(f"  Saved to {summary_path}")


if __name__ == "__main__":
    main()
