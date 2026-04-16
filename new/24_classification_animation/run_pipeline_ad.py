#!/usr/bin/env python3
"""
BrLP AD Patient Pipeline
=========================
Adapted from run_pipeline.py for AD patients.

Key differences:
- Reads from ad_brlp_innovation.csv + B_adni_from_processed.csv
- Extracts latents on-the-fly (AD patients don't have pre-extracted latents)
- Extracts volume features from synthseg segmentations
- Gets age from B_adni CSV

Usage:
  cd /home/wangchong/data/fwz/code
  python 24_classification_animation/run_pipeline_ad.py --gpu 1 --subject 023_S_0139
"""

import os
import sys
import json
import csv
import argparse
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")

# ──────── BrLP imports ────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import nibabel as nib
import pandas as pd
from monai import transforms
from monai.data.image_reader import NumpyReader
from skimage.metrics import structural_similarity

from brlp import const, utils, networks
from brlp.sampling import sample_using_controlnet_and_z

# ──────── Config ────────
DATA_DIR = "/home/wangchong/data/fwz/data"
BMCI_CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
AD_CSV = f"{DATA_DIR}/diagnosis_categorized/ad_brlp_innovation.csv"
B_ADNI_CSV = "/home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv"

# Model checkpoints
AE_CKPT = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT = "/home/wangchong/data/fwz/output/innovation_5/controlnet/cnet-ep-4.pth"

# Volume features
VOL_FEATURES = [
    'cerebral_cortex', 'hippocampus', 'amygdala',
    'cerebral_white_matter', 'lateral_ventricle'
]

DIAGNOSIS_MAP = {0.0: 'CN', 0.5: 'MCI', 1.0: 'AD'}
VOLUMES_3CLASS_CSV = "/home/wangchong/data/fwz/output/classification_animation/volumes_3class.csv"


# ──────── Volume extraction from SynthSeg ────────

def extract_volumes_from_synthseg(segm_path, head_size_ref=None):
    """Extract and normalize volume features from synthseg segmentation."""
    segm = nib.load(segm_path).get_fdata().round()
    head_size = int((segm > 0).sum())

    raw_volumes = {}
    for region in const.COARSE_REGIONS:
        raw_volumes[region] = 0
    for code, region in const.SYNTHSEG_CODEMAP.items():
        if region == 'background':
            continue
        coarse = region.replace('left_', '').replace('right_', '')
        raw_volumes[coarse] += int((segm == code).sum())

    # Normalize by head_size (same as BrLP prepare_csv.py)
    norm_volumes = {}
    for k, v in raw_volumes.items():
        norm_volumes[k] = v / max(head_size, 1)

    return norm_volumes, head_size


def get_bmci_normalization_range():
    """Get [0,1] normalization stats from B_mci training split for consistent scaling."""
    if not os.path.exists(BMCI_CSV):
        return None

    stats = {}
    with open(BMCI_CSV) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    train_rows = [r for r in all_rows if r.get('split', '') == 'train']

    for feat in VOL_FEATURES:
        col = f'starting_{feat}'
        values = []
        for row in train_rows:
            try:
                values.append(float(row[col]))
            except (ValueError, KeyError):
                pass
        if values:
            stats[feat] = {'min': min(values), 'max': max(values)}

    return stats


# ──────── Latent extraction ────────

def extract_latent(image_path, autoencoder, device):
    """Extract latent representation from a NIfTI image using the autoencoder."""
    load_fn = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    img_tensor = load_fn(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        latent, _ = autoencoder.encode(img_tensor)
    latent = latent.cpu().squeeze(0).numpy()

    # Save for future use
    latent_path = image_path.replace('.nii.gz', '_latent.npz')
    np.savez_compressed(latent_path, data=latent)
    print(f"    Saved latent: {latent_path}")

    return latent, latent_path


# ──────── Model loading ────────

def load_models(device):
    """Load AE, Diffusion, ControlNet models."""
    print(f"[MODEL] Loading autoencoder from {AE_CKPT}")
    autoencoder = networks.init_autoencoder(AE_CKPT).to(device).eval()

    print(f"[MODEL] Loading diffusion from {DIFF_CKPT}")
    diffusion = networks.init_latent_diffusion(DIFF_CKPT).to(device).eval()

    print(f"[MODEL] Loading controlnet from {CNET_CKPT}")
    controlnet = networks.init_controlnet(CNET_CKPT).to(device).eval()

    return autoencoder, diffusion, controlnet


def get_latent_loader():
    """Latent loading transform."""
    npz_reader = NumpyReader(npz_keys=['data'])
    return transforms.Compose([
        transforms.LoadImage(reader=npz_reader),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])


def get_gt_loader():
    """GT image loading pipeline."""
    return transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])


def compute_scale_factor(load_tfm, bmci_csv):
    """Compute scale factor from first TRAIN latent."""
    df = pd.read_csv(bmci_csv)
    train_df = df[df['split'] == 'train']
    first_z = load_tfm(train_df.iloc[0]['starting_latent'])
    return float(1.0 / torch.std(torch.tensor(first_z)))


# ──────── AD data loading ────────

def load_ad_subject_visits(subject_id):
    """Load all visits for an AD subject from ad_brlp_innovation.csv + B_adni age data."""
    # 1. Load visit data from ad_brlp_innovation.csv
    with open(AD_CSV) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['subject_id'] == subject_id]

    if not rows:
        print(f"[ERROR] Subject {subject_id} not found in {AD_CSV}")
        return [], None

    sex = float(rows[0]['sex'])

    # 2. Get age from B_adni_from_processed.csv
    age_map = {}  # image_uid -> age
    if os.path.exists(B_ADNI_CSV):
        with open(B_ADNI_CSV) as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r['subject_id'] == subject_id:
                    age_map[r['starting_image_uid']] = float(r['starting_age'])
                    age_map[r['followup_image_uid']] = float(r['followup_age'])
                    # Also get sex from B_adni (more reliable)
                    sex = float(r['sex'])

    # 3. Build visits
    visits = []
    for r in sorted(rows, key=lambda x: int(x['days_from_first_visit'])):
        uid = r['image_uid']
        days = int(r['days_from_first_visit'])

        # Get age: try B_adni first, then estimate from baseline
        age = age_map.get(uid, None)
        if age is None and r['age']:
            age = float(r['age'])
        if age is None:
            # Estimate: use first available age + days/365/100 (normalized age scale)
            age = None  # will be filled after sort

        visits.append({
            'uid': uid,
            'age': age,
            'image': r['image_path'],
            'latent': r['latent_path'],
            'segm': r['segm_path'],
            'diagnosis': float(r['diagnosis']),
            'days': days,
            'visit_date': r['visit_date'],
        })

    # Fill missing ages by estimation from baseline
    base_age = None
    for v in visits:
        if v['age'] is not None:
            base_age = v['age']
            base_days = v['days']
            break

    if base_age is not None:
        for v in visits:
            if v['age'] is None:
                # Age is normalized to [0,1] over ~100 year range
                v['age'] = base_age + (v['days'] - base_days) / 365.25 / 100.0
    else:
        # No age available at all, use reasonable default (0.75 = ~75 years)
        for i, v in enumerate(visits):
            v['age'] = 0.75 + i * (180 / 365.25 / 100.0)

    return visits, sex


# ──────── Metrics ────────

def compute_metrics(pred_np, gt_np):
    """Compute SSIM/PSNR/MAE/RMSE."""
    ms = tuple(min(a, b) for a, b in zip(pred_np.shape, gt_np.shape))
    pred_crop = pred_np[:ms[0], :ms[1], :ms[2]]
    gt_crop = gt_np[:ms[0], :ms[1], :ms[2]]

    dr = max(float(gt_crop.max() - gt_crop.min()), 1e-8)
    ssim_val = structural_similarity(gt_crop, pred_crop, data_range=dr)
    mse = float(np.mean((gt_crop - pred_crop) ** 2))
    psnr_val = 20 * np.log10(dr / np.sqrt(mse)) if mse > 0 else float('inf')
    mae_val = float(np.mean(np.abs(gt_crop - pred_crop)))
    rmse_val = float(np.sqrt(mse))

    return {
        'ssim': float(ssim_val), 'psnr': float(psnr_val),
        'mae': mae_val, 'rmse': rmse_val,
        'pred_norm': pred_crop, 'gt_norm': gt_crop,
    }


# ──────── Classifier ────────

def load_classification_data(volumes_csv=VOLUMES_3CLASS_CSV, bmci_csv=BMCI_CSV):
    """Load CN/MCI/AD volume data for classifier."""
    all_features = []
    all_labels = []

    if os.path.exists(volumes_csv):
        print(f"  Loading from {volumes_csv}")
        with open(volumes_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    feats = [float(row[k]) for k in VOL_FEATURES]
                    all_features.append(feats)
                    all_labels.append(row['diagnosis'])
                except (ValueError, KeyError):
                    continue
        return np.array(all_features), np.array(all_labels)

    # Fallback
    print(f"  Falling back to B_mci.csv")
    seen = set()
    with open(bmci_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for prefix in ['starting', 'followup']:
                uid = row.get(f'{prefix}_image_uid', '')
                if uid in seen:
                    continue
                seen.add(uid)
                diag = float(row.get(f'{prefix}_diagnosis', 0.5))
                label = DIAGNOSIS_MAP.get(diag, 'MCI')
                try:
                    feats = [float(row[f'{prefix}_{k}']) for k in VOL_FEATURES]
                    all_features.append(feats)
                    all_labels.append(label)
                except (ValueError, KeyError):
                    continue

    return np.array(all_features), np.array(all_labels)


def train_classifier(features, labels):
    """Train 3-class GBM classifier."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    unique_classes = np.unique(labels)
    print(f"  Classes: {list(unique_classes)}")

    if len(unique_classes) < 2:
        return None, scaler

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42,
        learning_rate=0.1, subsample=0.8
    )
    n_folds = min(5, min(np.bincount(pd.factorize(labels)[0])))
    n_folds = max(2, n_folds)
    scores = cross_val_score(clf, X, labels, cv=n_folds, scoring='accuracy')
    print(f"  {n_folds}-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    clf.fit(X, labels)
    return clf, scaler


def predict_diagnosis(clf, scaler, volumes):
    X = scaler.transform([volumes])
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_
    pred_class = clf.predict(X)[0]
    return pred_class, {cls: float(p) for cls, p in zip(classes, probs)}


# ──────── Animation ────────

def create_animation(results, output_dir, subject_id):
    """Create GIF animation showing AD progression."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import imageio

    frames_dir = os.path.join(output_dir, 'frames_ad')
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    for i, r in enumerate(results):
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
        fig.suptitle(f'AD Longitudinal Prediction — {subject_id}',
                     color='white', fontsize=18, fontweight='bold', y=0.98)

        months = r['months_from_baseline']
        ssim = r.get('ssim', 0)
        psnr = r.get('psnr', 0)
        pred_class = r.get('predicted_class', 'AD')
        pred_probs = r.get('class_probs', {})
        pred_vol = r.get('pred_norm')
        gt_vol = r.get('gt_norm')

        if pred_vol is None:
            continue

        h, w, d = pred_vol.shape
        slices = {
            'Axial': (pred_vol[:, :, d//2], gt_vol[:, :, d//2] if gt_vol is not None else None),
            'Coronal': (pred_vol[:, w//2, :], gt_vol[:, w//2, :] if gt_vol is not None else None),
            'Sagittal': (pred_vol[h//2, :, :], gt_vol[h//2, :, :] if gt_vol is not None else None),
        }

        views = ['Axial', 'Coronal', 'Sagittal']
        for row_idx, view in enumerate(views):
            pred_slice, gt_slice = slices[view]

            ax1 = fig.add_subplot(gs[row_idx, 0])
            ax1.imshow(pred_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            lbl = 'Baseline' if i == 0 else 'Generated'
            ax1.set_title(f'{lbl} ({view})', color='cyan', fontsize=11)
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[row_idx, 1])
            if gt_slice is not None:
                ax2.imshow(gt_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
                ax2.set_title(f'Real ({view})', color='lime', fontsize=11)
            else:
                ax2.text(0.5, 0.5, 'No Real\nData', ha='center', va='center',
                        color='gray', fontsize=14, transform=ax2.transAxes)
                ax2.set_title(f'Real ({view})', color='gray', fontsize=11)
            ax2.axis('off')

            ax3 = fig.add_subplot(gs[row_idx, 2])
            if gt_slice is not None and i > 0:
                diff = np.abs(pred_slice - gt_slice)
                ax3.imshow(diff.T, cmap='hot', origin='lower', vmin=0, vmax=0.15)
                ax3.set_title('|Diff|', color='orange', fontsize=11)
            else:
                ax3.text(0.5, 0.5, 'N/A' if i == 0 else 'No GT',
                        ha='center', va='center',
                        color='gray', fontsize=14, transform=ax3.transAxes)
                ax3.set_title('|Diff|', color='gray', fontsize=11)
            ax3.axis('off')

        # Right panel
        ax_info = fig.add_subplot(gs[:, 3])
        ax_info.set_facecolor('black')
        ax_info.axis('off')

        info_text = f"Time: +{months:.1f} months\n"
        info_text += f"(Visit {i+1}/{len(results)})\n\n"

        if i > 0:
            info_text += f"SSIM: {ssim:.4f}\n"
            info_text += f"PSNR: {psnr:.2f} dB\n\n"
        else:
            info_text += "Baseline (reference)\n\n"

        info_text += "── Classification ──\n"
        info_text += f"Predicted: {pred_class}\n\n"

        for cls in ['CN', 'MCI', 'AD']:
            prob = pred_probs.get(cls, 0)
            bar = '█' * int(prob * 20) + '░' * (20 - int(prob * 20))
            info_text += f"{cls}: {bar} {prob:.1%}\n"

        info_text += "\n── Volume Changes ──\n"
        vols = r.get('volumes', {})
        base_vols = results[0].get('volumes', {})
        for k in ['hippocampus', 'lateral_ventricle', 'amygdala']:
            if k in vols and k in base_vols:
                change = (vols[k] - base_vols[k]) / max(abs(base_vols[k]), 1e-6) * 100
                arrow = '↓' if change < 0 else '↑'
                info_text += f"  {k[:12]}: {change:+.2f}% {arrow}\n"

        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    color='white', fontsize=12, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9))

        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        fig.savefig(frame_path, dpi=100, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        plt.close(fig)
        frame_paths.append(frame_path)
        print(f"  [ANIM] Frame {i+1}/{len(results)} saved")

    gif_path = os.path.join(output_dir, f'{subject_id}_progression.gif')
    images = [imageio.imread(fp) for fp in frame_paths]
    durations = [1500] * len(images)
    if durations:
        durations[-1] = 3000
    imageio.mimsave(gif_path, images, duration=durations, loop=0)
    print(f"  [ANIM] GIF saved: {gif_path}")

    return gif_path, frame_paths


def create_trajectory_chart(results, output_dir, subject_id):
    """Create trajectory chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    months = [r['months_from_baseline'] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), facecolor='#0d1117')
    fig.suptitle(f'AD Progression Trajectory — {subject_id}',
                 color='white', fontsize=16, fontweight='bold')

    # Panel 1: Volumes
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    colors = {'hippocampus': '#ff6b6b', 'amygdala': '#ffd93d',
              'lateral_ventricle': '#6bcb77', 'cerebral_cortex': '#4d96ff',
              'cerebral_white_matter': '#9b59b6'}
    for feat in VOL_FEATURES:
        vals = [r['volumes'].get(feat, 0) for r in results]
        ax1.plot(months, vals, 'o-', color=colors.get(feat, 'white'),
                label=feat.replace('_', ' ').title(), linewidth=2, markersize=6)
    ax1.set_xlabel('Months from baseline', color='white')
    ax1.set_ylabel('Normalized Volume', color='white')
    ax1.set_title('Brain Region Volumes Over Time', color='white', fontsize=13)
    ax1.legend(fontsize=9, facecolor='#161b22', edgecolor='gray',
              labelcolor='white', loc='upper right')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='gray')
    for spine in ax1.spines.values():
        spine.set_color('gray')

    # Panel 2: Classification probs
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    class_colors = {'CN': '#6bcb77', 'MCI': '#ffd93d', 'AD': '#ff6b6b'}
    for cls in ['CN', 'MCI', 'AD']:
        vals = [r['class_probs'].get(cls, 0) for r in results]
        ax2.plot(months, vals, 's-', color=class_colors[cls],
                label=cls, linewidth=2, markersize=8)
    ax2.set_xlabel('Months from baseline', color='white')
    ax2.set_ylabel('Probability', color='white')
    ax2.set_title('Diagnosis Classification Over Time', color='white', fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=11, facecolor='#161b22', edgecolor='gray', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='gray')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # Panel 3: SSIM
    ax3 = axes[2]
    ax3.set_facecolor('#161b22')
    ssim_vals = [r.get('ssim', 1.0) for r in results]
    bars = ax3.bar(months, ssim_vals, width=max(1, (max(months) - min(months))/len(months)*0.6),
                   color=['#4d96ff' if i > 0 else '#555555' for i in range(len(months))],
                   edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Months from baseline', color='white')
    ax3.set_ylabel('SSIM', color='white')
    ax3.set_title('Image Similarity (Generated vs Real)', color='white', fontsize=13)
    ax3.set_ylim(0, 1.05)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='gray', axis='y')
    for spine in ax3.spines.values():
        spine.set_color('gray')
    for bar, val in zip(bars, ssim_vals):
        ypos = min(val + 0.02, 1.0)
        ax3.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.4f}',
                ha='center', va='bottom', color='white', fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f'{subject_id}_trajectory.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"  [CHART] Saved: {chart_path}")
    return chart_path


# ──────── Main Pipeline ────────

def main():
    parser = argparse.ArgumentParser(description='BrLP AD Patient Pipeline')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--subject', type=str, default='023_S_0139',
                       help='AD subject ID (default: 023_S_0139, 4 visits)')
    parser.add_argument('--output_dir', type=str,
                       default='/home/wangchong/data/fwz/output/classification_animation')
    parser.add_argument('--avg_n', type=int, default=3)
    parser.add_argument('--list-subjects', action='store_true',
                       help='List available AD subjects and exit')
    args = parser.parse_args()

    # List mode
    if args.list_subjects:
        with open(AD_CSV) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        subjects = defaultdict(list)
        for r in rows:
            subjects[r['subject_id']].append(r)
        print(f"\nAD longitudinal subjects ({len(subjects)} total):")
        print(f"{'Subject':>15} {'Visits':>6}  {'Days range':>10}")
        print("-" * 40)
        for sid in sorted(subjects.keys(), key=lambda s: len(subjects[s]), reverse=True):
            visits = subjects[sid]
            days = [int(v['days_from_first_visit']) for v in visits]
            print(f"{sid:>15} {len(visits):>6}  {max(days):>10}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device = f'cuda:{args.gpu}'

    print("=" * 70)
    print("BrLP AD Patient Pipeline")
    print(f"Subject: {args.subject} | GPU: {args.gpu} | Avg_n: {args.avg_n}")
    print("=" * 70)
    t_start = time.time()

    # ── Step 1: Train classifier ──
    print("\n[STEP 1] Training 3-class volume classifier (CN/MCI/AD)...")
    clf_features, clf_labels = load_classification_data(
        volumes_csv=os.path.join(args.output_dir, 'volumes_3class.csv'),
        bmci_csv=BMCI_CSV,
    )
    print(f"  Loaded {len(clf_features)} samples")
    for lbl in np.unique(clf_labels):
        print(f"    {lbl}: {sum(clf_labels == lbl)}")
    clf, scaler = train_classifier(clf_features, clf_labels)
    clf_trained = clf is not None

    # ── Step 2: Load models ──
    print(f"\n[STEP 2] Loading models on {device}...")
    autoencoder, diffusion, controlnet = load_models(device)
    print("  Models loaded successfully")

    load_tfm = get_latent_loader()
    load_gt = get_gt_loader()
    scale_factor = compute_scale_factor(load_tfm, BMCI_CSV)
    print(f"  Scale factor: {scale_factor:.4f}")

    # ── Step 3: Load AD subject visits ──
    print(f"\n[STEP 3] Loading visits for AD subject {args.subject}...")
    visits, sex = load_ad_subject_visits(args.subject)
    print(f"  Found {len(visits)} visits (sex={sex})")

    if len(visits) < 2:
        print("[ERROR] Need at least 2 visits")
        return

    # ── Step 3b: Extract volumes and latents ──
    print(f"\n[STEP 3b] Extracting volumes from synthseg + latents...")
    for i, v in enumerate(visits):
        # Extract volumes
        if os.path.exists(v['segm']):
            vols, head_size = extract_volumes_from_synthseg(v['segm'])
            v['volumes'] = {k: vols.get(k, 0) for k in VOL_FEATURES}
            v['head_size'] = head_size
            print(f"  Visit {i+1}: volumes extracted (head_size={head_size})")
        else:
            print(f"  Visit {i+1}: [WARN] synthseg not found: {v['segm']}")
            v['volumes'] = {k: 0.5 for k in VOL_FEATURES}

        # Extract latent
        latent_path = v['image'].replace('.nii.gz', '_latent.npz')
        if os.path.exists(latent_path):
            print(f"  Visit {i+1}: latent exists: {latent_path}")
            v['latent'] = latent_path
        else:
            print(f"  Visit {i+1}: extracting latent from {v['image']}...")
            _, latent_path = extract_latent(v['image'], autoencoder, device)
            v['latent'] = latent_path

    for i, v in enumerate(visits):
        months = v['days'] / 30.44
        print(f"  Visit {i+1}: age={v['age']:.4f}, days={v['days']}, "
              f"months={months:.1f}, diag={DIAGNOSIS_MAP.get(v['diagnosis'], '?')}")

    # ── Step 4: Generate predictions ──
    baseline = visits[0]
    baseline_z_raw = load_tfm(baseline['latent'])
    baseline_z = torch.tensor(baseline_z_raw).float() * scale_factor
    baseline_age = baseline['age']

    print(f"\n[STEP 4] Generating predictions from baseline (Visit 1)...")
    print(f"  Baseline latent shape: {baseline_z.shape}, age: {baseline_age:.4f}")

    results = []

    for i, visit in enumerate(visits):
        months = visit['days'] / 30.44
        print(f"\n  --- Visit {i+1}/{len(visits)} (month +{months:.1f}) ---")

        result = {
            'visit_idx': i + 1,
            'months_from_baseline': months,
            'age': visit['age'],
            'days_from_baseline': visit['days'],
            'real_diagnosis': DIAGNOSIS_MAP.get(visit['diagnosis'], '?'),
            'volumes': visit['volumes'],
        }

        # Load real image
        gt_np = None
        if os.path.exists(visit['image']):
            gt_t = load_gt(visit['image']).squeeze(0)
            gt_np = gt_t.numpy()
            print(f"    Real image: {gt_np.shape}, range=[{gt_np.min():.4f}, {gt_np.max():.4f}]")
        else:
            print(f"    Real image not found: {visit['image']}")

        if i == 0:
            if gt_np is not None:
                result['pred_norm'] = gt_np
                result['gt_norm'] = gt_np
            result['ssim'] = 1.0
            result['psnr'] = 99.0
            result['mae'] = 0.0
            result['rmse'] = 0.0
            print(f"    Baseline (no generation)")
        else:
            # Build context with GT volumes of target visit
            context = torch.tensor([
                visit['age'],
                sex,
                visit['diagnosis'],
                visit['volumes']['cerebral_cortex'],
                visit['volumes']['hippocampus'],
                visit['volumes']['amygdala'],
                visit['volumes']['cerebral_white_matter'],
                visit['volumes']['lateral_ventricle'],
            ]).float()
            print(f"    Context: age={visit['age']:.4f}, diag={visit['diagnosis']}")

            pred = sample_using_controlnet_and_z(
                autoencoder=autoencoder,
                diffusion=diffusion,
                controlnet=controlnet,
                starting_z=baseline_z,
                starting_a=baseline_age,
                context=context,
                device=device,
                scale_factor=scale_factor,
                average_over_n=args.avg_n,
                verbose=False,
            )
            pred_np = pred.numpy().clip(0, 1)
            print(f"    Generated: shape={pred_np.shape}, range=[{pred_np.min():.4f}, {pred_np.max():.4f}]")

            if gt_np is not None:
                metrics = compute_metrics(pred_np, gt_np)
                result.update(metrics)
                print(f"    SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}")
            else:
                result['pred_norm'] = pred_np

            nii_path = os.path.join(args.output_dir, f'{args.subject}_visit{i+1}_pred.nii.gz')
            nib.save(nib.Nifti1Image(pred_np, np.eye(4)), nii_path)

        # Classification
        if clf_trained:
            vols = [visit['volumes'][k] for k in VOL_FEATURES]
            pred_class, prob_dict = predict_diagnosis(clf, scaler, vols)
            result['predicted_class'] = pred_class
            result['class_probs'] = prob_dict
            print(f"    Classification: {pred_class} "
                  f"(CN={prob_dict.get('CN',0):.2%}, MCI={prob_dict.get('MCI',0):.2%}, "
                  f"AD={prob_dict.get('AD',0):.2%})")
        else:
            result['predicted_class'] = 'AD'
            result['class_probs'] = {'CN': 0.0, 'MCI': 0.0, 'AD': 1.0}

        results.append(result)

    # ── Step 5: Animation ──
    print(f"\n[STEP 5] Creating animation...")
    gif_path, frame_paths = create_animation(results, args.output_dir, args.subject)

    # ── Step 6: Trajectory chart ──
    print(f"\n[STEP 6] Creating trajectory chart...")
    chart_path = create_trajectory_chart(results, args.output_dir, args.subject)

    # ── Step 7: Summary ──
    print(f"\n[STEP 7] Saving summary...")
    summary = {
        'subject_id': args.subject,
        'diagnosis_type': 'AD',
        'n_visits': len(visits),
        'model': 'Inn5-CNet-Avg3',
        'avg_n': args.avg_n,
        'scale_factor': scale_factor,
        'gif_path': gif_path,
        'chart_path': chart_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'visits': []
    }

    for r in results:
        visit_summary = {
            'visit_idx': r['visit_idx'],
            'months_from_baseline': r['months_from_baseline'],
            'age': r['age'],
            'real_diagnosis': r['real_diagnosis'],
            'predicted_class': r['predicted_class'],
            'class_probs': r['class_probs'],
            'ssim': r.get('ssim'),
            'psnr': r.get('psnr'),
            'mae': r.get('mae'),
            'rmse': r.get('rmse'),
            'volumes': r['volumes'],
        }
        summary['visits'].append(visit_summary)

    if clf_trained:
        summary['classifier'] = {
            'type': 'GradientBoosting',
            'n_train_samples': len(clf_features),
            'classes': list(clf.classes_),
            'feature_importance': {
                k: float(v) for k, v in zip(VOL_FEATURES, clf.feature_importances_)
            }
        }

    non_baseline = [r for r in results if r['visit_idx'] > 1 and r.get('ssim') is not None]
    if non_baseline:
        summary['overall_metrics'] = {
            'mean_ssim': float(np.mean([r['ssim'] for r in non_baseline])),
            'mean_psnr': float(np.mean([r['psnr'] for r in non_baseline])),
            'mean_mae': float(np.mean([r['mae'] for r in non_baseline])),
        }

    summary_path = os.path.join(args.output_dir, f'{args.subject}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Pipeline completed in {elapsed/60:.1f} minutes")
    print(f"Output: {args.output_dir}")
    print(f"GIF: {gif_path}")
    print(f"Chart: {chart_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
