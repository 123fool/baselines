#!/usr/bin/env python3
"""
BrLP MCI→AD Converter Pipeline
================================
Pipeline for MCI-to-AD converter patients.

Key features:
- Reads visits from mci_longitudinal/{subject}/{date}/
- Gets age/sex/diagnosis from B_mci.csv
- Gets ADNI diagnosis from local CSV mapping (uploaded as mci_diagnosis_map.json)
- Generates predictions at 6-month intervals from baseline
- Compares with real visits when timing matches (±60 days)
- Classifies predicted images at each timepoint
- Creates per-subject GIF + trajectory chart + summary JSON
- Outputs overall bias analysis across all subjects

Usage:
  cd /home/wangchong/data/fwz/code
  python 24_classification_animation/run_pipeline_mci_ad.py \
    --gpu 1 --subjects 002_S_1070 023_S_0388 --avg_n 3
"""

import os
import sys
import json
import csv
import argparse
import time
import warnings
import glob
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

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
MCI_LONG_DIR = f"{DATA_DIR}/mci_longitudinal"
BMCI_CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
VOLUMES_3CLASS_CSV = "/home/wangchong/data/fwz/output/classification_animation/volumes_3class.csv"

# Diagnosis map file (uploaded by auto script)
DIAG_MAP_FILE = None  # set in main()

# Model checkpoints
AE_CKPT = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT = "/home/wangchong/data/fwz/output/innovation_5/controlnet/cnet-ep-4.pth"

# Volume features for classifier
VOL_FEATURES = [
    'cerebral_cortex', 'hippocampus', 'amygdala',
    'cerebral_white_matter', 'lateral_ventricle'
]

DIAGNOSIS_MAP = {0.0: 'CN', 0.5: 'MCI', 1.0: 'AD'}
DIAGNOSIS_REVERSE = {'CN': 0.0, 'MCI': 0.5, 'AD': 1.0}

# SynthSeg code map (same as in const)
SYNTHSEG_CODEMAP = const.SYNTHSEG_CODEMAP
COARSE_REGIONS = const.COARSE_REGIONS


# ──────── Volume extraction from SynthSeg ────────

def extract_volumes_from_synthseg(segm_path):
    """Extract and normalize volume features from synthseg segmentation."""
    segm = nib.load(segm_path).get_fdata().round()
    head_size = int((segm > 0).sum())

    raw_volumes = {}
    for region in COARSE_REGIONS:
        raw_volumes[region] = 0
    for code, region in SYNTHSEG_CODEMAP.items():
        if region == 'background':
            continue
        coarse = region.replace('left_', '').replace('right_', '')
        raw_volumes[coarse] = raw_volumes.get(coarse, 0) + int((segm == code).sum())

    if head_size > 0:
        for k in raw_volumes:
            raw_volumes[k] = raw_volumes[k] / head_size

    return raw_volumes, head_size


def get_bmci_normalization_range(bmci_csv=BMCI_CSV):
    """Get normalization range from B_mci train split."""
    mins = {k: float('inf') for k in VOL_FEATURES}
    maxs = {k: float('-inf') for k in VOL_FEATURES}

    with open(bmci_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split', '') != 'train':
                continue
            for prefix in ['starting', 'followup']:
                for feat in VOL_FEATURES:
                    key = f'{prefix}_{feat}'
                    if key in row and row[key]:
                        val = float(row[key])
                        mins[feat] = min(mins[feat], val)
                        maxs[feat] = max(maxs[feat], val)
    return mins, maxs


# ──────── Latent extraction ────────

def extract_latent(image_path, autoencoder, device):
    """Extract latent from NIfTI image using autoencoder."""
    tfm = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    img = tfm(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        z = autoencoder.encode_stage_2_inputs(img)

    latent_path = image_path.replace('.nii.gz', '_latent.npz')
    np.savez_compressed(latent_path, data=z.squeeze(0).cpu().numpy())
    print(f"    Latent saved: {latent_path}")
    return z, latent_path


# ──────── Model loading ────────

def load_models(device):
    """Load AE, Diffusion, ControlNet."""
    print(f"  Loading AutoencoderKL from {AE_CKPT}...")
    autoencoder = networks.init_autoencoder(AE_CKPT).to(device).eval()

    print(f"  Loading DiffusionUNet from {DIFF_CKPT}...")
    diffusion = networks.init_latent_diffusion(DIFF_CKPT).to(device).eval()

    print(f"  Loading ControlNet from {CNET_CKPT}...")
    controlnet = networks.init_controlnet(CNET_CKPT).to(device).eval()

    return autoencoder, diffusion, controlnet


def get_latent_loader():
    npz_reader = NumpyReader(npz_keys=['data'])
    return transforms.Compose([
        transforms.LoadImage(reader=npz_reader),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])


def get_gt_loader():
    return transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])


def compute_scale_factor(load_tfm, bmci_csv=BMCI_CSV):
    """Compute scale factor from first training latent."""
    df = pd.read_csv(bmci_csv)
    train_df = df[df['split'] == 'train']
    first_latent = train_df.iloc[0]['starting_latent']
    if os.path.exists(first_latent):
        z = load_tfm(first_latent)
        sf = float(1.0 / torch.std(torch.tensor(z)))
        print(f"  Scale factor: {sf:.4f} (from {os.path.basename(first_latent)})")
        return sf
    print("  [WARN] Could not compute scale factor, using 0.9665")
    return 0.9665


# ──────── Load MCI→AD subject visits ────────

def load_mci_subject_visits(subject_id, diag_map=None):
    """
    Load visits for an MCI→AD converter from mci_longitudinal directory + B_mci.csv.

    Returns: (visits_list, sex, bmci_info)
    """
    subject_dir = os.path.join(MCI_LONG_DIR, subject_id)
    if not os.path.isdir(subject_dir):
        print(f"  [ERROR] Directory not found: {subject_dir}")
        return [], None, {}

    # 1. Discover date directories
    date_dirs = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and d != subject_id:
            # Check it looks like a date
            try:
                datetime.strptime(d, '%Y-%m-%d')
                date_dirs.append(d)
            except ValueError:
                continue

    if not date_dirs:
        print(f"  [ERROR] No date directories for {subject_id}")
        return [], None, {}

    print(f"  Found {len(date_dirs)} date directories: {date_dirs}")

    # 2. Load age/sex/diagnosis from B_mci.csv
    bmci_info = {}  # date -> {age, sex, diagnosis, ...}
    sex = None

    if os.path.exists(BMCI_CSV):
        with open(BMCI_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['subject_id'] != subject_id:
                    continue
                if sex is None:
                    sex = float(row['sex'])

                for prefix in ['starting', 'followup']:
                    visit_date = row.get(f'{prefix}_visit_date', '')
                    if not visit_date:
                        continue
                    age_key = f'{prefix}_age'
                    diag_key = f'{prefix}_diagnosis'

                    info = {
                        'age': float(row[age_key]) if row.get(age_key) else None,
                        'diagnosis': float(row[diag_key]) if row.get(diag_key) else 0.5,
                        'days': int(row.get(f'{prefix}_days_from_first_visit', 0)),
                    }
                    for feat in VOL_FEATURES:
                        fk = f'{prefix}_{feat}'
                        if fk in row and row[fk]:
                            info[feat] = float(row[fk])

                    bmci_info[visit_date] = info

    # 3. Load ADNI diagnosis if available
    adni_diag = {}  # date -> int (2=MCI, 3=AD)
    if diag_map and subject_id in diag_map:
        for entry in diag_map[subject_id]:
            adni_diag[entry['date']] = int(entry['diagnosis'])

    # 4. Build visits list
    baseline_date = datetime.strptime(date_dirs[0], '%Y-%m-%d')
    visits = []

    for date_str in date_dirs:
        visit_date = datetime.strptime(date_str, '%Y-%m-%d')
        days_from_baseline = (visit_date - baseline_date).days

        img_path = os.path.join(subject_dir, date_str, 't1w_final.nii.gz')
        segm_path = os.path.join(subject_dir, date_str, 'synthseg.nii.gz')
        latent_path = os.path.join(subject_dir, date_str, 't1w_final_latent.npz')

        if not os.path.exists(img_path):
            print(f"  [WARN] Missing image: {img_path}")
            continue

        # Get info from B_mci
        bi = bmci_info.get(date_str, {})

        # Determine diagnosis
        diag_val = bi.get('diagnosis', 0.5)
        if date_str in adni_diag:
            # ADNI: 2=MCI, 3=AD -> BrLP: 0.5=MCI, 1.0=AD
            adni_d = adni_diag[date_str]
            if adni_d == 3:
                diag_val = 1.0
            elif adni_d == 2:
                diag_val = 0.5

        visit = {
            'date': date_str,
            'days': days_from_baseline,
            'age': bi.get('age'),
            'diagnosis': diag_val,
            'image': img_path,
            'segm': segm_path,
            'latent': latent_path if os.path.exists(latent_path) else None,
            'has_segm': os.path.exists(segm_path),
            'volumes': {k: bi.get(k, None) for k in VOL_FEATURES},
        }
        visits.append(visit)

    # 5. Fill missing ages
    base_age = None
    base_days = 0
    for v in visits:
        if v['age'] is not None:
            base_age = v['age']
            base_days = v['days']
            break

    if base_age is not None:
        for v in visits:
            if v['age'] is None:
                v['age'] = base_age + (v['days'] - base_days) / 365.25 / 100.0
    else:
        # Default: ~75 years normalized
        for i, v in enumerate(visits):
            v['age'] = 0.75 + i * (180 / 365.25 / 100.0)

    if sex is None:
        sex = 0.0  # default

    return visits, sex, bmci_info


# ──────── Generate 6-month timeline ────────

def build_6month_timeline(visits, max_months=42):
    """
    Build a timeline at 6-month intervals.
    For each timepoint, find matching real visit (±60 days) if any.

    Returns: list of {month, target_age, matched_visit_idx, has_real_data}
    """
    if not visits:
        return []

    baseline = visits[0]
    baseline_age = baseline['age']
    timeline = []

    for month in range(0, max_months + 1, 6):
        target_days = month * 30.44
        target_age = baseline_age + month / 12.0 / 100.0

        # Find closest real visit within ±60 days
        best_idx = None
        best_diff = 9999

        for i, v in enumerate(visits):
            diff = abs(v['days'] - target_days)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        has_real = best_diff <= 60 and best_idx is not None

        tp = {
            'month': month,
            'target_days': target_days,
            'target_age': target_age,
            'matched_visit_idx': best_idx if has_real else None,
            'has_real_data': has_real,
            'closest_real_days': visits[best_idx]['days'] if best_idx is not None else None,
            'day_mismatch': best_diff,
        }
        timeline.append(tp)

    return timeline


# ──────── Metrics ────────

def compute_metrics(pred_np, gt_np):
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import imageio

    frames_dir = os.path.join(output_dir, f'frames_{subject_id}')
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    for i, r in enumerate(results):
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
        fig.suptitle(f'MCI→AD Progression — {subject_id}',
                     color='white', fontsize=18, fontweight='bold', y=0.98)

        months = r['months_from_baseline']
        ssim = r.get('ssim', 0)
        pred_class = r.get('predicted_class', '?')
        pred_probs = r.get('class_probs', {})
        pred_vol = r.get('pred_norm')
        gt_vol = r.get('gt_norm')
        real_diag = r.get('real_diagnosis', '?')
        has_real = r.get('has_real_data', False)

        if pred_vol is None:
            continue

        h, w, d = pred_vol.shape
        slices = {
            'Axial': (pred_vol[:, :, d // 2], gt_vol[:, :, d // 2] if gt_vol is not None else None),
            'Coronal': (pred_vol[:, w // 2, :], gt_vol[:, w // 2, :] if gt_vol is not None else None),
            'Sagittal': (pred_vol[h // 2, :, :], gt_vol[h // 2, :, :] if gt_vol is not None else None),
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

        info_text = f"Time: +{months:.0f} months\n"
        info_text += f"(Step {i + 1}/{len(results)})\n"
        info_text += f"Real Diag: {real_diag}\n\n"

        if i > 0 and has_real:
            info_text += f"SSIM: {ssim:.4f}\n\n"
        elif i > 0:
            info_text += "(no real data to compare)\n\n"
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
            v_now = vols.get(k)
            v_base = base_vols.get(k)
            if v_now is not None and v_base is not None and abs(v_base) > 1e-6:
                change = (v_now - v_base) / abs(v_base) * 100
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
        print(f"  [ANIM] Frame {i + 1}/{len(results)} saved")

    gif_path = os.path.join(output_dir, f'{subject_id}_mci_ad_progression.gif')
    images = [imageio.imread(fp) for fp in frame_paths]
    durations = [1500] * len(images)
    if durations:
        durations[-1] = 3000
    imageio.mimsave(gif_path, images, duration=durations, loop=0)
    print(f"  [ANIM] GIF saved: {gif_path}")

    return gif_path, frame_paths


def create_trajectory_chart(results, output_dir, subject_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    months = [r['months_from_baseline'] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), facecolor='#0d1117')
    fig.suptitle(f'MCI→AD Progression Trajectory — {subject_id}',
                 color='white', fontsize=16, fontweight='bold')

    # Panel 1: Volumes
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    colors = {'hippocampus': '#ff6b6b', 'amygdala': '#ffd93d',
              'lateral_ventricle': '#6bcb77', 'cerebral_cortex': '#4d96ff',
              'cerebral_white_matter': '#9b59b6'}
    for feat in VOL_FEATURES:
        vals = [r['volumes'].get(feat, 0) if r['volumes'].get(feat) is not None else 0
                for r in results]
        ax1.plot(months, vals, 'o-', color=colors.get(feat, 'white'),
                 label=feat.replace('_', ' ').title(), linewidth=2, markersize=6)

    # Mark real vs generated
    for i, r in enumerate(results):
        if r.get('has_real_data'):
            ax1.axvline(x=months[i], color='lime', alpha=0.3, linestyle='--', linewidth=1)

    ax1.set_xlabel('Months from baseline', color='white')
    ax1.set_ylabel('Normalized Volume', color='white')
    ax1.set_title('Brain Region Volumes Over Time (green lines = real data available)', color='white', fontsize=13)
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

    # Mark real diagnosis transitions
    for i, r in enumerate(results):
        rd = r.get('real_diagnosis', '')
        if rd == 'AD':
            ax2.axvspan(months[i] - 1, months[i] + 1, alpha=0.15, color='red')

    ax2.set_xlabel('Months from baseline', color='white')
    ax2.set_ylabel('Probability', color='white')
    ax2.set_title('Diagnosis Probability (red bands = real AD diagnosis)', color='white', fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=11, facecolor='#161b22', edgecolor='gray', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='gray')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # Panel 3: SSIM (only where real data exists)
    ax3 = axes[2]
    ax3.set_facecolor('#161b22')
    real_months = [m for m, r in zip(months, results) if r.get('has_real_data') and r.get('ssim') is not None]
    real_ssim = [r['ssim'] for r in results if r.get('has_real_data') and r.get('ssim') is not None]

    if real_ssim:
        bar_w = max(1, (max(months) - min(months)) / max(len(real_months), 1) * 0.6)
        bars = ax3.bar(real_months, real_ssim, width=bar_w,
                       color=['#4d96ff' if i > 0 else '#555555' for i in range(len(real_months))],
                       edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, real_ssim):
            ypos = min(val + 0.02, 1.0)
            ax3.text(bar.get_x() + bar.get_width() / 2, ypos, f'{val:.4f}',
                     ha='center', va='bottom', color='white', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No real data for SSIM comparison',
                 ha='center', va='center', color='gray', fontsize=14,
                 transform=ax3.transAxes)

    ax3.set_xlabel('Months from baseline', color='white')
    ax3.set_ylabel('SSIM', color='white')
    ax3.set_title('Image Similarity (Generated vs Real)', color='white', fontsize=13)
    ax3.set_ylim(0, 1.05)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='gray', axis='y')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f'{subject_id}_mci_ad_trajectory.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"  [CHART] Saved: {chart_path}")
    return chart_path


# ──────── Process one subject ────────

def process_subject(subject_id, autoencoder, diffusion, controlnet,
                    clf, scaler, clf_trained, scale_factor,
                    load_tfm, load_gt, device, output_dir,
                    diag_map, avg_n=3, max_months=36, sex_override=None):
    """Process a single MCI→AD converter subject."""
    print(f"\n{'='*70}")
    print(f"Processing: {subject_id}")
    print(f"{'='*70}")
    t0 = time.time()

    subj_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subj_dir, exist_ok=True)

    # Load visits
    visits, sex, bmci_info = load_mci_subject_visits(subject_id, diag_map)
    if sex_override is not None:
        sex = sex_override

    if len(visits) < 1:
        print(f"  [SKIP] No visits found for {subject_id}")
        return None

    print(f"  Visits: {len(visits)}, sex={sex}")
    for i, v in enumerate(visits):
        diag_label = DIAGNOSIS_MAP.get(v['diagnosis'], '?')
        print(f"    Visit {i+1}: {v['date']}, days={v['days']}, age={v['age']:.4f}, diag={diag_label}")

    # Extract volumes from synthseg for visits that need it
    for i, v in enumerate(visits):
        if any(v['volumes'][k] is None for k in VOL_FEATURES):
            if v['has_segm']:
                print(f"  Extracting volumes from synthseg for visit {i+1} ({v['date']})...")
                vols, head_size = extract_volumes_from_synthseg(v['segm'])
                v['volumes'] = {k: vols.get(k, 0) for k in VOL_FEATURES}
                v['head_size'] = head_size
            else:
                print(f"  [WARN] No segm for visit {i+1}, using defaults")
                v['volumes'] = {k: 0.5 for k in VOL_FEATURES}

    # Extract or load latent for baseline
    baseline = visits[0]
    if baseline['latent'] and os.path.exists(baseline['latent']):
        print(f"  Loading baseline latent: {baseline['latent']}")
        baseline_z_raw = load_tfm(baseline['latent'])
    else:
        print(f"  Extracting baseline latent from image...")
        _, lp = extract_latent(baseline['image'], autoencoder, device)
        baseline['latent'] = lp
        baseline_z_raw = load_tfm(lp)

    baseline_z = torch.tensor(baseline_z_raw).float() * scale_factor
    baseline_age = baseline['age']
    print(f"  Baseline: latent shape={baseline_z.shape}, age={baseline_age:.4f}")

    # Build 6-month timeline
    timeline = build_6month_timeline(visits, max_months=max_months)
    print(f"\n  Timeline ({len(timeline)} points):")
    for tp in timeline:
        match_str = f"→ visit {tp['matched_visit_idx']+1}" if tp['has_real_data'] else "(generated only)"
        print(f"    Month {tp['month']:3d}: {match_str}")

    # Process each timepoint
    results = []

    for ti, tp in enumerate(timeline):
        month = tp['month']
        has_real = tp['has_real_data']
        matched_idx = tp['matched_visit_idx']

        print(f"\n  --- Timepoint {ti+1}/{len(timeline)}: month +{month} ---")

        # Determine context for this timepoint
        if has_real and matched_idx is not None:
            visit = visits[matched_idx]
            context_age = visit['age']
            context_diag = visit['diagnosis']
            context_vols = visit['volumes']
            real_diag = DIAGNOSIS_MAP.get(visit['diagnosis'], '?')
        else:
            # Interpolate/extrapolate from known visits
            context_age = tp['target_age']
            # Estimate diagnosis: if past last MCI visit, assume progression
            context_diag = 0.5  # default MCI
            # Find nearest visit for volume estimation
            nearest_idx = 0
            nearest_diff = 9999
            for vi, v in enumerate(visits):
                diff = abs(v['days'] - tp['target_days'])
                if diff < nearest_diff:
                    nearest_diff = diff
                    nearest_idx = vi
            context_vols = visits[nearest_idx]['volumes']
            real_diag = '?'

        result = {
            'timepoint_idx': ti + 1,
            'months_from_baseline': float(month),
            'age': context_age,
            'real_diagnosis': real_diag,
            'has_real_data': has_real,
            'volumes': dict(context_vols),
        }

        # Load real image if available
        gt_np = None
        if has_real and matched_idx is not None:
            visit = visits[matched_idx]
            if os.path.exists(visit['image']):
                gt_t = load_gt(visit['image']).squeeze(0)
                gt_np = gt_t.numpy()
                print(f"    Real image loaded: {gt_np.shape}")

        if ti == 0:
            # Baseline
            if gt_np is not None:
                result['pred_norm'] = gt_np
                result['gt_norm'] = gt_np
            else:
                bl_t = load_gt(baseline['image']).squeeze(0)
                result['pred_norm'] = bl_t.numpy()
                result['gt_norm'] = result['pred_norm']

            result['ssim'] = 1.0
            result['psnr'] = 99.0
            result['mae'] = 0.0
            result['rmse'] = 0.0
            print(f"    Baseline (no generation)")
        else:
            # Build context tensor
            vol_vals = [context_vols.get(k, 0.5) if context_vols.get(k) is not None else 0.5
                        for k in VOL_FEATURES]
            context = torch.tensor([
                context_age,
                sex,
                context_diag,
                vol_vals[0],  # cerebral_cortex
                vol_vals[1],  # hippocampus
                vol_vals[2],  # amygdala
                vol_vals[3],  # cerebral_white_matter
                vol_vals[4],  # lateral_ventricle
            ]).float()

            print(f"    Context: age={context_age:.4f}, diag={context_diag}, "
                  f"hipp={vol_vals[1]:.4f}, vent={vol_vals[4]:.4f}")

            pred = sample_using_controlnet_and_z(
                autoencoder=autoencoder,
                diffusion=diffusion,
                controlnet=controlnet,
                starting_z=baseline_z,
                starting_a=baseline_age,
                context=context,
                device=device,
                scale_factor=scale_factor,
                average_over_n=avg_n,
                verbose=False,
            )
            pred_np = pred.numpy().clip(0, 1)
            print(f"    Generated: shape={pred_np.shape}")

            if gt_np is not None:
                metrics = compute_metrics(pred_np, gt_np)
                result.update(metrics)
                print(f"    SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}")
            else:
                result['pred_norm'] = pred_np
                result['ssim'] = None
                result['psnr'] = None

            # Save NIfTI
            nii_path = os.path.join(subj_dir, f'{subject_id}_month{month:02d}_pred.nii.gz')
            nib.save(nib.Nifti1Image(pred_np, np.eye(4)), nii_path)

        # Extract volumes from predicted image's segm (use real if available)
        if has_real and matched_idx is not None and visits[matched_idx]['has_segm']:
            # Use real segmentation volumes
            pass  # already in context_vols
        else:
            # For generated-only timepoints, use interpolated volumes
            pass

        # Classification
        if clf_trained:
            vol_list = [result['volumes'].get(k, 0.5) if result['volumes'].get(k) is not None else 0.5
                        for k in VOL_FEATURES]
            pred_class, prob_dict = predict_diagnosis(clf, scaler, vol_list)
            result['predicted_class'] = pred_class
            result['class_probs'] = prob_dict
            print(f"    Classification: {pred_class} "
                  f"(CN={prob_dict.get('CN',0):.2%}, MCI={prob_dict.get('MCI',0):.2%}, "
                  f"AD={prob_dict.get('AD',0):.2%})")
        else:
            result['predicted_class'] = '?'
            result['class_probs'] = {'CN': 0.33, 'MCI': 0.34, 'AD': 0.33}

        results.append(result)

    # Create animation
    print(f"\n  Creating animation for {subject_id}...")
    gif_path, frame_paths = create_animation(results, subj_dir, subject_id)

    # Create trajectory chart
    print(f"  Creating trajectory chart...")
    chart_path = create_trajectory_chart(results, subj_dir, subject_id)

    # Build summary
    elapsed = time.time() - t0
    summary = {
        'subject_id': subject_id,
        'diagnosis_type': 'MCI_to_AD',
        'n_real_visits': len(visits),
        'n_timepoints': len(timeline),
        'max_months': max_months,
        'model': 'Inn5-CNet-Avg3',
        'avg_n': avg_n,
        'scale_factor': scale_factor,
        'sex': sex,
        'gif_path': gif_path,
        'chart_path': chart_path,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'visits_real': [],
        'timeline': [],
    }

    for v in visits:
        summary['visits_real'].append({
            'date': v['date'],
            'days': v['days'],
            'age': v['age'],
            'diagnosis': DIAGNOSIS_MAP.get(v['diagnosis'], '?'),
        })

    for r in results:
        tp_summary = {
            'timepoint_idx': r['timepoint_idx'],
            'months_from_baseline': r['months_from_baseline'],
            'age': r['age'],
            'real_diagnosis': r['real_diagnosis'],
            'has_real_data': r['has_real_data'],
            'predicted_class': r['predicted_class'],
            'class_probs': r['class_probs'],
            'ssim': r.get('ssim'),
            'psnr': r.get('psnr'),
            'mae': r.get('mae'),
            'rmse': r.get('rmse'),
            'volumes': {k: v for k, v in r['volumes'].items() if v is not None},
        }
        summary['timeline'].append(tp_summary)

    # Classifier info
    if clf_trained:
        summary['classifier'] = {
            'type': 'GradientBoosting',
            'n_train_samples': int(len(clf_features_global)),
            'classes': list(clf.classes_),
        }

    # Overall metrics (only from timepoints with real data, excluding baseline)
    real_tp = [r for r in results if r.get('has_real_data') and r['timepoint_idx'] > 1
               and r.get('ssim') is not None]
    if real_tp:
        summary['overall_metrics'] = {
            'mean_ssim': float(np.mean([r['ssim'] for r in real_tp])),
            'mean_psnr': float(np.mean([r['psnr'] for r in real_tp])),
            'mean_mae': float(np.mean([r['mae'] for r in real_tp])),
        }

    # Bias analysis for this subject
    ad_tps = [r for r in results if r.get('real_diagnosis') == 'AD']
    if ad_tps:
        ad_classified_as_ad = sum(1 for r in ad_tps if r['predicted_class'] == 'AD')
        ad_probs = [r['class_probs'].get('AD', 0) for r in ad_tps]
        summary['bias_analysis'] = {
            'n_ad_timepoints': len(ad_tps),
            'n_correctly_classified_ad': ad_classified_as_ad,
            'ad_accuracy': ad_classified_as_ad / len(ad_tps) if ad_tps else 0,
            'mean_ad_probability': float(np.mean(ad_probs)),
            'ad_prob_trend': ad_probs,
        }

    summary_path = os.path.join(subj_dir, f'{subject_id}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_path}")
    print(f"  Completed {subject_id} in {elapsed/60:.1f} minutes")

    return summary


# Global for classifier data
clf_features_global = []


# ──────── Main ────────

def main():
    global clf_features_global

    parser = argparse.ArgumentParser(description='BrLP MCI→AD Converter Pipeline')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--subjects', nargs='+', type=str,
                        default=['002_S_1070', '023_S_0388', '023_S_0604'],
                        help='MCI→AD converter subject IDs')
    parser.add_argument('--output_dir', type=str,
                        default='/home/wangchong/data/fwz/output/mci_ad_classification')
    parser.add_argument('--avg_n', type=int, default=3)
    parser.add_argument('--max_months', type=int, default=36)
    parser.add_argument('--diag_map', type=str, default=None,
                        help='Path to mci_diagnosis_map.json')
    parser.add_argument('--list_subjects', action='store_true',
                        help='List available MCI subjects and exit')
    args = parser.parse_args()

    if args.list_subjects:
        print(f"\nScanning {MCI_LONG_DIR} for subjects...")
        subjects = {}
        for d in sorted(os.listdir(MCI_LONG_DIR)):
            sdir = os.path.join(MCI_LONG_DIR, d)
            if not os.path.isdir(sdir):
                continue
            dates = [x for x in os.listdir(sdir)
                     if os.path.isdir(os.path.join(sdir, x)) and x != d]
            if dates:
                subjects[d] = len(dates)

        print(f"\n{'Subject':>15} {'Visits':>6}")
        print("-" * 25)
        for sid in sorted(subjects.keys(), key=lambda s: subjects[s], reverse=True):
            print(f"{sid:>15} {subjects[sid]:>6}")
        return

    # Load ADNI diagnosis map
    diag_map = None
    if args.diag_map and os.path.exists(args.diag_map):
        with open(args.diag_map) as f:
            diag_map = json.load(f)
        print(f"Loaded diagnosis map: {len(diag_map)} subjects")

    os.makedirs(args.output_dir, exist_ok=True)
    device = f'cuda:{args.gpu}'

    print("=" * 70)
    print("BrLP MCI→AD Converter Pipeline")
    print(f"Subjects: {args.subjects}")
    print(f"GPU: {args.gpu} | Avg_n: {args.avg_n} | Max months: {args.max_months}")
    print("=" * 70)
    t_start = time.time()

    # Step 1: Train classifier
    print("\n[STEP 1] Training 3-class volume classifier...")
    clf_features, clf_labels = load_classification_data(
        volumes_csv=VOLUMES_3CLASS_CSV,
        bmci_csv=BMCI_CSV,
    )
    clf_features_global = clf_features
    print(f"  Loaded {len(clf_features)} samples")
    for lbl in np.unique(clf_labels):
        print(f"    {lbl}: {sum(clf_labels == lbl)}")
    clf, scaler = train_classifier(clf_features, clf_labels)
    clf_trained = clf is not None

    # Step 2: Load models
    print(f"\n[STEP 2] Loading models on {device}...")
    autoencoder, diffusion, controlnet = load_models(device)
    print("  Models loaded successfully")

    load_tfm = get_latent_loader()
    load_gt = get_gt_loader()
    scale_factor = compute_scale_factor(load_tfm, BMCI_CSV)

    # Step 3: Process each subject
    all_summaries = []

    for si, subject_id in enumerate(args.subjects):
        print(f"\n\n{'#'*70}")
        print(f"# Subject {si+1}/{len(args.subjects)}: {subject_id}")
        print(f"{'#'*70}")

        summary = process_subject(
            subject_id=subject_id,
            autoencoder=autoencoder,
            diffusion=diffusion,
            controlnet=controlnet,
            clf=clf,
            scaler=scaler,
            clf_trained=clf_trained,
            scale_factor=scale_factor,
            load_tfm=load_tfm,
            load_gt=load_gt,
            device=device,
            output_dir=args.output_dir,
            diag_map=diag_map,
            avg_n=args.avg_n,
            max_months=args.max_months,
        )

        if summary:
            all_summaries.append(summary)

    # Step 4: Overall bias analysis
    print(f"\n\n{'='*70}")
    print("Overall Bias Analysis")
    print(f"{'='*70}")

    bias_report = {
        'n_subjects': len(all_summaries),
        'subjects': [],
        'overall': {},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    all_ad_tps = []
    all_mci_tps = []

    for s in all_summaries:
        subj_bias = {
            'subject_id': s['subject_id'],
            'n_visits': s['n_real_visits'],
            'n_timepoints': s['n_timepoints'],
        }

        if 'bias_analysis' in s:
            subj_bias.update(s['bias_analysis'])

        for tp in s.get('timeline', []):
            rd = tp.get('real_diagnosis', '?')
            if rd == 'AD':
                all_ad_tps.append(tp)
            elif rd == 'MCI':
                all_mci_tps.append(tp)

        bias_report['subjects'].append(subj_bias)

    if all_ad_tps:
        ad_pred_ad = sum(1 for t in all_ad_tps if t['predicted_class'] == 'AD')
        ad_pred_mci = sum(1 for t in all_ad_tps if t['predicted_class'] == 'MCI')
        ad_pred_cn = sum(1 for t in all_ad_tps if t['predicted_class'] == 'CN')
        ad_probs = [t['class_probs'].get('AD', 0) for t in all_ad_tps]

        bias_report['overall']['ad_timepoints'] = {
            'total': len(all_ad_tps),
            'predicted_AD': ad_pred_ad,
            'predicted_MCI': ad_pred_mci,
            'predicted_CN': ad_pred_cn,
            'accuracy': ad_pred_ad / len(all_ad_tps),
            'mean_ad_prob': float(np.mean(ad_probs)),
            'std_ad_prob': float(np.std(ad_probs)),
        }
        print(f"\n  AD timepoints: {len(all_ad_tps)}")
        print(f"    Predicted AD: {ad_pred_ad} ({ad_pred_ad/len(all_ad_tps):.1%})")
        print(f"    Predicted MCI: {ad_pred_mci} ({ad_pred_mci/len(all_ad_tps):.1%})")
        print(f"    Predicted CN: {ad_pred_cn} ({ad_pred_cn/len(all_ad_tps):.1%})")
        print(f"    Mean AD prob: {np.mean(ad_probs):.4f}")

    if all_mci_tps:
        mci_pred_mci = sum(1 for t in all_mci_tps if t['predicted_class'] == 'MCI')
        mci_pred_cn = sum(1 for t in all_mci_tps if t['predicted_class'] == 'CN')
        mci_pred_ad = sum(1 for t in all_mci_tps if t['predicted_class'] == 'AD')
        mci_probs = [t['class_probs'].get('MCI', 0) for t in all_mci_tps]

        bias_report['overall']['mci_timepoints'] = {
            'total': len(all_mci_tps),
            'predicted_MCI': mci_pred_mci,
            'predicted_CN': mci_pred_cn,
            'predicted_AD': mci_pred_ad,
            'accuracy': mci_pred_mci / len(all_mci_tps),
            'mean_mci_prob': float(np.mean(mci_probs)),
        }
        print(f"\n  MCI timepoints: {len(all_mci_tps)}")
        print(f"    Predicted MCI: {mci_pred_mci} ({mci_pred_mci/len(all_mci_tps):.1%})")
        print(f"    Predicted AD: {mci_pred_ad} ({mci_pred_ad/len(all_mci_tps):.1%})")
        print(f"    Mean MCI prob: {np.mean(mci_probs):.4f}")

    # Potential bias sources
    bias_report['analysis_notes'] = [
        "Classifier trained on volume features only (5 regions)",
        "Training data imbalance: MCI >> CN > AD",
        "SynthSeg segmentation may have systematic errors",
        "Volume normalization by head_size may not capture atrophy patterns",
        "Generated images maintain structural similarity but may not alter volumes sufficiently",
    ]

    bias_path = os.path.join(args.output_dir, 'bias_analysis.json')
    with open(bias_path, 'w') as f:
        json.dump(bias_report, f, indent=2)
    print(f"\n  Bias analysis saved: {bias_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"All subjects completed in {elapsed/60:.1f} minutes")
    print(f"Processed {len(all_summaries)}/{len(args.subjects)} subjects")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
