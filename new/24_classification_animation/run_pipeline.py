#!/usr/bin/env python3
"""
BrLP Classification + Animation Pipeline
==========================================
1) Load trained Inn5-CNet model
2) For selected subjects with multiple visits:
   - Generate predicted follow-up MRI at each real visit time
   - Compute SSIM/PSNR against real follow-up MRI
3) Train 3-class volume-based classifier (CN vs MCI vs AD)
4) Apply classifier to generated images' brain volumes → diagnosis trajectory
5) Generate GIF animations (generated vs real, + classification overlay)
6) Output JSON summary with all metrics

Usage:
  cd /home/wangchong/data/fwz/code/brlp_src
  python new/24_classification_animation/run_pipeline.py \
    --gpu 1 --subject 005_S_0572

Requires: torch, monai, nibabel, numpy, scipy, sklearn, matplotlib, imageio
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

# ──────── BrLP imports (add project root to path) ────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # BrLP-main/
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

# Model checkpoints
AE_CKPT = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT = "/home/wangchong/data/fwz/output/innovation_5/controlnet/cnet-ep-4.pth"

# Volume features used for context and classification
VOL_FEATURES = [
    'cerebral_cortex', 'hippocampus', 'amygdala',
    'cerebral_white_matter', 'lateral_ventricle'
]

DIAGNOSIS_MAP = {0.0: 'CN', 0.5: 'MCI', 1.0: 'AD'}

# Pre-extracted 3-class volume CSV (created by extract_volumes_for_classification.py)
VOLUMES_3CLASS_CSV = "/home/wangchong/data/fwz/output/classification_animation/volumes_3class.csv"


def load_models(device):
    """Load AE, Diffusion, ControlNet models (matching evaluate_all_methods.py)."""
    print(f"[MODEL] Loading autoencoder from {AE_CKPT}")
    autoencoder = networks.init_autoencoder(AE_CKPT).to(device).eval()

    print(f"[MODEL] Loading diffusion from {DIFF_CKPT}")
    diffusion = networks.init_latent_diffusion(DIFF_CKPT).to(device).eval()

    print(f"[MODEL] Loading controlnet from {CNET_CKPT}")
    controlnet = networks.init_controlnet(CNET_CKPT).to(device).eval()

    return autoencoder, diffusion, controlnet


def get_latent_loader():
    """Create the latent loading transform (matching evaluate_all_methods.py)."""
    npz_reader = NumpyReader(npz_keys=['data'])
    load_tfm = transforms.Compose([
        transforms.LoadImage(reader=npz_reader),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])
    return load_tfm


def get_gt_loader():
    """Create GT image loading pipeline (matching eval_fixed.py exactly).
    
    LoadImage returns MetaTensor with affine, so Spacing correctly
    identifies the image is already at 1.5mm and performs a no-op.
    Then ResizeWithPadOrCrop to (122,146,122) and ScaleIntensity to [0,1].
    """
    load_gt = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])
    return load_gt


def compute_scale_factor(load_tfm, bmci_csv):
    """Compute scale factor from first TRAIN latent (matching eval_fixed.py)."""
    df = pd.read_csv(bmci_csv)
    train_df = df[df['split'] == 'train']
    first_z = load_tfm(train_df.iloc[0]['starting_latent'])
    scale_factor = 1.0 / torch.std(torch.tensor(first_z))
    return float(scale_factor)


def load_subject_visits(bmci_csv, subject_id):
    """Load all visits for a subject from B_mci.csv."""
    with open(bmci_csv) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['subject_id'] == subject_id]
    
    if not rows:
        return [], None
    
    sex = float(rows[0]['sex'])
    
    # Collect unique visits
    visits = {}
    for r in rows:
        # Starting visit
        s_uid = r['starting_image_uid']
        if s_uid not in visits:
            visits[s_uid] = {
                'uid': s_uid,
                'age': float(r['starting_age']),
                'image': r['starting_image'],
                'latent': r.get('starting_latent', ''),
                'diagnosis': float(r['starting_diagnosis']),
                'days': int(r['starting_days_from_first_visit']),
                'volumes': {k: float(r[f'starting_{k}']) for k in VOL_FEATURES},
            }
        # Follow-up visit
        f_uid = r['followup_image_uid']
        if f_uid not in visits:
            visits[f_uid] = {
                'uid': f_uid,
                'age': float(r['followup_age']),
                'image': r['followup_image'],
                'latent': r.get('followup_latent', ''),
                'diagnosis': float(r['followup_diagnosis']),
                'days': int(r['followup_days_from_first_visit']),
                'volumes': {k: float(r[f'followup_{k}']) for k in VOL_FEATURES},
            }

    # Sort by age
    sorted_visits = sorted(visits.values(), key=lambda v: v['age'])
    return sorted_visits, sex


def build_context_original(visit, sex):
    """Build 8-D context vector (matching evaluate_all_methods original mode)."""
    return torch.tensor([
        visit['age'],
        sex,
        visit['diagnosis'],
        visit['volumes']['cerebral_cortex'],
        visit['volumes']['hippocampus'],
        visit['volumes']['amygdala'],
        visit['volumes']['cerebral_white_matter'],
        visit['volumes']['lateral_ventricle'],
    ]).float()


def compute_metrics(pred_np, gt_np):
    """Compute SSIM/PSNR/MAE/RMSE (matching eval_fixed.py exactly).
    
    pred_np: clipped [0,1], shape (122,146,122)
    gt_np: ScaleIntensity [0,1], shape (122,146,122)
    Both are already in the same space and shape.
    """
    # Crop to common shape (should already both be 122,146,122)
    ms = tuple(min(a, b) for a, b in zip(pred_np.shape, gt_np.shape))
    pred_crop = pred_np[:ms[0], :ms[1], :ms[2]]
    gt_crop = gt_np[:ms[0], :ms[1], :ms[2]]
    
    # data_range from GT (after ScaleIntensity, should be ~1.0)
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


# ──────── 3-class Classifier (CN vs MCI vs AD) ────────

def load_classification_data(volumes_csv=VOLUMES_3CLASS_CSV, bmci_csv=BMCI_CSV):
    """Load CN/MCI/AD volume data for classifier training.
    
    Uses pre-extracted volumes_3class.csv if available,
    otherwise falls back to B_mci.csv (MCI only).
    """
    all_features = []
    all_labels = []
    
    # Try pre-extracted 3-class CSV first
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
    
    # Fallback: use B_mci.csv (all MCI, but with baseline volumes from different visits)
    print(f"  Falling back to B_mci.csv (MCI-only classifier)")
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
    """Train a 3-class classifier (CN vs MCI vs AD) using volume features."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    # Count unique classes
    unique_classes = np.unique(labels)
    print(f"  Classes: {list(unique_classes)}")
    
    if len(unique_classes) < 2:
        print(f"  [WARN] Only {len(unique_classes)} class(es), cannot train classifier")
        return None, scaler
    
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42, 
        learning_rate=0.1, subsample=0.8
    )
    
    # Cross-validation
    n_folds = min(5, min(np.bincount(pd.factorize(labels)[0])))
    n_folds = max(2, n_folds)
    scores = cross_val_score(clf, X, labels, cv=n_folds, scoring='accuracy')
    print(f"  {n_folds}-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    
    # Train on full data
    clf.fit(X, labels)
    return clf, scaler


def predict_diagnosis(clf, scaler, volumes):
    """Predict diagnosis probabilities from brain volumes."""
    X = scaler.transform([volumes])
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_
    pred_class = clf.predict(X)[0]
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    return pred_class, prob_dict


# ──────── Animation Generator ────────

def create_animation(results, output_dir, subject_id):
    """Create GIF animation showing MCI progression."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import imageio
    
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_paths = []
    
    for i, r in enumerate(results):
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
        fig.suptitle(f'MCI Longitudinal Prediction — {subject_id}',
                     color='white', fontsize=18, fontweight='bold', y=0.98)
        
        months = r['months_from_baseline']
        ssim = r.get('ssim', 0)
        psnr = r.get('psnr', 0)
        pred_class = r.get('predicted_class', 'MCI')
        pred_probs = r.get('class_probs', {})
        
        # Get normalized volumes for display
        pred_vol = r.get('pred_norm')
        gt_vol = r.get('gt_norm')
        
        if pred_vol is None:
            continue  # Skip if no volume available
        
        h, w, d = pred_vol.shape
        slices = {
            'Axial': (pred_vol[:, :, d//2], gt_vol[:, :, d//2] if gt_vol is not None else None),
            'Coronal': (pred_vol[:, w//2, :], gt_vol[:, w//2, :] if gt_vol is not None else None),
            'Sagittal': (pred_vol[h//2, :, :], gt_vol[h//2, :, :] if gt_vol is not None else None),
        }
        
        views = ['Axial', 'Coronal', 'Sagittal']
        for row_idx, view in enumerate(views):
            pred_slice, gt_slice = slices[view]
            
            # Generated / Predicted
            ax1 = fig.add_subplot(gs[row_idx, 0])
            ax1.imshow(pred_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            lbl = 'Baseline' if i == 0 else 'Generated'
            ax1.set_title(f'{lbl} ({view})', color='cyan', fontsize=11)
            ax1.axis('off')
            
            # Real
            ax2 = fig.add_subplot(gs[row_idx, 1])
            if gt_slice is not None:
                ax2.imshow(gt_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
                ax2.set_title(f'Real ({view})', color='lime', fontsize=11)
            else:
                ax2.text(0.5, 0.5, 'No Real\nData', ha='center', va='center',
                        color='gray', fontsize=14, transform=ax2.transAxes)
                ax2.set_title(f'Real ({view})', color='gray', fontsize=11)
            ax2.axis('off')
            
            # Difference
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
        
        # Right panel: metrics + classification
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
        
        # Save frame
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        fig.savefig(frame_path, dpi=100, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close(fig)
        frame_paths.append(frame_path)
        print(f"  [ANIM] Frame {i+1}/{len(results)} saved")
    
    # Create GIF
    gif_path = os.path.join(output_dir, f'{subject_id}_progression.gif')
    images = [imageio.imread(fp) for fp in frame_paths]
    # Each frame shows for 1.5 seconds, last frame for 3 seconds
    durations = [1500] * len(images)
    if len(durations) > 0:
        durations[-1] = 3000
    imageio.mimsave(gif_path, images, duration=durations, loop=0)
    print(f"  [ANIM] GIF saved: {gif_path}")
    
    return gif_path, frame_paths


# ──────── Main Pipeline ────────

def main():
    parser = argparse.ArgumentParser(description='BrLP Classification + Animation Pipeline')
    parser.add_argument('--gpu', type=int, default=1, help='GPU index')
    parser.add_argument('--subject', type=str, default='005_S_0572',
                       help='Subject ID for animation')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/wangchong/data/fwz/output/classification_animation',
                       help='Output directory')
    parser.add_argument('--avg_n', type=int, default=3, help='LAS averaging count')
    parser.add_argument('--bmci_csv', type=str, default=BMCI_CSV)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = f'cuda:{args.gpu}'
    
    print("=" * 70)
    print("BrLP Classification + Animation Pipeline")
    print("=" * 70)
    t_start = time.time()
    
    # ── Step 1: Train classifier ──
    print("\n[STEP 1] Training 3-class volume classifier (CN/MCI/AD)...")
    clf_features, clf_labels = load_classification_data(
        volumes_csv=os.path.join(args.output_dir, 'volumes_3class.csv'),
        bmci_csv=args.bmci_csv,
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
    
    # Latent loading transform + scale factor
    load_tfm = get_latent_loader()
    load_gt = get_gt_loader()
    scale_factor = compute_scale_factor(load_tfm, args.bmci_csv)
    print(f"  Scale factor: {scale_factor:.4f}")
    
    # ── Step 3: Load subject visits ──
    print(f"\n[STEP 3] Loading visits for {args.subject}...")
    visits, sex = load_subject_visits(args.bmci_csv, args.subject)
    print(f"  Found {len(visits)} unique visits (sex={sex})")
    for i, v in enumerate(visits):
        months = v['days'] / 30.44
        print(f"    Visit {i+1}: age={v['age']:.4f}, days={v['days']}, "
              f"months={months:.1f}, diag={DIAGNOSIS_MAP.get(v['diagnosis'], '?')}")
    
    if len(visits) < 2:
        print("[ERROR] Need at least 2 visits for animation")
        return
    
    # ── Step 4: Generate predictions from baseline ──
    baseline = visits[0]
    
    # Load baseline latent (matching evaluate_all_methods.py exactly)
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
        
        # Load real image for comparison using proper MONAI pipeline
        # (LoadImage preserves affine → Spacing is no-op → ResizeWithPadOrCrop → ScaleIntensity [0,1])
        real_path = visit['image']
        gt_np = None
        if os.path.exists(real_path):
            gt_t = load_gt(real_path).squeeze(0)
            gt_np = gt_t.numpy()
            print(f"    Real image loaded: {gt_np.shape}, range=[{gt_np.min():.4f}, {gt_np.max():.4f}]")
        else:
            print(f"    Real image not found: {real_path}")
        
        if i == 0:
            # Baseline → no generation needed, just load real image
            if gt_np is not None:
                result['pred_norm'] = gt_np
                result['gt_norm'] = gt_np
            result['ssim'] = 1.0
            result['psnr'] = 99.0
            result['mae'] = 0.0
            result['rmse'] = 0.0
            print(f"    Baseline visit (no generation needed)")
        else:
            # Build context using GT volumes of target visit
            context = build_context_original(visit, sex)
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
            pred_np = pred.numpy().clip(0, 1)  # clip to [0,1] matching eval_fixed.py
            print(f"    Generated: shape={pred_np.shape}, "
                  f"range=[{pred_np.min():.4f}, {pred_np.max():.4f}]")
            
            # Compute metrics against real (both in [0,1] range, shape (122,146,122))
            if gt_np is not None:
                metrics = compute_metrics(pred_np, gt_np)
                result.update(metrics)
                print(f"    SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}")
            else:
                # No GT available, use pred directly for animation
                result['pred_norm'] = pred_np
            
            # Save generated NIfTI
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
            result['predicted_class'] = 'MCI'
            result['class_probs'] = {'CN': 0.0, 'MCI': 1.0, 'AD': 0.0}
        
        results.append(result)
    
    # ── Step 5: Create animation ──
    print(f"\n[STEP 5] Creating animation...")
    gif_path, frame_paths = create_animation(results, args.output_dir, args.subject)
    
    # ── Step 6: Create trajectory chart ──
    print(f"\n[STEP 6] Creating trajectory chart...")
    chart_path = create_trajectory_chart(results, args.output_dir, args.subject)
    
    # ── Step 7: Save summary JSON ──
    print(f"\n[STEP 7] Saving summary...")
    summary = {
        'subject_id': args.subject,
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
    
    # Classifier info
    if clf_trained:
        summary['classifier'] = {
            'type': 'GradientBoosting',
            'n_train_samples': len(clf_features),
            'classes': list(clf.classes_),
            'feature_importance': {
                k: float(v) for k, v in zip(VOL_FEATURES, clf.feature_importances_)
            }
        }
    
    # Overall metrics (excluding baseline)
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


def create_trajectory_chart(results, output_dir, subject_id):
    """Create a trajectory chart showing volume changes + classification over time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    months = [r['months_from_baseline'] for r in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), facecolor='#0d1117')
    fig.suptitle(f'MCI Progression Trajectory — {subject_id}',
                 color='white', fontsize=16, fontweight='bold')
    
    # Panel 1: Brain volumes over time
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    colors = {'hippocampus': '#ff6b6b', 'amygdala': '#ffd93d', 
              'lateral_ventricle': '#6bcb77', 'cerebral_cortex': '#4d96ff',
              'cerebral_white_matter': '#9b59b6'}
    for feat in VOL_FEATURES:
        vals = [r['volumes'][feat] for r in results]
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
    
    # Panel 2: Classification probabilities over time
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    class_colors = {'CN': '#6bcb77', 'MCI': '#ffd93d', 'AD': '#ff6b6b'}
    for cls in ['CN', 'MCI', 'AD']:
        vals = [r['class_probs'].get(cls, 0) for r in results]
        ax2.plot(months, vals, 's-', color=class_colors[cls], 
                label=cls, linewidth=2.5, markersize=8)
        ax2.fill_between(months, vals, alpha=0.1, color=class_colors[cls])
    ax2.set_xlabel('Months from baseline', color='white')
    ax2.set_ylabel('Classification Probability', color='white')
    ax2.set_title('Diagnosis Classification Over Time', color='white', fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=11, facecolor='#161b22', edgecolor='gray', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='gray')
    for spine in ax2.spines.values():
        spine.set_color('gray')
    
    # Panel 3: Generation quality metrics over time (skip baseline)
    ax3 = axes[2]
    ax3.set_facecolor('#161b22')
    non_base = [r for r in results if r['visit_idx'] > 1]
    if non_base:
        m = [r['months_from_baseline'] for r in non_base]
        ssims = [r.get('ssim', 0) for r in non_base]
        psnrs = [r.get('psnr', 0) for r in non_base]
        
        ax3_twin = ax3.twinx()
        l1 = ax3.plot(m, ssims, 'o-', color='#00d2ff', label='SSIM', 
                      linewidth=2.5, markersize=8)
        l2 = ax3_twin.plot(m, psnrs, 's-', color='#ff9500', label='PSNR (dB)',
                          linewidth=2.5, markersize=8)
        ax3.set_xlabel('Months from baseline', color='white')
        ax3.set_ylabel('SSIM', color='#00d2ff')
        ax3_twin.set_ylabel('PSNR (dB)', color='#ff9500')
        ax3.set_title('Generation Quality vs Time Gap', color='white', fontsize=13)
        
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, fontsize=11, facecolor='#161b22', 
                  edgecolor='gray', labelcolor='white')
        ax3.tick_params(colors='white')
        ax3_twin.tick_params(colors='white')
        ax3.grid(True, alpha=0.2, color='gray')
        for spine in ax3.spines.values():
            spine.set_color('gray')
        for spine in ax3_twin.spines.values():
            spine.set_color('gray')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, f'{subject_id}_trajectory.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"  Trajectory chart saved: {chart_path}")
    return chart_path


if __name__ == '__main__':
    main()
