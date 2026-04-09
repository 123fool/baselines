"""
Data Preparation for Innovation 5 ControlNet Training.

Full pipeline:
  1. Extract latents from all images using fine-tuned AE
  2. Fill missing age column (baseline_age + days_from_first_visit/365.25)
  3. Compute normalized SynthSeg region volumes (CSV A)
  4. Create paired starting/followup records (CSV B)
  5. Rename columns for BrLP compatibility

Usage:
    python prepare_data.py \
        --input_csv /path/to/mci_brlp_clean.csv \
        --output_dir /path/to/prepared/ \
        --aekl_ckpt /path/to/autoencoder-ep-2.pth
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
from monai import transforms

# Fix PyTorch 2.7 weights_only default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.environ.get('BRLP_SRC',
    os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src')))
sys.path.insert(0, BRLP_SRC)

from brlp import const
from brlp.networks import init_autoencoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASELINE_AGE = 70.0


def extract_latents(df, aekl_ckpt):
    """Extract latent representations for all images using fine-tuned AE."""
    print(f"\n=== Step 1: Extracting latents ===")
    print(f"  Checkpoint: {aekl_ckpt}")
    autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    extracted, skipped, failed = 0, 0, 0
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            image_path = row['image_path']
            destpath = image_path.replace('.nii.gz', '_latent.npz').replace(
                '.nii', '_latent.npz')
            if os.path.exists(destpath):
                skipped += 1
                continue
            try:
                mri_tensor = transforms_fn({'image_path': image_path})['image']
                mri_tensor = mri_tensor.to(DEVICE)
                mri_latent, _ = autoencoder.encode(mri_tensor.unsqueeze(0))
                mri_latent = mri_latent.cpu().squeeze(0).numpy()
                np.savez_compressed(destpath, data=mri_latent)
                extracted += 1
            except Exception as e:
                print(f"  WARN: {image_path}: {e}")
                failed += 1

    print(f"  Done: {extracted} extracted, {skipped} skipped, {failed} failed")


def fill_age(df):
    """Fill missing age column using days_from_first_visit."""
    if df['age'].notna().all():
        print("  Age already filled.")
        return df
    print(f"\n=== Step 2: Filling age (baseline={BASELINE_AGE}) ===")
    raw_age = BASELINE_AGE + df['days_from_first_visit'] / 365.25
    # BrLP convention: age = age_years / 100 (see REPR-DATA.md)
    df['age'] = raw_age / 100.0
    print(f"  Range: {df['age'].min():.4f} - {df['age'].max():.4f} (normalized)")
    return df


def make_csv_a(df):
    """Create CSV A: add normalized SynthSeg region volumes."""
    print("\n=== Step 3: Computing SynthSeg region volumes ===")
    coarse_regions = const.COARSE_REGIONS
    code_map = const.SYNTHSEG_CODEMAP

    records = []
    for record in tqdm(df.to_dict(orient='records'), desc="Volumes"):
        segm_path = record.get('segm_path') or record.get('segm')
        if not segm_path or not os.path.exists(str(segm_path)):
            for region in coarse_regions:
                record[region] = 0
            record['head_size'] = 0
            records.append(record)
            continue

        segm = nib.load(segm_path).get_fdata().round()
        record['head_size'] = int((segm > 0).sum())

        for region in coarse_regions:
            record[region] = 0

        for code, region in code_map.items():
            if region == 'background':
                continue
            coarse = region.replace('left_', '').replace('right_', '')
            record[coarse] += int((segm == code).sum())

        records.append(record)

    csv_a = pd.DataFrame(records)

    for region in coarse_regions:
        train_vals = csv_a[csv_a.split == 'train'][region]
        minv, maxv = train_vals.min(), train_vals.max()
        if maxv > minv:
            csv_a[region] = (csv_a[region] - minv) / (maxv - minv)
        else:
            csv_a[region] = 0.5

    print(f"  CSV A: {len(csv_a)} rows, {len(coarse_regions)} regions added")
    return csv_a


def make_csv_b(df):
    """Create CSV B: all chronological pairs per subject."""
    print("\n=== Step 4: Creating paired records (CSV B) ===")
    sorting_field = 'age'

    data = []
    for subject_id in tqdm(df.subject_id.unique(), desc="Pairing"):
        subj = df[df.subject_id == subject_id].sort_values(
            sorting_field, ascending=True)
        for i in range(len(subj)):
            for j in range(i + 1, len(subj)):
                s = subj.iloc[i]
                e = subj.iloc[j]
                rec = {'subject_id': s.subject_id, 'sex': s.sex,
                       'split': s.split}
                remaining = set(df.columns) - set(rec.keys())
                for col in remaining:
                    rec[f'starting_{col}'] = s[col]
                    rec[f'followup_{col}'] = e[col]
                data.append(rec)

    csv_b = pd.DataFrame(data)
    print(f"  CSV B: {len(csv_b)} pairs from {df.subject_id.nunique()} subjects")
    return csv_b


def verify_paths(df, cols):
    """Verify file paths exist."""
    for col in cols:
        if col not in df.columns:
            continue
        n = df[col].apply(
            lambda p: os.path.exists(str(p)) if pd.notna(p) else False).sum()
        print(f"  {col}: {n}/{len(df)} exist")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare data for ControlNet')
    parser.add_argument('--input_csv', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt', default=None, type=str,
                        help='Fine-tuned AE checkpoint for latent extraction')
    parser.add_argument('--skip_latents', action='store_true')
    parser.add_argument('--skip_volumes', action='store_true',
                        help='Use dummy 0.5 for all region volumes')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows, {df.subject_id.nunique()} subjects")
    print(f"  Splits: {df.split.value_counts().to_dict()}")

    # Step 1: Extract latents
    if not args.skip_latents:
        if args.aekl_ckpt is None:
            parser.error("--aekl_ckpt required unless --skip_latents")
        extract_latents(df, args.aekl_ckpt)

    # Step 2: Fill age
    df = fill_age(df)

    # Step 3: Compute volumes → CSV A
    if args.skip_volumes:
        print("\n=== Skipping volumes, using dummy 0.5 ===")
        for r in const.COARSE_REGIONS:
            df[r] = 0.5
        df['head_size'] = 1000000
        csv_a = df
    else:
        csv_a = make_csv_a(df)

    # Rename *_path → * for BrLP compatibility
    renames = {}
    for col in csv_a.columns:
        if col.endswith('_path'):
            renames[col] = col.replace('_path', '')
    if renames:
        print(f"\n  Renaming: {renames}")
        csv_a = csv_a.rename(columns=renames)

    csv_a_path = os.path.join(args.output_dir, 'A_mci.csv')
    csv_a.to_csv(csv_a_path, index=False)
    print(f"\n  Saved CSV A: {csv_a_path}")

    # Step 4: Create CSV B (paired)
    csv_b = make_csv_b(csv_a)
    csv_b_path = os.path.join(args.output_dir, 'B_mci.csv')
    csv_b.to_csv(csv_b_path, index=False)
    print(f"  Saved CSV B: {csv_b_path}")

    # Verify
    required = ['starting_latent', 'followup_latent', 'starting_age',
                'followup_age', 'sex', 'split', 'followup_diagnosis']
    missing = [c for c in required if c not in csv_b.columns]
    if missing:
        print(f"\n  WARNING: Missing columns: {missing}")
    else:
        print(f"\n  All required ControlNet columns present")

    if args.verify:
        print("\nVerifying CSV B paths...")
        verify_paths(csv_b, ['starting_image', 'followup_image',
                             'starting_segm', 'followup_segm',
                             'starting_latent', 'followup_latent'])

    train_b = len(csv_b[csv_b.split == 'train'])
    valid_b = len(csv_b[csv_b.split == 'valid'])
    test_b = len(csv_b[csv_b.split == 'test'])
    print(f"\n  Final: train={train_b}, valid={valid_b}, test={test_b}")
    print("Done!")
