#!/usr/bin/env python3
"""
Extract brain volumes from SynthSeg segmentations for CN/MCI/AD subjects.
Creates a combined CSV with diagnosis labels + normalized brain volumes
for training a 3-class classifier.

Usage:
  python extract_volumes_for_classification.py --output /path/to/volumes.csv

Reads from diagnosis_categorized CSVs + their synthseg.nii.gz files.
Uses B_mci.csv normalization stats for consistent scaling.
"""

import os
import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# BrLP imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brlp.const import SYNTHSEG_CODEMAP, COARSE_REGIONS

DATA_DIR = "/home/wangchong/data/fwz/data"
CN_CSV = f"{DATA_DIR}/diagnosis_categorized/cn_brlp_innovation.csv"
AD_CSV = f"{DATA_DIR}/diagnosis_categorized/ad_brlp_innovation.csv"
MCI_CSV = f"{DATA_DIR}/diagnosis_categorized/mci_brlp_innovation.csv"
BMCI_CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"

# We only need these 5 features for classification (matching model's context)
KEY_FEATURES = [
    'cerebral_cortex', 'hippocampus', 'amygdala',
    'cerebral_white_matter', 'lateral_ventricle'
]


def extract_volumes_from_seg(segm_path):
    """Extract coarse region volumes from a SynthSeg segmentation file."""
    import nibabel as nib
    segm = nib.load(segm_path).get_fdata().round()
    
    volumes = {}
    volumes['head_size'] = int((segm > 0).sum())
    
    for region in COARSE_REGIONS:
        volumes[region] = 0
    
    for code, region in SYNTHSEG_CODEMAP.items():
        if region == 'background':
            continue
        coarse_region = region.replace('left_', '').replace('right_', '')
        volumes[coarse_region] += int((segm == code).sum())
    
    return volumes


def get_normalization_stats_from_bmci(bmci_csv):
    """Get min/max normalization stats from B_mci.csv training data."""
    stats = {}
    
    with open(bmci_csv) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    # Collect values per region from starting_ columns (training split only)
    for region in KEY_FEATURES:
        col = f'starting_{region}'
        values = []
        for row in all_rows:
            if row.get('split', '') == 'train':
                try:
                    values.append(float(row[col]))
                except (ValueError, KeyError):
                    pass
        
        if values:
            # These are already normalized in B_mci.csv
            # We need RAW volumes min/max for normalizing new data
            # Since B_mci is already normalized, we'll normalize our extracted
            # volumes the same way: using the TRAINING split min/max of raw counts
            stats[region] = {
                'min': min(values),
                'max': max(values),
                'already_normalized': True
            }
    
    return stats, all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, 
                       default='/home/wangchong/data/fwz/output/classification_animation/volumes_3class.csv')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Step 1: Collect volumes from existing B_mci.csv (already normalized)
    print("[1/3] Loading MCI volumes from B_mci.csv...")
    records = []
    seen = set()  # avoid duplicates
    
    with open(BMCI_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for prefix in ['starting', 'followup']:
                uid = row.get(f'{prefix}_image_uid', '')
                if uid in seen:
                    continue
                seen.add(uid)
                
                diag = float(row.get(f'{prefix}_diagnosis', 0.5))
                diag_label = {0.0: 'CN', 0.5: 'MCI', 1.0: 'AD'}.get(diag, 'MCI')
                
                record = {
                    'subject_id': row['subject_id'],
                    'image_uid': uid,
                    'diagnosis': diag_label,
                    'split': row.get('split', 'train'),
                }
                for feat in KEY_FEATURES:
                    try:
                        record[feat] = float(row[f'{prefix}_{feat}'])
                    except (ValueError, KeyError):
                        record[feat] = 0.0
                records.append(record)
    
    n_mci = sum(1 for r in records if r['diagnosis'] == 'MCI')
    print(f"  MCI samples from B_mci.csv: {n_mci}")
    
    # Step 2: Extract volumes for CN and AD subjects from segmentations
    for csv_path, diag_label in [(CN_CSV, 'CN'), (AD_CSV, 'AD')]:
        print(f"\n[2/3] Processing {diag_label} data from {os.path.basename(csv_path)}...")
        
        if not os.path.exists(csv_path):
            print(f"  WARN: {csv_path} not found, skipping")
            continue
        
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            diag_rows = list(reader)
        
        count = 0
        for row in diag_rows:
            uid = row.get('image_uid', '')
            if uid in seen:
                continue
            
            segm_path = row.get('segm_path', '')
            if not segm_path or not os.path.exists(segm_path):
                continue
            
            try:
                volumes = extract_volumes_from_seg(segm_path)
            except Exception as e:
                print(f"  ERROR loading {segm_path}: {e}")
                continue
            
            seen.add(uid)
            record = {
                'subject_id': row.get('subject_id', ''),
                'image_uid': uid,
                'diagnosis': diag_label,
                'split': row.get('split', 'train'),
            }
            for feat in KEY_FEATURES:
                record[feat] = volumes.get(feat, 0)
            records.append(record)
            count += 1
            
            if count % 20 == 0:
                print(f"  Processed {count} {diag_label} subjects...")
        
        print(f"  Total {diag_label} samples extracted: {count}")
    
    # Step 3: Normalize CN/AD volumes using training split statistics
    print(f"\n[3/3] Normalizing volumes...")
    
    # Compute normalization stats from training split
    for feat in KEY_FEATURES:
        train_values = [r[feat] for r in records 
                       if r['split'] == 'train' and r['diagnosis'] == 'MCI']
        if not train_values:
            train_values = [r[feat] for r in records if r['split'] == 'train']
        
        if train_values:
            # MCI values from B_mci.csv are already normalized [0,1]
            # CN/AD values are raw voxel counts → need same normalization
            # Get the scale from MCI normalized range
            mci_norm = [r[feat] for r in records 
                       if r['diagnosis'] == 'MCI' and r['split'] == 'train']
            cn_ad_raw = [r[feat] for r in records 
                        if r['diagnosis'] != 'MCI']
            
            if mci_norm and cn_ad_raw:
                # MCI values are in [0,1] range (already normalized)
                # CN/AD values are raw counts (thousands or more)
                # Detect if normalization is needed by checking magnitude
                avg_mci = np.mean(mci_norm)
                avg_raw = np.mean([r[feat] for r in records if r['diagnosis'] != 'MCI']) if cn_ad_raw else 0
                
                if avg_raw > 10 and avg_mci < 10:  # raw counts vs normalized
                    # Need to normalize CN/AD values
                    # Get raw range from CN/AD training data
                    raw_train = [r[feat] for r in records 
                                if r['diagnosis'] != 'MCI' and r['split'] == 'train']
                    if raw_train:
                        minv = min(raw_train)
                        maxv = max(raw_train)
                        if maxv > minv:
                            for r in records:
                                if r['diagnosis'] != 'MCI':
                                    r[feat] = (r[feat] - minv) / (maxv - minv)
                            print(f"  Normalized {feat}: raw range [{minv:.0f}, {maxv:.0f}]")
    
    # Save combined CSV
    fieldnames = ['subject_id', 'image_uid', 'diagnosis', 'split'] + KEY_FEATURES
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
    
    # Summary
    diag_counts = defaultdict(int)
    for r in records:
        diag_counts[r['diagnosis']] += 1
    
    print(f"\nSaved {len(records)} records to {args.output}")
    for d, c in sorted(diag_counts.items()):
        print(f"  {d}: {c}")


if __name__ == '__main__':
    main()
