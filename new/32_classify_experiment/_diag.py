#!/usr/bin/env python3
"""Quick diagnostics for classify_experiment issues."""
import csv, sys, os

BMCI = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
CONVERTER_SUBJECTS = [
    '002_S_1070','023_S_0388','023_S_0604','027_S_0835',
    '053_S_0507','023_S_0331','016_S_1326','023_S_1247'
]

print("=== 1. Converter subjects in B_mci.csv ===")
found = set()
with open(BMCI) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['subject_id'] in CONVERTER_SUBJECTS:
            found.add(row['subject_id'])
            ds = row.get('starting_diagnosis','?')
            df = row.get('followup_diagnosis','?')
            print(f"  {row['subject_id']}: s_diag={ds} f_diag={df}")
print(f"Found: {sorted(found)}")
print(f"Missing: {sorted(set(CONVERTER_SUBJECTS) - found)}")

print("\n=== 2. xgboost / imblearn import ===")
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.8/site-packages'))
try:
    import xgboost; print(f"  xgboost OK: {xgboost.__version__}")
except Exception as e:
    print(f"  xgboost FAIL: {e}")
try:
    from imblearn.over_sampling import SMOTE; print("  imblearn OK")
except Exception as e:
    print(f"  imblearn FAIL: {e}")

print("\n=== 3. NL sample volumes vs confs minmax ===")
import nibabel as nib
sys.path.insert(0, '/home/wangchong/data/fwz/code/src')
from brlp.const import SYNTHSEG_CODEMAP, COARSE_REGIONS

MINMAX = {
    'cerebral_cortex': [370876, 744801],
    'hippocampus': [5006, 13955],
    'amygdala': [1462, 5828],
    'cerebral_white_matter': [323328, 696723],
    'lateral_ventricle': [10404, 191374],
}

# Check one AD NL subject
ad_dir = '/home/wangchong/data/fwz/data/ad_non_longitudinal'
subjects = sorted(os.listdir(ad_dir))[:3]
for sid in subjects:
    sp = os.path.join(ad_dir, sid)
    tps = sorted([d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp,d))])
    for tp in tps[:1]:
        seg_path = os.path.join(sp, tp, 'synthseg.nii.gz')
        if not os.path.exists(seg_path):
            continue
        segm = nib.load(seg_path).get_fdata().round()
        hs = int((segm > 0).sum())
        vols = {}
        for code, region in SYNTHSEG_CODEMAP.items():
            if region == 'background': continue
            c = region.replace('left_','').replace('right_','')
            vols[c] = vols.get(c, 0) + int((segm == code).sum())
        print(f"\n  {sid}/{tp}: head_size={hs}")
        for feat in ['cerebral_cortex','hippocampus','amygdala','cerebral_white_matter','lateral_ventricle']:
            raw = vols.get(feat, 0)
            mn, mx = MINMAX[feat]
            norm = (raw - mn) / (mx - mn)
            print(f"    {feat}: raw={raw}, norm={norm:.4f}")

# Check one longitudinal MCI subject for comparison
print("\n=== 4. Longitudinal MCI sample ===")
mci_dir = '/home/wangchong/data/fwz/data/mci_longitudinal'
lsubs = sorted(os.listdir(mci_dir))[:2]
for sid in lsubs:
    sp = os.path.join(mci_dir, sid)
    tps = sorted([d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp,d))])
    for tp in tps[:1]:
        seg_path = os.path.join(sp, tp, 'synthseg.nii.gz')
        if not os.path.exists(seg_path):
            continue
        segm = nib.load(seg_path).get_fdata().round()
        hs = int((segm > 0).sum())
        vols = {}
        for code, region in SYNTHSEG_CODEMAP.items():
            if region == 'background': continue
            c = region.replace('left_','').replace('right_','')
            vols[c] = vols.get(c, 0) + int((segm == code).sum())
        print(f"\n  {sid}/{tp}: head_size={hs}")
        for feat in ['cerebral_cortex','hippocampus','amygdala','cerebral_white_matter','lateral_ventricle']:
            raw = vols.get(feat, 0)
            mn, mx = MINMAX[feat]
            norm = (raw - mn) / (mx - mn)
            print(f"    {feat}: raw={raw}, norm={norm:.4f}")

print("\n=== 5. Find actual confs.yaml ===")
import subprocess
result = subprocess.run(['find', '/home/wangchong/data/fwz', '-name', 'confs.yaml', '-o', '-name', 'confs.yml'], capture_output=True, text=True, timeout=10)
print(result.stdout.strip() if result.stdout.strip() else "  No confs.yaml found")

print("\nDONE")
