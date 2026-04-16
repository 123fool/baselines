#!/usr/bin/env python3
"""
AD/MCI/CN Classification Improvement Experiment  (v2)
=====================================================
Fixes from v1:
  - Normalization: use BrLP's minmax_params, not self min-max
  - Converter eval: B_mci.csv has diagnosis=0.5 for ALL rows; test
    converter subjects' latest timepoints regardless of diagnosis
  - XGBoost / imblearn: now installed for Python 3.9 in fwz env
  - Filter bad synthseg (head_size < threshold)
  - Added Method F: threshold tuning for AD probability

Methods:
  A: Baseline GradientBoosting (original 640 samples)
  B: Balanced class weights (multiple classifiers, original data)
  C: Expanded AD data (add NL AD, correctly normalized) + balanced
  D: SMOTE oversampling on original data
  E: Full expansion (NL AD+CN+MCI, correctly normalized) + balanced
  F: Threshold tuning (lower AD probability threshold)

Usage:
  python classify_experiment_v2.py
"""

import os
import sys
import csv
import json
import time
import warnings
import numpy as np
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──
BASE = '/home/wangchong/data/fwz'
DATA = f'{BASE}/data'
VOLUMES_CSV = f'{BASE}/output/classification_animation/volumes_3class.csv'
BMCI_CSV = f'{BASE}/output/innovation_5/prepared/B_mci.csv'
OUTPUT_DIR = f'{BASE}/output/classify_experiment'

# BrLP imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / 'src'))
from brlp.const import SYNTHSEG_CODEMAP, COARSE_REGIONS

KEY_FEATURES = [
    'cerebral_cortex', 'hippocampus', 'amygdala',
    'cerebral_white_matter', 'lateral_ventricle'
]

# BrLP training-set min-max params (from confs.example.yaml)
MINMAX_PARAMS = {
    'cerebral_cortex':       [370876, 744801],
    'hippocampus':           [5006,   13955],
    'amygdala':              [1462,   5828],
    'cerebral_white_matter': [323328, 696723],
    'lateral_ventricle':     [10404,  191374],
}

# MCI→AD converter subjects (Section 25)
CONVERTER_SUBJECTS = [
    '002_S_1070', '023_S_0388', '023_S_0604', '027_S_0835',
    '053_S_0507', '023_S_0331', '016_S_1326', '023_S_1247'
]

# Minimum head_size to accept (filter bad synthseg)
MIN_HEAD_SIZE = 100000


# ═══════════════════════════════════════════════════════════
# Volume extraction & normalization
# ═══════════════════════════════════════════════════════════

def extract_volumes_from_synthseg(segm_path):
    """Extract coarse region volumes from SynthSeg segmentation."""
    import nibabel as nib
    segm = nib.load(segm_path).get_fdata().round()
    head_size = int((segm > 0).sum())
    volumes = {r: 0 for r in COARSE_REGIONS}
    for code, region in SYNTHSEG_CODEMAP.items():
        if region == 'background':
            continue
        coarse = region.replace('left_', '').replace('right_', '')
        volumes[coarse] += int((segm == code).sum())
    return volumes, head_size


def normalize_with_minmax(raw_val, feat_name):
    """Normalize a raw voxel count using BrLP's minmax_params."""
    mn, mx = MINMAX_PARAMS[feat_name]
    return (raw_val - mn) / (mx - mn)


def load_existing_volumes():
    """Load existing volumes_3class.csv (already normalized)."""
    records = []
    with open(VOLUMES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                'subject_id': row['subject_id'],
                'diagnosis': row['diagnosis'],
            }
            for feat in KEY_FEATURES:
                rec[feat] = float(row.get(feat, 0))
            records.append(rec)
    return records


def extract_nonlong_volumes(dx_label, dx_dir):
    """Extract volumes from non-longitudinal subjects, normalize with BrLP minmax."""
    import nibabel as nib
    records = []
    skipped = 0
    p = os.path.join(DATA, dx_dir)
    if not os.path.isdir(p):
        print(f"  [WARN] {p} not found")
        return records, skipped

    subjects = sorted(os.listdir(p))
    for sid in subjects:
        sp = os.path.join(p, sid)
        if not os.path.isdir(sp):
            continue
        tps = sorted([d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp, d))])
        for tp in tps:
            segm_path = os.path.join(sp, tp, 'synthseg.nii.gz')
            if not os.path.exists(segm_path):
                continue
            try:
                volumes, head_size = extract_volumes_from_synthseg(segm_path)
                if head_size < MIN_HEAD_SIZE:
                    skipped += 1
                    continue
                rec = {
                    'subject_id': sid,
                    'diagnosis': dx_label,
                    'head_size': head_size,
                }
                for feat in KEY_FEATURES:
                    rec[feat] = normalize_with_minmax(volumes.get(feat, 0), feat)
                records.append(rec)
            except Exception as e:
                print(f"  [ERR] {segm_path}: {e}")
                skipped += 1
    return records, skipped


# ═══════════════════════════════════════════════════════════
# Classifier training & evaluation
# ═══════════════════════════════════════════════════════════

def prepare_Xy(records):
    """Convert records -> numpy arrays. Labels: CN=0, MCI=1, AD=2."""
    X = np.array([[r[f] for f in KEY_FEATURES] for r in records])
    label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    y = np.array([label_map.get(r['diagnosis'], 1) for r in records])
    return X, y


def get_classifiers(balanced=False, include_xgb=True):
    """Return dict of classifier name -> instance."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    clfs = {}
    if not balanced:
        clfs['GradBoosting'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42)
    else:
        clfs['GradBoosting_bal'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42)

    clfs['RF_bal'] = RandomForestClassifier(
        n_estimators=300, max_depth=5, class_weight='balanced', random_state=42)
    clfs['SVM_rbf_bal'] = SVC(
        kernel='rbf', C=10, gamma='scale', class_weight='balanced',
        probability=True, random_state=42)
    clfs['SVM_linear_bal'] = SVC(
        kernel='linear', C=1, class_weight='balanced',
        probability=True, random_state=42)
    clfs['LogReg_bal'] = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42)

    if include_xgb:
        try:
            from xgboost import XGBClassifier
            clfs['XGB_bal'] = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                eval_metric='mlogloss', use_label_encoder=False,
                random_state=42)
        except ImportError:
            print("  [INFO] XGBoost not available")

    return clfs


def compute_sample_weight(y):
    """Compute balanced sample weights for classifiers that don't support class_weight."""
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    return np.array([total / (n_classes * counts[yi]) for yi in y])


def cross_validate(X, y, clfs, n_folds=5):
    """Run stratified k-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for name, clf in clfs.items():
        fold_accs, fold_f1s = [], []
        fold_f1_per = {'CN': [], 'MCI': [], 'AD': []}

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if 'GradBoosting' in name or 'XGB' in name:
                sw = compute_sample_weight(y_tr)
                clf.fit(X_tr, y_tr, sample_weight=sw)
            else:
                clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_val)
            from sklearn.metrics import accuracy_score
            fold_accs.append(accuracy_score(y_val, y_pred))
            fold_f1s.append(f1_score(y_val, y_pred, average='macro'))
            f1_per = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2])
            for i, lbl in enumerate(['CN', 'MCI', 'AD']):
                fold_f1_per[lbl].append(f1_per[i])

        results[name] = {
            'accuracy': float(np.mean(fold_accs)),
            'accuracy_std': float(np.std(fold_accs)),
            'macro_f1': float(np.mean(fold_f1s)),
            'per_class_f1': {lbl: float(np.mean(v)) for lbl, v in fold_f1_per.items()},
        }
    return results


def evaluate_on_converters(clf, X_train, y_train, clf_name, sample_weight=None):
    """
    Evaluate classifier on MCI→AD converter subjects.

    B_mci.csv has diagnosis=0.5 (MCI) for ALL rows of these subjects.
    We extract their LATEST followup features (already normalized) and
    test whether the classifier predicts them as AD.
    """
    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)

    if not os.path.exists(BMCI_CSV):
        return None

    # Collect ALL timepoint features for converter subjects
    # For each converter, keep the LATEST followup (max followup_age)
    latest = {}  # subject_id -> (followup_age, features_dict)
    all_timepoints = []  # All individual timepoints

    with open(BMCI_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row['subject_id']
            if sid not in CONVERTER_SUBJECTS:
                continue

            # Extract followup features (already normalized by BrLP)
            fu_feats = {}
            for feat in KEY_FEATURES:
                fu_feats[feat] = float(row.get(f'followup_{feat}', 0))

            fu_age = float(row.get('followup_age', 0))
            conv_label = int(float(row.get('followup_mci_conversion_label', 0)))

            tp_rec = {
                'subject_id': sid,
                'features': fu_feats,
                'age': fu_age,
                'conv_label': conv_label,
            }
            all_timepoints.append(tp_rec)

            # Track latest followup per subject
            if sid not in latest or fu_age > latest[sid][0]:
                latest[sid] = (fu_age, fu_feats, conv_label)

    if not latest:
        return None

    # Predict on LATEST timepoints of each converter subject
    latest_X = []
    latest_sids = []
    for sid in sorted(latest.keys()):
        _, feats, _ = latest[sid]
        latest_X.append([feats[f] for f in KEY_FEATURES])
        latest_sids.append(sid)

    latest_X = np.array(latest_X)
    latest_pred = clf.predict(latest_X)
    latest_prob = clf.predict_proba(latest_X) if hasattr(clf, 'predict_proba') else None

    label_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    n_total = len(latest_pred)
    n_ad = int((latest_pred == 2).sum())
    n_mci = int((latest_pred == 1).sum())
    n_cn = int((latest_pred == 0).sum())

    result = {
        'n_subjects': n_total,
        'pred_AD': n_ad,
        'pred_MCI': n_mci,
        'pred_CN': n_cn,
        'ad_rate': float(n_ad / n_total) if n_total > 0 else 0,
    }

    if latest_prob is not None:
        result['mean_AD_prob'] = float(latest_prob[:, 2].mean())
        result['mean_MCI_prob'] = float(latest_prob[:, 1].mean())
        # Per-subject detail
        details = []
        for i, sid in enumerate(latest_sids):
            details.append({
                'subject_id': sid,
                'pred': label_names[latest_pred[i]],
                'AD_prob': float(latest_prob[i, 2]),
                'MCI_prob': float(latest_prob[i, 1]),
            })
        result['details'] = details

    # Also predict on ALL timepoints for convergence analysis
    if all_timepoints:
        all_X = np.array([[tp['features'][f] for f in KEY_FEATURES] for tp in all_timepoints])
        all_pred = clf.predict(all_X)
        all_prob = clf.predict_proba(all_X) if hasattr(clf, 'predict_proba') else None
        result['all_timepoints_total'] = len(all_pred)
        result['all_timepoints_AD'] = int((all_pred == 2).sum())
        result['all_timepoints_AD_rate'] = float((all_pred == 2).mean())
        if all_prob is not None:
            result['all_timepoints_mean_AD_prob'] = float(all_prob[:, 2].mean())

    return result


# ═══════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    print("=" * 70)
    print("AD/MCI/CN Classification Experiment v2")
    print("=" * 70)

    # ── Step 1: Load existing data ──
    print("\n[1] Loading existing volumes_3class.csv ...")
    existing = load_existing_volumes()
    print(f"  Loaded {len(existing)} records")
    diag_counts = Counter(r['diagnosis'] for r in existing)
    print(f"  Distribution: {dict(diag_counts)}")

    # Print feature stats for reference
    for feat in KEY_FEATURES:
        vals = [r[feat] for r in existing]
        print(f"  {feat}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} "
              f"[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    # ── Step 2: Extract non-longitudinal volumes with BrLP normalization ──
    print("\n[2] Extracting non-longitudinal AD volumes (BrLP minmax norm) ...")
    t0 = time.time()
    ad_nl, ad_skip = extract_nonlong_volumes('AD', 'ad_non_longitudinal')
    print(f"  Extracted {len(ad_nl)} records, skipped {ad_skip} (bad synthseg) in {time.time()-t0:.1f}s")

    if ad_nl:
        for feat in KEY_FEATURES:
            vals = [r[feat] for r in ad_nl]
            print(f"  AD_NL {feat}: mean={np.mean(vals):.4f} [{np.min(vals):.4f}, {np.max(vals):.4f}]")

    print("\n[3] Extracting non-longitudinal CN volumes ...")
    t0 = time.time()
    cn_nl, cn_skip = extract_nonlong_volumes('CN', 'cn_non_longitudinal')
    print(f"  Extracted {len(cn_nl)} records, skipped {cn_skip} in {time.time()-t0:.1f}s")

    print("\n[4] Extracting non-longitudinal MCI volumes ...")
    t0 = time.time()
    mci_nl, mci_skip = extract_nonlong_volumes('MCI', 'mci_non_longitudinal')
    print(f"  Extracted {len(mci_nl)} records, skipped {mci_skip} in {time.time()-t0:.1f}s")

    # ── Check scale alignment ──
    print("\n[5] Scale alignment check ...")
    for feat in KEY_FEATURES:
        ex_vals = [r[feat] for r in existing]
        nl_vals = [r[feat] for r in ad_nl + cn_nl + mci_nl]
        if nl_vals:
            print(f"  {feat}: existing mean={np.mean(ex_vals):.4f}, NL mean={np.mean(nl_vals):.4f}, "
                  f"overlap={'YES' if abs(np.mean(ex_vals)-np.mean(nl_vals))<0.3 else 'NO (MISMATCH)'}")

    # ═══════════════════════════════════════════════════════
    # Method A: Baseline
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method A: Baseline GradientBoosting (original data)")
    print("=" * 70)

    from sklearn.ensemble import GradientBoostingClassifier
    X_orig, y_orig = prepare_Xy(existing)

    clf_a = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    res_a = cross_validate(X_orig, y_orig, {'GradBoosting': clf_a})
    all_results['A_baseline'] = res_a

    # Converter eval
    clf_a_full = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    conv_a = evaluate_on_converters(clf_a_full, X_orig, y_orig, 'A_baseline')
    all_results['A_baseline_converter'] = conv_a

    r = res_a['GradBoosting']
    print(f"  CV: Acc={r['accuracy']:.4f}±{r['accuracy_std']:.4f}  "
          f"F1={r['macro_f1']:.4f}  AD_F1={r['per_class_f1']['AD']:.4f}")
    if conv_a:
        print(f"  Converter: {conv_a['pred_AD']}/{conv_a['n_subjects']} predicted AD "
              f"(rate={conv_a['ad_rate']:.2%})")
        if 'mean_AD_prob' in conv_a:
            print(f"  Mean AD prob on converters: {conv_a['mean_AD_prob']:.4f}")
        if 'details' in conv_a:
            for d in conv_a['details']:
                print(f"    {d['subject_id']}: pred={d['pred']} AD_prob={d['AD_prob']:.4f}")

    # ═══════════════════════════════════════════════════════
    # Method B: Balanced weights, multiple classifiers
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method B: Balanced weights, multiple classifiers")
    print("=" * 70)

    clfs_b = get_classifiers(balanced=True)
    res_b = cross_validate(X_orig, y_orig, clfs_b)
    all_results['B_balanced'] = res_b

    for name, r in res_b.items():
        print(f"  [{name}] Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  "
              f"AD_F1={r['per_class_f1']['AD']:.4f}")

    # Best classifier converter eval
    best_b = max(res_b, key=lambda k: res_b[k]['per_class_f1']['AD'])
    best_b_clf = get_classifiers(balanced=True)[best_b]
    if 'GradBoosting' in best_b or 'XGB' in best_b:
        sw = compute_sample_weight(y_orig)
        conv_b = evaluate_on_converters(best_b_clf, X_orig, y_orig, best_b, sample_weight=sw)
    else:
        conv_b = evaluate_on_converters(best_b_clf, X_orig, y_orig, best_b)
    all_results['B_converter'] = {'best_clf': best_b, **(conv_b or {})}

    if conv_b:
        print(f"  Best [{best_b}] -> Converter: {conv_b['pred_AD']}/{conv_b['n_subjects']} AD "
              f"| mean_AD_prob={conv_b.get('mean_AD_prob', 0):.4f}")
        if 'details' in conv_b:
            for d in conv_b['details']:
                print(f"    {d['subject_id']}: pred={d['pred']} AD_prob={d['AD_prob']:.4f}")

    # ═══════════════════════════════════════════════════════
    # Method C: Expanded AD + balanced (only if scale aligned)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method C: Expanded AD + balanced weights")
    print("=" * 70)

    # Check if NL AD data is scale-aligned
    if ad_nl:
        ex_cc = np.mean([r['cerebral_cortex'] for r in existing])
        nl_cc = np.mean([r['cerebral_cortex'] for r in ad_nl])
        scale_ok = abs(ex_cc - nl_cc) < 0.3
        print(f"  Scale check: existing CC mean={ex_cc:.4f}, NL AD CC mean={nl_cc:.4f} -> {'OK' if scale_ok else 'MISMATCH'}")

        expanded_c = existing + ad_nl
        diag_c = Counter(r['diagnosis'] for r in expanded_c)
        print(f"  Data: {dict(diag_c)} (total={len(expanded_c)})")
        X_c, y_c = prepare_Xy(expanded_c)

        clfs_c = get_classifiers(balanced=True)
        res_c = cross_validate(X_c, y_c, clfs_c)
        all_results['C_expand_ad'] = res_c

        for name, r in res_c.items():
            print(f"  [{name}] Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  "
                  f"AD_F1={r['per_class_f1']['AD']:.4f}")

        best_c = max(res_c, key=lambda k: res_c[k]['per_class_f1']['AD'])
        best_c_clf = get_classifiers(balanced=True)[best_c]
        if 'GradBoosting' in best_c or 'XGB' in best_c:
            sw = compute_sample_weight(y_c)
            conv_c = evaluate_on_converters(best_c_clf, X_c, y_c, best_c, sample_weight=sw)
        else:
            conv_c = evaluate_on_converters(best_c_clf, X_c, y_c, best_c)
        all_results['C_converter'] = {'best_clf': best_c, **(conv_c or {})}
        if conv_c:
            print(f"  [{best_c}] Converter: {conv_c['pred_AD']}/{conv_c['n_subjects']} AD "
                  f"| mean_AD_prob={conv_c.get('mean_AD_prob', 0):.4f}")
    else:
        print("  [SKIP] No NL AD data")
        all_results['C_expand_ad'] = 'SKIPPED'

    # ═══════════════════════════════════════════════════════
    # Method D: SMOTE
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method D: SMOTE oversampling + balanced")
    print("=" * 70)

    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_sm, y_sm = smote.fit_resample(X_orig, y_orig)
        print(f"  After SMOTE: {dict(Counter(y_sm))} (total={len(y_sm)})")

        clfs_d = get_classifiers(balanced=True)
        res_d = cross_validate(X_sm, y_sm, clfs_d)
        all_results['D_smote'] = res_d

        for name, r in res_d.items():
            print(f"  [{name}] Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  "
                  f"AD_F1={r['per_class_f1']['AD']:.4f}")

        best_d = max(res_d, key=lambda k: res_d[k]['per_class_f1']['AD'])
        best_d_clf = get_classifiers(balanced=True)[best_d]
        if 'GradBoosting' in best_d or 'XGB' in best_d:
            sw = compute_sample_weight(y_sm)
            conv_d = evaluate_on_converters(best_d_clf, X_sm, y_sm, best_d, sample_weight=sw)
        else:
            conv_d = evaluate_on_converters(best_d_clf, X_sm, y_sm, best_d)
        all_results['D_converter'] = {'best_clf': best_d, **(conv_d or {})}
        if conv_d:
            print(f"  [{best_d}] Converter: {conv_d['pred_AD']}/{conv_d['n_subjects']} AD "
                  f"| mean_AD_prob={conv_d.get('mean_AD_prob', 0):.4f}")
    except ImportError:
        print("  [SKIP] imblearn not installed")
        all_results['D_smote'] = 'SKIPPED'

    # ═══════════════════════════════════════════════════════
    # Method E: Full expansion + balanced
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method E: Full expansion (NL AD+CN+MCI) + balanced")
    print("=" * 70)

    expanded_e = existing + ad_nl + cn_nl + mci_nl
    diag_e = Counter(r['diagnosis'] for r in expanded_e)
    print(f"  Data: {dict(diag_e)} (total={len(expanded_e)})")
    X_e, y_e = prepare_Xy(expanded_e)

    clfs_e = get_classifiers(balanced=True)
    res_e = cross_validate(X_e, y_e, clfs_e)
    all_results['E_full_expand'] = res_e

    for name, r in res_e.items():
        print(f"  [{name}] Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  "
              f"AD_F1={r['per_class_f1']['AD']:.4f}")

    best_e = max(res_e, key=lambda k: res_e[k]['per_class_f1']['AD'])
    best_e_clf = get_classifiers(balanced=True)[best_e]
    if 'GradBoosting' in best_e or 'XGB' in best_e:
        sw = compute_sample_weight(y_e)
        conv_e = evaluate_on_converters(best_e_clf, X_e, y_e, best_e, sample_weight=sw)
    else:
        conv_e = evaluate_on_converters(best_e_clf, X_e, y_e, best_e)
    all_results['E_converter'] = {'best_clf': best_e, **(conv_e or {})}
    if conv_e:
        print(f"  [{best_e}] Converter: {conv_e['pred_AD']}/{conv_e['n_subjects']} AD "
              f"| mean_AD_prob={conv_e.get('mean_AD_prob', 0):.4f}")

    # ═══════════════════════════════════════════════════════
    # Method F: Threshold tuning (lower AD probability threshold)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method F: AD threshold tuning on baseline")
    print("=" * 70)

    # Use the best baseline classifier (GradientBoosting with balanced weights)
    from sklearn.ensemble import GradientBoostingClassifier
    clf_f = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    sw_f = compute_sample_weight(y_orig)
    clf_f.fit(X_orig, y_orig, sample_weight=sw_f)

    # Get converter features (latest timepoints)
    conv_features = []
    conv_sids = []
    if os.path.exists(BMCI_CSV):
        latest = {}
        with open(BMCI_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row['subject_id']
                if sid not in CONVERTER_SUBJECTS:
                    continue
                fu_age = float(row.get('followup_age', 0))
                fu_feats = {feat: float(row.get(f'followup_{feat}', 0)) for feat in KEY_FEATURES}
                if sid not in latest or fu_age > latest[sid][0]:
                    latest[sid] = (fu_age, fu_feats)
        for sid in sorted(latest.keys()):
            _, feats = latest[sid]
            conv_features.append([feats[f] for f in KEY_FEATURES])
            conv_sids.append(sid)

    if conv_features:
        conv_X = np.array(conv_features)
        conv_prob = clf_f.predict_proba(conv_X)

        print(f"  Converter subjects AD probabilities (default threshold 0.333):")
        for i, sid in enumerate(conv_sids):
            print(f"    {sid}: AD_prob={conv_prob[i,2]:.4f} MCI_prob={conv_prob[i,1]:.4f} CN_prob={conv_prob[i,0]:.4f}")

        # Test different thresholds
        thresholds = [0.33, 0.25, 0.20, 0.15, 0.10, 0.05]
        threshold_results = {}
        for thresh in thresholds:
            pred_ad = (conv_prob[:, 2] >= thresh).sum()
            rate = pred_ad / len(conv_prob)

            # Also evaluate on full CV data to see trade-off
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import f1_score
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_f1s = []
            for train_idx, val_idx in skf.split(X_orig, y_orig):
                X_tr, X_val = X_orig[train_idx], X_orig[val_idx]
                y_tr, y_val = y_orig[train_idx], y_orig[val_idx]
                clf_tmp = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
                sw_tmp = compute_sample_weight(y_tr)
                clf_tmp.fit(X_tr, y_tr, sample_weight=sw_tmp)
                prob_val = clf_tmp.predict_proba(X_val)
                # Custom prediction with threshold
                y_pred_custom = np.argmax(prob_val, axis=1)
                # Override: if AD prob >= thresh, predict AD
                y_pred_custom[prob_val[:, 2] >= thresh] = 2
                fold_f1s.append(f1_score(y_val, y_pred_custom, average='macro'))

            avg_f1 = np.mean(fold_f1s)
            threshold_results[thresh] = {
                'converter_AD_rate': float(rate),
                'converter_AD_count': int(pred_ad),
                'cv_macro_f1': float(avg_f1),
            }
            print(f"  Threshold={thresh:.2f}: {pred_ad}/{len(conv_prob)} converters->AD "
                  f"({rate:.0%}) | CV F1={avg_f1:.4f}")

        all_results['F_threshold'] = threshold_results

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    summary_rows = []
    for method_key in ['A_baseline', 'B_balanced', 'C_expand_ad', 'D_smote', 'E_full_expand']:
        res = all_results.get(method_key)
        if not res or isinstance(res, str):
            continue
        conv_key = method_key[0] + '_converter'
        conv = all_results.get(conv_key, {})

        for clf_name, r in res.items():
            ad_rate = conv.get('ad_rate', 'N/A') if isinstance(conv, dict) else 'N/A'
            ad_prob = conv.get('mean_AD_prob', 'N/A') if isinstance(conv, dict) else 'N/A'
            conv_n = conv.get('n_subjects', 0) if isinstance(conv, dict) else 0
            pred_ad = conv.get('pred_AD', 0) if isinstance(conv, dict) else 0

            row = {
                'method': method_key,
                'classifier': clf_name,
                'cv_accuracy': f"{r['accuracy']:.4f}",
                'macro_f1': f"{r['macro_f1']:.4f}",
                'AD_f1': f"{r['per_class_f1']['AD']:.4f}",
                'MCI_f1': f"{r['per_class_f1']['MCI']:.4f}",
                'CN_f1': f"{r['per_class_f1']['CN']:.4f}",
                'conv_AD_pred': f"{pred_ad}/{conv_n}",
                'conv_AD_rate': f"{ad_rate:.2%}" if isinstance(ad_rate, float) else ad_rate,
                'conv_mean_AD_prob': f"{ad_prob:.4f}" if isinstance(ad_prob, float) else ad_prob,
            }
            summary_rows.append(row)
            print(f"  {method_key:16s} | {clf_name:16s} | Acc={row['cv_accuracy']} "
                  f"| AD_F1={row['AD_f1']} | Conv={row['conv_AD_pred']} "
                  f"({row['conv_AD_rate']}) | AD_prob={row['conv_mean_AD_prob']}")

    # Save summary CSV
    csv_path = os.path.join(OUTPUT_DIR, 'classification_summary_v2.csv')
    if summary_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\n  Summary saved to {csv_path}")

    # Save full results JSON
    json_path = os.path.join(OUTPUT_DIR, 'classification_results_v2.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Full results saved to {json_path}")

    print("\n  DONE!")


if __name__ == '__main__':
    main()
