#!/usr/bin/env python3
"""
AD/MCI/CN Classification Improvement Experiment
================================================
Tests multiple approaches to improve AD classification accuracy,
especially for MCI→AD converter subjects.

Methods tested:
  A: Baseline GradientBoosting (original 640 samples)
  B: Balanced class weights (multiple classifiers, original data)
  C: Expanded AD data (add non-longitudinal AD) + balanced weights
  D: SMOTE oversampling on original data
  E: Full expansion (add non-longitudinal AD+CN+MCI) + balanced weights

Evaluation:
  - 5-fold cross-validation accuracy, macro-F1, per-class F1
  - MCI→AD converter test (8 subjects from Section 25)

Usage:
  python classify_experiment.py
"""

import os
import sys
import csv
import json
import time
import warnings
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──
BASE = '/home/wangchong/data/fwz'
DATA = f'{BASE}/data'
VOLUMES_CSV = f'{BASE}/output/classification_animation/volumes_3class.csv'
BMCI_CSV = f'{BASE}/output/innovation_5/prepared/B_mci.csv'
OUTPUT_DIR = f'{BASE}/output/classify_experiment'

# BrLP imports for volume extraction
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / 'src'))
from brlp.const import SYNTHSEG_CODEMAP, COARSE_REGIONS

KEY_FEATURES = [
    'cerebral_cortex', 'hippocampus', 'amygdala',
    'cerebral_white_matter', 'lateral_ventricle'
]

# MCI→AD converter subjects for evaluation (from Section 25)
CONVERTER_SUBJECTS = [
    '002_S_1070', '023_S_0388', '023_S_0604', '027_S_0835',
    '053_S_0507', '023_S_0331', '016_S_1326', '023_S_1247'
]


# ═══════════════════════════════════════════════════════════
# Volume extraction
# ═══════════════════════════════════════════════════════════

def extract_volumes_from_synthseg(segm_path):
    """Extract coarse region volumes from a SynthSeg segmentation."""
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


def load_existing_volumes():
    """Load existing volumes_3class.csv."""
    records = []
    if not os.path.exists(VOLUMES_CSV):
        print(f"[WARN] {VOLUMES_CSV} not found")
        return records

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


def extract_nonlong_volumes(dx_label, dx_dir, max_subjects=None):
    """Extract volumes from non-longitudinal subject folders."""
    import nibabel as nib
    records = []
    p = os.path.join(DATA, dx_dir)
    if not os.path.isdir(p):
        print(f"[WARN] {p} not found")
        return records

    subjects = sorted(os.listdir(p))
    if max_subjects:
        subjects = subjects[:max_subjects]

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
                rec = {
                    'subject_id': sid,
                    'diagnosis': dx_label,
                    'head_size': head_size,
                }
                for feat in KEY_FEATURES:
                    rec[feat] = volumes.get(feat, 0)
                records.append(rec)
            except Exception as e:
                print(f"  [ERR] {segm_path}: {e}")
    return records


def normalize_raw_volumes(records, ref_records):
    """
    Normalize raw voxel-count records using the same min-max range
    as the existing (already-normalized) reference records.
    We detect normalization status by magnitude: if mean>10, it's raw.
    """
    for feat in KEY_FEATURES:
        ref_vals = [r[feat] for r in ref_records if r[feat] is not None]
        new_vals = [r[feat] for r in records if r[feat] is not None]
        if not ref_vals or not new_vals:
            continue

        ref_mean = np.mean(ref_vals)
        new_mean = np.mean(new_vals)

        if new_mean > 10 and ref_mean < 10:
            # new records are raw counts, need normalization
            raw_vals = new_vals
            minv = min(raw_vals)
            maxv = max(raw_vals)
            if maxv > minv:
                for r in records:
                    r[feat] = (r[feat] - minv) / (maxv - minv)
                print(f"  Normalized {feat}: raw[{minv:.0f},{maxv:.0f}] -> [0,1]")

    return records


# ═══════════════════════════════════════════════════════════
# Classifier training & evaluation
# ═══════════════════════════════════════════════════════════

def prepare_Xy(records):
    """Convert records -> numpy arrays."""
    X = np.array([[r[f] for f in KEY_FEATURES] for r in records])
    label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    y = np.array([label_map.get(r['diagnosis'], 1) for r in records])
    return X, y


def get_classifiers(balanced=False):
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
        # GradientBoosting doesn't support class_weight; we'll use sample_weight

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

    try:
        from xgboost import XGBClassifier
        counts = None  # will be set per fold
        clfs['XGB_bal'] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            eval_metric='mlogloss', use_label_encoder=False,
            random_state=42)
    except ImportError:
        print("[INFO] XGBoost not available, skipping")

    return clfs


def cross_validate(X, y, clfs, n_folds=5):
    """Run stratified k-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for name, clf in clfs.items():
        fold_accs = []
        fold_f1s = []
        fold_f1_per_class = {'CN': [], 'MCI': [], 'AD': []}
        all_y_true = []
        all_y_pred = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # For GradientBoosting, compute sample weights manually
            if 'GradBoosting' in name:
                counts = Counter(y_tr)
                total = len(y_tr)
                n_classes = len(counts)
                weights = np.array([total / (n_classes * counts[yi]) for yi in y_tr])
                clf.fit(X_tr, y_tr, sample_weight=weights)
            elif 'XGB' in name:
                counts = Counter(y_tr)
                total = len(y_tr)
                n_classes = len(counts)
                weights = np.array([total / (n_classes * counts[yi]) for yi in y_tr])
                clf.fit(X_tr, y_tr, sample_weight=weights)
            else:
                clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_val)
            fold_accs.append(accuracy_score(y_val, y_pred))
            fold_f1s.append(f1_score(y_val, y_pred, average='macro'))

            f1_per = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2])
            for i, lbl in enumerate(['CN', 'MCI', 'AD']):
                fold_f1_per_class[lbl].append(f1_per[i])

            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)

        results[name] = {
            'accuracy': float(np.mean(fold_accs)),
            'accuracy_std': float(np.std(fold_accs)),
            'macro_f1': float(np.mean(fold_f1s)),
            'macro_f1_std': float(np.std(fold_f1s)),
            'per_class_f1': {
                lbl: float(np.mean(vals))
                for lbl, vals in fold_f1_per_class.items()
            },
            'confusion': {
                'y_true': all_y_true,
                'y_pred': all_y_pred,
            }
        }

    return results


def evaluate_on_converters(clf, X_train, y_train, clf_name,
                           sample_weight=None):
    """
    Train classifier on full training data, then evaluate on
    MCI→AD converter subjects using their real volumes from B_mci.csv
    or extracted from synthseg.
    """
    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)

    # Load converter data from B_mci.csv
    converter_records = []
    if os.path.exists(BMCI_CSV):
        with open(BMCI_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['subject_id'] not in CONVERTER_SUBJECTS:
                    continue
                for prefix in ['starting', 'followup']:
                    diag_val = float(row.get(f'{prefix}_diagnosis', 0.5))
                    diag_label = {0.0: 'CN', 0.5: 'MCI', 1.0: 'AD'}.get(diag_val, 'MCI')
                    rec = {
                        'subject_id': row['subject_id'],
                        'diagnosis': diag_label,
                    }
                    for feat in KEY_FEATURES:
                        rec[feat] = float(row.get(f'{prefix}_{feat}', 0))
                    converter_records.append(rec)

    if not converter_records:
        return None

    X_conv, y_conv = prepare_Xy(converter_records)
    y_pred = clf.predict(X_conv)
    y_prob = clf.predict_proba(X_conv) if hasattr(clf, 'predict_proba') else None

    label_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    ad_mask = y_conv == 2
    mci_mask = y_conv == 1

    result = {
        'total_samples': int(len(y_conv)),
        'ad_samples': int(ad_mask.sum()),
        'mci_samples': int(mci_mask.sum()),
    }

    if ad_mask.sum() > 0:
        ad_correct = int((y_pred[ad_mask] == 2).sum())
        result['ad_accuracy'] = float(ad_correct / ad_mask.sum())
        result['ad_correct'] = ad_correct
        result['ad_pred_dist'] = {
            label_names[i]: int((y_pred[ad_mask] == i).sum()) for i in range(3)
        }
        if y_prob is not None:
            result['ad_mean_prob'] = float(y_prob[ad_mask, 2].mean())

    if mci_mask.sum() > 0:
        mci_correct = int((y_pred[mci_mask] == 1).sum())
        result['mci_accuracy'] = float(mci_correct / mci_mask.sum())

    overall_acc = float((y_pred == y_conv).mean())
    result['overall_accuracy'] = overall_acc

    return result


# ═══════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    print("=" * 70)
    print("AD/MCI/CN Classification Improvement Experiment")
    print("=" * 70)

    # ── Step 1: Load existing data ──
    print("\n[1] Loading existing volumes_3class.csv ...")
    existing = load_existing_volumes()
    print(f"  Loaded {len(existing)} records")
    diag_counts = Counter(r['diagnosis'] for r in existing)
    print(f"  Distribution: {dict(diag_counts)}")

    # ── Step 2: Extract non-longitudinal AD volumes ──
    print("\n[2] Extracting non-longitudinal AD volumes ...")
    t0 = time.time()
    ad_nl_records = extract_nonlong_volumes('AD', 'ad_non_longitudinal')
    print(f"  Extracted {len(ad_nl_records)} AD_NL records in {time.time()-t0:.1f}s")

    # Normalize raw volumes to match existing data range
    if ad_nl_records:
        ad_nl_records = normalize_raw_volumes(ad_nl_records, existing)

    # ── Step 3: Extract non-longitudinal CN volumes ──
    print("\n[3] Extracting non-longitudinal CN volumes ...")
    t0 = time.time()
    cn_nl_records = extract_nonlong_volumes('CN', 'cn_non_longitudinal')
    print(f"  Extracted {len(cn_nl_records)} CN_NL records in {time.time()-t0:.1f}s")

    if cn_nl_records:
        cn_nl_records = normalize_raw_volumes(cn_nl_records, existing)

    # ── Step 4: Extract non-longitudinal MCI volumes (sample) ──
    print("\n[4] Extracting non-longitudinal MCI volumes ...")
    t0 = time.time()
    mci_nl_records = extract_nonlong_volumes('MCI', 'mci_non_longitudinal')
    print(f"  Extracted {len(mci_nl_records)} MCI_NL records in {time.time()-t0:.1f}s")

    if mci_nl_records:
        mci_nl_records = normalize_raw_volumes(mci_nl_records, existing)

    # ═══════════════════════════════════════════════════════
    # Method A: Baseline (original GradientBoosting, no balancing)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method A: Baseline GradientBoosting (original data, no balancing)")
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

    print(f"  CV Accuracy: {res_a['GradBoosting']['accuracy']:.4f} "
          f"± {res_a['GradBoosting']['accuracy_std']:.4f}")
    print(f"  Macro F1:    {res_a['GradBoosting']['macro_f1']:.4f}")
    print(f"  Per-class:   {res_a['GradBoosting']['per_class_f1']}")
    if conv_a:
        print(f"  Converter AD acc: {conv_a.get('ad_accuracy', 'N/A')}")
        print(f"  Converter AD dist: {conv_a.get('ad_pred_dist', 'N/A')}")

    # ═══════════════════════════════════════════════════════
    # Method B: Balanced class weights, original data, multiple classifiers
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method B: Balanced weights, original data, multiple classifiers")
    print("=" * 70)

    clfs_b = get_classifiers(balanced=True)
    res_b = cross_validate(X_orig, y_orig, clfs_b)
    all_results['B_balanced'] = res_b

    for name, r in res_b.items():
        print(f"\n  [{name}]")
        print(f"    CV Acc: {r['accuracy']:.4f} ± {r['accuracy_std']:.4f}")
        print(f"    Macro F1: {r['macro_f1']:.4f}")
        print(f"    Per-class F1: {r['per_class_f1']}")

    # Best B classifier converter eval
    best_b_name = max(res_b, key=lambda k: res_b[k]['per_class_f1']['AD'])
    best_b_clf = get_classifiers(balanced=True)[best_b_name]
    counts_b = Counter(y_orig)
    total_b = len(y_orig)
    n_cls_b = len(counts_b)
    if 'GradBoosting' in best_b_name or 'XGB' in best_b_name:
        sw = np.array([total_b / (n_cls_b * counts_b[yi]) for yi in y_orig])
        conv_b = evaluate_on_converters(best_b_clf, X_orig, y_orig, best_b_name, sample_weight=sw)
    else:
        conv_b = evaluate_on_converters(best_b_clf, X_orig, y_orig, best_b_name)
    all_results['B_balanced_converter'] = {'best_clf': best_b_name, **conv_b} if conv_b else None
    if conv_b:
        print(f"\n  Best for AD: [{best_b_name}]")
        print(f"    Converter AD acc: {conv_b.get('ad_accuracy', 'N/A')}")
        print(f"    Converter AD dist: {conv_b.get('ad_pred_dist', 'N/A')}")

    # ═══════════════════════════════════════════════════════
    # Method C: Expanded AD data + balanced weights
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method C: Expanded AD + balanced weights")
    print("=" * 70)

    expanded_c = existing + ad_nl_records
    diag_c = Counter(r['diagnosis'] for r in expanded_c)
    print(f"  Data: {dict(diag_c)} (total={len(expanded_c)})")
    X_c, y_c = prepare_Xy(expanded_c)

    clfs_c = get_classifiers(balanced=True)
    res_c = cross_validate(X_c, y_c, clfs_c)
    all_results['C_expand_ad'] = res_c

    for name, r in res_c.items():
        print(f"\n  [{name}]")
        print(f"    CV Acc: {r['accuracy']:.4f} ± {r['accuracy_std']:.4f}")
        print(f"    Macro F1: {r['macro_f1']:.4f}")
        print(f"    Per-class F1: {r['per_class_f1']}")

    best_c_name = max(res_c, key=lambda k: res_c[k]['per_class_f1']['AD'])
    best_c_clf = get_classifiers(balanced=True)[best_c_name]
    counts_c = Counter(y_c)
    total_c = len(y_c)
    n_cls_c = len(counts_c)
    if 'GradBoosting' in best_c_name or 'XGB' in best_c_name:
        sw = np.array([total_c / (n_cls_c * counts_c[yi]) for yi in y_c])
        conv_c = evaluate_on_converters(best_c_clf, X_c, y_c, best_c_name, sample_weight=sw)
    else:
        conv_c = evaluate_on_converters(best_c_clf, X_c, y_c, best_c_name)
    all_results['C_expand_ad_converter'] = {'best_clf': best_c_name, **conv_c} if conv_c else None
    if conv_c:
        print(f"\n  Best for AD: [{best_c_name}]")
        print(f"    Converter AD acc: {conv_c.get('ad_accuracy', 'N/A')}")
        print(f"    Converter AD dist: {conv_c.get('ad_pred_dist', 'N/A')}")

    # ═══════════════════════════════════════════════════════
    # Method D: SMOTE oversampling on original data
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method D: SMOTE oversampling + balanced classifiers")
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
            print(f"\n  [{name}]")
            print(f"    CV Acc: {r['accuracy']:.4f} ± {r['accuracy_std']:.4f}")
            print(f"    Macro F1: {r['macro_f1']:.4f}")
            print(f"    Per-class F1: {r['per_class_f1']}")

        best_d_name = max(res_d, key=lambda k: res_d[k]['per_class_f1']['AD'])
        best_d_clf = get_classifiers(balanced=True)[best_d_name]
        # Train on SMOTE data, test on converters
        if 'GradBoosting' in best_d_name or 'XGB' in best_d_name:
            counts_d = Counter(y_sm)
            total_d = len(y_sm)
            sw = np.array([total_d / (3 * counts_d[yi]) for yi in y_sm])
            conv_d = evaluate_on_converters(best_d_clf, X_sm, y_sm, best_d_name, sample_weight=sw)
        else:
            conv_d = evaluate_on_converters(best_d_clf, X_sm, y_sm, best_d_name)
        all_results['D_smote_converter'] = {'best_clf': best_d_name, **conv_d} if conv_d else None
        if conv_d:
            print(f"\n  Best for AD: [{best_d_name}]")
            print(f"    Converter AD acc: {conv_d.get('ad_accuracy', 'N/A')}")
    except ImportError:
        print("  [SKIP] imblearn not installed, skipping SMOTE")
        all_results['D_smote'] = 'SKIPPED (imblearn not installed)'

    # ═══════════════════════════════════════════════════════
    # Method E: Full expansion (all non-longitudinal) + balanced
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Method E: Full expansion (NL AD+CN+MCI) + balanced weights")
    print("=" * 70)

    expanded_e = existing + ad_nl_records + cn_nl_records + mci_nl_records
    diag_e = Counter(r['diagnosis'] for r in expanded_e)
    print(f"  Data: {dict(diag_e)} (total={len(expanded_e)})")
    X_e, y_e = prepare_Xy(expanded_e)

    clfs_e = get_classifiers(balanced=True)
    res_e = cross_validate(X_e, y_e, clfs_e)
    all_results['E_full_expand'] = res_e

    for name, r in res_e.items():
        print(f"\n  [{name}]")
        print(f"    CV Acc: {r['accuracy']:.4f} ± {r['accuracy_std']:.4f}")
        print(f"    Macro F1: {r['macro_f1']:.4f}")
        print(f"    Per-class F1: {r['per_class_f1']}")

    best_e_name = max(res_e, key=lambda k: res_e[k]['per_class_f1']['AD'])
    best_e_clf = get_classifiers(balanced=True)[best_e_name]
    counts_e = Counter(y_e)
    total_e = len(y_e)
    n_cls_e = len(counts_e)
    if 'GradBoosting' in best_e_name or 'XGB' in best_e_name:
        sw = np.array([total_e / (n_cls_e * counts_e[yi]) for yi in y_e])
        conv_e = evaluate_on_converters(best_e_clf, X_e, y_e, best_e_name, sample_weight=sw)
    else:
        conv_e = evaluate_on_converters(best_e_clf, X_e, y_e, best_e_name)
    all_results['E_full_expand_converter'] = {'best_clf': best_e_name, **conv_e} if conv_e else None
    if conv_e:
        print(f"\n  Best for AD: [{best_e_name}]")
        print(f"    Converter AD acc: {conv_e.get('ad_accuracy', 'N/A')}")
        print(f"    Converter AD dist: {conv_e.get('ad_pred_dist', 'N/A')}")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_rows = []
    for method_key in ['A_baseline', 'B_balanced', 'C_expand_ad', 'D_smote', 'E_full_expand']:
        res = all_results.get(method_key)
        if not res or isinstance(res, str):
            continue
        for clf_name, r in res.items():
            conv_key = f'{method_key}_converter'
            conv = all_results.get(conv_key, {})
            ad_acc = conv.get('ad_accuracy', 'N/A') if conv and isinstance(conv, dict) else 'N/A'
            mci_acc = conv.get('mci_accuracy', 'N/A') if conv and isinstance(conv, dict) else 'N/A'

            row = {
                'method': method_key,
                'classifier': clf_name,
                'cv_accuracy': f"{r['accuracy']:.4f}",
                'macro_f1': f"{r['macro_f1']:.4f}",
                'AD_f1': f"{r['per_class_f1']['AD']:.4f}",
                'MCI_f1': f"{r['per_class_f1']['MCI']:.4f}",
                'CN_f1': f"{r['per_class_f1']['CN']:.4f}",
                'converter_AD_acc': f"{ad_acc:.4f}" if isinstance(ad_acc, float) else ad_acc,
                'converter_MCI_acc': f"{mci_acc:.4f}" if isinstance(mci_acc, float) else mci_acc,
            }
            summary_rows.append(row)
            print(f"  {method_key:20s} | {clf_name:20s} | Acc={row['cv_accuracy']} "
                  f"| F1={row['macro_f1']} | AD_F1={row['AD_f1']} "
                  f"| Conv_AD={row['converter_AD_acc']}")

    # Save summary CSV
    csv_path = os.path.join(OUTPUT_DIR, 'classification_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\n  Summary saved to {csv_path}")

    # Save full results JSON (without confusion matrix arrays for readability)
    json_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean = {}
            for kk, vv in v.items():
                if isinstance(vv, dict) and 'confusion' in vv:
                    vv_copy = {kkk: vvv for kkk, vvv in vv.items() if kkk != 'confusion'}
                    clean[kk] = vv_copy
                else:
                    clean[kk] = vv
            json_results[k] = clean
        else:
            json_results[k] = v

    json_path = os.path.join(OUTPUT_DIR, 'classification_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"  Full results saved to {json_path}")

    print("\n  DONE!")


if __name__ == '__main__':
    main()
