"""
TPN 评估脚本 — 对比 TPN vs Leaspy 体积预测精度
=================================================
用验证集/测试集的真实 followup 体积作为 ground truth，
分别评估 TPN 和 Leaspy 的预测 MAE。

Usage:
    python evaluate_tpn.py \
        --dataset_csv /path/to/B_mci.csv \
        --tpn_ckpt    /path/to/tpn_best.pth \
        --leaspy_dir  /path/to/leaspy/models/ \
        --output_dir  /path/to/output/tpn

日志格式 (用于 dashboard 解析):
    TPN MAE: 0.0312 | Leaspy MAE: 0.0345 | R²: 0.9521
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
BRLP_SRC_ALT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'brlp_src'))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, BRLP_SRC_ALT)

from tpn import TemporalProgressionNetwork
from brlp import const

VOLUME_REGIONS = const.CONDITIONING_REGIONS


def load_tpn(ckpt_path, device='cpu'):
    model = TemporalProgressionNetwork(in_dim=14, hidden_dim=128, out_dim=5, n_layers=3, dropout=0.0)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


def evaluate_tpn_on_csv(model, df, device='cpu'):
    """用 TPN 对 CSV B 中的配对做体积预测，返回 per-region MAE 和 overall MAE。"""
    all_preds = []
    all_targets = []

    for _, row in df.iterrows():
        current_age = row['starting_age']
        target_age = row['followup_age']

        sex_raw = row['sex']
        sex_norm = (sex_raw - const.SEX_MIN) / const.SEX_DELTA

        diag_col = None
        for col_name in ['starting_diagnosis', 'starting_last_diagnosis']:
            if col_name in df.columns:
                diag_col = col_name
                break
        diag_raw = row[diag_col] if diag_col else 0.5
        diag_norm = diag_raw if diag_raw <= 1 else (diag_raw - const.DIA_MIN) / const.DIA_DELTA

        current_vols = [row[f'starting_{r}'] for r in VOLUME_REGIONS]
        target_vols = [row[f'followup_{r}'] for r in VOLUME_REGIONS]
        age_gap = target_age - current_age
        # v2 新增特征
        age_ratio = age_gap / (current_age + 1e-8)
        vol_mean = np.mean(current_vols)
        vol_std = np.std(current_vols)
        age_gap_sq = age_gap ** 2

        x = torch.tensor([
            current_age, target_age, sex_norm, diag_norm,
            *current_vols, age_gap, age_ratio, vol_mean, vol_std, age_gap_sq
        ], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x).cpu().numpy().flatten()

        all_preds.append(pred)
        all_targets.append(target_vols)

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Per-region MAE
    region_mae = {}
    for i, region in enumerate(VOLUME_REGIONS):
        region_mae[region] = float(np.mean(np.abs(preds[:, i] - targets[:, i])))

    overall_mae = float(np.mean(np.abs(preds - targets)))

    # R² score
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "overall_mae": overall_mae,
        "region_mae": region_mae,
        "r2_score": r2,
        "n_samples": len(df),
    }


def evaluate_leaspy_on_csv(df, leaspy_dir):
    """用 Leaspy 对 CSV B 中的配对做体积预测 (如果可用)。"""
    try:
        from leaspy import Leaspy, AlgorithmSettings, Data
    except ImportError:
        print("[Eval] Leaspy not available, skipping Leaspy comparison")
        return None

    # 判断诊断类型
    diag_col = None
    for col_name in ['starting_diagnosis', 'starting_last_diagnosis']:
        if col_name in df.columns:
            diag_col = col_name
            break

    all_preds = []
    all_targets = []

    # 按 subject_id 分组
    for subject_id, group in df.groupby('subject_id'):
        group = group.sort_values('starting_age')
        first_row = group.iloc[0]

        # 确定使用哪个 Leaspy 模型
        if diag_col:
            diag_raw = first_row[diag_col]
            if diag_raw <= 0.25:
                model_name = 'dcm_cn.json'
            elif diag_raw <= 0.75:
                model_name = 'dcm_mci.json'
            else:
                model_name = 'dcm_ad.json'
        else:
            model_name = 'dcm_mci.json'

        model_path = os.path.join(leaspy_dir, model_name)
        if not os.path.exists(model_path):
            continue

        try:
            leaspy = Leaspy.load(model_path)
        except Exception:
            continue

        # 构建历史记录
        records = []
        seen_ages = set()
        for _, row in group.iterrows():
            age = row['starting_age'] * 100  # 反归一化
            if age in seen_ages:
                age += 0.001
            seen_ages.add(age)

            record = {'ID': str(subject_id), 'TIME': age}
            for region in VOLUME_REGIONS:
                val = row[f'starting_{region}']
                if region != 'lateral_ventricle':
                    val = 1 - val  # Leaspy 需要反转
                record[region] = val
            records.append(record)

        if not records:
            continue

        hist_df = pd.DataFrame(records)
        hist_df = hist_df.set_index(['ID', 'TIME'], verify_integrity=False).sort_index()
        hist_df = hist_df[VOLUME_REGIONS]

        try:
            data = Data.from_dataframe(hist_df)
            settings = AlgorithmSettings('scipy_minimize')
            ip = leaspy.personalize(data, settings)
        except Exception:
            continue

        # 对每一对做预测
        for _, row in group.iterrows():
            target_age = row['followup_age'] * 100
            target_vols = [row[f'followup_{r}'] for r in VOLUME_REGIONS]

            try:
                estimates = leaspy.estimate({str(subject_id): [target_age]}, ip)
                pred = estimates[str(subject_id)][0]
                # 反转回原始方向
                pred_corrected = []
                for j, region in enumerate(VOLUME_REGIONS):
                    v = pred[j]
                    if region != 'lateral_ventricle':
                        v = 1 - v
                    pred_corrected.append(v)
                all_preds.append(pred_corrected)
                all_targets.append(target_vols)
            except Exception:
                continue

    if not all_preds:
        return None

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    region_mae = {}
    for i, region in enumerate(VOLUME_REGIONS):
        region_mae[region] = float(np.mean(np.abs(preds[:, i] - targets[:, i])))

    overall_mae = float(np.mean(np.abs(preds - targets)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "overall_mae": overall_mae,
        "region_mae": region_mae,
        "r2_score": r2,
        "n_samples": len(all_preds),
    }


def main(args):
    print(f"[TPN Eval] 开始评估 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'eval.log')
    log_f = open(log_path, 'w', buffering=1)

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    # 加载数据
    df = pd.read_csv(args.dataset_csv)
    if 'split' in df.columns:
        test_df = df[df.split != 'train'].reset_index(drop=True)
    else:
        test_df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    log(f"[TPN Eval] Test samples: {len(test_df)}")

    # TPN 评估
    log("[TPN Eval] Evaluating TPN...")
    model = load_tpn(args.tpn_ckpt, device)
    tpn_results = evaluate_tpn_on_csv(model, test_df, device)

    log(f"[TPN Eval] TPN Overall MAE: {tpn_results['overall_mae']:.4f}")
    log(f"[TPN Eval] TPN R² Score: {tpn_results['r2_score']:.4f}")
    for region, mae in tpn_results['region_mae'].items():
        log(f"  {region}: MAE={mae:.4f}")

    # Leaspy 评估 (如果可用)
    leaspy_results = None
    if args.leaspy_dir and os.path.isdir(args.leaspy_dir):
        log("[TPN Eval] Evaluating Leaspy...")
        leaspy_results = evaluate_leaspy_on_csv(test_df, args.leaspy_dir)
        if leaspy_results:
            log(f"[TPN Eval] Leaspy Overall MAE: {leaspy_results['overall_mae']:.4f}")
            log(f"[TPN Eval] Leaspy R² Score: {leaspy_results['r2_score']:.4f}")
            for region, mae in leaspy_results['region_mae'].items():
                log(f"  {region}: MAE={mae:.4f}")

    # 对比摘要
    if leaspy_results:
        tpn_mae = tpn_results['overall_mae']
        leaspy_mae = leaspy_results['overall_mae']
        improvement = (leaspy_mae - tpn_mae) / leaspy_mae * 100
        log(f"\n=== 对比摘要 ===")
        log(f"TPN MAE: {tpn_mae:.4f} | Leaspy MAE: {leaspy_mae:.4f} | R²: {tpn_results['r2_score']:.4f}")
        log(f"MAE 改善: {improvement:+.2f}%")
        if tpn_mae < leaspy_mae:
            log(f"✅ TPN 优于 Leaspy ({improvement:.1f}% 更低的 MAE)")
        else:
            log(f"⚠️  TPN 劣于 Leaspy ({-improvement:.1f}% 更高的 MAE)")
    else:
        log(f"\nTPN MAE: {tpn_results['overall_mae']:.4f} | Leaspy MAE: N/A | R²: {tpn_results['r2_score']:.4f}")

    # 保存结果
    results = {
        "tpn": tpn_results,
        "leaspy": leaspy_results,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\nEvaluation complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TPN vs Leaspy')
    parser.add_argument('--dataset_csv', type=str, required=True)
    parser.add_argument('--tpn_ckpt', type=str, required=True, help='Path to trained TPN checkpoint')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--leaspy_dir', type=str, default=None, help='Directory with dcm_*.json Leaspy models')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
