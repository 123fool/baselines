#!/usr/bin/env python3
"""
查找 AD 转化患者 + 运行 pipeline + 下载结果

一键脚本：在服务器上查找合适的 AD 转化患者，运行预测 pipeline，
自动选择最佳候选人。

用法 (在服务器上运行):
  cd /home/wangchong/data/fwz/code/brlp_src
  python new/24_classification_animation/find_and_run_ad.py --gpu 1
"""

import os
import sys
import csv
import json
from collections import defaultdict
from pathlib import Path

BMCI_CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
DIAGNOSIS_MAP = {0.0: 'CN', 0.5: 'MCI', 1.0: 'AD'}

def find_ad_conversion_subjects():
    """在 B_mci.csv 中查找转化为 AD 的患者"""
    subjects = defaultdict(lambda: {'visits': {}, 'sex': None, 'has_ad': False, 'split': None})

    with open(BMCI_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r['subject_id']
            subjects[sid]['sex'] = r['sex']
            subjects[sid]['split'] = r['split']
            
            # Starting visit
            s_uid = r['starting_image_uid']
            s_diag = float(r['starting_diagnosis'])
            s_days = int(r['starting_days_from_first_visit'])
            subjects[sid]['visits'][s_uid] = {'diag': s_diag, 'days': s_days, 'uid': s_uid}
            if s_diag == 1.0:
                subjects[sid]['has_ad'] = True
            
            # Followup visit
            f_uid = r['followup_image_uid']
            f_diag = float(r['followup_diagnosis'])
            f_days = int(r['followup_days_from_first_visit'])
            subjects[sid]['visits'][f_uid] = {'diag': f_diag, 'days': f_days, 'uid': f_uid}
            if f_diag == 1.0:
                subjects[sid]['has_ad'] = True

    # 筛选: >=3 visits + 有 AD 诊断
    candidates = []
    for sid, info in subjects.items():
        if not info['has_ad']:
            continue
        
        visits_sorted = sorted(info['visits'].values(), key=lambda v: v['days'])
        n_visits = len(visits_sorted)
        diag_seq = [DIAGNOSIS_MAP.get(v['diag'], '?') for v in visits_sorted]
        days_range = visits_sorted[-1]['days'] - visits_sorted[0]['days']
        
        if n_visits >= 3:
            # 优先选 MCI→AD 转化的
            has_mci_to_ad = False
            for i in range(len(diag_seq) - 1):
                if diag_seq[i] == 'MCI' and diag_seq[i+1] == 'AD':
                    has_mci_to_ad = True
                    break
            
            candidates.append({
                'sid': sid,
                'n_visits': n_visits,
                'diag_seq': diag_seq,
                'diag_str': ' → '.join(diag_seq),
                'days_range': days_range,
                'split': info['split'],
                'has_mci_to_ad': has_mci_to_ad,
            })

    # 排序: MCI→AD 转化优先, 然后按 visit 数量
    candidates.sort(key=lambda x: (x['has_mci_to_ad'], x['n_visits'], x['days_range']), reverse=True)
    return candidates


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--list-only', action='store_true', help='只列出候选人，不运行 pipeline')
    parser.add_argument('--subject', type=str, default=None, help='指定患者 ID')
    args = parser.parse_args()

    print("=" * 70)
    print("查找 AD 转化患者")
    print("=" * 70)
    
    candidates = find_ad_conversion_subjects()
    
    print(f"\n共找到 {len(candidates)} 个 AD 转化候选人 (>=3 visits):\n")
    print(f"{'#':>3} {'Subject ID':<15} {'Visits':>6} {'Days':>6} {'Split':<6} {'MCI→AD':>6}  Trajectory")
    print("-" * 90)
    for i, s in enumerate(candidates[:20]):
        mci_ad = '★' if s['has_mci_to_ad'] else ''
        print(f"{i+1:>3} {s['sid']:<15} {s['n_visits']:>6} {s['days_range']:>6} {s['split']:<6} {mci_ad:>6}  {s['diag_str']}")
    
    if args.list_only:
        return
    
    # 选择最佳候选
    if args.subject:
        chosen = args.subject
        print(f"\n使用指定患者: {chosen}")
    else:
        # 自动选择: 优先 MCI→AD 转化 + visits 最多的
        chosen = candidates[0]['sid']
        chosen_info = candidates[0]
        print(f"\n自动选择: {chosen}")
        print(f"  Visits: {chosen_info['n_visits']}")
        print(f"  Trajectory: {chosen_info['diag_str']}")
        print(f"  Days range: {chosen_info['days_range']}")
    
    # 运行 pipeline
    print(f"\n{'='*70}")
    print(f"运行 pipeline: subject={chosen}, gpu={args.gpu}")
    print(f"{'='*70}\n")
    
    import subprocess
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(script_dir / 'run_pipeline.py'),
        '--gpu', str(args.gpu),
        '--subject', chosen,
        '--avg_n', '3',
    ]
    
    print(f"命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✅ Pipeline 完成!")
        print(f"结果目录: /home/wangchong/data/fwz/output/classification_animation/")
        print(f"\n下载命令 (在本地 Windows 执行):")
        print(f'  scp -P 2638 wangchong@10.96.27.109:/home/wangchong/data/fwz/output/classification_animation/{chosen}_* "C:\\Users\\PC\\Desktop\\baselines\\BrLP-main\\new\\24_classification_animation\\results_v3\\"')
    else:
        print(f"\n❌ Pipeline 失败 (exit code: {result.returncode})")


if __name__ == '__main__':
    main()
