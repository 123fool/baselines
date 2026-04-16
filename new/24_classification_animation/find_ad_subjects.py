#!/usr/bin/env python3
"""查找转化为 AD 的患者（有多次随访）"""
import csv
from collections import defaultdict

BMCI_CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
DIAGNOSIS_MAP = {0.0: 'CN', 0.5: 'MCI', 1.0: 'AD'}

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
        subjects[sid]['visits'][s_uid] = {'diag': s_diag, 'days': s_days}
        if s_diag == 1.0:
            subjects[sid]['has_ad'] = True
        
        # Followup visit
        f_uid = r['followup_image_uid']
        f_diag = float(r['followup_diagnosis'])
        f_days = int(r['followup_days_from_first_visit'])
        subjects[sid]['visits'][f_uid] = {'diag': f_diag, 'days': f_days}
        if f_diag == 1.0:
            subjects[sid]['has_ad'] = True

# 找转化为 AD 的患者（有 MCI→AD 或 CN→AD 的过程）
print("=" * 80)
print("转化为 AD 的患者（多次随访，且含 diagnosis=1.0 的 visit）:")
print("=" * 80)

ad_subjects = []
for sid, info in subjects.items():
    if not info['has_ad']:
        continue
    
    visits_sorted = sorted(info['visits'].values(), key=lambda v: v['days'])
    n_visits = len(visits_sorted)
    diag_seq = [DIAGNOSIS_MAP.get(v['diag'], '?') for v in visits_sorted]
    days_range = visits_sorted[-1]['days'] - visits_sorted[0]['days']
    
    # 必须有 >= 3 次随访 且 有至少1个 AD 诊断
    if n_visits >= 3:
        ad_subjects.append({
            'sid': sid,
            'n_visits': n_visits,
            'diag_seq': ' → '.join(diag_seq),
            'days_range': days_range,
            'split': info['split'],
        })

ad_subjects.sort(key=lambda x: x['n_visits'], reverse=True)

print(f"\n共找到 {len(ad_subjects)} 个 AD 转化患者 (>=3 visits):\n")
print(f"{'Subject ID':<15} {'Visits':>6} {'Days':>6} {'Split':<6}  Diagnosis Trajectory")
print("-" * 80)
for s in ad_subjects[:30]:
    print(f"{s['sid']:<15} {s['n_visits']:>6} {s['days_range']:>6} {s['split']:<6}  {s['diag_seq']}")

# 推荐最佳候选
print("\n\n推荐候选（visits 最多 + 有明确 MCI→AD 转化）:")
print("=" * 80)
for s in ad_subjects[:10]:
    if 'MCI' in s['diag_seq'] and 'AD' in s['diag_seq']:
        print(f"  ★ {s['sid']} — {s['n_visits']} visits, {s['days_range']} days, {s['diag_seq']}")
