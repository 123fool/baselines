"""Find MCI->AD converters from ADNI MCI CSV."""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open(r'E:\ADNI\MCI\MCI_all_timepoint_standardized_latest.csv', encoding='utf-8-sig')))
# Fix quoted column names
fixed_rows = []
for r in rows:
    fixed = {}
    for k, v in r.items():
        key = k.strip('"')
        fixed[key] = v.strip('"') if v else v
    fixed_rows.append(fixed)
rows = fixed_rows

subjects = defaultdict(list)
for r in rows:
    subjects[r['PTID']].append(r)

converters = []
for ptid, visits in subjects.items():
    visits_sorted = sorted(visits, key=lambda x: int(x['DeltaDays']) if x['DeltaDays'] else 0)
    diags = [v['DIAGNOSIS'] for v in visits_sorted]
    if '2' in diags and '3' in diags:
        first_mci = next(i for i, d in enumerate(diags) if d == '2')
        first_ad = next(i for i, d in enumerate(diags) if d == '3')
        if first_mci < first_ad:
            converters.append({
                'ptid': ptid,
                'n_visits': len(visits_sorted),
                'mci_visits': sum(1 for d in diags if d == '2'),
                'ad_visits': sum(1 for d in diags if d == '3'),
                'dates': [v['TimepointDate'] for v in visits_sorted],
                'diags': diags,
                'viscodes': [v['VISCODE'] for v in visits_sorted],
                'delta_days': [int(v['DeltaDays']) for v in visits_sorted],
            })

converters.sort(key=lambda x: x['n_visits'], reverse=True)
print(f'MCI->AD converters: {len(converters)}')
print('Top 20 by visit count:')
for c in converters[:20]:
    print(f"  {c['ptid']}: {c['n_visits']} visits (MCI:{c['mci_visits']}, AD:{c['ad_visits']}), viscodes={c['viscodes']}, diags={c['diags']}")
    
# Check which have images on local disk (E:\ADNI\MCI\by_subject)
import os
print('\n--- Check local image availability ---')
for c in converters[:20]:
    subdir = os.path.join(r'E:\ADNI\MCI\by_subject', c['ptid'])
    if os.path.exists(subdir):
        contents = os.listdir(subdir)
        print(f"  {c['ptid']}: LOCAL EXISTS, files={len(contents)}, items={contents[:5]}")
    else:
        print(f"  {c['ptid']}: NO LOCAL DATA")
