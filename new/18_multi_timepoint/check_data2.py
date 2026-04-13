"""Check real time intervals using days_from_first_visit and original CSV."""
import paramiko

REMOTE_SCRIPT = r'''
/home/wangchong/miniconda3/envs/fwz/bin/python - <<'PY'
import pandas as pd
import numpy as np
import os

# 1. Check the original CSV with real ages
orig_csv = '/home/wangchong/data/fwz/data/diagnosis_categorized/mci_brlp_innovation_filtered.csv'
df_orig = pd.read_csv(orig_csv)
print("=== Original MCI CSV ===")
print(f"Rows: {len(df_orig)}, Subjects: {df_orig['subject_id'].nunique()}")
print(f"Age range: {df_orig['age'].min():.1f} - {df_orig['age'].max():.1f}")
print(f"Columns: {list(df_orig.columns)}")
print()

# Visits per subject in original data
visits = df_orig.groupby('subject_id').agg(
    n_visits=('age', 'count'),
    min_age=('age', 'min'),
    max_age=('age', 'max'),
    age_span=('age', lambda x: x.max() - x.min())
)
print("Visits per subject:")
for n in sorted(visits['n_visits'].unique()):
    cnt = (visits['n_visits'] == n).sum()
    if cnt > 0:
        print(f"  {n} visits: {cnt} subjects")
print()

# 2. Check the prepared CSV with days_from_first_visit
mci_csv = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
df = pd.read_csv(mci_csv)

# Real time gaps using days
days_gap = df['followup_days_from_first_visit'] - df['starting_days_from_first_visit']
years_gap = days_gap / 365.25
print("=== Time gaps in prepared pairs (days_from_first_visit) ===")
print(f"Days gap: mean={days_gap.mean():.0f}, std={days_gap.std():.0f}, min={days_gap.min():.0f}, max={days_gap.max():.0f}")
print(f"Years gap: mean={years_gap.mean():.2f}, std={years_gap.std():.2f}, min={years_gap.min():.2f}, max={years_gap.max():.2f}")
print()

# Distribution of time gaps
print("Time gap distribution:")
for months in [3, 6, 9, 12, 18, 24, 36, 48]:
    cnt = ((days_gap >= months*30 - 45) & (days_gap < months*30 + 45)).sum()
    print(f"  ~{months}mo ({months*30-45}-{months*30+45} days): {cnt} pairs")
print(f"  >4 years: {(days_gap > 1460).sum()} pairs")
print()

# 3. Multi-visit subjects with real time intervals
print("=== Multi-visit subjects (from original CSV) ===")
multi = visits[visits['n_visits'] >= 3].sort_values('n_visits', ascending=False)
print(f"3+ visit subjects: {len(multi)}")
print(f"4+ visit subjects: {(multi['n_visits'] >= 4).sum()}")
print(f"5+ visit subjects: {(multi['n_visits'] >= 5).sum()}")
print(f"6+ visit subjects: {(multi['n_visits'] >= 6).sum()}")
print()

# Show subjects with most visits
print("Top multi-visit subjects (real ages):")
for sid in multi.head(15).index:
    subj_data = df_orig[df_orig['subject_id'] == sid].sort_values('age')
    ages = subj_data['age'].tolist()
    intervals = [f"{ages[i+1]-ages[i]:.1f}y" for i in range(len(ages)-1)]
    latent_ok = all(os.path.exists(p) if isinstance(p, str) else False for p in subj_data['latent_path'])
    print(f"  {sid}: {len(ages)} visits, ages={[f'{a:.1f}' for a in ages]}, intervals={intervals}, latent={latent_ok}")

# 4. Check const.py for age normalization
print()
print("=== Age normalization check ===")
print(f"In prepared CSV: starting_age range = {df['starting_age'].min():.4f} - {df['starting_age'].max():.4f}")
print(f"In original CSV: age range = {df_orig['age'].min():.1f} - {df_orig['age'].max():.1f}")
# Check if normalized: (age - AGE_MIN) / (AGE_MAX - AGE_MIN) where AGE_MIN=0, AGE_MAX=100
print(f"Denormalized starting_ages range = {df['starting_age'].min()*100:.1f} - {df['starting_age'].max()*100:.1f}")
print()

# 5. Understand test split
test_subjs = df[df['split'] == 'test']['subject_id'].unique() if 'split' in df.columns else []
print(f"Test split subjects: {len(test_subjs)}")
test_multi = [s for s in test_subjs if s in multi.index]
print(f"Test subjects with 3+ visits: {len(test_multi)}")
for sid in test_multi[:10]:
    subj_data = df_orig[df_orig['subject_id'] == sid].sort_values('age')
    ages = subj_data['age'].tolist()
    span = ages[-1] - ages[0]
    print(f"  {sid}: {len(ages)} visits, ages={[f'{a:.1f}' for a in ages]}, span={span:.1f}y")
PY
'''

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')
stdin, stdout, stderr = ssh.exec_command(REMOTE_SCRIPT)
print(stdout.read().decode('utf-8', errors='replace'))
err = stderr.read().decode('utf-8', errors='replace')
if err.strip():
    print('ERR:', err[-2000:])
ssh.close()
