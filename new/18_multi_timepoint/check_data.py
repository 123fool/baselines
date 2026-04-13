"""Check multi-visit subjects in the dataset for temporal sequence evaluation."""
import paramiko

REMOTE_SCRIPT = r'''
/home/wangchong/miniconda3/envs/fwz/bin/python - <<'PY'
import pandas as pd
import numpy as np
import os

# Load all diagnosis CSVs
data_dir = '/home/wangchong/data/fwz/data/diagnosis_categorized'
mci_csv = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'

# 1. Check the prepared MCI CSV structure
df = pd.read_csv(mci_csv)
print("=== Prepared MCI CSV ===")
print(f"Rows: {len(df)}, Unique subjects: {df['subject_id'].nunique()}")
print(f"Columns with 'age': {[c for c in df.columns if 'age' in c.lower()]}")
print(f"Columns with 'visit': {[c for c in df.columns if 'visit' in c.lower()]}")
print()

# How many visits per subject?
visits_per_subj = df.groupby('subject_id').size()
print(f"Pairs per subject distribution:")
for n in sorted(visits_per_subj.unique()):
    count = (visits_per_subj == n).sum()
    print(f"  {n} pairs: {count} subjects")
print()

# Age gaps
age_gap = df['followup_age'] - df['starting_age']
print(f"Age gap: mean={age_gap.mean():.3f}y, std={age_gap.std():.3f}y, min={age_gap.min():.3f}y, max={age_gap.max():.3f}y")
print()

# 2. Look at the original full dataset for multi-visit subjects
for fname in os.listdir(data_dir):
    if fname.endswith('.csv'):
        full = pd.read_csv(os.path.join(data_dir, fname))
        print(f"=== {fname} ===")
        print(f"Rows: {len(full)}")
        print(f"Columns: {list(full.columns[:10])}...")
        break

# 3. Check if we can get subjects with 3+ actual scan visits
# Each row in B_mci.csv is a (starting, followup) pair
# A subject with 3 visits would have pairs: (v1,v2), (v1,v3), (v2,v3) etc.
# Let's find subjects with enough unique timepoints
subjects_visits = {}
for _, row in df.iterrows():
    sid = row['subject_id']
    if sid not in subjects_visits:
        subjects_visits[sid] = set()
    subjects_visits[sid].add(round(row['starting_age'], 2))
    subjects_visits[sid].add(round(row['followup_age'], 2))

multi_visit = {s: sorted(v) for s, v in subjects_visits.items() if len(v) >= 3}
print(f"\n=== Multi-visit subjects (3+ unique ages) ===")
print(f"Total: {len(multi_visit)}")
for n_visits in [3, 4, 5, 6, 7, 8]:
    count = sum(1 for v in multi_visit.values() if len(v) == n_visits)
    if count > 0:
        print(f"  {n_visits} visits: {count} subjects")

# Show example subjects
print("\nExample multi-visit subjects:")
for sid, ages in list(multi_visit.items())[:5]:
    age_span = ages[-1] - ages[0]
    print(f"  {sid}: ages={[f'{a:.1f}' for a in ages]}, span={age_span:.1f}y")

# 4. Check which subjects have latent files
print("\n=== Latent file availability for multi-visit subjects ===")
latent_dir = '/home/wangchong/data/fwz/data'
# Get all unique starting/followup latent paths
latent_cols = [c for c in df.columns if 'latent' in c.lower() and 'exists' not in c.lower()]
print(f"Latent columns: {latent_cols}")

# Check a sample of latent paths
sample_rows = df[df['subject_id'].isin(list(multi_visit.keys())[:3])]
for _, row in sample_rows.head(6).iterrows():
    sl = row.get('starting_latent', 'N/A')
    fl = row.get('followup_latent', 'N/A')
    sl_exists = os.path.exists(sl) if isinstance(sl, str) and sl != 'N/A' else False
    fl_exists = os.path.exists(fl) if isinstance(fl, str) and fl != 'N/A' else False
    print(f"  {row['subject_id']}: s_age={row['starting_age']:.1f} f_age={row['followup_age']:.1f} s_lat={sl_exists} f_lat={fl_exists}")

# 5. Find subjects with 4+ visits that have all latents
print("\n=== Best candidates (4+ visits, all latents exist) ===")
good_subjects = []
for sid, ages in multi_visit.items():
    if len(ages) < 4:
        continue
    subj_rows = df[df['subject_id'] == sid]
    all_latents = set()
    for _, row in subj_rows.iterrows():
        sl = row.get('starting_latent', '')
        fl = row.get('followup_latent', '')
        if isinstance(sl, str) and os.path.exists(sl):
            all_latents.add((round(row['starting_age'], 2), sl))
        if isinstance(fl, str) and os.path.exists(fl):
            all_latents.add((round(row['followup_age'], 2), fl))
    if len(all_latents) >= 4:
        good_subjects.append((sid, sorted(all_latents)))

print(f"Subjects with 4+ visits and all latents: {len(good_subjects)}")
for sid, visits in good_subjects[:10]:
    ages_str = [f"{a:.1f}" for a, _ in visits]
    print(f"  {sid}: {len(visits)} visits, ages={ages_str}")
PY
'''

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')
stdin, stdout, stderr = ssh.exec_command(REMOTE_SCRIPT)
print(stdout.read().decode('utf-8', errors='replace'))
err = stderr.read().decode('utf-8', errors='replace')
if err.strip():
    print('ERR:', err[-1000:])
ssh.close()
