"""Find multi-timepoint subjects from prepared MCI CSV using days_from_first_visit."""
import paramiko

REMOTE_SCRIPT = r'''
/home/wangchong/miniconda3/envs/fwz/bin/python - <<'PY'
import pandas as pd
import numpy as np
import os

mci_csv = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
df = pd.read_csv(mci_csv)

# Reconstruct unique visits per subject using days_from_first_visit
subjects_timeline = {}
for _, row in df.iterrows():
    sid = row['subject_id']
    if sid not in subjects_timeline:
        subjects_timeline[sid] = {}
    
    s_days = int(round(row['starting_days_from_first_visit']))
    f_days = int(round(row['followup_days_from_first_visit']))
    s_age = row['starting_age']
    f_age = row['followup_age']
    s_lat = row['starting_latent']
    f_lat = row['followup_latent']
    s_img = row['starting_image']
    f_img = row['followup_image']
    
    # Use days as key (round to nearest 30 days to group nearby visits)
    s_key = round(s_days / 30) * 30
    f_key = round(f_days / 30) * 30
    
    if s_key not in subjects_timeline[sid]:
        subjects_timeline[sid][s_key] = {
            'days': s_days, 'age': s_age, 'latent': s_lat, 'image': s_img,
            'followup_days': f_days  # not needed just tracking
        }
    if f_key not in subjects_timeline[sid]:
        subjects_timeline[sid][f_key] = {
            'days': f_days, 'age': f_age, 'latent': f_lat, 'image': f_img,
            'followup_days': None
        }

# Find subjects with 3+ unique timepoints
multi_subjects = {}
for sid, timeline in subjects_timeline.items():
    if len(timeline) >= 3:
        sorted_visits = sorted(timeline.items(), key=lambda x: x[0])
        multi_subjects[sid] = sorted_visits

print(f"=== Multi-timepoint subjects ===")
print(f"Total subjects: {len(subjects_timeline)}")
print(f"3+ timepoints: {len([s for s in multi_subjects if len(multi_subjects[s]) >= 3])}")
print(f"4+ timepoints: {len([s for s in multi_subjects if len(multi_subjects[s]) >= 4])}")
print(f"5+ timepoints: {len([s for s in multi_subjects if len(multi_subjects[s]) >= 5])}")
print(f"6+ timepoints: {len([s for s in multi_subjects if len(multi_subjects[s]) >= 6])}")
print()

# Check split info
print("=== By split ===")
split_info = {}
for _, row in df.iterrows():
    split_info[row['subject_id']] = row['split']

for split in ['train', 'valid', 'test']:
    subjs = [s for s in multi_subjects if split_info.get(s) == split]
    print(f"{split}: {len(subjs)} multi-visit subjects")
    
print()

# Show multi-visit subjects with details
print("=== Multi-visit subjects details ===")
for sid in sorted(multi_subjects.keys(), key=lambda s: -len(multi_subjects[s])):
    visits = multi_subjects[sid]
    if len(visits) < 3:
        continue
    split = split_info.get(sid, '?')
    days_list = [v[1]['days'] for v in visits]
    ages_denorm = [v[1]['age'] * 100 for v in visits]
    latents_exist = [os.path.exists(v[1]['latent']) if isinstance(v[1]['latent'], str) else False for v in visits]
    images_exist = [os.path.exists(v[1]['image']) if isinstance(v[1]['image'], str) else False for v in visits]
    intervals = [f"{(days_list[i+1]-days_list[i])/30:.0f}mo" for i in range(len(days_list)-1)]
    total_span = (days_list[-1] - days_list[0]) / 365.25
    print(f"{sid} [{split}]: {len(visits)} visits, span={total_span:.1f}y")
    print(f"  ages: {[f'{a:.1f}' for a in ages_denorm]}")
    print(f"  days: {days_list}")
    print(f"  intervals: {intervals}")
    print(f"  latents: {latents_exist}")
    print(f"  images: {images_exist}")
    print()
    if sum(1 for s in multi_subjects if len(multi_subjects[s]) >= 3) > 20:
        break  # only show first ones

# Get all volume info for first multi-visit subject
regions = ['cerebral_cortex','hippocampus','amygdala','cerebral_white_matter','lateral_ventricle']
print("=== Volume trajectory example ===")
example_sid = list(multi_subjects.keys())[0]
example_visits = multi_subjects[example_sid]
for key, info in example_visits:
    subj_rows = df[(df['subject_id'] == example_sid)]
    # Get volumes from rows where this visit appears as starting or followup
    for _, row in subj_rows.iterrows():
        if abs(row['starting_days_from_first_visit'] - info['days']) < 30:
            vols = [row[f'starting_{r}'] for r in regions]
            print(f"  day={info['days']}: vols={[f'{v:.4f}' for v in vols]}")
            break
        elif abs(row['followup_days_from_first_visit'] - info['days']) < 30:
            vols = [row[f'followup_{r}'] for r in regions]
            print(f"  day={info['days']}: vols={[f'{v:.4f}' for v in vols]}")
            break
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
