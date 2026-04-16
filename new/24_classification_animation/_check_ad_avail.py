"""检查 AD 患者的 latent/volume 数据可用性"""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

check = r"""
import os, csv
from collections import defaultdict

# 使用 ad_brlp_innovation.csv (单行格式)
csv_path = '/home/wangchong/data/fwz/data/diagnosis_categorized/ad_brlp_innovation.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 按 subject 分组
subjects = defaultdict(list)
for r in rows:
    subjects[r['subject_id']].append(r)

# 只看 4-visit subjects
target_sids = [sid for sid, visits in subjects.items() if len(visits) >= 4]
print(f"4-visit AD subjects: {target_sids}")

for sid in target_sids:
    visits = sorted(subjects[sid], key=lambda x: x['days_from_first_visit'])
    print(f"\n{'='*60}")
    print(f"Subject: {sid} ({len(visits)} visits)")
    print(f"{'='*60}")
    for v in visits:
        img_path = v['image_path']
        latent_path = v['latent_path']
        segm_path = v['segm_path']
        img_exists = os.path.exists(img_path)
        lat_exists = os.path.exists(latent_path)
        seg_exists = os.path.exists(segm_path)
        print(f"  Visit {v['visit_order']} ({v['visit_date']}, day {v['days_from_first_visit']}):")
        print(f"    Image:  {'OK' if img_exists else 'MISSING'} - {img_path}")
        print(f"    Latent: {'OK' if lat_exists else 'MISSING'} - {latent_path}")
        print(f"    Segm:   {'OK' if seg_exists else 'MISSING'} - {segm_path}")
        print(f"    Age: {v['age']}, Sex: {v['sex']}, Diag: {v['diagnosis']}")

# Also check B_adni paths
print(f"\n{'='*60}")
print("B_adni AD data paths:")
print(f"{'='*60}")
b_csv = '/home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv'
with open(b_csv) as f:
    reader = csv.DictReader(f)
    b_rows = list(reader)

b_ad = [r for r in b_rows if r['starting_diagnosis'] == '1.0']
b_subs = defaultdict(list)
for r in b_ad:
    b_subs[r['subject_id']].append(r)

for sid in target_sids:
    if sid in b_subs:
        print(f"\n{sid} in B_adni:")
        for r in b_subs[sid][:2]:
            s_exists = os.path.exists(r['starting_image_path'])
            f_exists = os.path.exists(r['followup_image_path'])
            print(f"  starting: {'OK' if s_exists else 'MISS'} {r['starting_image_path']}")
            print(f"  followup: {'OK' if f_exists else 'MISS'} {r['followup_image_path']}")
            print(f"  age: {r['starting_age']} -> {r['followup_age']}")

# Check also adni-ad-data dir
print(f"\n{'='*60}")
print("adni-ad-data directory check:")
print(f"{'='*60}")
for sid in target_sids:
    adni_dir = f"/home/wangchong/data/fwz/adni-ad-data/{sid}"
    ad_long_dir = f"/home/wangchong/data/fwz/data/ad_longitudinal/{sid}"
    print(f"\n{sid}:")
    for d in [adni_dir, ad_long_dir]:
        if os.path.isdir(d):
            subdirs = sorted(os.listdir(d))
            print(f"  {d}: {subdirs}")
            for sd in subdirs:
                files = os.listdir(os.path.join(d, sd))
                print(f"    {sd}/: {files}")
        else:
            print(f"  {d}: NOT FOUND")
"""

sftp = client.open_sftp()
with sftp.file('/tmp/check_ad_data.py', 'w') as f:
    f.write(check)
sftp.close()

cmd = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 /tmp/check_ad_data.py"
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    for line in err.split('\n'):
        if 'Warning' not in line and line.strip():
            print(f'[err] {line}')
client.close()
