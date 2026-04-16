"""分析 AD CSV 数据和 B_adni 详情"""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

analyze = r"""
import csv
from collections import defaultdict, Counter

# 1. ad_brlp_innovation.csv
print("="*60)
print("ad_brlp_innovation.csv:")
print("="*60)
csv_path = '/home/wangchong/data/fwz/data/diagnosis_categorized/ad_brlp_innovation.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Rows: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")

# Show sample row
print("\nSample row:")
for k, v in rows[0].items():
    print(f"  {k}: {v}")

# Subject stats
subjects = defaultdict(set)
for r in rows:
    sid = r['subject_id']
    if 'starting_image_uid' in r:
        subjects[sid].add(r['starting_image_uid'])
        subjects[sid].add(r['followup_image_uid'])
    elif 'image_uid' in r:
        subjects[sid].add(r['image_uid'])

print(f"\nSubjects: {len(subjects)}")
vc = Counter(len(v) for v in subjects.values())
print(f"Visit distribution: {dict(sorted(vc.items()))}")

# Top multi-visit subjects
multi = sorted(subjects.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("\nTop subjects by visits:")
for sid, uids in multi:
    print(f"  {sid}: {len(uids)} visits")

# 2. B_adni_from_processed.csv - AD subjects
print("\n" + "="*60)
print("B_adni_from_processed.csv (AD only):")
print("="*60)
csv_path = '/home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

ad_rows = [r for r in rows if r['starting_diagnosis'] == '1.0']
print(f"Total rows: {len(rows)}, AD rows: {len(ad_rows)}")

# AD sample
if ad_rows:
    print("\nSample AD row:")
    for k, v in ad_rows[0].items():
        print(f"  {k}: {v}")

# AD subjects with multiple visits
subjects = defaultdict(lambda: {'uids': set(), 'rows': []})
for r in ad_rows:
    sid = r['subject_id']
    subjects[sid]['uids'].add(r['starting_image_uid'])
    subjects[sid]['uids'].add(r['followup_image_uid'])
    subjects[sid]['rows'].append(r)

print(f"\nAD subjects: {len(subjects)}")
vc = Counter(len(s['uids']) for s in subjects.values())
print(f"Visit distribution: {dict(sorted(vc.items()))}")

multi = sorted(subjects.items(), key=lambda x: len(x[1]['uids']), reverse=True)[:10]
print("\nTop AD subjects by visits:")
for sid, info in multi:
    print(f"  {sid}: {len(info['uids'])} visits")
    # Show image paths
    r = info['rows'][0]
    print(f"    starting: {r['starting_image_path']}")
    print(f"    followup: {r['followup_image_path']}")

# 3. 检查 B_adni 是否有 volume 特征（hippocampus 等）
print("\n" + "="*60)
print("B_adni columns 是否包含 volume features:")
print("="*60)
cols = list(rows[0].keys())
vol_cols = [c for c in cols if any(x in c for x in ['hippocampus', 'amygdala', 'cerebral', 'lateral_ventricle', 'volume', 'latent'])]
print(f"Volume/latent columns: {vol_cols}")

# 4. 检查 ad_brlp_innovation.csv 是否有完整特征
print("\n" + "="*60)
print("ad_brlp_innovation.csv 是否包含所有需要的特征:")
print("="*60)
csv_path = '/home/wangchong/data/fwz/data/diagnosis_categorized/ad_brlp_innovation.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    ad_rows2 = list(reader)
cols2 = list(ad_rows2[0].keys())
needed = ['hippocampus', 'amygdala', 'cerebral_cortex', 'cerebral_white_matter', 'lateral_ventricle', 'latent', 'image', 'age', 'diagnosis']
for n in needed:
    matching = [c for c in cols2 if n in c.lower()]
    print(f"  {n}: {matching if matching else 'MISSING'}")

# Check latent path
for r in ad_rows2[:3]:
    if 'starting_latent' in r:
        print(f"\n  Starting latent: {r['starting_latent']}")
        print(f"  Followup latent: {r['followup_latent']}")
    elif 'latent' in r:
        print(f"  Latent: {r['latent']}")
"""

sftp = client.open_sftp()
with sftp.file('/tmp/analyze_ad2.py', 'w') as f:
    f.write(analyze)
sftp.close()

cmd = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 /tmp/analyze_ad2.py"
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    for line in err.split('\n'):
        if 'Warning' not in line and line.strip():
            print(f'[err] {line}')
client.close()
