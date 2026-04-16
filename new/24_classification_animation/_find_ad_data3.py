"""查找 AD 纵向患者数据和 B_adni CSV"""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

def run(cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode().strip()

# 1. ad_longitudinal 结构
print("="*60)
print("1. ad_longitudinal 目录前10个患者:")
print("="*60)
out = run("ls /home/wangchong/data/fwz/data/ad_longitudinal/ | head -10")
print(out)

# 每个患者有什么
first_ad = run("ls /home/wangchong/data/fwz/data/ad_longitudinal/ | head -1")
print(f"\n第一个患者 {first_ad} 的内容:")
out = run(f"ls -la /home/wangchong/data/fwz/data/ad_longitudinal/{first_ad}/")
print(out)

# 看它的 visit 内容
first_visit = run(f"ls /home/wangchong/data/fwz/data/ad_longitudinal/{first_ad}/ | head -1")
out = run(f"ls -la /home/wangchong/data/fwz/data/ad_longitudinal/{first_ad}/{first_visit}/")
print(f"\nVisit {first_visit} 内容:")
print(out)

# 检查有 latent 文件吗
out = run(f"find /home/wangchong/data/fwz/data/ad_longitudinal/{first_ad}/ -name '*latent*' -type f")
print(f"\nLatent 文件:")
print(out if out else "(none)")

# 有 synthseg 吗
out = run(f"find /home/wangchong/data/fwz/data/ad_longitudinal/{first_ad}/ -name 'synthseg*' -type f")
print(f"\nSynthseg 文件:")
print(out if out else "(none)")

# 2. B_adni_from_processed.csv
print("\n" + "="*60)
print("2. B_adni_from_processed.csv:")
print("="*60)
out = run("head -1 /home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv")
print("Columns:", out[:500])
nrows = run("wc -l < /home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv")
print(f"Rows: {nrows}")

# 检查 diagnosis 分布
analyze_script = r"""
import csv
from collections import Counter, defaultdict

# B_adni
csv_path = '/home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

cols = list(rows[0].keys())
print('Columns:', cols[:20], '...')

# 找 diagnosis 相关列
diag_cols = [c for c in cols if 'diagnosis' in c.lower() or 'diag' in c.lower() or 'label' in c.lower() or 'folder' in c.lower()]
print('Diagnosis-like columns:', diag_cols)

for dc in diag_cols:
    vals = Counter(r[dc] for r in rows)
    print(f'\n  {dc}: {dict(vals)}')

# 统计 subject folders
if 'starting_folder' in cols:
    folders = Counter(r['starting_folder'] for r in rows)
    print(f'\nstarting_folder: {dict(folders)}')

# 找 AD 相关
subjects = defaultdict(lambda: {'uids': set(), 'folders': set(), 'diags': set()})
for r in rows:
    sid = r['subject_id']
    subjects[sid]['uids'].add(r.get('starting_image_uid', ''))
    subjects[sid]['uids'].add(r.get('followup_image_uid', ''))
    if 'starting_folder' in r:
        subjects[sid]['folders'].add(r['starting_folder'])
        subjects[sid]['folders'].add(r.get('followup_folder', ''))
    if 'starting_diagnosis' in r:
        subjects[sid]['diags'].add(r['starting_diagnosis'])
    if 'followup_diagnosis' in r:
        subjects[sid]['diags'].add(r['followup_diagnosis'])

# AD folder subjects
ad_subs = {sid: info for sid, info in subjects.items() 
           if any('ad' in f.lower() for f in info['folders'])}
print(f'\nSubjects from AD folders: {len(ad_subs)}')

# AD subjects with >= 3 visits
ad_multi = {sid: info for sid, info in ad_subs.items() if len(info['uids']) >= 3}
print(f'AD subjects with >= 3 visits: {len(ad_multi)}')
for sid, info in sorted(ad_multi.items(), key=lambda x: len(x[1]['uids']), reverse=True)[:15]:
    print(f'  {sid}: {len(info["uids"])} visits, folders={info["folders"]}, diags={info["diags"]}')

# If no multi-visit AD, show >=2
if not ad_multi:
    ad_two = {sid: info for sid, info in ad_subs.items() if len(info['uids']) >= 2}
    print(f'AD subjects with >= 2 visits: {len(ad_two)}')
    for sid, info in sorted(ad_two.items(), key=lambda x: len(x[1]['uids']), reverse=True)[:15]:
        print(f'  {sid}: {len(info["uids"])} visits, folders={info["folders"]}, diags={info["diags"]}')
"""

sftp = client.open_sftp()
with sftp.file('/tmp/analyze_ad.py', 'w') as f:
    f.write(analyze_script)
sftp.close()

cmd = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 /tmp/analyze_ad.py"
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    for line in err.split('\n'):
        if 'Warning' not in line and line.strip():
            print(f'[err] {line}')

# 3. diagnosis_categorized 目录
print("\n" + "="*60)
print("3. diagnosis_categorized 目录:")
print("="*60)
out = run("ls -la /home/wangchong/data/fwz/data/diagnosis_categorized/")
print(out)
out = run("find /home/wangchong/data/fwz/data/diagnosis_categorized/ -name '*.csv' -o -name '*.json' | head -10")
print("\nCSV/JSON:", out if out else "(none)")

# 4. A_mci.csv 中有 AD 吗
print("\n" + "="*60)
print("4. A_mci.csv 诊断分布:")
print("="*60)
a_script = r"""
import csv
from collections import Counter
with open('/home/wangchong/data/fwz/output/innovation_5/prepared/A_mci.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
diag_cols = [c for c in rows[0].keys() if 'diagnosis' in c.lower()]
for dc in diag_cols:
    vals = Counter(r[dc] for r in rows)
    print(f'{dc}: {dict(vals)}')
"""
sftp = client.open_sftp()
with sftp.file('/tmp/analyze_amci.py', 'w') as f:
    f.write(a_script)
sftp.close()

cmd = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 /tmp/analyze_amci.py"
stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
print(stdout.read().decode())

client.close()
