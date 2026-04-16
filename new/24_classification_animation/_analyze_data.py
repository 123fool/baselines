"""快速分析服务器上 B_mci.csv 数据"""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

# 直接用 awk 快速分析
cmds = [
    # 1. 总行数和列名
    "head -1 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv",
    "wc -l /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv",
    # 2. 统计 starting/followup diagnosis
    "awk -F, 'NR>1{print $3}' /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | sort | uniq -c",
    "awk -F, 'NR>1{print $6}' /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | sort | uniq -c",
]

labels = ["Columns (header):", "Row count:", "Starting diagnosis distribution:", "Followup diagnosis distribution:"]
for label, cmd in zip(labels, cmds):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    out = stdout.read().decode().strip()
    print(f"\n{label}")
    print(out)

# 3. 用 Python 做更详细分析
analyze_script = r"""
import csv
from collections import defaultdict, Counter

csv_path = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

cols = list(rows[0].keys())
print('All columns:', cols)

# 统计每个 subject 的 visit 数
subjects = defaultdict(lambda: {'uids': set(), 'diags': set(), 'rows': []})
for r in rows:
    sid = r['subject_id']
    subjects[sid]['uids'].add(r['starting_image_uid'])
    subjects[sid]['uids'].add(r['followup_image_uid'])
    subjects[sid]['diags'].add(r['starting_diagnosis'])
    subjects[sid]['diags'].add(r['followup_diagnosis'])
    subjects[sid]['rows'].append(r)

# Visit count distribution
vc = Counter(len(s['uids']) for s in subjects.values())
print('\nVisit count distribution:', dict(sorted(vc.items())))

# Subjects with AD (diagnosis = 1.0)
ad_subs = {}
for sid, info in subjects.items():
    if '1.0' in info['diags'] or '1' in info['diags']:
        ad_subs[sid] = len(info['uids'])

print(f'\nSubjects with AD: {len(ad_subs)}')
for sid, nv in sorted(ad_subs.items(), key=lambda x: x[1], reverse=True)[:15]:
    diags = subjects[sid]['diags']
    print(f'  {sid}: {nv} visits, diags={diags}')

# 降低门槛: >=2 visits + has AD
ad2 = {sid: nv for sid, nv in ad_subs.items() if nv >= 2}
print(f'\nAD subjects with >=2 visits: {len(ad2)}')
for sid, nv in sorted(ad2.items(), key=lambda x: x[1], reverse=True)[:15]:
    diags = subjects[sid]['diags']
    print(f'  {sid}: {nv} visits, diags={diags}')
"""

# 上传并执行
sftp = client.open_sftp()
with sftp.file('/tmp/analyze_bmci.py', 'w') as f:
    f.write(analyze_script)
sftp.close()

cmd = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 /tmp/analyze_bmci.py"
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print("\n" + "="*60)
print("详细分析:")
print("="*60)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    for line in err.split('\n'):
        if 'Warning' not in line and line.strip():
            print(f'[err] {line}')

client.close()
