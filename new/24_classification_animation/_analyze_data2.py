"""分析 B_mci.csv 中真正的 AD 转化信息"""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

analyze_script = r"""
import csv
from collections import defaultdict, Counter

csv_path = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 1. 检查 last_diagnosis 列值
last_diag_s = Counter(r['starting_last_diagnosis'] for r in rows)
last_diag_f = Counter(r['followup_last_diagnosis'] for r in rows)
print("starting_last_diagnosis values:", dict(last_diag_s))
print("followup_last_diagnosis values:", dict(last_diag_f))

# 2. 检查 mci_conversion_label 列值
conv_s = Counter(r['starting_mci_conversion_label'] for r in rows)
conv_f = Counter(r['followup_mci_conversion_label'] for r in rows)
print("\nstarting_mci_conversion_label:", dict(conv_s))
print("followup_mci_conversion_label:", dict(conv_f))

# 3. 检查 diagnosis 列的范围
diag_s = [float(r['starting_diagnosis']) for r in rows]
diag_f = [float(r['followup_diagnosis']) for r in rows]
print(f"\nstarting_diagnosis: min={min(diag_s):.4f}, max={max(diag_s):.4f}")
print(f"followup_diagnosis: min={min(diag_f):.4f}, max={max(diag_f):.4f}")

# 4. 检查 mci_conversion_prob 列
prob_s = [float(r['starting_mci_conversion_prob']) for r in rows if r['starting_mci_conversion_prob']]
prob_f = [float(r['followup_mci_conversion_prob']) for r in rows if r['followup_mci_conversion_prob']]
print(f"\nstarting_mci_conversion_prob: min={min(prob_s):.4f}, max={max(prob_s):.4f}")
print(f"followup_mci_conversion_prob: min={min(prob_f):.4f}, max={max(prob_f):.4f}")

# 5. 找 AD 转化患者 (last_diagnosis = 'AD' 或 1.0)
subjects = defaultdict(lambda: {'uids': set(), 'last_diags': set(), 'conv_labels': set(), 
                                  'diag_vals': [], 'rows': []})
for r in rows:
    sid = r['subject_id']
    subjects[sid]['uids'].add(r['starting_image_uid'])
    subjects[sid]['uids'].add(r['followup_image_uid'])
    subjects[sid]['last_diags'].add(r['starting_last_diagnosis'])
    subjects[sid]['last_diags'].add(r['followup_last_diagnosis'])
    subjects[sid]['conv_labels'].add(r['starting_mci_conversion_label'])
    subjects[sid]['conv_labels'].add(r['followup_mci_conversion_label'])
    subjects[sid]['diag_vals'].append(float(r['starting_diagnosis']))
    subjects[sid]['diag_vals'].append(float(r['followup_diagnosis']))
    subjects[sid]['rows'].append(r)

# 按诊断值最高的排序
print("\n" + "="*80)
print("按最高 diagnosis 值排名 (top 20):")
print("="*80)
ranked = sorted(subjects.items(), key=lambda x: max(x[1]['diag_vals']), reverse=True)
for sid, info in ranked[:20]:
    n_visits = len(info['uids'])
    max_diag = max(info['diag_vals'])
    min_diag = min(info['diag_vals'])
    last_d = info['last_diags']
    conv = info['conv_labels']
    print(f"  {sid}: {n_visits} visits, diag=[{min_diag:.3f}, {max_diag:.3f}], last_diag={last_d}, conv={conv}")

# 6. 找 last_diagnosis 为 Dementia/AD 的
print("\n" + "="*80)
print("有 'Dementia' 或 'AD' 在 last_diagnosis 中的 subjects:")
print("="*80)
for sid, info in subjects.items():
    for d in info['last_diags']:
        if 'dementia' in d.lower() or 'ad' in d.lower() or d == '1.0':
            n_visits = len(info['uids'])
            max_diag = max(info['diag_vals'])
            print(f"  {sid}: {n_visits} visits, max_diag={max_diag:.3f}, last_diag={info['last_diags']}")
            break

# 7. 找 conversion_label = 1 (或类似) 的
print("\n" + "="*80)
print("mci_conversion_label = 1 的 subjects (converted to AD):")
print("="*80)
for sid, info in subjects.items():
    if '1' in info['conv_labels'] or '1.0' in info['conv_labels'] or 'True' in info['conv_labels']:
        n_visits = len(info['uids'])
        max_diag = max(info['diag_vals'])
        print(f"  {sid}: {n_visits} visits, max_diag={max_diag:.3f}, conv={info['conv_labels']}")

# 8. 打印一行完整数据做参考
print("\n" + "="*80)
print("Sample row (first row with highest diag):")
print("="*80)
best_sid = ranked[0][0]
r = ranked[0][1]['rows'][0]
for k, v in r.items():
    print(f"  {k}: {v}")
"""

sftp = client.open_sftp()
with sftp.file('/tmp/analyze_bmci2.py', 'w') as f:
    f.write(analyze_script)
sftp.close()

cmd = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 /tmp/analyze_bmci2.py"
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    for line in err.split('\n'):
        if 'Warning' not in line and line.strip():
            print(f'[err] {line}')
client.close()
