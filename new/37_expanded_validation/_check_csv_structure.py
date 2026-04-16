#!/usr/bin/env python3
"""Check CSV structure for cross-validation / alternative test sets"""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

script = '''
import pandas as pd
import json

df = pd.read_csv('/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv')

print("=== SPLIT COUNTS ===")
print(df['split'].value_counts().to_string())
print(f"Total: {len(df)}")

print("\\n=== TEST SUBJECT IDS ===")
test_df = df[df['split'] == 'test']
print(test_df['subject_id'].tolist())

print("\\n=== VALID SUBJECT IDS ===")
valid_df = df[df['split'] == 'valid']
print(valid_df['subject_id'].tolist())

print("\\n=== UNIQUE SUBJECTS PER SPLIT ===")
for s in ['train', 'valid', 'test']:
    sub = df[df['split'] == s]
    print(f"  {s}: {len(sub)} rows, {sub['subject_id'].nunique()} unique subjects")

print("\\n=== ANY OVERLAP BETWEEN SPLITS? ===")
train_ids = set(df[df['split'] == 'train']['subject_id'])
valid_ids = set(df[df['split'] == 'valid']['subject_id'])
test_ids = set(df[df['split'] == 'test']['subject_id'])
print(f"  train & test overlap: {train_ids & test_ids}")
print(f"  train & valid overlap: {train_ids & valid_ids}")
print(f"  valid & test overlap: {valid_ids & test_ids}")

print("\\n=== FOLDER DISTRIBUTION ===")
print(df.groupby('split')['starting_folder'].value_counts().to_string())
'''

stdin, stdout, stderr = ssh.exec_command(
    f'/home/wangchong/miniconda3/envs/fwz/bin/python -c "{script}"',
    timeout=15
)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(out)
if err: print(f"[ERR] {err}")

ssh.close()
