#!/usr/bin/env python3
"""直接查看所有eval结果JSON的完整内容"""
import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

EVAL = '/home/wangchong/data/fwz/output/37_expanded_validation/eval'

for name in ['S35best_noref_50subj', 'S35best_noref_valid44', 
             'S36_RefC_H1a30_50subj', 'S36_RefC_H1a30_valid44']:
    _, out, _ = ssh.exec_command(f'cat {EVAL}/{name}.json', timeout=10)
    data = out.read().decode().strip()
    if data:
        r = json.loads(data)
        print(f"\n{'='*60}")
        print(f"=== {name} ===")
        # 打印所有key
        for k, v in sorted(r.items()):
            if isinstance(v, dict):
                print(f"\n  [{k}]:")
                for k2, v2 in sorted(v.items()):
                    if isinstance(v2, float):
                        print(f"    {k2}: {v2:.4f}")
                    else:
                        print(f"    {k2}: {v2}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, str) and len(v) > 100:
                print(f"  {k}: {v[:80]}...")
            else:
                print(f"  {k}: {v}")

ssh.close()
