#!/usr/bin/env python3
"""Quick check: GPU status + process list + training logs"""
import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

# GPU status
print("=== GPU STATUS ===")
_, out, _ = ssh.exec_command('nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader', timeout=10)
print(out.read().decode().strip())

# All python processes
print("\n=== PYTHON PROCESSES ===")
_, out, _ = ssh.exec_command('ps aux | grep python | grep -v grep', timeout=10)
print(out.read().decode().strip() or '(none)')

# Check resume logs
print("\n=== RESUME LOGS (last 10 lines) ===")
for exp in ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']:
    print(f'\n--- {exp} ---')
    _, out, _ = ssh.exec_command(
        f'tail -20 /home/wangchong/data/fwz/output/37_expanded_validation/{exp}_resume_train.log 2>/dev/null',
        timeout=10)
    log = out.read().decode().strip()
    print(log if log else '(no log file)')

# Training log epoch counts
print("\n=== TRAINING HISTORY ===")
for exp in ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']:
    _, out, _ = ssh.exec_command(
        f'cat /home/wangchong/data/fwz/output/37_expanded_validation/{exp}/training_log.json 2>/dev/null',
        timeout=10)
    data = out.read().decode().strip()
    if data:
        hist = json.loads(data)
        n = len(hist.get('train_loss', []))
        print(f'{exp}: {n} epochs completed')
    else:
        print(f'{exp}: no training_log.json')

ssh.close()
