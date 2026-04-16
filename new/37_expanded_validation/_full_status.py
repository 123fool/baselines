#!/usr/bin/env python3
"""Comprehensive status check: training + evaluation"""
import paramiko, json, os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

OUT = '/home/wangchong/data/fwz/output/37_expanded_validation'
EVAL = f'{OUT}/eval'

# GPU status
print("=" * 60)
print("GPU STATUS")
print("=" * 60)
_, out, _ = ssh.exec_command('nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader', timeout=10)
print(out.read().decode().strip())

# Training progress
print("\n" + "=" * 60)
print("TRAINING PROGRESS")
print("=" * 60)
for exp in ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']:
    _, out, _ = ssh.exec_command(f'cat {OUT}/{exp}/training_log.json 2>/dev/null', timeout=10)
    data = out.read().decode().strip()
    if data:
        h = json.loads(data)
        n = len(h.get('train_loss', []))
        # Check if "resumed" history (starts fresh)
        vl = h.get('val_loss', [])
        tl = h.get('train_loss', [])
        print(f'{exp}: {n} epochs')
        if vl:
            print(f'  val_loss: {[f"{x:.4f}" for x in vl[-5:]]}')
            print(f'  best val: {min(vl):.4f}')
    else:
        print(f'{exp}: no log yet')

# Training processes
print("\n" + "=" * 60)
print("TRAINING PROCESSES")
print("=" * 60)
_, out, _ = ssh.exec_command('ps aux | grep train_refinement_v2 | grep python | grep -v grep', timeout=10)
procs = out.read().decode().strip()
if procs:
    for line in procs.split('\n'):
        parts = line.split()
        pid = parts[1]
        cpu = parts[2]
        # Extract exp_name
        if '--exp_name' in line:
            idx = line.index('--exp_name') + len('--exp_name')
            name = line[idx:].split()[0]
        else:
            name = '?'
        print(f'  PID {pid}: {name} (CPU: {cpu}%)')
else:
    print('  NO TRAINING PROCESSES!')

# Eval progress
print("\n" + "=" * 60)
print("EVALUATION PROGRESS")
print("=" * 60)
_, out, _ = ssh.exec_command(f'ls {EVAL}/*.json 2>/dev/null', timeout=10)
files = out.read().decode().strip().split('\n')
for f in sorted(files):
    if not f or '_progress' in f:
        continue
    fname = os.path.basename(f)
    _, out2, _ = ssh.exec_command(f'cat {f}', timeout=10)
    data = out2.read().decode().strip()
    if data:
        try:
            r = json.loads(data)
            if 'results' in r:
                overall = r['results'].get('overall_mean_dice', '?')
                ad_comp = r['results'].get('AD_composite_mean', '?')
                print(f'{fname}: Overall={overall:.4f}, AD-Comp={ad_comp:.4f}')
            else:
                print(f'{fname}: {data[:100]}...')
        except:
            print(f'{fname}: (parse error)')
    else:
        print(f'{fname}: EMPTY')

# Check eval progress files
_, out, _ = ssh.exec_command(f'ls {EVAL}/*_progress.json 2>/dev/null', timeout=10)
pfiles = out.read().decode().strip().split('\n')
for f in sorted(pfiles):
    if not f:
        continue
    fname = os.path.basename(f)
    _, out2, _ = ssh.exec_command(f'cat {f}', timeout=10)
    data = out2.read().decode().strip()
    if data:
        try:
            p = json.loads(data)
            done = p.get('completed', 0)
            total = p.get('total', '?')
            print(f'  {fname}: {done}/{total} subjects done')
        except:
            print(f'  {fname}: {data[:80]}')

# Eval processes
_, out, _ = ssh.exec_command('ps aux | grep evaluate_refinement_v2 | grep python | grep -v grep', timeout=10)
eprocs = out.read().decode().strip()
if eprocs:
    for line in eprocs.split('\n'):
        parts = line.split()
        pid = parts[1]
        cpu = parts[2]
        if '--label' in line:
            idx = line.index('--label') + len('--label')
            name = line[idx:].split()[0]
        else:
            name = '?'
        print(f'  PID {pid}: {name} (CPU: {cpu}%)')
else:
    print('  No eval processes running')

# Eval logs - check for errors
print("\n" + "=" * 60)
print("EVAL LOGS (errors check)")
print("=" * 60)
for label in ['S36_RefC_H1a30_valid44', 'S35best_noref_valid44']:
    _, out, _ = ssh.exec_command(f'tail -5 {EVAL}/{label}.log 2>/dev/null', timeout=10)
    log = out.read().decode().strip()
    if log:
        print(f'\n--- {label} ---')
        print(log)

ssh.close()
