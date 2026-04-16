#!/usr/bin/env python3
"""Check training_log.json state after multiple failed starts"""
import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

OUT = '/home/wangchong/data/fwz/output/37_expanded_validation'

for exp in ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']:
    print(f'\n=== {exp} ===')
    # Check current log
    _, out, _ = ssh.exec_command(f'cat {OUT}/{exp}/training_log.json 2>/dev/null', timeout=10)
    data = out.read().decode().strip()
    if data:
        h = json.loads(data)
        n = len(h.get('train_loss', []))
        print(f'  current: {n} epochs, val_loss={[f"{x:.4f}" for x in h["val_loss"]]}')
    else:
        print(f'  current: EMPTY/MISSING')
    
    # Check backup
    _, out, _ = ssh.exec_command(f'cat {OUT}/{exp}/training_log_before_outage.json 2>/dev/null', timeout=10)
    data2 = out.read().decode().strip()
    if data2:
        h2 = json.loads(data2)
        n2 = len(h2.get('train_loss', []))
        print(f'  backup:  {n2} epochs, val_loss={[f"{x:.4f}" for x in h2["val_loss"]]}')
    else:
        print(f'  backup:  NONE')
    
    # Check checkpoint
    _, out, _ = ssh.exec_command(f'ls -la {OUT}/{exp}/refnet-{exp}-best.pth 2>/dev/null', timeout=10)
    ckpt = out.read().decode().strip()
    print(f'  ckpt: {ckpt if ckpt else "MISSING"}')
    
    # Check resume log last lines
    _, out, _ = ssh.exec_command(f'tail -5 {OUT}/{exp}_resume_train.log 2>/dev/null', timeout=10)
    log = out.read().decode().strip()
    if log:
        print(f'  resume_log tail:\n    {log.replace(chr(10), chr(10) + "    ")}')

ssh.close()
