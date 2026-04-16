#!/usr/bin/env python3
"""Investigate server shutdown reason"""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

cmds = [
    # Last journal entries before shutdown (boot -1)
    'journalctl -b -1 --since "2026-04-16 00:25" --no-pager 2>/dev/null | tail -50',
    # Shutdown/poweroff/reboot related entries
    'journalctl -b -1 --no-pager 2>/dev/null | grep -iE "shutdown|poweroff|halt|reboot|power.off" | tail -15',
    # Auth log: who triggered shutdown
    'grep -iE "shutdown|poweroff|halt" /var/log/auth.log 2>/dev/null | tail -10 || echo "no-auth-log-match"',
    # Syslog around shutdown time
    'grep "Apr 16 00:2[5-9]\\|Apr 16 00:3[0-5]" /var/log/syslog.1 2>/dev/null | tail -30 || echo "no-syslog-match"',
    # Check if it was a kernel panic (MCE/panic/OOM)
    'journalctl -b -1 --no-pager 2>/dev/null | grep -iE "panic|oom|mce|error.*hardware|temperature|thermal" | tail -10',
    # Last logged-in users around that time
    'last -t 20260416003030 | head -10',
    # Check training logs - did they finish or get interrupted?
    'tail -5 /home/wangchong/data/fwz/output/37_expanded_validation/RefC_v2_cont_train.log 2>/dev/null',
    'tail -5 /home/wangchong/data/fwz/output/37_expanded_validation/RefC_v2_fresh_train.log 2>/dev/null',
    'tail -5 /home/wangchong/data/fwz/output/37_expanded_validation/RefD_v2_highnoise_train.log 2>/dev/null',
    # Check training_log.json for last epoch
    'python3 -c "import json; d=json.load(open(\'/home/wangchong/data/fwz/output/37_expanded_validation/RefC_v2_cont/training_log.json\')); print(f\'RefC_v2_cont: {len(d)} epochs, last: {d[-1]}\')" 2>/dev/null || echo "no RefC_v2_cont log"',
    'python3 -c "import json; d=json.load(open(\'/home/wangchong/data/fwz/output/37_expanded_validation/RefC_v2_fresh/training_log.json\')); print(f\'RefC_v2_fresh: {len(d)} epochs, last: {d[-1]}\')" 2>/dev/null || echo "no RefC_v2_fresh log"',
    'python3 -c "import json; d=json.load(open(\'/home/wangchong/data/fwz/output/37_expanded_validation/RefD_v2_highnoise/training_log.json\')); print(f\'RefD_v2_highnoise: {len(d)} epochs, last: {d[-1]}\')" 2>/dev/null || echo "no RefD_v2_highnoise log"',
    # Check if any eval results exist for v2 models
    'ls -la /home/wangchong/data/fwz/output/37_expanded_validation/eval/ 2>/dev/null',
    # Check GPU status now
    'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null',
    # Check if any training processes survived (they wouldn't after reboot)
    'ps aux | grep -E "train_refinement|evaluate_refinement" | grep -v grep',
    # CSV split info
    '/home/wangchong/miniconda3/envs/fwz/bin/python -c "import pandas as pd; df=pd.read_csv(\'/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv\'); print(df[\'split\'].value_counts()); print(f\'\\nTotal: {len(df)}\')"',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    label = cmd[:90] + ('...' if len(cmd) > 90 else '')
    print(f'\n=== {label} ===')
    if out: print(out)
    if err and 'Hint:' not in err and 'adm' not in err: print(f'[ERR] {err}')

ssh.close()
