#!/usr/bin/env python3
"""Check training checkpoints and more shutdown details"""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

cmds = [
    # Check all checkpoints saved
    'echo "=== CHECKPOINTS ==="; for d in RefC_v2_cont RefC_v2_fresh RefD_v2_highnoise; do echo "--- $d ---"; ls -lh /home/wangchong/data/fwz/output/37_expanded_validation/$d/*.pth 2>/dev/null || echo "  no checkpoints"; done',
    # Check training_log.json existence and content
    'echo "=== TRAINING LOGS ==="; for d in RefC_v2_cont RefC_v2_fresh RefD_v2_highnoise; do echo "--- $d ---"; cat /home/wangchong/data/fwz/output/37_expanded_validation/$d/training_log.json 2>/dev/null || echo "  no training_log.json"; done',
    # Check the EXACT shutdown - last journal entry timestamp
    'echo "=== LAST JOURNAL ENTRIES BOOT -1 ==="; journalctl -b -1 -n 5 --no-pager 2>/dev/null',
    # Check if there's a scheduled shutdown via cron
    'echo "=== CRONTAB ==="; crontab -l 2>/dev/null || echo "no crontab"',
    # Check /etc/crontab for system-wide scheduled events
    'echo "=== SYSTEM CRONTAB ==="; grep -v "^#" /etc/crontab 2>/dev/null',
    # Check anacron or systemd timers
    'echo "=== SYSTEMD TIMERS ==="; systemctl list-timers --all 2>/dev/null | head -15 || echo "cannot list timers"',
    # Was it a normal shutdown (power button pressed?) - check ACPI
    'echo "=== ACPI EVENTS ==="; journalctl -b -1 --no-pager 2>/dev/null | grep -iE "acpi|button|lid|power.button" | tail -5',
    # Test CSV columns for understanding test set structure
    '/home/wangchong/miniconda3/envs/fwz/bin/python -c "import pandas as pd; df=pd.read_csv(\'/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv\'); test_df=df[df[\'split\']==\'test\']; print(f\'Test set: {len(test_df)} subjects\'); print(f\'Columns: {list(df.columns)[:10]}\'); print(test_df.head(3).to_string())" 2>/dev/null || echo "CSV check failed"',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'\n{out}')
    if err and 'Hint:' not in err and 'adm' not in err: 
        print(f'[ERR] {err}')

ssh.close()
