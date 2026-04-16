import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    # Find ALL csv files
    'find /home/wangchong/data/fwz -name "*.csv" -type f 2>/dev/null',
    # Check what evaluate_btr.py logs used
    'find /home/wangchong/data/fwz/output -name "eval*.log" -type f 2>/dev/null | head -10',
    'head -20 /home/wangchong/data/fwz/output/innovation_2/eval.log 2>/dev/null || echo NO_EVAL_LOG',
    # Check brlp-data directory
    'ls -la /home/wangchong/data/fwz/brlp-data/ 2>/dev/null || echo NO_DIR',
    # Find latent files
    'find /home/wangchong/data/fwz -name "*.npz" -type f 2>/dev/null | head -5',
    # Check oasis data
    'ls /home/wangchong/data/fwz/oasis-eval-v2/ 2>/dev/null | head -10',
    'head -2 /home/wangchong/data/fwz/oasis-eval-v2/eval_results.csv 2>/dev/null || echo MISSING',
    # Find CSV with 'split' column (the dataset format we need)
    'find /home/wangchong/data/fwz -name "*.csv" -exec grep -l "split" {} \\; 2>/dev/null | head -10',
]
for cmd in cmds:
    _, stdout, stderr = c.exec_command(cmd, timeout=30, get_pty=True)
    out = stdout.read().decode().strip()
    print(f'CMD: {cmd[:120]}')
    print(f'  OUT: {out[:800]}')
    print()
c.close()
