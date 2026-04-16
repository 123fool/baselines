import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    # Find diffusion model (different names)
    'find /home/wangchong/data/fwz -name "diff*" -type f 2>/dev/null | head -10',
    'find /home/wangchong/data/fwz/brlp-train -type f 2>/dev/null',
    'ls -la /home/wangchong/data/fwz/brlp-train/pretrained/',
    # Find dataset CSV used by existing eval scripts
    'find /home/wangchong/data/fwz -name "dataset*" -type f 2>/dev/null | head -10',
    'find /home/wangchong/data/fwz -name "input*csv" -type f 2>/dev/null | head -10',
    # Check what the existing evaluation scripts used
    'grep -r "dataset_csv\\|input.*csv" /home/wangchong/data/fwz/output/innovation_2/*.log 2>/dev/null | tail -5',
    # Check conda fwz has monai generative
    'source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python -c "from generative.networks.schedulers import DDIMScheduler; print(\'generative OK\')" 2>&1',
    # Check for the OASIS CSV
    'find /home/wangchong/data/fwz -name "*.csv" -type f 2>/dev/null | grep -i "oasis\\|input\\|dataset" | head -10',
]
for cmd in cmds:
    _, stdout, stderr = c.exec_command(cmd, timeout=30, get_pty=True)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'CMD: {cmd[:120]}')
    print(f'  OUT: {out[:600]}')
    if err:
        print(f'  ERR: {err[:300]}')
    print()
c.close()
