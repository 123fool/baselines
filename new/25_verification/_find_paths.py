import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    # Find autoencoder checkpoint
    'find /home/wangchong/data/fwz -name "autoencoder*" -type f 2>/dev/null | head -10',
    # Find diffusion checkpoint  
    'find /home/wangchong/data/fwz -name "diffusion*" -type f 2>/dev/null | head -10',
    # Find dataset CSV
    'find /home/wangchong/data/fwz -name "*.csv" -type f 2>/dev/null | head -10',
    # List brlp-train dir
    'ls -la /home/wangchong/data/fwz/brlp-train/ 2>/dev/null || echo NO_DIR',
    # Find conda/python  
    'which python3 2>/dev/null; which python 2>/dev/null; ls /home/wangchong/anaconda3/envs/ 2>/dev/null || ls /home/wangchong/miniconda3/envs/ 2>/dev/null || echo NO_CONDA',
    # Check common conda locations
    'source /home/wangchong/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate fwz && python -c "import torch; print(torch.__version__)" || echo "conda_activate_failed"',
    'source /home/wangchong/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate fwz && python -c "import torch; print(torch.__version__)" || echo "miniconda_activate_failed"',
]
for cmd in cmds:
    _, stdout, stderr = c.exec_command(cmd, timeout=30, get_pty=True)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'CMD: {cmd[:120]}')
    print(f'  OUT: {out[:500]}')
    if err:
        print(f'  ERR: {err[:300]}')
    print()
c.close()
