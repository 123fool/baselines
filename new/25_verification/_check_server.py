import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    'ls -la /home/wangchong/data/fwz/brlp-train/autoencoder-ep-2.pth 2>/dev/null || echo MISSING',
    'ls -la /home/wangchong/data/fwz/brlp-train/diffusion-ep-63.pth 2>/dev/null || echo MISSING',
    'ls /home/wangchong/data/fwz/output/innovation_2/controlnet/ 2>/dev/null || echo NO_DIR',
    'head -1 /home/wangchong/data/fwz/brlp-data/dataset.csv 2>/dev/null || echo MISSING',
    'conda run -n fwz python -c "import torch; print(torch.__version__)" 2>&1 | tail -3',
    'ls /home/wangchong/data/fwz/code/verification/scripts/*.py',
]
for cmd in cmds:
    _, stdout, stderr = c.exec_command(cmd, timeout=30)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'CMD: {cmd[:80]}')
    print(f'  OUT: {out[:300]}')
    if err:
        print(f'  ERR: {err[:200]}')
    print()
c.close()
