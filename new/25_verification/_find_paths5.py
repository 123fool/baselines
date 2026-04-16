import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    # Check what eval_ep1.log says about which CSV/checkpoints were used
    'head -20 /home/wangchong/data/fwz/output/innovation_2/eval_ep1.log 2>/dev/null',
    # Check B_mci.csv has needed columns: starting_age, followup_age, starting_latent
    'head -1 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | tr "," "\\n" | grep -n "age\\|latent\\|hippocampus\\|amygdala\\|cerebral_cortex\\|ventricle\\|white_matter"',
    # Check the Innovation 5 AE checkpoint (the improved one)
    'ls -la /home/wangchong/data/fwz/output/innovation_5/ae/ 2>/dev/null',
    # Check the eval dir for Innovation 2
    'ls -la /home/wangchong/data/fwz/output/innovation_2/eval/ 2>/dev/null',
    # Check Innovation 2 controlnet files
    'ls -la /home/wangchong/data/fwz/output/innovation_2/controlnet/*.pth 2>/dev/null',
]
for cmd in cmds:
    _, stdout, stderr = c.exec_command(cmd, timeout=30, get_pty=True)
    out = stdout.read().decode().strip()
    print(f'CMD: {cmd[:120]}')
    print(f'  OUT: {out[:800]}')
    print()
c.close()
