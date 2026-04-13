"""Check checkpoint paths on server."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

cmds = [
    "ls -la /home/wangchong/data/fwz/output/innovation_5/autoencoder/ 2>/dev/null || echo 'NOT FOUND'",
    "find /home/wangchong/data/fwz/ -name 'autoencoder*.pth' -type f 2>/dev/null | head -10",
    "ls -la /home/wangchong/data/fwz/output/innovation_2/latent_diffusion/ 2>/dev/null || echo 'NOT FOUND'",
    "ls -la /home/wangchong/data/fwz/output/innovation_2/controlnet/ 2>/dev/null || echo 'NOT FOUND'",
    "ls -la /home/wangchong/data/fwz/output/tpn_v3b/tpn_best.pth 2>/dev/null || echo 'NOT FOUND'",
    "# Check what the no_aux experiment used:",
    "head -5 /home/wangchong/data/fwz/code/no_aux_model/run_eval.sh 2>/dev/null || cat /home/wangchong/data/fwz/output/no_aux_model/run.log 2>/dev/null | head -20",
]

for cmd in cmds:
    if cmd.startswith('#'):
        print(f"\n{cmd}")
        continue
    print(f"\n$ {cmd}")
    _, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out: print(out)
    if err: print(f"ERR: {err}")

ssh.close()
