"""Check more paths."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

cmds = [
    "cat /home/wangchong/data/fwz/code/no_aux_model/run_eval.sh",
    "find /home/wangchong/data/fwz/ -name 'diffusion*.pth' -type f 2>/dev/null | head -10",
]
for cmd in cmds:
    print(f"\n$ {cmd}")
    _, stdout, _ = ssh.exec_command(cmd)
    print(stdout.read().decode().strip())
ssh.close()
