"""Fetch key source files from server."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

files = [
    '/home/wangchong/data/fwz/code/no_aux_model/evaluate_no_aux.py',
    '/home/wangchong/data/fwz/code/tpn/brlp_src/brlp/sampling.py',
]

for filepath in files:
    stdin, stdout, stderr = ssh.exec_command(f'cat {filepath}')
    content = stdout.read().decode('utf-8', errors='replace')
    print(f"\n{'='*60}")
    print(f"FILE: {filepath}")
    print(f"{'='*60}")
    print(content)

ssh.close()
