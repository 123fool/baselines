import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    'grep -n "def encode\|def decode\|def sampling\|def forward" /home/wangchong/miniconda3/envs/fwz/lib/python3.9/site-packages/generative/networks/nets/autoencoderkl.py',
    'sed -n "190,350p" /home/wangchong/miniconda3/envs/fwz/lib/python3.9/site-packages/generative/networks/nets/autoencoderkl.py',
]

for cmd in cmds:
    print(f'=== {cmd[:80]} ===')
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    print(stdout.read().decode('utf-8', errors='replace'))

ssh.close()
