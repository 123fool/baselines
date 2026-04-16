import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    # Check AutoencoderKL encode/decode/sampling/forward methods
    'sed -n "718,810p" /home/wangchong/miniconda3/envs/fwz/lib/python3.9/site-packages/generative/networks/nets/autoencoderkl.py',
    # Check the full server losses.py (maybe it was modified)
    'cat /home/wangchong/data/fwz/brlp-code/src/brlp/losses.py',
]

for cmd in cmds:
    print(f'=== {cmd[:80]} ===')
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    print(stdout.read().decode('utf-8', errors='replace'))

ssh.close()
