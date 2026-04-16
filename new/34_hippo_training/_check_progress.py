import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
time.sleep(180)

for tag in ['H1_a10', 'H1_a50', 'H2_a30']:
    print(f'=== {tag} ===')
    _, o, _ = ssh.exec_command(f'tail -5 /home/wangchong/data/fwz/output/34_hippo_training/{tag}/train.log 2>/dev/null')
    out = o.read().decode().strip()
    print(out if out else '(not started yet)')
    print()

print('=== AE Decoder ===')
_, o, _ = ssh.exec_command('tail -5 /home/wangchong/data/fwz/output/34_hippo_training/AE_dec_a30/train.log 2>/dev/null')
print(o.read().decode())

_, o, _ = ssh.exec_command('nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader')
print('GPU:', o.read().decode())

_, o, _ = ssh.exec_command('find /home/wangchong/data/fwz/output/34_hippo_training -name "*.pth" | sort')
print('=== Checkpoints ===')
print(o.read().decode())

ssh.close()
