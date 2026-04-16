import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Get lines with epoch and val info
stdin, stdout, stderr = ssh.exec_command('grep -E "(\\[Epoch|val_|perc3d_w|Training complete)" /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log', timeout=10)
print("=== Key training milestones ===")
print(stdout.read().decode())

# Check for checkpoints saved
stdin, stdout, stderr = ssh.exec_command('ls -la /home/wangchong/data/fwz/output/innovation_4_v4/ae_training/*.pth 2>/dev/null', timeout=10)
print("=== Checkpoints ===")
print(stdout.read().decode())

# GPU
stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader', timeout=10)
print("=== GPU ===")
print(stdout.read().decode())

ssh.close()
