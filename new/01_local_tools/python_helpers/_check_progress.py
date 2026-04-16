import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check training progress
stdin, stdout, stderr = ssh.exec_command('tail -30 /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>/dev/null', timeout=10)
print("=== Training Log (last 30 lines) ===")
print(stdout.read().decode())

# GPU
stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader', timeout=10)
print("=== GPU Status ===")
print(stdout.read().decode())

# Process status
stdin, stdout, stderr = ssh.exec_command('ps aux | grep train_ae_v4 | grep -v grep | head -1', timeout=10)
print("=== Process ===")
print(stdout.read().decode())

ssh.close()
