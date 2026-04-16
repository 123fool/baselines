import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Kill all existing train_ae_v4 processes
print("Killing existing processes...")
stdin, stdout, stderr = ssh.exec_command('pkill -f train_ae_v4 2>/dev/null; echo "killed"', timeout=10)
print(stdout.read().decode())
time.sleep(2)

# Re-upload fixed script
sftp = ssh.open_sftp()
local_path = r'c:\Users\PC\Desktop\baselines\BrLP-main\new\07_innovation_4\train_ae_v4.py'
remote_path = '/home/wangchong/data/fwz/code/innovation_4_v4/scripts/train_ae_v4.py'
sftp.put(local_path, remote_path)
sftp.close()
print("Re-uploaded train_ae_v4.py")

# Clean up old cache and log
ssh.exec_command('rm -rf /home/wangchong/data/fwz/cache/innovation_4_v4/*')
ssh.exec_command('rm -f /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log')
time.sleep(1)

# Relaunch
print("Relaunching training...")
transport = ssh.get_transport()
channel = transport.open_session()
channel.exec_command(
    'nohup bash /home/wangchong/data/fwz/code/innovation_4_v4/launch_train.sh '
    '> /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>&1 &'
)
channel.close()

# Wait for initialization
print("Waiting 25 seconds for initialization...")
time.sleep(25)

# Check
stdin, stdout, stderr = ssh.exec_command('ps aux | grep train_ae_v4 | grep -v grep | wc -l', timeout=10)
nprocs = stdout.read().decode().strip()
print(f"Running processes: {nprocs}")

stdin, stdout, stderr = ssh.exec_command('tail -20 /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>/dev/null', timeout=10)
log_tail = stdout.read().decode()
print(f"\nLog tail:")
print(log_tail if log_tail.strip() else "(empty)")

stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader', timeout=10)
print(f"\nGPU status:")
print(stdout.read().decode())

ssh.close()
