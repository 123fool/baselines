"""Check experiment status."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check if running
_, o, _ = ssh.exec_command("ps aux | grep 'hippocampus_improvement' | grep -v grep | wc -l")
n = int(o.read().decode().strip())
print(f"Running instances: {n}")

# Show log
_, o, _ = ssh.exec_command("tail -50 /home/wangchong/data/fwz/output/33_hippocampus/run.log")
print(o.read().decode())

# Check results
_, o, _ = ssh.exec_command("ls -la /home/wangchong/data/fwz/output/33_hippocampus/")
print("\nFiles:")
print(o.read().decode())

ssh.close()
