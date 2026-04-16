"""Quick check if the fullscale experiment is running."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Check process
stdin, stdout, stderr = ssh.exec_command("ps aux | grep run_bon_fullscale | grep -v grep")
proc = stdout.read().decode().strip()
print("Process:", proc if proc else "NOT RUNNING")

# Check log
stdin, stdout, stderr = ssh.exec_command(
    "tail -15 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null || echo 'No log'"
)
print(f"\nLog tail:\n{stdout.read().decode()}")

# GPU status
stdin, stdout, stderr = ssh.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
print(f"GPU:\n{stdout.read().decode()}")

ssh.close()
