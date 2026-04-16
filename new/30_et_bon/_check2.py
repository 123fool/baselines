"""Check process status and full log tail."""
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.96.27.109", port=2638, username="wangchong", password="123456", timeout=15)

# Check if process is still running
_, so, _ = ssh.exec_command("ps aux | grep run_et_bon | grep -v grep")
procs = so.read().decode().strip()
print("=== PROCESS STATUS ===")
print(procs if procs else "NO PROCESS FOUND!")

# Raw tail of log file
_, so, _ = ssh.exec_command("tail -30 /home/wangchong/data/fwz/output/verification/et_bon/et_bon_experiment.log")
print("\n=== LOG TAIL (raw) ===")
print(so.read().decode())

# Check if results JSON exists
_, so, _ = ssh.exec_command("ls -la /home/wangchong/data/fwz/output/verification/et_bon/et_bon_results*.json 2>&1")
print("=== RESULTS FILES ===")
print(so.read().decode())

# GPU status
_, so, _ = ssh.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
print("=== GPU STATUS ===")
print(so.read().decode())

ssh.close()
