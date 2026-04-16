#!/usr/bin/env python3
"""Kill all training processes and restart clean"""
import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

# Kill ALL train_refinement_v2 processes
print("Killing all training processes...")
_, out, _ = ssh.exec_command("pkill -f train_refinement_v2", timeout=10)
out.read()
time.sleep(3)

# Verify
_, out, _ = ssh.exec_command("ps aux | grep train_refinement | grep -v grep", timeout=10)
procs = out.read().decode().strip()
print(f"Remaining processes: {procs if procs else 'NONE (clean)'}")

# Check GPU  
_, out, _ = ssh.exec_command("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader", timeout=10)
print(f"\nGPU status after kill:\n{out.read().decode().strip()}")

ssh.close()
print("\nAll killed. Ready for clean restart.")
