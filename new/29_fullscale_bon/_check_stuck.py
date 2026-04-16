"""Check if experiment is stuck."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Process status
stdin, stdout, stderr = ssh.exec_command(
    "ps aux | grep run_bon_fullscale | grep -v grep"
)
proc = stdout.read().decode().strip()
if proc:
    print("Process details:")
    for line in proc.split('\n'):
        parts = line.split()
        pid = parts[1]
        cpu = parts[2]
        mem = parts[3]
        print(f"  PID={pid} CPU={cpu}% MEM={mem}%")
else:
    print("PROCESS NOT RUNNING!")

# GPU status
stdin, stdout, stderr = ssh.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu "
    "--format=csv,noheader"
)
print(f"\nGPU status:\n{stdout.read().decode()}")

# Check for any Python errors in log
stdin, stdout, stderr = ssh.exec_command(
    "grep -i 'error\\|traceback\\|exception\\|killed\\|oom' "
    "/home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null || echo 'no errors'"
)
errs = stdout.read().decode().strip()
print(f"Errors in log: {errs}")

# Log tail
stdin, stdout, stderr = ssh.exec_command(
    "tail -5 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log"
)
print(f"\nLast 5 log lines:\n{stdout.read().decode()}")

# Log file size and modification time
stdin, stdout, stderr = ssh.exec_command(
    "ls -la /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log; "
    "stat --format='%Y' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log; "
    "date +%s"
)
print(f"Log info:\n{stdout.read().decode()}")

ssh.close()
