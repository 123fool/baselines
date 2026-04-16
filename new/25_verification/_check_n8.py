import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

n8_dir = "/home/wangchong/data/fwz/output/verification/bon_n8_full"

# Check N8 log lines
_, stdout, _ = c.exec_command(f"wc -l {n8_dir}/eval_verification.log 2>/dev/null", timeout=10)
wc = stdout.read().decode().strip()
print(f"N8 log lines: {wc}")

# Check last 10 lines of N8 log
_, stdout, _ = c.exec_command(f"tail -10 {n8_dir}/eval_verification.log 2>/dev/null", timeout=10)
tail = stdout.read().decode().strip()
print(f"\nN8 log tail:\n{tail}")

# Check N8 summary
_, stdout, _ = c.exec_command(f"ls -la {n8_dir}/summary_*.json 2>/dev/null", timeout=10)
summary = stdout.read().decode().strip()
print(f"\nN8 summary: {summary if summary else 'not yet'}")

# Runner log
_, stdout, _ = c.exec_command("tail -3 /home/wangchong/data/fwz/output/verification/n8_runner.log 2>/dev/null", timeout=10)
runner = stdout.read().decode().strip()
print(f"\nRunner: {runner}")

# Process check
_, stdout, _ = c.exec_command("ps aux | grep 'evaluate_verification\\|run_n8' | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
for line in ps.split('\n'):
    if line.strip():
        parts = line.split()
        print(f"PID={parts[1]} CPU={parts[2]}% MEM={parts[3]}%")

# GPU
_, stdout, _ = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=10)
gpu = stdout.read().decode().strip()
print(f"\nGPU: {gpu}")

c.close()
