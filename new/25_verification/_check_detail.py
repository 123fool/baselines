import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check weighted_compare eval log (full)
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/weighted_compare/eval_verification.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
print(f"=== weighted_compare eval log ({len(log)} bytes) ===")
if log:
    lines = log.split('\n')
    print(f"  Lines: {len(lines)}")
    # Print last 20 lines
    for line in lines[-20:]:
        print(f"  {line}")

# check GPU
_, stdout, _ = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=10)
gpu = stdout.read().decode().strip()
print(f"\nGPU status: {gpu}")

# Check process details
_, stdout, _ = c.exec_command("ps aux | grep evaluate_verification | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
for line in ps.split('\n'):
    if line.strip():
        parts = line.split()
        pid = parts[1]
        cpu = parts[2]
        mem = parts[3]
        print(f"PID={pid} CPU={cpu}% MEM={mem}%")

c.close()
