import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

base = "/home/wangchong/data/fwz/output/verification"
exps = ["quick_compare", "weighted_compare", "bon_n8_full"]

# Check running processes
_, stdout, _ = c.exec_command("ps aux | grep -E 'evaluate_verification|run_all_exps' | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
print("=== RUNNING PROCESSES ===")
for line in ps.split('\n'):
    if line.strip():
        parts = line.split()
        print(f"  PID={parts[1]}")

# Check each experiment
for exp in exps:
    exp_dir = f"{base}/{exp}"
    print(f"\n=== {exp} ===")
    
    # Check log
    _, stdout, _ = c.exec_command(f"tail -5 {exp_dir}/eval_verification.log 2>/dev/null", timeout=10)
    log = stdout.read().decode().strip()
    if log:
        print(f"  Log tail: {log[-200:]}")
    else:
        print("  No log yet")
    
    # Check summary
    _, stdout, _ = c.exec_command(f"ls -la {exp_dir}/summary_*.json 2>/dev/null", timeout=10)
    summary = stdout.read().decode().strip()
    if summary:
        print(f"  Summary found: {summary}")
    else:
        print("  Summary: not yet")

# Check runner log
_, stdout, _ = c.exec_command(f"tail -5 {base}/all_exps.log 2>/dev/null", timeout=10)
runner = stdout.read().decode().strip()
print(f"\n=== Runner Log ===\n{runner}")

# GPU usage
_, stdout, _ = c.exec_command("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader", timeout=10)
gpu = stdout.read().decode().strip()
print(f"\nGPU: {gpu}")

c.close()
