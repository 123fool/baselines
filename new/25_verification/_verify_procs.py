import paramiko, time
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check processes
_, stdout, _ = c.exec_command("ps aux | grep evaluate_verification | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
print("Running evaluate_verification processes:")
if ps:
    for line in ps.split('\n'):
        parts = line.split()
        pid = parts[1]
        mem = parts[3]
        cpu = parts[2]
        print(f"  PID={pid}  CPU={cpu}%  MEM={mem}%")
else:
    print("  NONE running!")

# Check GPU memory
_, stdout, _ = c.exec_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", timeout=10)
gpu = stdout.read().decode().strip()
print(f"\nGPU memory: {gpu}")

# Check if exp2 log exists
_, stdout, _ = c.exec_command("tail -5 /home/wangchong/data/fwz/output/verification/weighted_roundtrip/eval_verification.log 2>/dev/null", timeout=10)
log2 = stdout.read().decode().strip()
print(f"\nExp2 log: {log2 if log2 else 'no log yet'}")

# Check if exp3 log exists
_, stdout, _ = c.exec_command("tail -5 /home/wangchong/data/fwz/output/verification/bon_n8_full/eval_verification.log 2>/dev/null", timeout=10)
log3 = stdout.read().decode().strip()
print(f"\nExp3 log: {log3 if log3 else 'no log yet'}")

c.close()
