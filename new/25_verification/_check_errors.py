import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Full error log for weighted_compare
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/weighted_compare/eval_verification.log 2>/dev/null | tail -30", timeout=10)
log = stdout.read().decode().strip()
print("=== weighted_compare FULL ERROR ===")
print(log)

# Check what's using GPU 0
_, stdout, _ = c.exec_command("nvidia-smi", timeout=10)
smi = stdout.read().decode().strip()
print(f"\n=== nvidia-smi ===")
print(smi[:2000])

# Check which GPU the quick_compare used
_, stdout, _ = c.exec_command("head -3 /home/wangchong/data/fwz/output/verification/quick_compare/eval_verification.log", timeout=10)
log_head = stdout.read().decode().strip()
print(f"\n=== quick_compare log head ===")
print(log_head)

c.close()
