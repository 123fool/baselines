import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check eval log for latest entries
_, stdout, _ = c.exec_command('cat /home/wangchong/data/fwz/output/verification/quick_compare/eval_verification.log 2>/dev/null | tail -15', timeout=10)
log = stdout.read().decode().strip()
print("Log tail:")
print(log)

# Check if summary exists
_, stdout, _ = c.exec_command('cat /home/wangchong/data/fwz/output/verification/quick_compare/summary_quick_compare.json 2>/dev/null', timeout=10)
summary = stdout.read().decode().strip()
if summary:
    print("\n=== SUMMARY ===")
    print(summary)
else:
    print("\nSummary not yet generated")

# Check dir contents
_, stdout, _ = c.exec_command('ls -la /home/wangchong/data/fwz/output/verification/quick_compare/ 2>/dev/null', timeout=10)
files = stdout.read().decode().strip()
print("\nOutput dir:")
print(files)

# Check process
_, stdout, _ = c.exec_command('ps aux | grep evaluate_verification | grep -v grep', timeout=10)
ps = stdout.read().decode().strip()
status = "RUNNING" if ps else "FINISHED"
print(f"\nProcess: {status}")

c.close()
