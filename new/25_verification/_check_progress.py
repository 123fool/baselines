import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check if still running
_, stdout, _ = c.exec_command("ps aux | grep evaluate_verification | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
print(f"Process: {'RUNNING' if ps else 'FINISHED'}")
if ps:
    print(ps[:200])

# Check log
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/runner.log 2>/dev/null | tail -30", timeout=10)
log = stdout.read().decode().strip()
print(f"\nLog tail:\n{log}")

# Check if summary exists
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/quick_compare/summary_quick_compare.json 2>/dev/null", timeout=10)
summary = stdout.read().decode().strip()
if summary:
    print(f"\nSummary:\n{summary[:1000]}")
else:
    print("\nSummary not yet generated")

# Check eval log
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/quick_compare/eval_verification.log 2>/dev/null | tail -20", timeout=10)
elog = stdout.read().decode().strip()
if elog:
    print(f"\nEval log tail:\n{elog}")

c.close()
