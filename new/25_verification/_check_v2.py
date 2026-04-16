import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check log file size
_, stdout, _ = c.exec_command("wc -l /home/wangchong/data/fwz/output/verification/weighted_compare/eval_verification.log 2>/dev/null", timeout=10)
wc = stdout.read().decode().strip()
print(f"Log lines: {wc}")

# Check if summary exists
_, stdout, _ = c.exec_command("ls -la /home/wangchong/data/fwz/output/verification/weighted_compare/summary_*.json 2>/dev/null", timeout=10)
summary = stdout.read().decode().strip()
print(f"Summary: {summary if summary else 'not yet'}")

# Check if comparison CSV exists
_, stdout, _ = c.exec_command("ls -la /home/wangchong/data/fwz/output/verification/weighted_compare/comparison*.csv 2>/dev/null", timeout=10)
csv_f = stdout.read().decode().strip()
print(f"Comparison CSV: {csv_f if csv_f else 'not yet'}")

# Last 10 lines of log
_, stdout, _ = c.exec_command("tail -10 /home/wangchong/data/fwz/output/verification/weighted_compare/eval_verification.log 2>/dev/null", timeout=10)
tail = stdout.read().decode().strip()
print(f"\nLast 10 lines:\n{tail}")

# Check runner log (the shell script)
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/all_exps.log 2>/dev/null", timeout=10)
runner = stdout.read().decode().strip()
print(f"\nRunner log: {runner}")

# Check if PID 2836695 is still alive
_, stdout, _ = c.exec_command("ps -p 2836695 -o pid,state,etime,pcpu 2>/dev/null", timeout=10)
proc = stdout.read().decode().strip()
print(f"\nProcess 2836695: {proc}")

c.close()
