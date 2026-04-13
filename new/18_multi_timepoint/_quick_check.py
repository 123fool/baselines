"""Quick status check for multi-timepoint experiment."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Just check the log tail
_, stdout, _ = ssh.exec_command("cat /home/wangchong/data/fwz/output/multi_timepoint/eval_multi_tp.log 2>/dev/null | tail -20")
log = stdout.read().decode().strip()
print(log)

# Count lines
_, stdout2, _ = ssh.exec_command("wc -l /home/wangchong/data/fwz/output/multi_timepoint/eval_multi_tp.log 2>/dev/null")
count = stdout2.read().decode().strip()
print(f"\nTotal log lines: {count}")

# Process still running?
_, stdout3, _ = ssh.exec_command("ps aux | grep 'evaluate_multi_timepoint' | grep python | grep -v grep | wc -l")
n = stdout3.read().decode().strip()
print(f"Process running: {'Yes' if n.strip() == '1' else 'No'}")

ssh.close()
