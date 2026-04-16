import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Get latest validation metrics
cmd = 'grep -E "\\[Epoch [0-9]+\\] val_" /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log | tail -3'
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
print("=== Latest val lines ===")
print(stdout.read().decode())

# Check current progress
cmd2 = 'tail -3 /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log'
stdin, stdout, stderr = ssh.exec_command(cmd2, timeout=10)
print("=== Current progress ===")
print(stdout.read().decode())

# Check process status
stdin, stdout, stderr = ssh.exec_command('ps -p 1885197 -o pid,stat,etime --no-headers', timeout=10)
proc = stdout.read().decode().strip()
print("=== Process ===")
print(proc if proc else "FINISHED")

# Check training completion marker
stdin, stdout, stderr = ssh.exec_command('grep -E "Training complete|Best val_combined" /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log', timeout=10)
done_mark = stdout.read().decode().strip()
print("=== Train completion marker ===")
print(done_mark if done_mark else "NOT_COMPLETED")

# Check auto-eval process
stdin, stdout, stderr = ssh.exec_command('ps -p 4003673 -o pid,stat,etime,cmd --no-headers', timeout=10)
auto_proc = stdout.read().decode().strip()
print("=== Auto-eval process ===")
print(auto_proc if auto_proc else "AUTO_EVAL_EXITED")

# Check auto-eval monitor log
stdin, stdout, stderr = ssh.exec_command('tail -20 /home/wangchong/data/fwz/output/innovation_4_v4/auto_eval_monitor.log 2>/dev/null', timeout=10)
print("=== Auto-eval monitor tail ===")
print(stdout.read().decode())

# Check final evaluation summary
stdin, stdout, stderr = ssh.exec_command('cat /home/wangchong/data/fwz/output/innovation_4_v4/eval/summary_innovation_4_v4.json 2>/dev/null', timeout=10)
summary = stdout.read().decode().strip()
print("=== Eval summary ===")
print(summary if summary else "NO_EVAL_SUMMARY")

ssh.close()
