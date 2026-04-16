"""Monitor with more lines and direct last line extraction."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Check process alive
stdin, stdout, stderr = ssh.exec_command("ps aux | grep run_bon_fullscale | grep -v grep | wc -l")
n_procs = stdout.read().decode().strip()
print(f"Process count: {n_procs}")

# Total lines in log
stdin, stdout, stderr = ssh.exec_command(
    "wc -l /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null"
)
print(f"Log lines: {stdout.read().decode().strip()}")

# Last 20 lines
stdin, stdout, stderr = ssh.exec_command(
    "tail -20 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null"
)
print(f"Last 20 lines:\n{stdout.read().decode()}")

# Count done
stdin, stdout, stderr = ssh.exec_command(
    "grep -c 'done (' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null || echo 0"
)
print(f"Pairs done: {stdout.read().decode().strip()}/50")

# Count wins
stdin, stdout, stderr = ssh.exec_command(
    "grep 'Winner: bon' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null | wc -l"
)
bon_w = stdout.read().decode().strip()
stdin, stdout, stderr = ssh.exec_command(
    "grep 'Winner: las' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null | wc -l"
)
las_w = stdout.read().decode().strip()
print(f"Wins: bon_weighted={bon_w}, las={las_w}")

ssh.close()
