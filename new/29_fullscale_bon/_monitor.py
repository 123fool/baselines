"""Monitor fullscale experiment progress in detail."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Get log tail - last 40 lines
stdin, stdout, stderr = ssh.exec_command(
    "tail -40 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null"
)
log = stdout.read().decode()
print(log)

# Count completed pairs
stdin, stdout, stderr = ssh.exec_command(
    "grep -c 'done (' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null || echo 0"
)
n_done = stdout.read().decode().strip()
print(f"Pairs completed: {n_done}/50")

# Count wins
stdin, stdout, stderr = ssh.exec_command(
    "grep 'Winner:' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log | "
    "grep -c bon_weighted 2>/dev/null || echo 0"
)
bon_wins = stdout.read().decode().strip()
stdin, stdout, stderr = ssh.exec_command(
    "grep 'Winner:' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log | "
    "grep -c 'las$' 2>/dev/null || echo 0"
)
las_wins = stdout.read().decode().strip()
print(f"Wins: bon_weighted={bon_wins}, las={las_wins}")

ssh.close()
