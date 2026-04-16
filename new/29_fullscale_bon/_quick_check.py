"""Quick time and progress check."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Server time
stdin, stdout, stderr = ssh.exec_command("date '+%H:%M:%S'")
print(f"Server time: {stdout.read().decode().strip()}")

# Log tail + file info
stdin, stdout, stderr = ssh.exec_command(
    "tail -3 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log; "
    "echo '---'; "
    "wc -l /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log; "
    "echo '---'; "
    "stat --format='Last modified: %y' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log"
)
print(stdout.read().decode())

# Process CPU
stdin, stdout, stderr = ssh.exec_command(
    "ps -p $(pgrep -f run_bon_fullscale | head -1) -o pid,pcpu,pmem,etime --no-headers 2>/dev/null || echo 'not found'"
)
print(f"Process: {stdout.read().decode().strip()}")

ssh.close()
