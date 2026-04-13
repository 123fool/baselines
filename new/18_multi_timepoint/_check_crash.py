"""Check crash reason."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

_, stdout, _ = ssh.exec_command("tail -60 /home/wangchong/data/fwz/output/multi_timepoint/run.log 2>/dev/null")
print(stdout.read().decode().strip())
ssh.close()
