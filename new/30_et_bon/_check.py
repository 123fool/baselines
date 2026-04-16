"""Check log progress."""
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.96.27.109", port=2638, username="wangchong", password="123456", timeout=15)
_, so, _ = ssh.exec_command("grep '^\\[' /home/wangchong/data/fwz/output/verification/et_bon/et_bon_experiment.log | tail -25")
print(so.read().decode())
ssh.close()
