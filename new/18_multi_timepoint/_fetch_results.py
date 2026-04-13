"""Fetch final results from server: summary JSON, eval CSV, and full eval log."""
import paramiko

host = "10.96.27.109"
port = 2638
user = "wangchong"
passwd = "123456"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username=user, password=passwd)

output_dir = "/home/wangchong/data/fwz/output/multi_timepoint"

# Fetch summary JSON
print("=" * 60)
print("SUMMARY JSON:")
print("=" * 60)
stdin, stdout, stderr = ssh.exec_command(f"cat {output_dir}/summary_multi_timepoint.json")
print(stdout.read().decode())

# Fetch full eval log  
print("=" * 60)
print("FULL EVAL LOG:")
print("=" * 60)
stdin, stdout, stderr = ssh.exec_command(f"cat {output_dir}/eval_multi_tp.log")
print(stdout.read().decode())

# Fetch run.log (stdout) for any additional output
print("=" * 60)
print("RUN.LOG (stdout):")
print("=" * 60)
stdin, stdout, stderr = ssh.exec_command(f"tail -50 {output_dir}/run.log")
print(stdout.read().decode())

# Check CSV exists and get row count
stdin, stdout, stderr = ssh.exec_command(f"wc -l {output_dir}/eval_multi_timepoint.csv")
print("CSV line count:", stdout.read().decode().strip())

ssh.close()
