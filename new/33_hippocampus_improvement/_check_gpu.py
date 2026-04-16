"""Check GPU availability and existing processes."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

check_script = """
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
print("GPU Status:")
print(result.stdout)

# Check running python processes
result2 = subprocess.run(['bash', '-c', "ps aux | grep python | grep -v grep | awk '{print $2, $4, $11, $12, $13}'"], capture_output=True, text=True)
print("Running Python processes:")
print(result2.stdout)
"""

sftp = ssh.open_sftp()
with sftp.open('/tmp/check_gpu.py', 'w') as f:
    f.write(check_script)
sftp.close()

_, o, e = ssh.exec_command("python /tmp/check_gpu.py")
print(o.read().decode())
ssh.close()
