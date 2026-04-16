import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Find where evaluate_verification.py is
_, stdout, _ = c.exec_command("find /home/wangchong/data/fwz/code -name 'evaluate_verification.py' 2>/dev/null", timeout=15)
found = stdout.read().decode().strip()
print(f"Found evaluate_verification.py at:\n{found}")

# List entire code directory
_, stdout, _ = c.exec_command("find /home/wangchong/data/fwz/code -type f 2>/dev/null", timeout=15)
all_files = stdout.read().decode().strip()
print(f"\nAll files under code/:\n{all_files}")

c.close()
