import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Get all column names
_, stdout, _ = c.exec_command('head -1 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | tr "," "\\n" | cat -n', timeout=20)
out = stdout.read().decode().strip()
print("ALL columns:")
print(out)

# Check first data row
_, stdout, _ = c.exec_command('sed -n "2p" /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | head -c 500', timeout=20)
out = stdout.read().decode().strip()
print("\nFirst data row (first 500 chars):")
print(out)

c.close()
