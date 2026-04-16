import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

def r(cmd):
    _, o, e = ssh.exec_command(cmd, timeout=30)
    return o.read().decode().strip()

# Check output dirs
print("=== Output directories ===")
print(r("find /home/wangchong/data/fwz/output/mci_ad_classification -maxdepth 2 -type f | sort"))
print()

# GPU status
print("=== GPU 1 ===")
print(r("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader -i 1"))
print()

# Process status
print("=== Process ===")
print(r("ps -p $(pgrep -f run_pipeline_mci) -o pid,etime,pcpu,pmem,rss --no-headers 2>/dev/null || echo 'NOT RUNNING'"))

ssh.close()
