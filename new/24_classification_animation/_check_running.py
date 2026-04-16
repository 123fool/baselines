import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

def r(cmd):
    _, o, e = ssh.exec_command(cmd, timeout=30)
    return o.read().decode().strip(), e.read().decode().strip()

# Check if the pipeline process is running
out, _ = r("ps aux | grep run_pipeline_mci | grep -v grep")
print("Running processes:", out or "NONE")

# Check GPU usage
out2, _ = r("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader")
print("\nGPU status:")
print(out2)

# Check if any output files exist yet
out3, _ = r("ls -la /home/wangchong/data/fwz/output/mci_ad_classification/ 2>/dev/null")
print("\nOutput dir:", out3 or "EMPTY/NOT_FOUND")

# Check for error log
out4, _ = r("ls -la /tmp/mci_ad_*.log 2>/dev/null")
print("\nTemp logs:", out4 or "NONE")

ssh.close()
