"""Launch fullscale BoN experiment on server."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Check GPU status first
stdin, stdout, stderr = ssh.exec_command(
    'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu '
    '--format=csv,noheader,nounits'
)
gpu_info = stdout.read().decode().strip()
print("GPU Status:")
print(gpu_info)

# Find best GPU (least memory used)
best_gpu = 0
min_mem = 999999
for line in gpu_info.strip().split('\n'):
    parts = [x.strip() for x in line.split(',')]
    idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
    if used < min_mem:
        min_mem = used
        best_gpu = idx
print(f"\nUsing GPU {best_gpu} (memory used: {min_mem} MiB)")

# Launch
cmd = (
    f"source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    f"conda activate fwz && "
    f"cd /home/wangchong/data/fwz/code/verification/scripts && "
    f"mkdir -p /home/wangchong/data/fwz/output/verification/fullscale_50 && "
    f"nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES={best_gpu} "
    f"python run_bon_fullscale.py "
    f"> /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>&1 &"
)
print(f"\nLaunching on GPU {best_gpu}...")
stdin, stdout, stderr = ssh.exec_command(cmd)
stdout.read()

# Verify process started
import time; time.sleep(3)
stdin, stdout, stderr = ssh.exec_command(
    "ps aux | grep run_bon_fullscale | grep -v grep"
)
proc = stdout.read().decode().strip()
if proc:
    print("Process running:")
    print(proc)
else:
    # Check log for errors
    stdin, stdout, stderr = ssh.exec_command(
        "tail -20 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log"
    )
    log = stdout.read().decode()
    print("Process NOT found. Log:")
    print(log)

ssh.close()
