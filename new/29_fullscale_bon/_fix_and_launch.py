"""Upload fixed run_bon_fullscale.py and re-launch experiment."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Upload fixed script
local_path = r"c:\Users\PC\Desktop\baselines\BrLP-main\new\29_fullscale_bon\scripts\run_bon_fullscale.py"
remote_path = "/home/wangchong/data/fwz/code/verification/scripts/run_bon_fullscale.py"

sftp = ssh.open_sftp()
sftp.put(local_path, remote_path)
print(f"Uploaded: {remote_path}")
sftp.close()

# Kill any old process
ssh.exec_command("pkill -f run_bon_fullscale || true")
import time; time.sleep(2)

# Verify fix — quick import test
stdin, stdout, stderr = ssh.exec_command(
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "cd /home/wangchong/data/fwz/code/verification/scripts && "
    "python -c \"import sys; sys.path.insert(0,'../src'); "
    "from brlp.networks import init_autoencoder, init_latent_diffusion, init_controlnet; "
    "from brlp.sampling import sample_using_controlnet_and_z; "
    "from sampling_bon_integrated import sample_bon_weighted; "
    "print('OK: imports good')\""
)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(f"Import test:\n{out}")
if 'Error' in err or 'Traceback' in err:
    print(f"ERRORS:\n{err}")
    ssh.close()
    exit(1)

# Check GPU status
stdin, stdout, stderr = ssh.exec_command(
    'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits'
)
gpu_info = stdout.read().decode().strip()
print(f"\nGPU Status:\n{gpu_info}")

# Find best GPU
best_gpu = 0
min_mem = 999999
for line in gpu_info.strip().split('\n'):
    parts = [x.strip() for x in line.split(',')]
    idx, used = int(parts[0]), int(parts[1])
    if used < min_mem:
        min_mem = used
        best_gpu = idx
print(f"Using GPU {best_gpu} (mem used: {min_mem} MiB)")

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
time.sleep(5)

# Verify running
stdin, stdout, stderr = ssh.exec_command("ps aux | grep run_bon_fullscale | grep -v grep")
proc = stdout.read().decode().strip()
if proc:
    print("Process running!")
    # Show initial log
    stdin, stdout, stderr = ssh.exec_command(
        "tail -10 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log"
    )
    print(f"Log:\n{stdout.read().decode()}")
else:
    stdin, stdout, stderr = ssh.exec_command(
        "tail -30 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log"
    )
    print(f"NOT RUNNING. Log:\n{stdout.read().decode()}")

ssh.close()
