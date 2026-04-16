import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Create a launch script on server
launch_script = """#!/bin/bash
export PYTHONPATH="/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
cd /home/wangchong/data/fwz/code/innovation_4_v4

/home/wangchong/miniconda3/envs/fwz/bin/python scripts/train_ae_v4.py \\
    --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/A_mci.csv \\
    --cache_dir /home/wangchong/data/fwz/cache/innovation_4_v4 \\
    --output_dir /home/wangchong/data/fwz/output/innovation_4_v4/ae_training \\
    --aekl_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/autoencoder.pth \\
    --mednet_ckpt /home/wangchong/data/fwz/code/innovation_4/pretrained/resnet_10_23dataset.pth \\
    --n_epochs 10 \\
    --max_batch_size 1 \\
    --batch_size 16 \\
    --lr 2e-5 \\
    --perc3d_weight 0.0005 \\
    --freq_weight 0.001 \\
    --ssim_weight 0.5 \\
    --l1_weight 1.5 \\
    --warmup_start 3 \\
    --warmup_end 6 \\
    --latent_noise_std 0.01 \\
    --latent_noise_prob 0.5
"""

# Write launch script to server
sftp = ssh.open_sftp()
with sftp.open('/home/wangchong/data/fwz/code/innovation_4_v4/launch_train.sh', 'w') as f:
    f.write(launch_script)
sftp.close()

ssh.exec_command('chmod +x /home/wangchong/data/fwz/code/innovation_4_v4/launch_train.sh')
time.sleep(1)

# Launch with nohup using a simple fire-and-forget approach
print("Launching training...")
transport = ssh.get_transport()
channel = transport.open_session()
channel.exec_command('nohup bash /home/wangchong/data/fwz/code/innovation_4_v4/launch_train.sh > /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>&1 &')
channel.close()
print("Training command sent!")

# Wait for initialization
print("Waiting 20 seconds for initialization...")
time.sleep(20)

# Check
stdin, stdout, stderr = ssh.exec_command('ps aux | grep train_ae_v4 | grep -v grep', timeout=10)
procs = stdout.read().decode()
print(f"\nProcess check:")
print(procs if procs.strip() else "(not found - checking log for errors)")

stdin, stdout, stderr = ssh.exec_command('wc -l /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>/dev/null', timeout=10)
print(f"Log lines: {stdout.read().decode().strip()}")

stdin, stdout, stderr = ssh.exec_command('tail -10 /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>/dev/null', timeout=10)
log_tail = stdout.read().decode()
print(f"\nLog tail:")
print(log_tail if log_tail.strip() else "(empty)")

stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader', timeout=10)
print(f"\nGPU status:")
print(stdout.read().decode())

ssh.close()
