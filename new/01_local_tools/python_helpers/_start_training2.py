import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Create all necessary directories
dirs = [
    '/home/wangchong/data/fwz/cache/innovation_4_v4',
    '/home/wangchong/data/fwz/output/innovation_4_v4',
    '/home/wangchong/data/fwz/output/innovation_4_v4/ae_training',
    '/home/wangchong/data/fwz/output/innovation_4_v4/eval',
]
for d in dirs:
    ssh.exec_command(f'mkdir -p {d}')
    print(f"Created: {d}")

time.sleep(1)

# Now start training with nohup
train_cmd = """cd /home/wangchong/data/fwz/code/innovation_4_v4 && \
nohup bash -c 'export PYTHONPATH="/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH" && \
export CUDA_VISIBLE_DEVICES=0 && \
/home/wangchong/miniconda3/envs/fwz/bin/python scripts/train_ae_v4.py \
    --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/A_mci.csv \
    --cache_dir /home/wangchong/data/fwz/cache/innovation_4_v4 \
    --output_dir /home/wangchong/data/fwz/output/innovation_4_v4/ae_training \
    --aekl_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/autoencoder.pth \
    --mednet_ckpt /home/wangchong/data/fwz/code/innovation_4/pretrained/resnet_10_23dataset.pth \
    --n_epochs 10 \
    --max_batch_size 1 \
    --batch_size 16 \
    --lr 2e-5 \
    --perc3d_weight 0.0005 \
    --freq_weight 0.001 \
    --ssim_weight 0.5 \
    --l1_weight 1.5 \
    --warmup_start 3 \
    --warmup_end 6 \
    --latent_noise_std 0.01 \
    --latent_noise_prob 0.5' \
> /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>&1 &
echo $!
"""

print("\nStarting training...")
stdin, stdout, stderr = ssh.exec_command(train_cmd, timeout=30)
pid = stdout.read().decode().strip()
print(f"PID: {pid}")

# Wait for process to initialize
time.sleep(15)

# Check if still running
stdin, stdout, stderr = ssh.exec_command('ps aux | grep train_ae_v4 | grep -v grep', timeout=10)
procs = stdout.read().decode()
print(f"\nProcess check:")
print(procs if procs.strip() else "(not found)")

# Check log
stdin, stdout, stderr = ssh.exec_command('head -30 /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log', timeout=10)
log = stdout.read().decode()
print(f"\nLog (first 30 lines):")
print(log if log.strip() else "(empty)")

# Check GPU
stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader', timeout=10)
print(f"\nGPU status:")
print(stdout.read().decode())

ssh.close()
