import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Start training with nohup
train_cmd = """cd /home/wangchong/data/fwz/code/innovation_4_v4 && \
nohup bash -c 'export PYTHONPATH="/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH" && \
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
echo "Training PID: $!"
"""

print("Starting training...")
stdin, stdout, stderr = ssh.exec_command(train_cmd, timeout=30)
out = stdout.read().decode('utf-8', errors='replace')
print(out)

# Wait a bit and check if it's running
time.sleep(5)

# Check process
stdin, stdout, stderr = ssh.exec_command('ps aux | grep train_ae_v4 | grep -v grep', timeout=10)
print("Running processes:")
print(stdout.read().decode())

# Check first few lines of log
time.sleep(3)
stdin, stdout, stderr = ssh.exec_command('head -20 /home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log 2>/dev/null', timeout=10)
log_start = stdout.read().decode()
print("Log start:")
print(log_start)

# Check GPU usage
stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader', timeout=10)
print("GPU status:")
print(stdout.read().decode())

ssh.close()
