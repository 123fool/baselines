import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Create output directory first
ssh.exec_command('mkdir -p /home/wangchong/data/fwz/output/innovation_4_v4/ae_training')

# Run directly (not nohup) to see errors, but with timeout
cmd = """cd /home/wangchong/data/fwz/code/innovation_4_v4 && \
export PYTHONPATH="/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH" && \
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
    --latent_noise_prob 0.5 \
2>&1 | head -60
"""

print("Running training (first 60 lines of output)...")
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print("STDOUT:")
print(out)
if err.strip():
    print("STDERR:")
    print(err[-2000:])

ssh.close()
