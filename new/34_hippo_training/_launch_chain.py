"""Launch remaining ControlNet training experiments on GPU 1."""
import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Create output dirs
for d in ['H1_a10', 'H1_a50', 'H2_a30']:
    ssh.exec_command(f'mkdir -p /home/wangchong/data/fwz/output/34_hippo_training/{d}')
time.sleep(1)

# Write a shell script on the server that runs all experiments sequentially
script = """#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
export CUDA_VISIBLE_DEVICES=1

COMMON="python /home/wangchong/data/fwz/code/34_hippo_training/scripts/train_controlnet_hippo.py \
  --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv \
  --cache_dir /home/wangchong/data/fwz/cache \
  --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
  --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
  --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth \
  --hippo_mask /home/wangchong/data/fwz/output/34_hippo_training/masks/hippo_latent_mask.npy \
  --n_epochs 3 --lr 1e-5 --batch_size 16"

echo "=== H1_a10 ===" >> /home/wangchong/data/fwz/output/34_hippo_training/chain.log
$COMMON --method H1 --alpha 10 --output_dir /home/wangchong/data/fwz/output/34_hippo_training/H1_a10 \
  > /home/wangchong/data/fwz/output/34_hippo_training/H1_a10/train.log 2>&1

echo "=== H1_a50 ===" >> /home/wangchong/data/fwz/output/34_hippo_training/chain.log
$COMMON --method H1 --alpha 50 --output_dir /home/wangchong/data/fwz/output/34_hippo_training/H1_a50 \
  > /home/wangchong/data/fwz/output/34_hippo_training/H1_a50/train.log 2>&1

echo "=== H2_a30 ===" >> /home/wangchong/data/fwz/output/34_hippo_training/chain.log
$COMMON --method H2 --alpha 30 --output_dir /home/wangchong/data/fwz/output/34_hippo_training/H2_a30 \
  > /home/wangchong/data/fwz/output/34_hippo_training/H2_a30/train.log 2>&1

echo "=== ALL DONE ===" >> /home/wangchong/data/fwz/output/34_hippo_training/chain.log
"""

sftp = ssh.open_sftp()
with sftp.open('/home/wangchong/data/fwz/code/34_hippo_training/run_chain.sh', 'w') as f:
    f.write(script)
sftp.close()

ssh.exec_command('chmod +x /home/wangchong/data/fwz/code/34_hippo_training/run_chain.sh')
time.sleep(1)

_, o, e = ssh.exec_command(
    'nohup bash /home/wangchong/data/fwz/code/34_hippo_training/run_chain.sh &')
print('Launched chain: H1_a10 -> H1_a50 -> H2_a30 on GPU 1')
ssh.close()
