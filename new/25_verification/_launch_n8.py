import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

scripts = "/home/wangchong/data/fwz/code/verification/scripts"

# Simple one-line launch for bon_n8_full on GPU 1
launcher = """#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd /home/wangchong/data/fwz/code/verification/scripts
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/wangchong/data/fwz/output/verification/bon_n8_full
echo "Starting bon_n8_full on GPU 1 at $(date)"
python evaluate_verification.py --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth --output_dir /home/wangchong/data/fwz/output/verification/bon_n8_full --n_candidates 8 --las_m 3 --max_pairs 10 --methods "las,single,bon_best1,bon_topk,bon_weighted" > /home/wangchong/data/fwz/output/verification/bon_n8_full/eval_verification.log 2>&1
echo "bon_n8_full DONE at $(date)"
"""

sftp = c.open_sftp()
with sftp.open(f'{scripts}/run_n8.sh', 'w') as f:
    f.write(launcher)
sftp.close()
c.close()

# Launch
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
transport = c2.get_transport()
channel = transport.open_session()
channel.exec_command(f"nohup bash {scripts}/run_n8.sh > /home/wangchong/data/fwz/output/verification/n8_runner.log 2>&1 &")

import time
time.sleep(3)

# Verify
c3 = paramiko.SSHClient()
c3.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c3.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
_, stdout, _ = c3.exec_command("tail -3 /home/wangchong/data/fwz/output/verification/n8_runner.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
print(f"Runner: {log}")

_, stdout, _ = c3.exec_command("ps aux | grep evaluate_verification | grep -v grep | wc -l", timeout=10)
cnt = stdout.read().decode().strip()
print(f"Running evaluate_verification processes: {cnt}")

c2.close()
c3.close()
