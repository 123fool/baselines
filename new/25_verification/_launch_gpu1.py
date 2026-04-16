import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

scripts = "/home/wangchong/data/fwz/code/verification/scripts"

# Upload updated launcher targeting GPU 1
launcher = """#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd /home/wangchong/data/fwz/code/verification/scripts
export CUDA_VISIBLE_DEVICES=1

BASE=/home/wangchong/data/fwz

# Exp 2: weighted_compare (5 pairs, N=5)
EXP2_DIR=$BASE/output/verification/weighted_compare
mkdir -p $EXP2_DIR
echo "[$(date)] Starting weighted_compare on GPU 1..."
python evaluate_verification.py \
    --dataset_csv $BASE/output/innovation_5/prepared/B_mci.csv \
    --aekl_ckpt $BASE/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt $BASE/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt $BASE/output/innovation_2/controlnet/cnet-btr-ep-1.pth \
    --output_dir $EXP2_DIR \
    --n_candidates 5 --las_m 3 --max_pairs 5 \
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \
    > $EXP2_DIR/eval_verification.log 2>&1
echo "[$(date)] weighted_compare DONE"

# Exp 3: N=8 full (10 pairs)
EXP3_DIR=$BASE/output/verification/bon_n8_full
mkdir -p $EXP3_DIR
echo "[$(date)] Starting bon_n8_full on GPU 1..."
python evaluate_verification.py \
    --dataset_csv $BASE/output/innovation_5/prepared/B_mci.csv \
    --aekl_ckpt $BASE/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt $BASE/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt $BASE/output/innovation_2/controlnet/cnet-btr-ep-1.pth \
    --output_dir $EXP3_DIR \
    --n_candidates 8 --las_m 3 --max_pairs 10 \
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \
    > $EXP3_DIR/eval_verification.log 2>&1
echo "[$(date)] bon_n8_full DONE"

echo "[$(date)] ALL EXPERIMENTS COMPLETE"
"""

sftp = c.open_sftp()
with sftp.open(f'{scripts}/run_all_exps.sh', 'w') as f:
    f.write(launcher)
print("Uploaded run_all_exps.sh with CUDA_VISIBLE_DEVICES=1")
sftp.close()
c.close()

# Launch with transport
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
transport = c2.get_transport()
channel = transport.open_session()
channel.exec_command(f"nohup bash {scripts}/run_all_exps.sh > /home/wangchong/data/fwz/output/verification/all_exps.log 2>&1 &")

import time
time.sleep(3)

# Verify
c3 = paramiko.SSHClient()
c3.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c3.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
_, stdout, _ = c3.exec_command("tail -3 /home/wangchong/data/fwz/output/verification/all_exps.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
print(f"Runner log: {log}")

_, stdout, _ = c3.exec_command("ps aux | grep evaluate_verification | grep -v grep | head -3", timeout=10)
ps = stdout.read().decode().strip()
print(f"Processes: {'RUNNING' if ps else 'waiting to start...'}")

c2.close()
c3.close()
