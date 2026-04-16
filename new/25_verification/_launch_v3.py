import paramiko, io

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

base = "/home/wangchong/data/fwz"
scripts = "/home/wangchong/data/fwz/code/verification/scripts"

# Step 1: Upload the launcher script via SFTP
launcher_content = """#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd /home/wangchong/data/fwz/code/verification/scripts

# Exp 2: weighted_compare (5 pairs, N=5, all 5 methods)
EXP2_DIR=/home/wangchong/data/fwz/output/verification/weighted_compare
mkdir -p $EXP2_DIR
echo "[$(date)] Starting weighted_compare..."
python evaluate_verification.py \
    --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth \
    --output_dir $EXP2_DIR \
    --n_candidates 5 --las_m 3 --max_pairs 5 \
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \
    > $EXP2_DIR/eval_verification.log 2>&1
echo "[$(date)] weighted_compare DONE"

# Exp 3: N=8 full (10 pairs, all 5 methods)
EXP3_DIR=/home/wangchong/data/fwz/output/verification/bon_n8_full
mkdir -p $EXP3_DIR
echo "[$(date)] Starting bon_n8_full..."
python evaluate_verification.py \
    --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth \
    --output_dir $EXP3_DIR \
    --n_candidates 8 --las_m 3 --max_pairs 10 \
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \
    > $EXP3_DIR/eval_verification.log 2>&1
echo "[$(date)] bon_n8_full DONE"

echo "[$(date)] ALL EXPERIMENTS COMPLETE"
"""

sftp = c.open_sftp()
with sftp.open(f'{scripts}/run_all_exps.sh', 'w') as f:
    f.write(launcher_content)
print("Uploaded run_all_exps.sh")
sftp.close()
c.close()

# Step 2: Use a new connection to launch
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Use transport for non-blocking exec
transport = c2.get_transport()
channel = transport.open_session()
channel.exec_command(f"chmod +x {scripts}/run_all_exps.sh; nohup bash {scripts}/run_all_exps.sh > /home/wangchong/data/fwz/output/verification/all_exps.log 2>&1 &")

import time
time.sleep(2)

# Verify with a separate connection
c3 = paramiko.SSHClient()
c3.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c3.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
_, stdout, _ = c3.exec_command("ps aux | grep 'run_all_exps\\|evaluate_verification' | grep -v grep | wc -l", timeout=10)
count = stdout.read().decode().strip()
print(f"Matching processes: {count}")

_, stdout, _ = c3.exec_command("tail -3 /home/wangchong/data/fwz/output/verification/all_exps.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
print(f"Log: {log}")

c2.close()
c3.close()
print("Done")
