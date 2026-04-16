import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

base = "/home/wangchong/data/fwz"
scripts = "/home/wangchong/data/fwz/code/verification/scripts"

# Create a launcher script on the server
launcher = f"""#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd {scripts}

# Exp 2: weighted (5 pairs, N=5)
EXP2_DIR={base}/output/verification/weighted_compare
mkdir -p $EXP2_DIR
python evaluate_verification.py \\
    --dataset_csv {base}/output/innovation_5/prepared/B_mci.csv \\
    --aekl_ckpt {base}/output/innovation_5/ae/autoencoder-ep-2.pth \\
    --diff_ckpt {base}/brlp-train/pretrained/latentdiffusion.pth \\
    --cnet_ckpt {base}/output/innovation_2/controlnet/cnet-btr-ep-1.pth \\
    --output_dir $EXP2_DIR \\
    --n_candidates 5 --las_m 3 --max_pairs 5 \\
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \\
    > $EXP2_DIR/eval_verification.log 2>&1

# Exp 3: N=8 full (10 pairs)
EXP3_DIR={base}/output/verification/bon_n8_full
mkdir -p $EXP3_DIR
python evaluate_verification.py \\
    --dataset_csv {base}/output/innovation_5/prepared/B_mci.csv \\
    --aekl_ckpt {base}/output/innovation_5/ae/autoencoder-ep-2.pth \\
    --diff_ckpt {base}/brlp-train/pretrained/latentdiffusion.pth \\
    --cnet_ckpt {base}/output/innovation_2/controlnet/cnet-btr-ep-1.pth \\
    --output_dir $EXP3_DIR \\
    --n_candidates 8 --las_m 3 --max_pairs 10 \\
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \\
    > $EXP3_DIR/eval_verification.log 2>&1
"""

# Upload launcher script
import io
sftp = c.open_sftp()
with sftp.open(f'{scripts}/run_all_exps.sh', 'w') as f:
    f.write(launcher)
sftp.close()

# Make executable and run in background
_, stdout, _ = c.exec_command(f"chmod +x {scripts}/run_all_exps.sh && nohup bash {scripts}/run_all_exps.sh > /home/wangchong/data/fwz/output/verification/all_exps.log 2>&1 & echo $!", timeout=10)
pid = stdout.read().decode().strip()
print(f"Launcher PID: {pid}")

import time
time.sleep(2)
_, stdout, _ = c.exec_command("ps aux | grep run_all_exps | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
print(f"Process running: {bool(ps)}")

c.close()
