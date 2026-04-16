import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

scripts = "/home/wangchong/data/fwz/code/verification/scripts"

# Roundtrip experiment on GPU 0 (now free!)
launcher = """#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd /home/wangchong/data/fwz/code/verification/scripts
export CUDA_VISIBLE_DEVICES=0
mkdir -p /home/wangchong/data/fwz/output/verification/roundtrip_test
echo "Starting roundtrip_test on GPU 0 at $(date)"
python evaluate_verification.py --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth --output_dir /home/wangchong/data/fwz/output/verification/roundtrip_test --n_candidates 5 --las_m 3 --max_pairs 5 --methods "las,bon_weighted,roundtrip_bon" > /home/wangchong/data/fwz/output/verification/roundtrip_test/eval_verification.log 2>&1
echo "roundtrip_test DONE at $(date)"
"""

sftp = c.open_sftp()
with sftp.open(f'{scripts}/run_roundtrip.sh', 'w') as f:
    f.write(launcher)
sftp.close()
c.close()

c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
transport = c2.get_transport()
channel = transport.open_session()
channel.exec_command(f"nohup bash {scripts}/run_roundtrip.sh > /home/wangchong/data/fwz/output/verification/roundtrip_runner.log 2>&1 &")

import time
time.sleep(3)

c3 = paramiko.SSHClient()
c3.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c3.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
_, stdout, _ = c3.exec_command("tail -2 /home/wangchong/data/fwz/output/verification/roundtrip_runner.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
print(f"Runner: {log}")
_, stdout, _ = c3.exec_command("ps aux | grep evaluate_verification | grep -v grep | wc -l", timeout=10)
cnt = stdout.read().decode().strip()
print(f"Running processes: {cnt}")

c2.close()
c3.close()
