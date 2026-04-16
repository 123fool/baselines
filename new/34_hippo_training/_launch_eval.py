"""Upload eval script and launch evaluation of completed checkpoints on GPU 1."""
import paramiko
import time
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Upload eval script
sftp = ssh.open_sftp()
sftp.put(
    r'c:\Users\PC\Desktop\baselines\BrLP-main\new\34_hippo_training\scripts\evaluate_checkpoint.py',
    '/home/wangchong/data/fwz/code/34_hippo_training/scripts/evaluate_checkpoint.py')
sftp.close()
print('Uploaded evaluate_checkpoint.py')

# Create eval output dir and shell script
ssh.exec_command('mkdir -p /home/wangchong/data/fwz/output/34_hippo_training/eval')
time.sleep(1)

# Write eval shell script
eval_script = """#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
export CUDA_VISIBLE_DEVICES=1

CSV=/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv
AE=/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth
DIFF=/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth
CACHE=/home/wangchong/data/fwz/cache
OUT=/home/wangchong/data/fwz/output/34_hippo_training
EVAL=$OUT/eval

CMD="python /home/wangchong/data/fwz/code/34_hippo_training/scripts/evaluate_checkpoint.py \
  --dataset_csv $CSV --aekl_ckpt $AE --diff_ckpt $DIFF \
  --cache_dir $CACHE --n_test 5 --m_las 3"

# Baseline BTR
echo ">>> Baseline BTR"
$CMD --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth \
  --label baseline_BTR --output_json $EVAL/baseline_BTR.json 2>&1 | tail -10

# H1_a10 ep2
echo ">>> H1_a10_ep2"
$CMD --cnet_ckpt $OUT/H1_a10/cnet-hippo-H1_a10-ep2.pth \
  --label H1_a10_ep2 --output_json $EVAL/H1_a10_ep2.json 2>&1 | tail -10

# H1_a30 ep2
echo ">>> H1_a30_ep2"
$CMD --cnet_ckpt $OUT/H1_a30/cnet-hippo-H1_a30-ep2.pth \
  --label H1_a30_ep2 --output_json $EVAL/H1_a30_ep2.json 2>&1 | tail -10

# H1_a50 ep2
echo ">>> H1_a50_ep2"
$CMD --cnet_ckpt $OUT/H1_a50/cnet-hippo-H1_a50-ep2.pth \
  --label H1_a50_ep2 --output_json $EVAL/H1_a50_ep2.json 2>&1 | tail -10

echo ">>> ALL EVALS DONE" >> $EVAL/eval_chain.log
echo "=== Evaluation complete ==="
"""

sftp = ssh.open_sftp()
with sftp.open('/home/wangchong/data/fwz/code/34_hippo_training/run_eval.sh', 'w') as f:
    f.write(eval_script)
sftp.close()

ssh.exec_command('chmod +x /home/wangchong/data/fwz/code/34_hippo_training/run_eval.sh')
time.sleep(1)

# Wait for H2_a30 to finish or detect GPU 1 is free
_, o, _ = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used --format=csv,noheader')
gpu_info = o.read().decode()
print('GPU Status:', gpu_info.strip())

# Check if H2_a30 is still running on GPU 1
_, o, _ = ssh.exec_command('pgrep -f "H2_a30" 2>/dev/null')
h2_pid = o.read().decode().strip()

if h2_pid:
    print(f'H2_a30 still running (PID {h2_pid}), waiting for it to finish before eval...')
    # Queue eval after H2_a30 finishes
    ssh.exec_command(
        f'while kill -0 {h2_pid} 2>/dev/null; do sleep 5; done; '
        'nohup bash /home/wangchong/data/fwz/code/34_hippo_training/run_eval.sh '
        '> /home/wangchong/data/fwz/output/34_hippo_training/eval/eval.log 2>&1 &')
    print('Queued evaluation to run after H2_a30 completes')
else:
    print('GPU 1 is free, launching evaluation now')
    ssh.exec_command(
        'nohup bash /home/wangchong/data/fwz/code/34_hippo_training/run_eval.sh '
        '> /home/wangchong/data/fwz/output/34_hippo_training/eval/eval.log 2>&1 &')
    print('Launched evaluation')

ssh.close()
