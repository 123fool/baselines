"""Launch RLP evaluation on server."""
import paramiko, sys, time

HOST = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'
PYTHON = '/home/wangchong/miniconda3/envs/fwz/bin/python'
BASE = '/home/wangchong/data/fwz/code/priority_2_rlp'

AE = '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'
DIFF = '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'
CSV = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'

RLP_CKPT = '/home/wangchong/data/fwz/output/priority_2_rlp/rlp_only/controlnet/cnet-rlp-ep-4.pth'
BTR_RLP_CKPT = '/home/wangchong/data/fwz/output/priority_2_rlp/btr_rlp/controlnet/cnet-btr-rlp-ep-4.pth'
EVAL_DIR = '/home/wangchong/data/fwz/output/priority_2_rlp/eval'

mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, PORT, USER, PASS, timeout=10)

# Build eval commands
cmds = []
if mode in ('rlp', 'both'):
    cmds.append(f"""
export CUDA_VISIBLE_DEVICES=1
cd {BASE}
{PYTHON} scripts/evaluate_rlp.py \\
    --dataset_csv {CSV} \\
    --aekl_ckpt {AE} \\
    --diff_ckpt {DIFF} \\
    --cnet_ckpt {RLP_CKPT} \\
    --output_dir {EVAL_DIR} \\
    --max_pairs 50 \\
    --model_name rlp_only_ep4
""")

if mode in ('btr', 'both'):
    cmds.append(f"""
export CUDA_VISIBLE_DEVICES=1
cd {BASE}
{PYTHON} scripts/evaluate_rlp.py \\
    --dataset_csv {CSV} \\
    --aekl_ckpt {AE} \\
    --diff_ckpt {DIFF} \\
    --cnet_ckpt {BTR_RLP_CKPT} \\
    --output_dir {EVAL_DIR} \\
    --max_pairs 50 \\
    --model_name btr_rlp_ep4
""")

# Join all commands with && and wrap in nohup
full_script = " && ".join(c.strip() for c in cmds)
launcher = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd {BASE}
mkdir -p {EVAL_DIR}
nohup bash -c '{full_script}' > {EVAL_DIR}/eval.log 2>&1 &
echo "PID=$!"
"""

# Upload and run launcher
sftp = client.open_sftp()
with sftp.open(f'{BASE}/_run_eval.sh', 'w') as f:
    f.write(launcher)
sftp.close()

stdin, stdout, stderr = client.exec_command(f'bash {BASE}/_run_eval.sh', timeout=15)
out = stdout.read().decode()
err = stderr.read().decode()
print(out)
if err:
    print("STDERR:", err)

# Quick check
time.sleep(2)
_, stdout2, _ = client.exec_command(f'ps aux | grep evaluate_rlp | grep -v grep')
procs = stdout2.read().decode().strip()
if procs:
    print("Evaluation running:")
    print(procs)
else:
    print("WARNING: No evaluation process found!")

client.close()
