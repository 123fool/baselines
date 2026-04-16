#!/usr/bin/env python3
"""Launch S36 and S35 baseline evaluations on valid set (44 subjects)
These models are already trained - can run now while v2 trains
"""
import paramiko, time

HOST, PORT, USER, PASS = '10.96.27.109', 2638, 'wangchong', '123456'
CSV       = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
AEKL_CKPT = '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'
DIFF_CKPT = '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'
CNET_CKPT = '/home/wangchong/data/fwz/output/34_hippo_training/H1_a30/cnet-hippo-H1_a30-ep2.pth'
CODE_DIR  = '/home/wangchong/data/fwz/code/37_expanded_validation/scripts'
OUTPUT_DIR= '/home/wangchong/data/fwz/output/37_expanded_validation'
EVAL_DIR  = f'{OUTPUT_DIR}/eval'
PYTHON    = '/home/wangchong/miniconda3/envs/fwz/bin/python'
S36_REFC  = '/home/wangchong/data/fwz/output/36_refinement/RefC/refnet-RefC-ep4.pth'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PASS, timeout=15)

# Create eval dir
stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {EVAL_DIR}', timeout=10)
stdout.read()

evals = [
    {
        'label': 'S36_RefC_H1a30_valid44',
        'ref_ckpt': S36_REFC,
        'gpu': 0,
    },
    {
        'label': 'S35best_noref_valid44',
        'ref_ckpt': None,  # no refinement
        'gpu': 1,
    },
]

for ev in evals:
    ref_arg = f"--ref_ckpt {ev['ref_ckpt']}" if ev['ref_ckpt'] else ''
    cmd = (
        f"cd {CODE_DIR} && "
        f"nohup {PYTHON} evaluate_refinement_v2.py "
        f"  --csv {CSV} "
        f"  --aekl_ckpt {AEKL_CKPT} "
        f"  --diff_ckpt {DIFF_CKPT} "
        f"  --cnet_ckpt {CNET_CKPT} "
        f"  {ref_arg} "
        f"  --eval_split valid "
        f"  --n_test 44 "
        f"  --m_las 3 "
        f"  --output_json {EVAL_DIR}/{ev['label']}.json "
        f"  --progress_file {EVAL_DIR}/{ev['label']}_progress.json "
        f"  --label {ev['label']} "
        f"  --gpu {ev['gpu']} "
        f"> {EVAL_DIR}/{ev['label']}.log 2>&1 &"
    )
    
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    time.sleep(1)  # fire-and-forget
    print(f"Launched: {ev['label']} on GPU {ev['gpu']}")
    time.sleep(3)

print("\nBoth valid-set evals launched!")
print("  S36_RefC_H1a30_valid44 → GPU 0 (alongside RefC_v2_cont training)")
print("  S35best_noref_valid44  → GPU 1 (alongside RefC_v2_fresh training)")

ssh.close()
