#!/usr/bin/env python3
"""
S37 Recovery & Extension Launcher
==================================
After server power outage at 00:30, training was interrupted at epochs 3-4/20.
This script:
1. Re-uploads updated scripts (with --eval_split support)
2. Resumes training from best checkpoints (remaining epochs)
3. Evaluates best models on BOTH test (50) and valid (44) sets
"""
import paramiko
import sys
import time

HOST, PORT, USER, PASS = '10.96.27.109', 2638, 'wangchong', '123456'
CSV       = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
AE_CKPT   = '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'
AE_DEC    = '/home/wangchong/data/fwz/output/35_multiregion/ExpC_l1ssim_multi/ae-v2-l1ssim_multi_a30-ep2.pth'
DIFF_CKPT = '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'
CNET_CKPT = '/home/wangchong/data/fwz/output/34_hippo_training/H1_a30/cnet-hippo-H1_a30-ep2.pth'
CODE_DIR  = '/home/wangchong/data/fwz/code/37_expanded_validation/scripts'
OUTPUT_DIR= '/home/wangchong/data/fwz/output/37_expanded_validation'
EVAL_DIR  = f'{OUTPUT_DIR}/eval'
PYTHON    = '/home/wangchong/miniconda3/envs/fwz/bin/python'

def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, timeout=15)
    return ssh

def run_cmd(ssh, cmd, label='', wait=True):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    if label:
        print(f'\n[{label}]')
    if wait:
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(out)
        if err and 'Hint:' not in err:
            print(f'[ERR] {err}')
        return out
    else:
        # Fire-and-forget for nohup commands
        time.sleep(1)
        return ''

def phase_upload(ssh):
    """Upload updated scripts"""
    import os
    sftp = ssh.open_sftp()
    local_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    run_cmd(ssh, f'mkdir -p {CODE_DIR}', 'mkdir')
    
    for fname in ['train_refinement_v2.py', 'evaluate_refinement_v2.py']:
        local_path = os.path.join(local_dir, fname)
        remote_path = f'{CODE_DIR}/{fname}'
        sftp.put(local_path, remote_path)
        print(f'  Uploaded: {fname}')
    sftp.close()
    print('Upload complete.')

def phase_resume_train(ssh):
    """Resume training from best checkpoints"""
    run_cmd(ssh, f'mkdir -p {OUTPUT_DIR}', 'mkdir')
    
    # Each experiment: resume from its best checkpoint, train remaining epochs
    experiments = [
        {
            'name': 'RefC_v2_cont',
            'gpu': 0,
            'resume': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',
            'lr': 3e-5,    # lower LR for fine-tuning (already partially trained)
            'noise_aug': 0.5,
            'epochs': 17,  # 20-3 = 17 remaining
        },
        {
            'name': 'RefC_v2_fresh',
            'gpu': 1,
            'resume': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',
            'lr': 5e-5,    # lower than initial 1e-4 since already 3 epochs in
            'noise_aug': 0.5,
            'epochs': 17,
        },
        {
            'name': 'RefD_v2_highnoise',
            'gpu': 2,
            'resume': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth',
            'lr': 5e-5,
            'noise_aug': 0.8,
            'epochs': 16,  # 20-4 = 16 remaining
        },
    ]
    
    for exp in experiments:
        # Backup old training log before overwriting
        old_log = f"{OUTPUT_DIR}/{exp['name']}/training_log.json"
        backup_log = f"{OUTPUT_DIR}/{exp['name']}/training_log_before_outage.json"
        run_cmd(ssh, f'cp {old_log} {backup_log} 2>/dev/null')
        
        cmd = (
            f"cd {CODE_DIR} && "
            f"nohup {PYTHON} train_refinement_v2.py "
            f"  --csv {CSV} "
            f"  --ae_ckpt {AE_CKPT} "
            f"  --ae_decoder_ckpt {AE_DEC} "
            f"  --resume_ckpt {exp['resume']} "
            f"  --output_dir {OUTPUT_DIR} "
            f"  --exp_name {exp['name']} "
            f"  --loss_type l1_ssim_region "
            f"  --noise_aug {exp['noise_aug']} "
            f"  --epochs {exp['epochs']} "
            f"  --lr {exp['lr']} "
            f"  --patience 5 "
            f"  --gpu {exp['gpu']} "
            f"> {OUTPUT_DIR}/{exp['name']}_resume_train.log 2>&1 &"
        )
        run_cmd(ssh, cmd, f"Resume {exp['name']} on GPU{exp['gpu']}", wait=False)
        print(f"  → {exp['name']}: LR={exp['lr']}, epochs={exp['epochs']}, noise={exp['noise_aug']}")
        time.sleep(3)
    
    print('\nAll 3 training experiments resumed!')

def phase_eval_test(ssh):
    """Evaluate best models on test set (50 subjects)"""
    run_cmd(ssh, f'mkdir -p {EVAL_DIR}', 'mkdir')
    
    evals = [
        # Evaluate each v2 model on TEST set
        {
            'label': 'RefC_v2_cont_50subj',
            'ref_ckpt': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',
            'eval_split': 'test',
            'n_test': 50,
            'gpu': 0,
        },
        {
            'label': 'RefC_v2_fresh_50subj',
            'ref_ckpt': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',
            'eval_split': 'test',
            'n_test': 50,
            'gpu': 1,
        },
        {
            'label': 'RefD_v2_highnoise_50subj',
            'ref_ckpt': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth',
            'eval_split': 'test',
            'n_test': 50,
            'gpu': 2,
        },
    ]
    
    for ev in evals:
        cmd = (
            f"cd {CODE_DIR} && "
            f"nohup {PYTHON} evaluate_refinement_v2.py "
            f"  --csv {CSV} "
            f"  --aekl_ckpt {AE_CKPT} "
            f"  --diff_ckpt {DIFF_CKPT} "
            f"  --cnet_ckpt {CNET_CKPT} "
            f"  --ref_ckpt {ev['ref_ckpt']} "
            f"  --eval_split {ev['eval_split']} "
            f"  --n_test {ev['n_test']} "
            f"  --m_las 3 "
            f"  --output_json {EVAL_DIR}/{ev['label']}.json "
            f"  --progress_file {EVAL_DIR}/{ev['label']}_progress.json "
            f"  --label {ev['label']} "
            f"  --gpu {ev['gpu']} "
            f"> {EVAL_DIR}/{ev['label']}.log 2>&1 &"
        )
        run_cmd(ssh, cmd, f"Eval {ev['label']}", wait=False)
        time.sleep(3)
    
    print('\nAll test-set evaluations launched!')

def phase_eval_valid(ssh):
    """Evaluate best models on valid set (44 subjects) as cross-validation"""
    run_cmd(ssh, f'mkdir -p {EVAL_DIR}', 'mkdir')
    
    # All models to evaluate on valid set: S36 best, S35 baseline, v2 models
    evals = [
        {
            'label': 'S36_RefC_H1a30_valid44',
            'ref_ckpt': '/home/wangchong/data/fwz/output/36_refinement/RefC/refnet-RefC-ep4.pth',
            'eval_split': 'valid',
            'n_test': 44,
            'gpu': 0,
        },
        {
            'label': 'S35best_noref_valid44',
            'ref_ckpt': '',  # no refinement
            'eval_split': 'valid',
            'n_test': 44,
            'gpu': 1,
        },
        {
            'label': 'RefC_v2_cont_valid44',
            'ref_ckpt': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',
            'eval_split': 'valid',
            'n_test': 44,
            'gpu': 0,
        },
        {
            'label': 'RefC_v2_fresh_valid44',
            'ref_ckpt': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',
            'eval_split': 'valid',
            'n_test': 44,
            'gpu': 1,
        },
        {
            'label': 'RefD_v2_highnoise_valid44',
            'ref_ckpt': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth',
            'eval_split': 'valid',
            'n_test': 44,
            'gpu': 2,
        },
    ]
    
    for ev in evals:
        ref_arg = f"--ref_ckpt {ev['ref_ckpt']}" if ev['ref_ckpt'] else ''
        cmd = (
            f"cd {CODE_DIR} && "
            f"nohup {PYTHON} evaluate_refinement_v2.py "
            f"  --csv {CSV} "
            f"  --aekl_ckpt {AE_CKPT} "
            f"  --diff_ckpt {DIFF_CKPT} "
            f"  --cnet_ckpt {CNET_CKPT} "
            f"  {ref_arg} "
            f"  --eval_split {ev['eval_split']} "
            f"  --n_test {ev['n_test']} "
            f"  --m_las 3 "
            f"  --output_json {EVAL_DIR}/{ev['label']}.json "
            f"  --progress_file {EVAL_DIR}/{ev['label']}_progress.json "
            f"  --label {ev['label']} "
            f"  --gpu {ev['gpu']} "
            f"> {EVAL_DIR}/{ev['label']}.log 2>&1 &"
        )
        run_cmd(ssh, cmd, f"Eval {ev['label']}", wait=False)
        time.sleep(3)
    
    print('\nAll valid-set evaluations launched!')


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'help'
    
    if phase == 'help':
        print("""Usage: python _recover_s37.py <phase>
Phases:
  upload       - Upload updated scripts (with --eval_split)
  resume       - Resume interrupted training (3 experiments)
  eval_test    - Evaluate v2 models on test set (50 subjects)
  eval_valid   - Evaluate all models on valid set (44 subjects, cross-validation)
  all_eval     - Run both eval_test and eval_valid
""")
        sys.exit(0)
    
    ssh = get_ssh()
    
    if phase == 'upload':
        phase_upload(ssh)
    elif phase == 'resume':
        phase_upload(ssh)  # always upload first
        phase_resume_train(ssh)
    elif phase == 'eval_test':
        phase_eval_test(ssh)
    elif phase == 'eval_valid':
        phase_eval_valid(ssh)
    elif phase == 'all_eval':
        phase_eval_test(ssh)
        time.sleep(5)
        phase_eval_valid(ssh)
    else:
        print(f'Unknown phase: {phase}')
    
    ssh.close()
