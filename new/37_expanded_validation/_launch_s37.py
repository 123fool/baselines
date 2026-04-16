"""
Section 37: Launch expanded-scale training and evaluation.
Uploads scripts and launches experiments on the server.
"""
import os, sys, time, json, argparse

# paramiko for SSH
import paramiko

HOST = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'
PYTHON = '/home/wangchong/miniconda3/envs/fwz/bin/python'

# Paths
CODE_DIR = '/home/wangchong/data/fwz/code/37_expanded_validation/scripts'
OUTPUT_DIR = '/home/wangchong/data/fwz/output/37_expanded_validation'
CSV = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
AE_CKPT = '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'
AE_DEC = '/home/wangchong/data/fwz/output/35_multiregion/ExpC_l1ssim_multi/ae-v2-l1ssim_multi_a30-ep2.pth'
DIFF_CKPT = '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'
CNET_H1A30 = '/home/wangchong/data/fwz/output/34_hippo_training/H1_a30/cnet-hippo-H1_a30-ep2.pth'
CNET_BTR = '/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth'

# S36 best checkpoint for resume
S36_REFC_BEST = '/home/wangchong/data/fwz/output/36_refinement/RefC/refnet-RefC-ep4.pth'

LOCAL_SCRIPTS = os.path.join(os.path.dirname(__file__), 'scripts')


def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, timeout=30)
    return ssh


def run_cmd(ssh, cmd, timeout=60):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err


def upload_scripts(ssh):
    sftp = ssh.open_sftp()
    # Create remote directory
    for d in ['/home/wangchong/data/fwz/code/37_expanded_validation',
              CODE_DIR, OUTPUT_DIR]:
        try:
            sftp.mkdir(d)
        except:
            pass

    for fname in os.listdir(LOCAL_SCRIPTS):
        if fname.endswith('.py'):
            local = os.path.join(LOCAL_SCRIPTS, fname)
            remote = f'{CODE_DIR}/{fname}'
            sftp.put(local, remote)
            print(f'  Uploaded: {fname}')
    sftp.close()


def launch_training(ssh, args):
    """Launch 3 training experiments on 3 GPUs."""
    experiments = [
        # Exp 1: Continue from S36 RefC (best), 20 more epochs
        {
            'name': 'RefC_v2_cont',
            'gpu': 0,
            'loss': 'l1_ssim_region',
            'noise_aug': 0.5,
            'resume': S36_REFC_BEST,
            'epochs': 20,
            'lr': 5e-5,  # Lower LR for fine-tuning
        },
        # Exp 2: Fresh training from scratch, 20 epochs
        {
            'name': 'RefC_v2_fresh',
            'gpu': 1,
            'loss': 'l1_ssim_region',
            'noise_aug': 0.5,
            'resume': None,
            'epochs': 20,
            'lr': 1e-4,
        },
        # Exp 3: Higher noise augmentation (0.8) for more robustness
        {
            'name': 'RefD_v2_highnoise',
            'gpu': 2,
            'loss': 'l1_ssim_region',
            'noise_aug': 0.8,
            'resume': None,
            'epochs': 20,
            'lr': 1e-4,
        },
    ]

    # Create shell script
    lines = ['#!/bin/bash', f'cd {CODE_DIR}', '']
    for exp in experiments:
        cmd = f'nohup {PYTHON} train_refinement_v2.py'
        cmd += f' --csv {CSV} --ae_ckpt {AE_CKPT} --ae_decoder_ckpt {AE_DEC}'
        cmd += f' --output_dir {OUTPUT_DIR} --exp_name {exp["name"]}'
        cmd += f' --loss_type {exp["loss"]} --noise_aug {exp["noise_aug"]}'
        cmd += f' --epochs {exp["epochs"]} --lr {exp["lr"]} --gpu {exp["gpu"]}'
        cmd += f' --patience 5'
        if exp.get('resume'):
            cmd += f' --resume_ckpt {exp["resume"]}'
        cmd += f' > {OUTPUT_DIR}/{exp["name"]}_train.log 2>&1 &'
        lines.append(cmd)
        lines.append(f'echo "Launched {exp["name"]} on GPU {exp["gpu"]}"')
        lines.append('')

    lines.append('echo "All 3 training experiments launched"')
    script = '\n'.join(lines)

    # Upload and run
    sftp = ssh.open_sftp()
    with sftp.open(f'{OUTPUT_DIR}/run_training.sh', 'w') as f:
        f.write(script)
    sftp.close()

    out, err = run_cmd(ssh, f'bash {OUTPUT_DIR}/run_training.sh', timeout=30)
    print(out)
    return experiments


def launch_evaluation(ssh, args):
    """Launch full-scale evaluations (50 test subjects) for multiple configs."""
    eval_configs = [
        # 1. S36 best (RefC ep4) - verify with 50 subjects
        {
            'label': 'S36_RefC_H1a30_50subj',
            'ref_ckpt': S36_REFC_BEST,
            'cnet': CNET_H1A30,
            'gpu': 0,
        },
        # 2. No refinement baseline - 50 subjects
        {
            'label': 'S35best_noref_50subj',
            'ref_ckpt': None,
            'cnet': CNET_H1A30,
            'gpu': 1,
        },
        # 3-5 will be the v2 trained models (best checkpoints) - added after training completes
    ]

    eval_dir = f'{OUTPUT_DIR}/eval'
    run_cmd(ssh, f'mkdir -p {eval_dir}')

    lines = ['#!/bin/bash', f'cd {CODE_DIR}', '']
    for cfg in eval_configs:
        cmd = f'{PYTHON} evaluate_refinement_v2.py'
        cmd += f' --csv {CSV} --aekl_ckpt {AE_CKPT} --diff_ckpt {DIFF_CKPT}'
        cmd += f' --cnet_ckpt {cfg["cnet"]} --ae_decoder_ckpt {AE_DEC}'
        cmd += f' --n_test 50 --m_las 3 --gpu {cfg["gpu"]}'
        cmd += f' --output_json {eval_dir}/{cfg["label"]}.json'
        cmd += f' --progress_file {eval_dir}/{cfg["label"]}_progress.json'
        cmd += f' --label {cfg["label"]}'
        if cfg.get('ref_ckpt'):
            cmd += f' --ref_ckpt {cfg["ref_ckpt"]}'
        cmd += f' > {eval_dir}/{cfg["label"]}.log 2>&1'  # Sequential, not background
        lines.append(cmd)
        lines.append(f'echo "Completed {cfg["label"]}"')
        lines.append('')

    # Run evaluations in parallel on different GPUs
    script = '\n'.join(lines)
    sftp = ssh.open_sftp()
    with sftp.open(f'{eval_dir}/run_eval.sh', 'w') as f:
        f.write(script)
    sftp.close()

    # Run in background
    out, _ = run_cmd(ssh, f'nohup bash {eval_dir}/run_eval.sh > {eval_dir}/eval_master.log 2>&1 &', timeout=10)
    print(f'Evaluation launched (2 configs × 50 subjects each)')
    return eval_configs


def launch_v2_evaluation(ssh, args):
    """Launch evaluations on the v2 trained models (after training completes)."""
    eval_dir = f'{OUTPUT_DIR}/eval'
    run_cmd(ssh, f'mkdir -p {eval_dir}')

    eval_configs = [
        {
            'label': 'RefC_v2_cont_50subj',
            'ref_ckpt': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',
            'cnet': CNET_H1A30,
            'gpu': 0,
        },
        {
            'label': 'RefC_v2_fresh_50subj',
            'ref_ckpt': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',
            'cnet': CNET_H1A30,
            'gpu': 1,
        },
        {
            'label': 'RefD_v2_highnoise_50subj',
            'ref_ckpt': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth',
            'cnet': CNET_H1A30,
            'gpu': 2,
        },
    ]

    lines = ['#!/bin/bash', f'cd {CODE_DIR}', '']
    for cfg in eval_configs:
        cmd = f'{PYTHON} evaluate_refinement_v2.py'
        cmd += f' --csv {CSV} --aekl_ckpt {AE_CKPT} --diff_ckpt {DIFF_CKPT}'
        cmd += f' --cnet_ckpt {cfg["cnet"]} --ae_decoder_ckpt {AE_DEC}'
        cmd += f' --n_test 50 --m_las 3 --gpu {cfg["gpu"]}'
        cmd += f' --output_json {eval_dir}/{cfg["label"]}.json'
        cmd += f' --progress_file {eval_dir}/{cfg["label"]}_progress.json'
        cmd += f' --label {cfg["label"]}'
        cmd += f' --ref_ckpt {cfg["ref_ckpt"]}'
        cmd += f' > {eval_dir}/{cfg["label"]}.log 2>&1 &'
        lines.append(cmd)
        lines.append(f'echo "Launched {cfg["label"]} on GPU {cfg["gpu"]}"')
        lines.append('')

    lines.append('wait')
    lines.append('echo "All v2 evaluations complete"')

    script = '\n'.join(lines)
    sftp = ssh.open_sftp()
    with sftp.open(f'{eval_dir}/run_eval_v2.sh', 'w') as f:
        f.write(script)
    sftp.close()

    out, _ = run_cmd(ssh, f'nohup bash {eval_dir}/run_eval_v2.sh > {eval_dir}/eval_v2_master.log 2>&1 &', timeout=10)
    print(f'V2 evaluation launched (3 configs × 50 subjects each, parallel on 3 GPUs)')
    return eval_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['upload', 'train', 'eval_s36', 'eval_v2'], required=True)
    args = parser.parse_args()

    ssh = get_ssh()

    if args.phase == 'upload':
        print('=== Uploading scripts ===')
        upload_scripts(ssh)
        print('Done')

    elif args.phase == 'train':
        print('=== Launching training ===')
        upload_scripts(ssh)
        launch_training(ssh, args)

    elif args.phase == 'eval_s36':
        print('=== Launching S36 full re-evaluation ===')
        upload_scripts(ssh)
        launch_evaluation(ssh, args)

    elif args.phase == 'eval_v2':
        print('=== Launching V2 model evaluations ===')
        launch_v2_evaluation(ssh, args)

    ssh.close()
