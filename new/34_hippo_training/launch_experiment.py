"""
Upload training scripts to server and run the hippocampus training pipeline.

Steps:
  1. Upload all scripts to /home/wangchong/data/fwz/code/34_hippo_training/
  2. Run prepare_hippo_mask.py
  3. Run check_ae_ceiling.py
  4. Run train_controlnet_hippo.py for each method
  5. Run evaluate_checkpoint.py for each checkpoint
"""
import paramiko
import os
import time

# ─── Config ───────────────────────────────────────────────────────────
HOST = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'

REMOTE_BASE = '/home/wangchong/data/fwz/code/34_hippo_training'
REMOTE_OUT = '/home/wangchong/data/fwz/output/34_hippo_training'
LOCAL_SCRIPTS = os.path.join(os.path.dirname(__file__), 'scripts')

# Paths on server
CSV = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
AE_CKPT = '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'
DIFF_CKPT = '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'
CNET_BTR = '/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth'
CACHE_DIR = '/home/wangchong/data/fwz/cache'
MASK_PATH = f'{REMOTE_OUT}/masks/hippo_latent_mask.npy'

CONDA_ACT = 'source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz'
GPU = '1'  # Free GPU


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return ssh


def run_cmd(ssh, cmd, timeout=600, label=''):
    """Run command on server and print output."""
    full_cmd = f'{CONDA_ACT} && CUDA_VISIBLE_DEVICES={GPU} {cmd}'
    print(f'\n{"="*60}')
    print(f'[{label}] Running: {cmd[:120]}...')
    print(f'{"="*60}')
    
    _, stdout, stderr = ssh.exec_command(full_cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    
    if out:
        print(out)
    if err:
        # Filter noisy warnings
        for line in err.split('\n'):
            if line.strip() and not any(w in line for w in 
                ['Warning', 'deprecated', 'FutureWarning', 'UserWarning']):
                print(f'ERR: {line}')
    
    return out, err


def upload_scripts(ssh):
    """Upload all scripts to server."""
    sftp = ssh.open_sftp()
    
    # Create dirs
    for d in [REMOTE_BASE, f'{REMOTE_BASE}/scripts', f'{REMOTE_OUT}/masks',
              f'{REMOTE_OUT}/eval']:
        try:
            sftp.mkdir(d)
        except IOError:
            pass
    
    # Upload scripts
    for fn in os.listdir(LOCAL_SCRIPTS):
        if fn.endswith('.py'):
            local = os.path.join(LOCAL_SCRIPTS, fn)
            remote = f'{REMOTE_BASE}/scripts/{fn}'
            sftp.put(local, remote)
            print(f'  Uploaded: {fn}')
    
    sftp.close()


def step1_prepare_mask(ssh):
    """Pre-compute hippocampus latent mask."""
    cmd = (f'cd {REMOTE_BASE} && python scripts/prepare_hippo_mask.py '
           f'--csv {CSV} --output {MASK_PATH} --n_samples 50')
    return run_cmd(ssh, cmd, timeout=120, label='MASK')


def step2_ae_ceiling(ssh):
    """Check AE reconstruction ceiling."""
    cmd = (f'cd {REMOTE_BASE} && python scripts/check_ae_ceiling.py '
           f'--dataset_csv {CSV} --aekl_ckpt {AE_CKPT} --n_test 10')
    return run_cmd(ssh, cmd, timeout=300, label='AE-CEILING')


def step3_train(ssh, method, alpha):
    """Train one configuration."""
    tag = f'{method}_a{int(alpha)}'
    out_dir = f'{REMOTE_OUT}/{tag}'
    cmd = (f'cd {REMOTE_BASE} && python scripts/train_controlnet_hippo.py '
           f'--dataset_csv {CSV} --cache_dir {CACHE_DIR} '
           f'--output_dir {out_dir} '
           f'--aekl_ckpt {AE_CKPT} --diff_ckpt {DIFF_CKPT} '
           f'--cnet_ckpt {CNET_BTR} --hippo_mask {MASK_PATH} '
           f'--method {method} --alpha {alpha} '
           f'--n_epochs 3 --lr 1e-5 --batch_size 16')
    return run_cmd(ssh, cmd, timeout=1800, label=f'TRAIN-{tag}')


def step4_evaluate(ssh, cnet_path, label):
    """Evaluate a checkpoint."""
    out_json = f'{REMOTE_OUT}/eval/{label}.json'
    cmd = (f'cd {REMOTE_BASE} && python scripts/evaluate_checkpoint.py '
           f'--dataset_csv {CSV} --aekl_ckpt {AE_CKPT} '
           f'--diff_ckpt {DIFF_CKPT} --cnet_ckpt {cnet_path} '
           f'--cache_dir {CACHE_DIR} --n_test 5 --m_las 3 '
           f'--output_json {out_json} --label {label}')
    return run_cmd(ssh, cmd, timeout=600, label=f'EVAL-{label}')


def main():
    import sys

    action = sys.argv[1] if len(sys.argv) > 1 else 'all'
    ssh = ssh_connect()

    if action in ('all', 'upload'):
        print('\n>>> Uploading scripts...')
        upload_scripts(ssh)

    if action in ('all', 'mask'):
        step1_prepare_mask(ssh)

    if action in ('all', 'ae_ceiling'):
        step2_ae_ceiling(ssh)

    if action in ('all', 'train'):
        # Train all methods
        configs = [
            ('H1', 10),
            ('H1', 30),
            ('H1', 50),
            ('H2', 30),  # with timestep bias
        ]
        for method, alpha in configs:
            step3_train(ssh, method, alpha)

    if action in ('all', 'eval'):
        # Evaluate baseline + all trained checkpoints
        evals = [
            (CNET_BTR, 'baseline_BTR'),
        ]
        # Add trained checkpoints (last epoch of each)
        for method, alpha in [('H1', 10), ('H1', 30), ('H1', 50), ('H2', 30)]:
            tag = f'{method}_a{int(alpha)}'
            for ep in range(3):
                ckpt = f'{REMOTE_OUT}/{tag}/cnet-hippo-{tag}-ep{ep}.pth'
                evals.append((ckpt, f'{tag}_ep{ep}'))

        for ckpt, label in evals:
            step4_evaluate(ssh, ckpt, label)

    if action == 'eval_best':
        # Evaluate specific checkpoints
        ckpt = sys.argv[2]
        label = sys.argv[3] if len(sys.argv) > 3 else 'custom'
        step4_evaluate(ssh, ckpt, label)

    ssh.close()
    print('\n>>> Done.')


if __name__ == '__main__':
    main()
