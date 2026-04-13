#!/usr/bin/env python3
"""
Master Deployment & Execution Script
=====================================
Uploads all method scripts to server and runs them sequentially.
Run this from the local Windows machine.

Usage:
  python deploy_and_run.py
"""

import subprocess
import sys
import time
import os

# Server config
SERVER = "10.96.27.109"
PORT = "2638"
USER = "wangchong"
PASS = "123456"
REMOTE_CODE = "/home/wangchong/data/fwz/code"
REMOTE_OUTPUT = "/home/wangchong/data/fwz/output"
CONDA_ENV = "fwz"

# Local paths
BASE = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.join(BASE, "..", "src")

def ssh_cmd(cmd, timeout=None):
    """Execute command on remote server via SSH."""
    full_cmd = f'sshpass -p "{PASS}" ssh -o StrictHostKeyChecking=no -p {PORT} {USER}@{SERVER} "{cmd}"'
    print(f"[SSH] {cmd}")
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True,
            timeout=timeout
        )
        if result.stdout:
            print(result.stdout[:2000])
        if result.returncode != 0 and result.stderr:
            print(f"[STDERR] {result.stderr[:1000]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Command timed out")
        return False


def scp_upload(local_path, remote_path):
    """Upload file/directory to server."""
    if os.path.isdir(local_path):
        cmd = f'sshpass -p "{PASS}" scp -o StrictHostKeyChecking=no -P {PORT} -r "{local_path}" {USER}@{SERVER}:{remote_path}'
    else:
        cmd = f'sshpass -p "{PASS}" scp -o StrictHostKeyChecking=no -P {PORT} "{local_path}" {USER}@{SERVER}:{remote_path}'
    print(f"[SCP] {local_path} -> {remote_path}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("MASTER DEPLOYMENT SCRIPT")
    print("=" * 60)

    # 1. Create remote directories
    print("\n[1/6] Creating remote directories...")
    dirs = [
        f"{REMOTE_CODE}/brlp_src/scripts",
        f"{REMOTE_OUTPUT}/method_b_time_aware/controlnet",
        f"{REMOTE_OUTPUT}/method_b_time_aware/eval",
        f"{REMOTE_OUTPUT}/method_c_identity/controlnet",
        f"{REMOTE_OUTPUT}/method_c_identity/eval",
        f"{REMOTE_OUTPUT}/method_d_freq/controlnet",
        f"{REMOTE_OUTPUT}/method_d_freq/eval",
        f"{REMOTE_OUTPUT}/enhanced_eval",
    ]
    for d in dirs:
        ssh_cmd(f"mkdir -p {d}")

    # 2. Upload brlp source code
    print("\n[2/6] Uploading brlp source code...")
    scp_upload(BRLP_SRC + "/brlp", f"{REMOTE_CODE}/brlp_src/brlp")

    # 3. Upload method scripts
    print("\n[3/6] Uploading method scripts...")
    scripts = {
        os.path.join(BASE, "20_method_b_time_aware", "train_time_aware.py"):
            f"{REMOTE_CODE}/brlp_src/scripts/method_b_time_aware.py",
        os.path.join(BASE, "20_method_b_time_aware", "evaluate_method_b.py"):
            f"{REMOTE_CODE}/brlp_src/scripts/evaluate_method_b.py",
        os.path.join(BASE, "21_method_c_identity", "train_identity.py"):
            f"{REMOTE_CODE}/brlp_src/scripts/method_c_identity.py",
        os.path.join(BASE, "22_method_d_frequency", "train_frequency.py"):
            f"{REMOTE_CODE}/brlp_src/scripts/method_d_frequency.py",
        os.path.join(BASE, "23_unified_eval", "evaluate_all_methods.py"):
            f"{REMOTE_CODE}/brlp_src/scripts/evaluate_all_methods.py",
        os.path.join(BASE, "19_enhanced_eval", "evaluate_enhanced.py"):
            f"{REMOTE_CODE}/brlp_src/scripts/evaluate_enhanced.py",
    }
    for local, remote in scripts.items():
        if os.path.exists(local):
            scp_upload(local, remote)
        else:
            print(f"  [SKIP] {local} not found")

    # 4. Run Method B: Time-Aware Context Training
    print("\n[4/6] Starting Method B training (time-aware context)...")
    train_b_cmd = (
        f"cd {REMOTE_CODE}/brlp_src && "
        f"source activate {CONDA_ENV} && "
        f"nohup python -m scripts.method_b_time_aware "
        f"--dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv "
        f"--cache_dir /home/wangchong/data/fwz/brlp-data/cache_time_aware "
        f"--output_dir {REMOTE_OUTPUT}/method_b_time_aware/controlnet "
        f"--aekl_ckpt {REMOTE_OUTPUT}/innovation_5/ae/autoencoder-ep-2.pth "
        f"--diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth "
        f"--n_epochs 5 --batch_size 4 --lr 2.5e-5 "
        f"> {REMOTE_OUTPUT}/method_b_time_aware/train.log 2>&1 &"
    )
    ssh_cmd(train_b_cmd)
    print("[4/6] Method B training started in background")

    # 5. Run Method C: Identity-Preserving Training  
    print("\n[5/6] Starting Method C training (identity-preserving)...")
    train_c_cmd = (
        f"cd {REMOTE_CODE}/brlp_src && "
        f"source activate {CONDA_ENV} && "
        f"nohup python -m scripts.method_c_identity "
        f"--dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv "
        f"--cache_dir /home/wangchong/data/fwz/brlp-data/cache_identity "
        f"--output_dir {REMOTE_OUTPUT}/method_c_identity/controlnet "
        f"--aekl_ckpt {REMOTE_OUTPUT}/innovation_5/ae/autoencoder-ep-2.pth "
        f"--diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth "
        f"--n_epochs 5 --batch_size 4 --lr 2.5e-5 "
        f"--lambda_id 0.1 --lambda_con 0.05 --context_mode time_aware "
        f"> {REMOTE_OUTPUT}/method_c_identity/train.log 2>&1 &"
    )
    ssh_cmd(train_c_cmd)
    print("[5/6] Method C training started in background")

    # 6. Run Method D: Frequency Loss Training
    print("\n[6/6] Starting Method D training (frequency loss)...")
    train_d_cmd = (
        f"cd {REMOTE_CODE}/brlp_src && "
        f"source activate {CONDA_ENV} && "  
        f"nohup python -m scripts.method_d_frequency "
        f"--dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv "
        f"--cache_dir /home/wangchong/data/fwz/brlp-data/cache_freq "
        f"--output_dir {REMOTE_OUTPUT}/method_d_freq/controlnet "
        f"--aekl_ckpt {REMOTE_OUTPUT}/innovation_5/ae/autoencoder-ep-2.pth "
        f"--diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth "
        f"--n_epochs 5 --batch_size 4 --lr 2.5e-5 "
        f"--lambda_freq 0.01 --lambda_smooth 0.005 "
        f"> {REMOTE_OUTPUT}/method_d_freq/train.log 2>&1 &"
    )
    ssh_cmd(train_d_cmd)
    print("[6/6] Method D training started in background")

    print("\n" + "=" * 60)
    print("ALL TRAINING JOBS STARTED IN BACKGROUND")
    print("=" * 60)
    print(f"\nMonitor with:")
    print(f"  ssh -p {PORT} {USER}@{SERVER}")
    print(f"  tail -f {REMOTE_OUTPUT}/method_b_time_aware/train.log")
    print(f"  tail -f {REMOTE_OUTPUT}/method_c_identity/train.log")
    print(f"  tail -f {REMOTE_OUTPUT}/method_d_freq/train.log")
    print(f"\nAfter training, run evaluation:")
    print(f"  python evaluate_all_methods.py --method_name Method-B ...")


if __name__ == "__main__":
    main()
