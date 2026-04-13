"""
Upload Combined Inn1+Inn2 (6ch+BTR) code to server and start training.

Server structure:
  /home/wangchong/data/fwz/code/combined_inn1_inn2/
    scripts/
      train_controlnet_6ch_btr.py
      evaluate_6ch_btr.py
    src/
      mci_conditioning.py
    brlp_src/
      brlp/  (BrLP source package)
    train.sh
    eval.sh
"""

import os
import sys
import time
import paramiko


SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

LOCAL_BASE = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(LOCAL_BASE, "..", "..", "src", "brlp"))

REMOTE_BASE = "/home/wangchong/data/fwz/code/combined_inn1_inn2"
OUTPUT_DIR = "/home/wangchong/data/fwz/output/combined_inn1_inn2"

# Server paths — use Innovation 1's prepared CSV (has rate columns)
DATASET_CSV = "/home/wangchong/data/fwz/output/innovation_1/prepared/B_mci_inn1.csv"
CACHE_DIR   = "/home/wangchong/data/fwz/cache/combined_inn1_inn2"
AEKL_CKPT   = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT   = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_4CH    = "/home/wangchong/data/fwz/brlp-train/pretrained/controlnet.pth"
PYTHON      = "/home/wangchong/miniconda3/envs/fwz/bin/python"


def upload_tree(sftp, local_dir, remote_dir, ext=(".py", ".sh")):
    """Recursively upload directory tree."""
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        sftp.mkdir(remote_dir)

    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = f"{remote_dir}/{item}"
        if os.path.isdir(local_path):
            if item == '__pycache__':
                continue
            upload_tree(sftp, local_path, remote_path, ext)
        elif os.path.isfile(local_path):
            if ext is None or any(item.endswith(e) for e in ext):
                sftp.put(local_path, remote_path)
                print(f"  Uploaded: {remote_path}")


def main():
    print("=" * 60)
    print("[Combined Inn1+Inn2] 6ch ControlNet + BTR — Upload & Train")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT,
                   username=SERVER_USER, password=SERVER_PASS, timeout=15)
    sftp = client.open_sftp()

    # Create directory structure
    for d in [
        REMOTE_BASE,
        f"{REMOTE_BASE}/scripts",
        f"{REMOTE_BASE}/src",
        f"{REMOTE_BASE}/brlp_src",
        f"{REMOTE_BASE}/brlp_src/brlp",
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/controlnet",
        f"{OUTPUT_DIR}/eval",
        CACHE_DIR,
    ]:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)
            print(f"  Created: {d}")

    # Upload combined scripts
    print("\n--- Combined scripts ---")
    upload_tree(sftp, os.path.join(LOCAL_BASE, "scripts"), f"{REMOTE_BASE}/scripts")

    # Upload source modules
    print("\n--- Source modules ---")
    upload_tree(sftp, os.path.join(LOCAL_BASE, "src"), f"{REMOTE_BASE}/src")

    # Upload BrLP source package
    print("\n--- BrLP source package ---")
    upload_tree(sftp, BRLP_SRC, f"{REMOTE_BASE}/brlp_src/brlp")

    # Create training shell script
    train_sh = f"""#!/bin/bash
# Combined Innovation 1+2: 6ch ControlNet + BTR Training
set -e
export CUDA_VISIBLE_DEVICES=1
cd {REMOTE_BASE}

echo "============================================"
echo "[Combined Inn1+Inn2] 6ch + BTR Training"
echo "Start: $(date)"
echo "============================================"

PYTHONPATH={REMOTE_BASE}/brlp_src:{REMOTE_BASE}/src:$PYTHONPATH \\
{PYTHON} scripts/train_controlnet_6ch_btr.py \\
    --dataset_csv    {DATASET_CSV} \\
    --cache_dir      {CACHE_DIR} \\
    --output_dir     {OUTPUT_DIR}/controlnet \\
    --aekl_ckpt      {AEKL_CKPT} \\
    --diff_ckpt      {DIFF_CKPT} \\
    --pretrained_cnet_4ch {CNET_4CH} \\
    --btc_weight     0.5 \\
    --n_epochs       5 \\
    --batch_size     16 \\
    --lr             2.5e-5 \\
    2>&1 | tee {OUTPUT_DIR}/train.log

echo "End: $(date)"
echo "[Combined Inn1+Inn2] Training complete."
"""
    with sftp.open(f"{REMOTE_BASE}/train.sh", "w") as f:
        f.write(train_sh)
    print(f"\n  Created: {REMOTE_BASE}/train.sh")

    # Create evaluation shell script
    eval_sh = f"""#!/bin/bash
# Combined Innovation 1+2: 6ch+BTR Evaluation
set -e
export CUDA_VISIBLE_DEVICES=1
cd {REMOTE_BASE}

EPOCH=${{1:-4}}
CNET_CKPT="{OUTPUT_DIR}/controlnet/cnet-6ch-btr-ep-${{EPOCH}}.pth"

echo "============================================"
echo "[Combined Inn1+Inn2] 6ch+BTR Evaluation — Epoch ${{EPOCH}}"
echo "Start: $(date)"
echo "============================================"
echo "  ControlNet: ${{CNET_CKPT}}"
echo "  AE: {AEKL_CKPT}"

if [ ! -f "${{CNET_CKPT}}" ]; then
    echo "ERROR: Checkpoint not found: ${{CNET_CKPT}}"
    echo "Available:"
    ls -la {OUTPUT_DIR}/controlnet/cnet-6ch-btr-ep-*.pth 2>/dev/null || echo "  none"
    exit 1
fi

PYTHONPATH={REMOTE_BASE}/brlp_src:{REMOTE_BASE}/src:$PYTHONPATH \\
{PYTHON} scripts/evaluate_6ch_btr.py \\
    --dataset_csv {DATASET_CSV} \\
    --aekl_ckpt   {AEKL_CKPT} \\
    --diff_ckpt   {DIFF_CKPT} \\
    --cnet_ckpt   ${{CNET_CKPT}} \\
    --output_dir  {OUTPUT_DIR}/eval \\
    --max_pairs   50 \\
    --model_name  combined_6ch_btr_ep${{EPOCH}} \\
    2>&1 | tee {OUTPUT_DIR}/eval.log

echo "End: $(date)"
echo "[Combined Inn1+Inn2] Evaluation complete."
"""
    with sftp.open(f"{REMOTE_BASE}/eval.sh", "w") as f:
        f.write(eval_sh)
    print(f"  Created: {REMOTE_BASE}/eval.sh")

    sftp.close()

    # Verify upload
    print("\n--- Verification ---")
    _, stdout, _ = client.exec_command(
        f"find {REMOTE_BASE} -name '*.py' -o -name '*.sh' | sort")
    out = stdout.read().decode().strip()
    print(out)

    # Check that the CSV exists
    print("\n--- Data verification ---")
    _, stdout, _ = client.exec_command(
        f"test -f {DATASET_CSV} && echo 'CSV OK: {DATASET_CSV}' || echo 'CSV MISSING!'")
    print(stdout.read().decode().strip())
    _, stdout, _ = client.exec_command(
        f"test -f {AEKL_CKPT} && echo 'AE OK' || echo 'AE MISSING!'")
    print(stdout.read().decode().strip())
    _, stdout, _ = client.exec_command(
        f"test -f {CNET_4CH} && echo 'ControlNet 4ch OK' || echo 'ControlNet MISSING!'")
    print(stdout.read().decode().strip())

    # Start training
    print("\n" + "=" * 60)
    print("[Combined Inn1+Inn2] Starting training...")
    print("=" * 60)

    _, stdout, stderr = client.exec_command(
        f"nohup bash {REMOTE_BASE}/train.sh > /dev/null 2>&1 &"
    )
    stdout.read()  # wait for command dispatch

    time.sleep(5)
    _, stdout, _ = client.exec_command(
        "ps aux | grep 'train_controlnet_6ch_btr' | grep -v grep"
    )
    proc_out = stdout.read().decode().strip()
    if proc_out:
        print(f"  Training process started!")
        print(f"  {proc_out[:150]}")
    else:
        print("  WARNING: No training process detected. Checking log...")
        _, stdout, _ = client.exec_command(
            f"tail -30 {OUTPUT_DIR}/train.log 2>/dev/null")
        log_out = stdout.read().decode().strip()
        print(f"  Log tail:\n{log_out}")

    client.close()
    print("\nDone. Monitor via dashboard or check train.log on server.")


if __name__ == "__main__":
    main()
