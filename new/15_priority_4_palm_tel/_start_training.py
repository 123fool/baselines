"""
上传 Priority 4 (PALM+TEL) 代码到服务器，并启动训练。

服务器结构:
  /home/wangchong/data/fwz/code/priority_4_palm_tel/
    scripts/
      train_controlnet_btc_palm_tel.py
      evaluate_palm_tel.py
    src/
      palm_tel.py
      sampling_palm_tel.py
    brlp_src/
      brlp/  (BrLP 源码包)
    innov2_src/
      bidirectional_temporal.py
    train.sh
"""

import os
import sys
import stat
import paramiko


SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

LOCAL_BASE = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(LOCAL_BASE, "..", "..", "src", "brlp"))
INNOV2_SRC = os.path.abspath(os.path.join(LOCAL_BASE, "..", "12_innovation_2", "src"))

REMOTE_BASE = "/home/wangchong/data/fwz/code/priority_4_palm_tel"
OUTPUT_DIR = "/home/wangchong/data/fwz/output/priority_4_palm_tel"

# Server paths
DATASET_CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
CACHE_DIR = "/home/wangchong/data/fwz/cache/innovation_5"
AEKL_CKPT = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"


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
    print("[Priority 4] Uploading PALM+TEL code to server")
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
        f"{REMOTE_BASE}/innov2_src",
        f"{OUTPUT_DIR}",
        f"{OUTPUT_DIR}/controlnet",
        f"{OUTPUT_DIR}/eval",
    ]:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)
            print(f"  Created: {d}")

    # Upload P4 source
    print("\n--- P4 source files ---")
    upload_tree(sftp, os.path.join(LOCAL_BASE, "src"), f"{REMOTE_BASE}/src")

    # Upload P4 scripts
    print("\n--- P4 training/eval scripts ---")
    upload_tree(sftp, os.path.join(LOCAL_BASE, "scripts"), f"{REMOTE_BASE}/scripts")

    # Upload BrLP source
    print("\n--- BrLP source package ---")
    upload_tree(sftp, BRLP_SRC, f"{REMOTE_BASE}/brlp_src/brlp")

    # Upload Innovation 2 bidirectional_temporal
    print("\n--- Innovation 2 BTR module ---")
    upload_tree(sftp, INNOV2_SRC, f"{REMOTE_BASE}/innov2_src")

    # Create training shell script
    train_sh = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd {REMOTE_BASE}

PYTHON=/home/wangchong/miniconda3/envs/fwz/bin/python

echo "[Priority 4] BTR + PALM + TEL Training"
echo "Start: $(date)"

PYTHONPATH={REMOTE_BASE}/brlp_src:{REMOTE_BASE}/innov2_src:{REMOTE_BASE}/src:$PYTHONPATH \\
$PYTHON scripts/train_controlnet_btc_palm_tel.py \\
    --dataset_csv {DATASET_CSV} \\
    --cache_dir   {CACHE_DIR} \\
    --output_dir  {OUTPUT_DIR}/controlnet \\
    --aekl_ckpt   {AEKL_CKPT} \\
    --diff_ckpt   {DIFF_CKPT} \\
    --btc_weight  0.5 \\
    --n_epochs    5 \\
    --batch_size  16 \\
    --lr          2.5e-5 \\
    2>&1 | tee {OUTPUT_DIR}/train.log

echo "End: $(date)"
echo "[Priority 4] Training complete."
"""
    with sftp.open(f"{REMOTE_BASE}/train.sh", "w") as f:
        f.write(train_sh)
    print(f"\n  Created: {REMOTE_BASE}/train.sh")

    # Create evaluation shell script
    eval_sh = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd {REMOTE_BASE}

PYTHON=/home/wangchong/miniconda3/envs/fwz/bin/python

echo "[Priority 4] BTR + PALM + TEL Evaluation"
echo "Start: $(date)"

# Find best checkpoint (last one)
BEST_CKPT=$(ls -1 {OUTPUT_DIR}/controlnet/cnet-btc-palm-tel-ep-*.pth 2>/dev/null | sort -V | tail -1)

if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi

echo "  Using checkpoint: $BEST_CKPT"

PYTHONPATH={REMOTE_BASE}/brlp_src:{REMOTE_BASE}/src:$PYTHONPATH \\
$PYTHON scripts/evaluate_palm_tel.py \\
    --dataset_csv {DATASET_CSV} \\
    --aekl_ckpt   {AEKL_CKPT} \\
    --diff_ckpt   {DIFF_CKPT} \\
    --cnet_ckpt   $BEST_CKPT \\
    --output_dir  {OUTPUT_DIR}/eval \\
    --max_pairs   50 \\
    --model_name  btc_palm_tel_ep4 \\
    2>&1 | tee {OUTPUT_DIR}/eval.log

echo "End: $(date)"
echo "[Priority 4] Evaluation complete."
"""
    with sftp.open(f"{REMOTE_BASE}/eval.sh", "w") as f:
        f.write(eval_sh)
    print(f"  Created: {REMOTE_BASE}/eval.sh")

    sftp.close()

    # Verify upload
    print("\n--- Verification ---")
    _, stdout, _ = client.exec_command(f"find {REMOTE_BASE} -name '*.py' -o -name '*.sh' | sort")
    out = stdout.read().decode().strip()
    print(out)

    # Start training
    print("\n" + "=" * 60)
    print("[Priority 4] Starting training...")
    print("=" * 60)

    _, stdout, stderr = client.exec_command(
        f"nohup bash {REMOTE_BASE}/train.sh > /dev/null 2>&1 &"
    )
    stdout.read()  # wait for command dispatch

    # Verify process started
    import time
    time.sleep(3)
    _, stdout, _ = client.exec_command(
        "ps aux | grep 'train_controlnet_btc_palm_tel' | grep -v grep"
    )
    proc_out = stdout.read().decode().strip()
    if proc_out:
        print(f"  Training process started!")
        print(f"  {proc_out[:120]}")
    else:
        print("  WARNING: No training process detected. Checking log...")
        _, stdout, _ = client.exec_command(f"tail -20 {OUTPUT_DIR}/train.log 2>/dev/null")
        log_out = stdout.read().decode().strip()
        print(f"  Log tail:\n{log_out}")

    client.close()
    print("\nDone. Use _monitor.py to track training progress.")


if __name__ == "__main__":
    main()
