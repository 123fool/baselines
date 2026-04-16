#!/usr/bin/env python3
"""
自动执行 AD Pipeline: paramiko SSH 全自动
1) 上传脚本到服务器
2) 确保 brlp 包可用
3) 运行 AD pipeline
4) 下载结果到本地
"""

import os
import sys
import time
import paramiko
from scp import SCPClient
from pathlib import Path

# ── 配置 ──
SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

REMOTE_CODE_DIR = "/home/wangchong/data/fwz/code/24_classification_animation"
REMOTE_OUTPUT_DIR = "/home/wangchong/data/fwz/output/classification_animation"
CONDA_ACTIVATE = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz"

LOCAL_DIR = Path(__file__).resolve().parent
LOCAL_RESULTS = LOCAL_DIR / "results_ad"

# AD 患者选择
SUBJECT = "023_S_0139"  # 4 visits, earliest data
GPU = 1
AVG_N = 3


def create_ssh():
    print(f"[SSH] Connecting to {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, timeout=30)
    print("[SSH] Connected!")
    return c


def run(client, cmd, timeout=1800, show=True):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    code = stdout.channel.recv_exit_status()
    if show and out.strip():
        print(out.strip())
    if show and err.strip():
        lines = [l for l in err.strip().split('\n')
                 if not any(w in l for w in ['UserWarning', 'FutureWarning', 'DeprecationWarning'])]
        if lines:
            print('\n'.join(f"  [stderr] {l}" for l in lines[:30]))
    return out, err, code


def step1_upload(client):
    print(f"\n{'='*60}")
    print(f"[STEP 1] Upload scripts to {REMOTE_CODE_DIR}")
    print(f"{'='*60}")

    run(client, f"mkdir -p {REMOTE_CODE_DIR}", show=False)

    files = [
        "run_pipeline_ad.py",
        "run_pipeline.py",
        "extract_volumes_for_classification.py",
    ]

    with SCPClient(client.get_transport()) as scp:
        for f in files:
            local = str(LOCAL_DIR / f)
            if os.path.exists(local):
                remote = f"{REMOTE_CODE_DIR}/{f}"
                scp.put(local, remote)
                print(f"  ✓ {f}")
            else:
                print(f"  ✗ {f} not found locally")

    # Verify
    out, _, _ = run(client, f"ls -la {REMOTE_CODE_DIR}/*.py", show=False)
    print(f"\n  Remote files:")
    for line in out.strip().split('\n'):
        fname = line.split('/')[-1] if '/' in line else line
        print(f"    {fname}")


def step2_setup_brlp(client):
    print(f"\n{'='*60}")
    print(f"[STEP 2] Ensure brlp package is available")
    print(f"{'='*60}")

    # run_pipeline_ad.py does: PROJECT_ROOT = SCRIPT_DIR.parent.parent
    # SCRIPT_DIR = REMOTE_CODE_DIR = /home/wangchong/data/fwz/code/24_classification_animation
    # PROJECT_ROOT = /home/wangchong/data/fwz/code
    # needs: /home/wangchong/data/fwz/code/src/brlp/

    setup_cmd = """
NEED_DIR="/home/wangchong/data/fwz/code/src/brlp"
if [ -d "$NEED_DIR" ] || [ -L "$NEED_DIR" ]; then
    echo "brlp already available at $NEED_DIR"
else
    BRLP_SRC=$(find /home/wangchong/data/fwz -maxdepth 6 -name 'const.py' -path '*/src/brlp/*' 2>/dev/null | head -1)
    if [ -n "$BRLP_SRC" ]; then
        BRLP_DIR=$(dirname "$BRLP_SRC")
        mkdir -p /home/wangchong/data/fwz/code/src
        ln -sf "$BRLP_DIR" "$NEED_DIR"
        echo "Created symlink: $NEED_DIR -> $BRLP_DIR"
    else
        echo "ERROR: Cannot find brlp source"
        exit 1
    fi
fi
ls -la /home/wangchong/data/fwz/code/src/
"""
    run(client, setup_cmd)

    # Quick import test
    test_cmd = f'{CONDA_ACTIVATE} && python3 -c "import sys; sys.path.insert(0, \'/home/wangchong/data/fwz/code/src\'); from brlp import const; print(\'brlp OK, RESOLUTION:\', const.RESOLUTION)"'
    out, err, code = run(client, test_cmd)
    if code != 0:
        print(f"  [ERROR] brlp import failed!")
        return False
    return True


def step3_run_pipeline(client):
    print(f"\n{'='*60}")
    print(f"[STEP 3] Run AD pipeline: {SUBJECT} on GPU {GPU}")
    print(f"{'='*60}")

    run(client, f"mkdir -p {REMOTE_OUTPUT_DIR}", show=False)

    cmd = (
        f"{CONDA_ACTIVATE} && "
        f"cd {REMOTE_CODE_DIR} && "
        f"python run_pipeline_ad.py "
        f"--gpu {GPU} --subject {SUBJECT} --avg_n {AVG_N} "
        f"--output_dir {REMOTE_OUTPUT_DIR} "
        f"2>&1"
    )

    print(f"  Running: python run_pipeline_ad.py --gpu {GPU} --subject {SUBJECT}")
    print(f"  This may take several minutes...\n")

    out, err, code = run(client, cmd, timeout=1800)

    if code != 0:
        print(f"\n  [ERROR] Pipeline failed (exit={code})")
        if err:
            print(f"  Last error: {err[-3000:]}")
        return False

    print(f"\n  ✓ Pipeline completed successfully!")
    return True


def step4_download(client):
    print(f"\n{'='*60}")
    print(f"[STEP 4] Download results to {LOCAL_RESULTS}")
    print(f"{'='*60}")

    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)

    # List result files
    out, _, _ = run(client, f"find {REMOTE_OUTPUT_DIR} -name '{SUBJECT}_*' -type f 2>/dev/null", show=False)
    remote_files = [f.strip() for f in out.strip().split('\n') if f.strip()]

    if not remote_files:
        print("  [WARN] No result files found!")
        out2, _, _ = run(client, f"ls -la {REMOTE_OUTPUT_DIR}/ 2>/dev/null", show=False)
        print(f"  Directory contents:\n{out2}")
        return

    print(f"  Found {len(remote_files)} files to download")

    with SCPClient(client.get_transport()) as scp:
        for rpath in remote_files:
            fname = os.path.basename(rpath)
            lpath = str(LOCAL_RESULTS / fname)
            try:
                scp.get(rpath, lpath)
                fsize = os.path.getsize(lpath)
                unit = 'KB' if fsize < 1048576 else 'MB'
                size = fsize / 1024 if fsize < 1048576 else fsize / 1048576
                print(f"  ✓ {fname} ({size:.1f} {unit})")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")

    print(f"\n  Local results: {LOCAL_RESULTS}")
    for f in sorted(LOCAL_RESULTS.iterdir()):
        print(f"    {f.name}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  BrLP AD Patient Pipeline - Automated Execution")
    print(f"  Subject: {SUBJECT} | GPU: {GPU}")
    print("=" * 60)

    client = create_ssh()
    try:
        step1_upload(client)

        if not step2_setup_brlp(client):
            print("\n[FATAL] brlp setup failed")
            return

        if not step3_run_pipeline(client):
            print("\n[FATAL] Pipeline failed")
            return

        step4_download(client)

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  All done! Elapsed: {elapsed/60:.1f} min")
        print(f"  Results: {LOCAL_RESULTS}")
        print(f"{'='*60}")
    finally:
        client.close()
        print("\n[SSH] Disconnected")


if __name__ == '__main__':
    main()
