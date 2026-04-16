#!/usr/bin/env python3
"""
自动执行 MCI→AD Pipeline: paramiko SSH 全自动
1) 生成 ADNI 诊断映射文件
2) 上传脚本+数据到服务器
3) 确保 brlp 可用
4) 运行 MCI→AD pipeline (多患者)
5) 下载结果到本地
"""

import os
import sys
import json
import csv
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
REMOTE_OUTPUT_DIR = "/home/wangchong/data/fwz/output/mci_ad_classification"
CONDA_ACTIVATE = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz"

LOCAL_DIR = Path(__file__).resolve().parent
LOCAL_RESULTS = LOCAL_DIR / "results_mci_ad"

# MCI→AD 转化患者 (从 _find_mci_converters.py 的结果)
SUBJECTS = [
    "002_S_1070",  # 6 visits, 4M/2A
    "023_S_0388",  # 6 visits, 3M/3A
    "023_S_0604",  # 6 visits, 3M/3A
    "027_S_0835",  # 6 visits, 4M/2A
    "053_S_0507",  # 6 visits, 2M/4A
    "023_S_0331",  # 6 visits, 5M/1A
    "016_S_1326",  # 5 visits, 3M/2A
    "023_S_1247",  # 5 visits, 2M/3A
]

GPU = 1
AVG_N = 3
MAX_MONTHS = 36

# ADNI MCI CSV (local)
ADNI_MCI_CSV = r"E:\ADNI\MCI\MCI_all_timepoint_standardized_latest.csv"


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
                 if not any(w in l for w in ['UserWarning', 'FutureWarning',
                                              'DeprecationWarning', 'TorchDynamo'])]
        if lines:
            for l in lines[:30]:
                print(f"  [stderr] {l}")
    return out, err, code


def step0_prepare_diag_map():
    """Generate ADNI diagnosis mapping JSON from local MCI CSV."""
    print(f"\n{'='*60}")
    print("[STEP 0] Generating ADNI diagnosis mapping")
    print(f"{'='*60}")

    diag_map = {}

    with open(ADNI_MCI_CSV, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers_raw = next(reader)
        headers = [h.strip().strip('"') for h in headers_raw]
        print(f"  CSV columns: {headers[:5]}...")

        for row_vals in reader:
            row = dict(zip(headers, [v.strip().strip('"') for v in row_vals]))
            ptid = row.get('PTID', '').strip()
            date_str = row.get('TimepointDate', '').strip()
            diag = row.get('DIAGNOSIS', '').strip()

            if not ptid or not date_str or not diag:
                continue

            if ptid not in diag_map:
                diag_map[ptid] = []
            diag_map[ptid].append({
                'date': date_str,
                'diagnosis': int(diag),
            })

    # Filter to only our subjects
    filtered = {s: diag_map[s] for s in SUBJECTS if s in diag_map}

    # Save locally
    map_path = LOCAL_DIR / "mci_diagnosis_map.json"
    with open(map_path, 'w') as f:
        json.dump(filtered, f, indent=2)

    print(f"  Diagnosis map for {len(filtered)} subjects saved: {map_path}")
    for s in SUBJECTS:
        entries = filtered.get(s, [])
        if entries:
            mci_n = sum(1 for e in entries if e['diagnosis'] == 2)
            ad_n = sum(1 for e in entries if e['diagnosis'] == 3)
            print(f"    {s}: {len(entries)} visits ({mci_n}M/{ad_n}A)")
        else:
            print(f"    {s}: NOT in ADNI CSV")

    return str(map_path)


def step1_upload(client, diag_map_path):
    print(f"\n{'='*60}")
    print(f"[STEP 1] Upload scripts to {REMOTE_CODE_DIR}")
    print(f"{'='*60}")

    run(client, f"mkdir -p {REMOTE_CODE_DIR}", show=False)

    files = [
        ("run_pipeline_mci_ad.py", f"{REMOTE_CODE_DIR}/run_pipeline_mci_ad.py"),
        ("run_pipeline_ad.py", f"{REMOTE_CODE_DIR}/run_pipeline_ad.py"),
        ("run_pipeline.py", f"{REMOTE_CODE_DIR}/run_pipeline.py"),
        ("extract_volumes_for_classification.py", f"{REMOTE_CODE_DIR}/extract_volumes_for_classification.py"),
    ]

    # Also upload diagnosis map
    if diag_map_path:
        files.append((os.path.basename(diag_map_path),
                       f"{REMOTE_CODE_DIR}/mci_diagnosis_map.json"))

    with SCPClient(client.get_transport()) as scp:
        for local_name, remote_path in files:
            local_path = str(LOCAL_DIR / local_name) if not os.path.isabs(local_name) else local_name
            if local_name == os.path.basename(diag_map_path or ''):
                local_path = diag_map_path
            if os.path.exists(local_path):
                scp.put(local_path, remote_path)
                print(f"  ✓ {local_name} -> {remote_path}")
            else:
                print(f"  ✗ {local_name} not found at {local_path}")

    out, _, _ = run(client, f"ls -la {REMOTE_CODE_DIR}/*.py {REMOTE_CODE_DIR}/*.json 2>/dev/null", show=False)
    print(f"\n  Remote files:")
    for line in out.strip().split('\n'):
        if line.strip():
            parts = line.strip().split()
            fname = parts[-1].split('/')[-1] if parts else line
            size = parts[4] if len(parts) > 4 else '?'
            print(f"    {fname} ({size} bytes)")


def step2_setup_brlp(client):
    print(f"\n{'='*60}")
    print("[STEP 2] Ensure brlp package is available")
    print(f"{'='*60}")

    setup_cmd = '''
NEED_DIR="/home/wangchong/data/fwz/code/src/brlp"
if [ -d "$NEED_DIR" ] || [ -L "$NEED_DIR" ]; then
    echo "brlp already available at $NEED_DIR"
else
    BRLP_SRC=$(find /home/wangchong/data/fwz -maxdepth 6 -name "const.py" -path "*/src/brlp/*" 2>/dev/null | head -1)
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
ls -la /home/wangchong/data/fwz/code/src/ 2>/dev/null
'''
    run(client, setup_cmd)

    test_cmd = f'{CONDA_ACTIVATE} && python3 -c "import sys; sys.path.insert(0, \'/home/wangchong/data/fwz/code/src\'); from brlp import const; print(\'brlp OK, RESOLUTION:\', const.RESOLUTION)"'
    out, err, code = run(client, test_cmd)
    if code != 0:
        print("  [ERROR] brlp import failed!")
        return False
    return True


def step3_run_pipeline(client):
    print(f"\n{'='*60}")
    print(f"[STEP 3] Run MCI→AD pipeline")
    print(f"  Subjects: {' '.join(SUBJECTS)}")
    print(f"  GPU: {GPU}, Avg_n: {AVG_N}, Max months: {MAX_MONTHS}")
    print(f"{'='*60}")

    run(client, f"mkdir -p {REMOTE_OUTPUT_DIR}", show=False)

    # Clean up any wrongly-extracted latent files (from previous buggy run)
    # These had batch dim [1,3,16,20,16] instead of [3,16,20,16]
    print("  Cleaning up potentially corrupted on-the-fly latent files...")
    for subj in SUBJECTS:
        # Only remove latent files that were created by our script (not original ones)
        # Original ones from B_mci.csv are named t1w_final_latent.npz and are correct
        # Our buggy run may have created new ones for subjects NOT in B_mci.csv
        cleanup_cmd = f'''
DATA_DIR="/home/wangchong/data/fwz/data/mci_longitudinal/{subj}"
if [ -d "$DATA_DIR" ]; then
    for d in "$DATA_DIR"/*/; do
        latent="$d/t1w_final_latent.npz"
        if [ -f "$latent" ]; then
            # Check if this latent has wrong shape (5 dims = batch dim present)
            python3 -c "
import numpy as np
z = np.load('$latent')['data']
if z.ndim == 4:
    print('OK: ' + str(z.shape))
elif z.ndim == 5:
    print('CORRUPTED (batch dim): ' + str(z.shape) + ' -> removing')
    import os
    os.remove('$latent')
else:
    print('UNKNOWN shape: ' + str(z.shape))
" 2>/dev/null
        fi
    done
fi'''
        out, _, _ = run(client, cleanup_cmd, show=False)
        if out.strip():
            for line in out.strip().split('\n'):
                if 'CORRUPTED' in line or 'removing' in line:
                    print(f"    [{subj}] {line.strip()}")

    # Also clean previous output to get fresh results
    print("  Cleaning previous output directory...")
    run(client, f"rm -rf {REMOTE_OUTPUT_DIR}/*", show=False)

    subjects_str = ' '.join(SUBJECTS)
    cmd = (
        f"{CONDA_ACTIVATE} && "
        f"cd {REMOTE_CODE_DIR} && "
        f"python run_pipeline_mci_ad.py "
        f"--gpu {GPU} "
        f"--subjects {subjects_str} "
        f"--avg_n {AVG_N} "
        f"--max_months {MAX_MONTHS} "
        f"--output_dir {REMOTE_OUTPUT_DIR} "
        f"--diag_map {REMOTE_CODE_DIR}/mci_diagnosis_map.json "
        f"2>&1"
    )

    print(f"\n  Command: python run_pipeline_mci_ad.py --subjects {subjects_str}")
    print(f"  This will take a while (~5-10 min per subject)...\n")

    out, err, code = run(client, cmd, timeout=14400)

    if code != 0:
        print(f"\n  [ERROR] Pipeline failed (exit={code})")
        if err:
            print(f"  Error output (last 3000 chars):")
            print(err[-3000:])
        return False

    print(f"\n  ✓ Pipeline completed successfully!")
    return True


def step4_download(client):
    print(f"\n{'='*60}")
    print(f"[STEP 4] Download results to {LOCAL_RESULTS}")
    print(f"{'='*60}")

    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)

    # Download bias_analysis.json first
    with SCPClient(client.get_transport()) as scp:
        try:
            scp.get(f"{REMOTE_OUTPUT_DIR}/bias_analysis.json",
                    str(LOCAL_RESULTS / "bias_analysis.json"))
            print("  ✓ bias_analysis.json")
        except Exception as e:
            print(f"  ✗ bias_analysis.json: {e}")

    # Download per-subject results
    for subject in SUBJECTS:
        subj_remote = f"{REMOTE_OUTPUT_DIR}/{subject}"
        subj_local = LOCAL_RESULTS / subject
        subj_local.mkdir(parents=True, exist_ok=True)

        out, _, _ = run(client, f"find {subj_remote} -type f 2>/dev/null", show=False)
        remote_files = [f.strip() for f in out.strip().split('\n') if f.strip()]

        if not remote_files:
            print(f"  [{subject}] No result files found")
            continue

        print(f"  [{subject}] Downloading {len(remote_files)} files...")

        # Only download key files (GIF, PNG, JSON), skip NIfTI to save time
        key_files = [f for f in remote_files
                     if f.endswith(('.gif', '.png', '.json'))]
        nifti_files = [f for f in remote_files if f.endswith('.nii.gz')]

        with SCPClient(client.get_transport()) as scp:
            for rpath in key_files:
                fname = os.path.basename(rpath)
                lpath = str(subj_local / fname)
                try:
                    scp.get(rpath, lpath)
                    fsize = os.path.getsize(lpath)
                    unit = 'KB' if fsize < 1048576 else 'MB'
                    size = fsize / 1024 if fsize < 1048576 else fsize / 1048576
                    print(f"    ✓ {fname} ({size:.1f} {unit})")
                except Exception as e:
                    print(f"    ✗ {fname}: {e}")

        if nifti_files:
            print(f"    ({len(nifti_files)} NIfTI files skipped, use --download_nifti to include)")

    # Print summary
    print(f"\n  Local results: {LOCAL_RESULTS}")
    for d in sorted(LOCAL_RESULTS.iterdir()):
        if d.is_dir():
            n_files = len(list(d.iterdir()))
            print(f"    {d.name}/ ({n_files} files)")
        else:
            print(f"    {d.name}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("MCI→AD Converter Pipeline — 自动执行")
    print(f"Subjects: {len(SUBJECTS)}")
    print(f"Server: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)

    # Step 0: Prepare diagnosis map
    diag_map_path = step0_prepare_diag_map()

    # Step 1-4: SSH workflow
    client = create_ssh()
    try:
        step1_upload(client, diag_map_path)

        if not step2_setup_brlp(client):
            print("\n[FATAL] Cannot setup brlp, aborting")
            return

        if not step3_run_pipeline(client):
            print("\n[WARN] Pipeline failed, attempting partial download")

        step4_download(client)

    finally:
        client.close()
        print("\n[SSH] Disconnected")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results: {LOCAL_RESULTS}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
