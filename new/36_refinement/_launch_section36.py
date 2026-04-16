"""
Section 36: Upload & Launch — Image-Space Refinement Network
=============================================================
Uploads scripts then runs training and evaluation on the server.

Phase 1: Train 3 refinement network variants (parallel on 3 GPUs)
  - RefA: L1 + region weighting (no SSIM, no noise aug)
  - RefB: L1 + SSIM + region weighting
  - RefC: L1 + SSIM + region weighting + noise augmentation (0.5)

Phase 2: Evaluate all variants + best from S35
"""
import paramiko, os, time, json
from datetime import datetime

# ── Server config ──
SERVER = {"host": "10.96.27.109", "port": 2638, "user": "wangchong", "pass": "123456"}
PY = "/home/wangchong/miniconda3/envs/fwz/bin/python"
ENV_PREFIX = f"export PATH=/home/wangchong/miniconda3/envs/fwz/bin:$PATH"

# ── Paths ──
CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
AE = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_BTR = "/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth"
CNET_H1A30 = "/home/wangchong/data/fwz/output/34_hippo_training/H1_a30/cnet-hippo-H1_a30-ep2.pth"
AE_S35_BEST = "/home/wangchong/data/fwz/output/35_multiregion/ExpC_l1ssim_multi/ae-v2-l1ssim_multi_a30-ep2.pth"

CODE_DIR = "/home/wangchong/data/fwz/code/36_refinement/scripts"
OUT_DIR = "/home/wangchong/data/fwz/output/36_refinement"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "scripts")

# ── Dashboard update ──
DASH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
DASH_STATE = os.path.join(DASH_DIR, "dashboard_state.json")


def update_dashboard(**kwargs):
    """Update dashboard state file."""
    if not os.path.exists(DASH_STATE):
        return
    with open(DASH_STATE, 'r', encoding='utf-8') as f:
        state = json.load(f)
    for key, val in kwargs.items():
        if key == 'add_op':
            val['time'] = datetime.now().strftime('%H:%M:%S')
            state.setdefault('ai_ops', []).append(val)
        else:
            state[key] = val
    with open(DASH_STATE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER["host"], port=SERVER["port"],
                   username=SERVER["user"], password=SERVER["pass"], timeout=30)
    return client


def ssh_exec(client, cmd, timeout=300):
    print(f"  $ {cmd[:120]}...")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        print(out[-500:] if len(out) > 500 else out)
    if err.strip() and 'warning' not in err.lower():
        print(f"  STDERR: {err[-300:]}")
    return out, err


def upload_files(client):
    sftp = client.open_sftp()
    # Create remote dirs
    for d in [CODE_DIR, OUT_DIR, f"{OUT_DIR}/eval", f"{OUT_DIR}/cache"]:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            ssh_exec(client, f"mkdir -p {d}")

    # Upload scripts
    for fname in ['train_refinement.py', 'evaluate_refinement.py']:
        local = os.path.join(LOCAL_DIR, fname)
        remote = f"{CODE_DIR}/{fname}"
        sftp.put(local, remote)
        print(f"  Uploaded: {remote}")

    # Also upload brlp source code needed
    sftp.close()


def create_and_run_phase1(client):
    """Phase 1: Train 3 refinement variants on 3 GPUs in parallel."""
    print("\n═══ Phase 1: Training Refinement Networks ═══")

    experiments = [
        {
            "name": "RefA",
            "loss_type": "l1_region",
            "noise_aug": 0.0,
            "gpu": 0,
        },
        {
            "name": "RefB",
            "loss_type": "l1_ssim_region",
            "noise_aug": 0.0,
            "gpu": 1,
        },
        {
            "name": "RefC",
            "loss_type": "l1_ssim_region",
            "noise_aug": 0.5,
            "gpu": 2,
        },
    ]

    # Create shell script for each experiment
    for exp in experiments:
        script = f"""#!/bin/bash
{ENV_PREFIX}
cd {OUT_DIR}

echo "=== {exp['name']}: {exp['loss_type']} noise={exp['noise_aug']} GPU={exp['gpu']} ==="
CUDA_VISIBLE_DEVICES={exp['gpu']} {PY} {CODE_DIR}/train_refinement.py \\
    --csv {CSV} \\
    --ae_ckpt {AE} \\
    --ae_decoder_ckpt {AE_S35_BEST} \\
    --output_dir {OUT_DIR} \\
    --exp_name {exp['name']} \\
    --loss_type {exp['loss_type']} \\
    --region_alpha 10.0 \\
    --ssim_weight 1.0 \\
    --noise_aug {exp['noise_aug']} \\
    --base_ch 32 \\
    --epochs 5 \\
    --lr 1e-4 \\
    --gpu 0 \\
    --n_test_hold 5 \\
    2>&1 | tee {OUT_DIR}/{exp['name']}_train.log

echo "DONE {exp['name']}"
"""
        script_path = f"{OUT_DIR}/train_{exp['name']}.sh"
        ssh_exec(client, f"cat > {script_path} << 'ENDSCRIPT'\n{script}\nENDSCRIPT")
        ssh_exec(client, f"chmod +x {script_path}")

    # Launch all 3 in parallel
    for exp in experiments:
        script_path = f"{OUT_DIR}/train_{exp['name']}.sh"
        ssh_exec(client, f"nohup bash {script_path} > /dev/null 2>&1 &")
        print(f"  Launched: {exp['name']} on GPU {exp['gpu']}")

    update_dashboard(
        add_op={'type': 'code', 'text': '3个精炼网络训练已启动 (GPU 0/1/2)'},
        tasks=[
            {'name': f'RefA: L1+region (GPU 0)', 'status': 'running', 'percent': 0, 'eta': '~15 min'},
            {'name': f'RefB: L1+SSIM+region (GPU 1)', 'status': 'running', 'percent': 0, 'eta': '~15 min'},
            {'name': f'RefC: +noise_aug (GPU 2)', 'status': 'running', 'percent': 0, 'eta': '~15 min'},
            {'name': 'Phase 2: 评估', 'status': 'queued', 'percent': 0, 'eta': '等待训练'},
        ]
    )


def create_and_run_phase2(client):
    """Phase 2: Evaluate all combinations."""
    print("\n═══ Phase 2: Evaluation ═══")

    evals = [
        # (label, ae_decoder_ckpt, cnet_ckpt, ref_ckpt, base_ch, m_las)
        ("RefA_H1a30", AE_S35_BEST, CNET_H1A30,
         f"{OUT_DIR}/RefA/refnet-RefA-ep4.pth", 32, 3),
        ("RefB_H1a30", AE_S35_BEST, CNET_H1A30,
         f"{OUT_DIR}/RefB/refnet-RefB-ep4.pth", 32, 3),
        ("RefC_H1a30", AE_S35_BEST, CNET_H1A30,
         f"{OUT_DIR}/RefC/refnet-RefC-ep4.pth", 32, 3),
        ("RefB_BTR", AE_S35_BEST, CNET_BTR,
         f"{OUT_DIR}/RefB/refnet-RefB-ep4.pth", 32, 3),
        # Also test with LAS=5
        ("RefB_H1a30_LAS5", AE_S35_BEST, CNET_H1A30,
         f"{OUT_DIR}/RefB/refnet-RefB-ep4.pth", 32, 5),
        # Baseline without refinement (S35 best for comparison)
        ("S35best_H1a30_noref", AE_S35_BEST, CNET_H1A30, None, 32, 3),
    ]

    eval_script = f"""#!/bin/bash
{ENV_PREFIX}
cd {OUT_DIR}
"""
    for i, (label, ae_dec, cnet, ref, bch, las) in enumerate(evals):
        ref_arg = f"--ref_ckpt {ref}" if ref else ""
        gpu = i % 3
        eval_script += f"""
echo "=== Eval: {label} ==="
CUDA_VISIBLE_DEVICES={gpu} {PY} {CODE_DIR}/evaluate_refinement.py \\
    --csv {CSV} \\
    --aekl_ckpt {AE} \\
    --diff_ckpt {DIFF} \\
    --cnet_ckpt {cnet} \\
    --ae_decoder_ckpt {ae_dec} \\
    {ref_arg} \\
    --base_ch {bch} \\
    --cache_dir {OUT_DIR}/cache \\
    --n_test 5 --m_las {las} \\
    --output_json {OUT_DIR}/eval/{label}.json \\
    --label {label} \\
    --gpu 0

"""

    script_path = f"{OUT_DIR}/run_eval.sh"
    ssh_exec(client, f"cat > {script_path} << 'ENDSCRIPT'\n{eval_script}\nENDSCRIPT")
    ssh_exec(client, f"chmod +x {script_path}")
    ssh_exec(client, f"nohup bash {script_path} > {OUT_DIR}/eval.log 2>&1 &")
    print("  Launched evaluation pipeline")

    update_dashboard(
        add_op={'type': 'test', 'text': f'评估流水线已启动: {len(evals)}个配置'},
        tasks=[
            {'name': f'RefA: L1+region (GPU 0)', 'status': 'completed', 'percent': 100, 'eta': '完成'},
            {'name': f'RefB: L1+SSIM+region (GPU 1)', 'status': 'completed', 'percent': 100, 'eta': '完成'},
            {'name': f'RefC: +noise_aug (GPU 2)', 'status': 'completed', 'percent': 100, 'eta': '完成'},
            {'name': f'Phase 2: 评估 ({len(evals)}组)', 'status': 'running', 'percent': 0, 'eta': '~30 min'},
        ]
    )


def main():
    print("Section 36: Image-Space Refinement Network")
    print("=" * 50)

    client = ssh_connect()
    print(f"Connected to {SERVER['host']}:{SERVER['port']}")

    # Step 1: Upload
    print("\n── Uploading scripts ──")
    upload_files(client)
    update_dashboard(add_op={'type': 'code', 'text': '脚本已上传到服务器'})

    # Step 2: Phase 1 — Training
    create_and_run_phase1(client)

    print("\n训练已启动。使用 _monitor36.py 监控进度。")
    print("训练完成后运行 Phase 2 评估:")
    print("  python _launch_section36.py --phase2")

    client.close()


def main_phase2():
    client = ssh_connect()
    create_and_run_phase2(client)
    print("\n评估已启动。请等待完成后收集结果。")
    client.close()


if __name__ == '__main__':
    import sys as _sys
    if '--phase2' in _sys.argv:
        main_phase2()
    else:
        main()
