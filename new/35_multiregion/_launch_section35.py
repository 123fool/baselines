"""
Section 35: Multi-Region Enhancement — Upload & Launch Script

Uploads scripts to server and launches:
  Phase 1: Baseline multi-region evaluation
  Phase 2: 3 AE decoder training experiments (parallel on 3 GPUs)
  Phase 3: Evaluation of all new models
"""
import paramiko
import os
import time

SERVER = {"host": "10.96.27.109", "port": 2638, "user": "wangchong", "pass": "123456"}
LOCAL_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
REMOTE_CODE = "/home/wangchong/data/fwz/code/35_multiregion/scripts"
REMOTE_OUT = "/home/wangchong/data/fwz/output/35_multiregion"

# Model checkpoints
AEKL = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_BTR = "/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth"
CNET_H1A30 = "/home/wangchong/data/fwz/output/34_hippo_training/H1_a30/cnet-hippo-H1_a30-ep2.pth"
AE_DEC_34 = "/home/wangchong/data/fwz/output/34_hippo_training/AE_dec_a30/ae-hippo-dec-a30-ep2.pth"
CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
CACHE = "/home/wangchong/data/fwz/cache"


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SERVER["host"], port=SERVER["port"],
                username=SERVER["user"], password=SERVER["pass"], timeout=30)
    return ssh


def ssh_exec(ssh, cmd, timeout=60):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return out, err


def upload_scripts(ssh):
    """Upload all scripts to server."""
    sftp = ssh.open_sftp()

    # Create remote directories
    for d in ["/home/wangchong/data/fwz/code/35_multiregion",
              REMOTE_CODE, REMOTE_OUT, f"{REMOTE_OUT}/eval"]:
        try:
            sftp.mkdir(d)
        except IOError:
            pass

    # Upload script files
    for fname in os.listdir(LOCAL_SCRIPTS):
        if fname.endswith('.py'):
            local = os.path.join(LOCAL_SCRIPTS, fname)
            remote = f"{REMOTE_CODE}/{fname}"
            sftp.put(local, remote)
            print(f"  Uploaded: {fname}")

    sftp.close()


def create_and_upload_run_script(ssh):
    """Create the bash script that runs all experiments."""
    sftp = ssh.open_sftp()

    PY = "/home/wangchong/miniconda3/envs/fwz/bin/python"

    # Phase 1: Baseline multi-region evaluation
    phase1_script = f"""#!/bin/bash
export PATH="/home/wangchong/miniconda3/envs/fwz/bin:$PATH"
cd {REMOTE_CODE}

echo "=== Phase 1: Baseline Multi-Region Evaluation ==="

# Baseline BTR
echo ">>> Evaluating Baseline BTR (multi-region)..."
CUDA_VISIBLE_DEVICES=0 {PY} evaluate_multiregion.py \\
    --dataset_csv {CSV} \\
    --aekl_ckpt {AEKL} \\
    --diff_ckpt {DIFF} \\
    --cnet_ckpt {CNET_BTR} \\
    --n_test 5 --m_las 3 \\
    --label "baseline_BTR_MR" \\
    --output_json {REMOTE_OUT}/eval/baseline_BTR_multiregion.json \\
    2>&1 | tee {REMOTE_OUT}/eval/eval_mr_baseline.log

# Best S34 model: AE_dec + H1_a30
echo ">>> Evaluating AE_dec+H1_a30 (multi-region)..."
CUDA_VISIBLE_DEVICES=0 {PY} evaluate_multiregion.py \\
    --dataset_csv {CSV} \\
    --aekl_ckpt {AEKL} \\
    --diff_ckpt {DIFF} \\
    --cnet_ckpt {CNET_H1A30} \\
    --ae_decoder_ckpt {AE_DEC_34} \\
    --n_test 5 --m_las 3 \\
    --label "AE_dec_H1a30_MR" \\
    --output_json {REMOTE_OUT}/eval/AE_dec_H1a30_multiregion.json \\
    2>&1 | tee {REMOTE_OUT}/eval/eval_mr_best34.log

echo "=== Phase 1 Complete ==="
"""

    # Phase 2: Training — 3 experiments on 3 GPUs
    phase2_script = f"""#!/bin/bash
export PATH="/home/wangchong/miniconda3/envs/fwz/bin:$PATH"
cd {REMOTE_CODE}

echo "=== Phase 2: AE Decoder V2 Training (3 experiments parallel) ==="

# Exp A: SSIM loss + hippocampus (GPU 0)
echo ">>> Starting Exp A: SSIM + hippo on GPU 0..."
CUDA_VISIBLE_DEVICES=0 {PY} train_ae_decoder_v2.py \\
    --dataset_csv {CSV} \\
    --cache_dir {CACHE}/ae_v2_a \\
    --output_dir {REMOTE_OUT}/ExpA_ssim_hippo \\
    --aekl_ckpt {AEKL} \\
    --loss_type ssim --regions hippo --alpha 30 \\
    --n_epochs 3 --lr 5e-5 \\
    --exp_name ssim_hippo_a30 \\
    2>&1 | tee {REMOTE_OUT}/train_expA.log &
PID_A=$!

# Exp B: L1 loss + multi-region (GPU 1)
echo ">>> Starting Exp B: L1 + multi on GPU 1..."
CUDA_VISIBLE_DEVICES=1 {PY} train_ae_decoder_v2.py \\
    --dataset_csv {CSV} \\
    --cache_dir {CACHE}/ae_v2_b \\
    --output_dir {REMOTE_OUT}/ExpB_l1_multi \\
    --aekl_ckpt {AEKL} \\
    --loss_type l1 --regions multi --alpha 30 \\
    --n_epochs 3 --lr 5e-5 \\
    --exp_name l1_multi_a30 \\
    2>&1 | tee {REMOTE_OUT}/train_expB.log &
PID_B=$!

# Exp C: L1+SSIM combined + multi-region (GPU 2)
echo ">>> Starting Exp C: L1+SSIM + multi on GPU 2..."
CUDA_VISIBLE_DEVICES=2 {PY} train_ae_decoder_v2.py \\
    --dataset_csv {CSV} \\
    --cache_dir {CACHE}/ae_v2_c \\
    --output_dir {REMOTE_OUT}/ExpC_l1ssim_multi \\
    --aekl_ckpt {AEKL} \\
    --loss_type l1_ssim --regions multi --alpha 30 --ssim_weight 1.0 \\
    --n_epochs 3 --lr 5e-5 \\
    --exp_name l1ssim_multi_a30 \\
    2>&1 | tee {REMOTE_OUT}/train_expC.log &
PID_C=$!

echo "Waiting for all training to complete..."
echo "PIDs: A=$PID_A B=$PID_B C=$PID_C"
wait $PID_A $PID_B $PID_C
echo "=== Phase 2 Complete ==="
"""

    # Phase 3: Evaluation of all new models
    phase3_script = f"""#!/bin/bash
export PATH="/home/wangchong/miniconda3/envs/fwz/bin:$PATH"
cd {REMOTE_CODE}

echo "=== Phase 3: Evaluate All New Models ==="

# Find best epoch checkpoints (ep2 typically)
for EXP in ExpA_ssim_hippo ExpB_l1_multi ExpC_l1ssim_multi; do
    CKPT=$(ls -t {REMOTE_OUT}/$EXP/ae-v2-*-ep2.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t {REMOTE_OUT}/$EXP/ae-v2-*.pth 2>/dev/null | head -1)
    fi
    if [ -n "$CKPT" ]; then
        echo ">>> Evaluating $EXP: $CKPT"

        # With baseline BTR ControlNet
        CUDA_VISIBLE_DEVICES=0 {PY} evaluate_multiregion.py \\
            --dataset_csv {CSV} \\
            --aekl_ckpt {AEKL} \\
            --diff_ckpt {DIFF} \\
            --cnet_ckpt {CNET_BTR} \\
            --ae_decoder_ckpt "$CKPT" \\
            --n_test 5 --m_las 3 \\
            --label "${{EXP}}_BTR" \\
            --output_json {REMOTE_OUT}/eval/${{EXP}}_BTR.json \\
            2>&1 | tee -a {REMOTE_OUT}/eval/eval_phase3.log

        # With H1_a30 ControlNet
        CUDA_VISIBLE_DEVICES=0 {PY} evaluate_multiregion.py \\
            --dataset_csv {CSV} \\
            --aekl_ckpt {AEKL} \\
            --diff_ckpt {DIFF} \\
            --cnet_ckpt {CNET_H1A30} \\
            --ae_decoder_ckpt "$CKPT" \\
            --n_test 5 --m_las 3 \\
            --label "${{EXP}}_H1a30" \\
            --output_json {REMOTE_OUT}/eval/${{EXP}}_H1a30.json \\
            2>&1 | tee -a {REMOTE_OUT}/eval/eval_phase3.log
    else
        echo "WARNING: No checkpoint found for $EXP"
    fi
done

echo "=== Phase 3 Complete ==="
echo "=== ALL PHASES COMPLETE ==="
"""

    # Upload scripts
    for name, content in [("run_phase1.sh", phase1_script),
                          ("run_phase2.sh", phase2_script),
                          ("run_phase3.sh", phase3_script)]:
        remote_path = f"{REMOTE_OUT}/{name}"
        with sftp.open(remote_path, 'w') as f:
            f.write(content)
        ssh_exec(ssh, f"chmod +x {remote_path}")
        print(f"  Created: {name}")

    # Master script that runs all phases
    master = f"""#!/bin/bash
export PATH="/home/wangchong/miniconda3/envs/fwz/bin:$PATH"
echo "=== Section 35: Multi-Region Enhancement ==="
echo "Started: $(date)"

bash {REMOTE_OUT}/run_phase1.sh
bash {REMOTE_OUT}/run_phase2.sh
bash {REMOTE_OUT}/run_phase3.sh

echo "Completed: $(date)"
echo "=== All Done ==="
"""
    with sftp.open(f"{REMOTE_OUT}/run_all.sh", 'w') as f:
        f.write(master)
    ssh_exec(ssh, f"chmod +x {REMOTE_OUT}/run_all.sh")
    print("  Created: run_all.sh (master)")

    sftp.close()


def launch_phase1(ssh):
    """Launch Phase 1 (baseline evaluation) via nohup."""
    cmd = f"nohup bash {REMOTE_OUT}/run_phase1.sh > {REMOTE_OUT}/phase1.log 2>&1 &"
    ssh_exec(ssh, cmd)
    print("Phase 1 launched (baseline multi-region evaluation)")


def launch_phase2(ssh):
    """Launch Phase 2 (training) via nohup."""
    cmd = f"nohup bash {REMOTE_OUT}/run_phase2.sh > {REMOTE_OUT}/phase2.log 2>&1 &"
    ssh_exec(ssh, cmd)
    print("Phase 2 launched (3 parallel training experiments)")


def launch_all(ssh):
    """Launch all phases sequentially via nohup."""
    cmd = f"nohup bash {REMOTE_OUT}/run_all.sh > {REMOTE_OUT}/master.log 2>&1 &"
    ssh_exec(ssh, cmd)
    print("All phases launched via master script")


if __name__ == '__main__':
    import sys

    ssh = ssh_connect()
    print("Connected to server")

    # Upload scripts
    print("\n--- Uploading scripts ---")
    upload_scripts(ssh)
    create_and_upload_run_script(ssh)

    # Determine what to launch
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode == 'phase1':
        print("\n--- Launching Phase 1 ---")
        launch_phase1(ssh)
    elif mode == 'phase2':
        print("\n--- Launching Phase 2 ---")
        launch_phase2(ssh)
    elif mode == 'all':
        print("\n--- Launching All Phases ---")
        launch_all(ssh)
    else:
        print(f"Unknown mode: {mode}. Use: phase1, phase2, all")

    ssh.close()
    print("\nDone. Monitor with: tail -f /home/wangchong/data/fwz/output/35_multiregion/master.log")
