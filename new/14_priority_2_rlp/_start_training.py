"""Start Priority 2 RLP training on the server - clean restart."""
import paramiko
import time
import sys

SERVER = "10.96.27.109"
PORT = 2638
USER = "wangchong"
PASS = "123456"
CODE_DIR = "/home/wangchong/data/fwz/code/priority_2_rlp"
PYTHON = "/home/wangchong/miniconda3/envs/fwz/bin/python"
BASE_OUT = "/home/wangchong/data/fwz/output/priority_2_rlp"
BASE_CACHE = "/home/wangchong/data/fwz/cache/priority_2_rlp"

def ssh_exec(client, cmd, timeout=15):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

def launch_training(client, sftp, name, script, extra_args="", gpu=1):
    """Launch a training script via shell launcher on the server."""
    out_dir = f"{BASE_OUT}/{name}/controlnet"
    cache_dir = f"{BASE_CACHE}/{name}"
    log_file = f"{BASE_OUT}/{name}/train.log"
    
    launcher = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES={gpu}
cd {CODE_DIR}
nohup {PYTHON} {CODE_DIR}/scripts/{script} \\
    --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv \\
    --cache_dir {cache_dir} \\
    --output_dir {out_dir} \\
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \\
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \\
    --cnet_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/controlnet.pth \\
    --n_epochs 5 --batch_size 8 --lr 2.5e-5 {extra_args} \\
    > {log_file} 2>&1 &
echo "PID=$!"
"""
    launcher_path = f"{CODE_DIR}/_run_{name}.sh"
    with sftp.open(launcher_path, 'w') as f:
        f.write(launcher)
    
    out, err = ssh_exec(client, f"chmod +x {launcher_path} && bash {launcher_path}")
    print(f"  {out}")
    if err:
        print(f"  Stderr: {err}")
    return log_file

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rlp_only"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {SERVER}:{PORT}...")
    client.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=15)
    sftp = client.open_sftp()

    # Create dirs
    for sub in ["rlp_only/controlnet", "btr_rlp/controlnet", "rlp_only/eval", "btr_rlp/eval"]:
        ssh_exec(client, f"mkdir -p {BASE_OUT}/{sub}")
    for sub in ["rlp_only", "btr_rlp"]:
        ssh_exec(client, f"mkdir -p {BASE_CACHE}/{sub}")

    # GPU status
    print("\nGPU Status:")
    out, _ = ssh_exec(client, "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader")
    print(f"  {out}")

    # Check existing
    out, _ = ssh_exec(client, "ps aux | grep -E 'train_controlnet|evaluate_rlp' | grep -v grep | wc -l")
    n = int(out.strip()) if out.strip().isdigit() else 0
    if n > 0:
        print(f"\nWARNING: {n} existing processes. Kill first with _cleanup.py")
        client.close()
        return

    if mode == "rlp_only":
        # Clean old output
        ssh_exec(client, f"rm -f {BASE_OUT}/rlp_only/controlnet/cnet-rlp-ep-*.pth")
        ssh_exec(client, f"rm -f {BASE_OUT}/rlp_only/train.log")
        
        print("\n=== Starting RLP-only training (GPU 1) ===")
        log_file = launch_training(client, sftp, "rlp_only", "train_controlnet_rlp.py", gpu=1)
        
    elif mode == "btr_rlp":
        ssh_exec(client, f"rm -f {BASE_OUT}/btr_rlp/controlnet/cnet-btr-rlp-ep-*.pth")
        ssh_exec(client, f"rm -f {BASE_OUT}/btr_rlp/train.log")
        
        print("\n=== Starting BTR+RLP training (GPU 1) ===")
        log_file = launch_training(client, sftp, "btr_rlp", "train_controlnet_btr_rlp.py", 
                                   extra_args="--btc_weight 0.5", gpu=1)
    else:
        print(f"Unknown mode: {mode}")
        client.close()
        return

    # Verify
    time.sleep(5)
    script_name = "train_controlnet_rlp" if mode == "rlp_only" else "train_controlnet_btr_rlp"
    out, _ = ssh_exec(client, f"ps aux | grep '{script_name}.py' | grep -v grep | head -1")
    if out:
        pid = out.split()[1]
        print(f"  Process running! PID: {pid}")
    else:
        print("  WARNING: Process not found!")

    # Check log
    time.sleep(3)
    out, _ = ssh_exec(client, f"tail -15 {log_file}")
    print(f"\n=== Log tail ===\n{out}")

    # Verify GPU usage
    out, _ = ssh_exec(client, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader")
    print(f"\n=== GPU memory ===\n{out}")

    sftp.close()
    client.close()
    print(f"\nDone! Monitor: tail -f {log_file}")


if __name__ == '__main__':
    main()
