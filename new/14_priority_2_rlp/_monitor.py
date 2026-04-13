"""Monitor training progress on server."""
import paramiko
import sys

SERVER = "10.96.27.109"
PORT = 2638
USER = "wangchong"
PASS = "123456"

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rlp_only"
    log_path = f"/home/wangchong/data/fwz/output/priority_2_rlp/{mode}/train.log"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=15)

    # GPU
    _, stdout, _ = client.exec_command("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader", timeout=10)
    print(f"GPU: {stdout.read().decode().strip()}")

    # Process
    _, stdout, _ = client.exec_command(f"ps aux | grep 'train_controlnet' | grep -v grep | wc -l", timeout=10)
    n = stdout.read().decode().strip()
    print(f"Training processes: {n}")

    # Extract key lines: Epoch summaries, checkpoints, completion messages
    _, stdout, _ = client.exec_command(
        f"grep -E '\\[Epoch|Checkpoint|complete|Scale factor|Device|btc_weight|Priority|BTR' {log_path} 2>/dev/null | tail -30",
        timeout=10)
    key_lines = stdout.read().decode().strip()
    print(f"\n=== Key Info ===\n{key_lines}")

    # Last 5 lines for current progress
    _, stdout, _ = client.exec_command(f"tail -3 {log_path} 2>/dev/null", timeout=10)
    tail = stdout.read().decode().strip()
    print(f"\n=== Current ===\n{tail}")

    # Checkpoints
    ckpt_dir = f"/home/wangchong/data/fwz/output/priority_2_rlp/{mode}/controlnet"
    _, stdout, _ = client.exec_command(f"ls -la {ckpt_dir}/*.pth 2>/dev/null", timeout=10)
    ckpts = stdout.read().decode().strip()
    if ckpts:
        print(f"\n=== Checkpoints ===\n{ckpts}")
    else:
        print("\nNo checkpoints yet.")

    client.close()

if __name__ == '__main__':
    main()
