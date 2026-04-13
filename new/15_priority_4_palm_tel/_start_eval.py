"""Start Priority 4 (PALM+TEL) evaluation on server."""

import paramiko
import time

SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

REMOTE_BASE = "/home/wangchong/data/fwz/code/priority_4_palm_tel"
OUTPUT_DIR = "/home/wangchong/data/fwz/output/priority_4_palm_tel"


def main():
    print("=" * 60)
    print("[Priority 4] Starting PALM+TEL Evaluation")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT,
                   username=SERVER_USER, password=SERVER_PASS, timeout=15)

    # Check for checkpoint
    _, stdout, _ = client.exec_command(
        f"ls -1 {OUTPUT_DIR}/controlnet/cnet-btc-palm-tel-ep-*.pth 2>/dev/null | sort -V | tail -1")
    best_ckpt = stdout.read().decode().strip()

    if not best_ckpt:
        print("ERROR: No checkpoint found! Train first.")
        client.close()
        return

    print(f"  Best checkpoint: {best_ckpt}")

    # Launch evaluation
    _, stdout, _ = client.exec_command(
        f"nohup bash {REMOTE_BASE}/eval.sh > /dev/null 2>&1 &")
    stdout.read()

    time.sleep(3)

    # Verify
    _, stdout, _ = client.exec_command(
        "ps aux | grep 'evaluate_palm_tel' | grep -v grep")
    proc = stdout.read().decode().strip()
    if proc:
        print(f"  Evaluation started!")
        print(f"  {proc[:120]}")
    else:
        print("  Checking log...")
        _, stdout, _ = client.exec_command(f"tail -10 {OUTPUT_DIR}/eval.log 2>/dev/null")
        print(stdout.read().decode().strip())

    client.close()
    print("\nUse _monitor_eval.py to track progress.")


if __name__ == "__main__":
    main()
