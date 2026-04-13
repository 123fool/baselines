"""Monitor Priority 4 (PALM+TEL) training progress on server."""

import paramiko

SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

OUTPUT_DIR = "/home/wangchong/data/fwz/output/priority_4_palm_tel"
CNET_DIR = f"{OUTPUT_DIR}/controlnet"


def ssh_exec(client, cmd, timeout=10):
    _, stdout, _ = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace").strip()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT,
                   username=SERVER_USER, password=SERVER_PASS, timeout=15)

    # GPU status
    gpu = ssh_exec(client,
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits")
    print("GPU:", gpu.replace('\n', '\n     '))

    # Training process
    proc = ssh_exec(client,
        "ps aux | grep 'train_controlnet_btc_palm_tel' | grep -v grep | wc -l")
    print(f"Train processes: {proc}")

    # Key info from log
    key_info = ssh_exec(client,
        f"grep -E '\\[Priority 4\\]|Scale factor|PALM params|TEL params|"
        f"Training:|Validation:|Checkpoint:|\\[Epoch' "
        f"{OUTPUT_DIR}/train.log 2>/dev/null | tail -20")
    if key_info:
        print(f"\n=== Key Info ===\n{key_info}")

    # Current progress (last tqdm line)
    current = ssh_exec(client,
        f"tail -5 {OUTPUT_DIR}/train.log 2>/dev/null")
    if current:
        print(f"\n=== Current ===\n{current}")

    # Checkpoints
    ckpts = ssh_exec(client,
        f"ls -la {CNET_DIR}/cnet-btc-palm-tel-ep-*.pth 2>/dev/null")
    if ckpts:
        print(f"\n=== Checkpoints ===\n{ckpts}")
    else:
        print("\n=== Checkpoints ===\nNone yet")

    client.close()


if __name__ == "__main__":
    main()
