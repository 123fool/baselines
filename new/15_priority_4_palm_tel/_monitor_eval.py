"""Monitor Priority 4 (PALM+TEL) evaluation progress on server."""

import paramiko

SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

OUTPUT_DIR = "/home/wangchong/data/fwz/output/priority_4_palm_tel"


def ssh_exec(client, cmd, timeout=10):
    _, stdout, _ = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace").strip()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT,
                   username=SERVER_USER, password=SERVER_PASS, timeout=15)

    # GPU
    gpu = ssh_exec(client,
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits")
    print("GPU:", gpu.replace('\n', '\n     '))

    # Eval process
    proc = ssh_exec(client,
        "ps aux | grep 'evaluate_palm_tel' | grep -v grep | wc -l")
    print(f"Eval processes: {proc}")

    # Key info from eval log
    key_info = ssh_exec(client,
        f"grep -E '\\[Priority 4\\]|Scale factor|Evaluating|Evaluation Results|"
        f"overall_ssim|overall_psnr|overall_mae|hippocampus|amygdala|roi_|"
        f"Pairs evaluated|Results saved' "
        f"{OUTPUT_DIR}/eval.log 2>/dev/null | tail -20")
    if key_info:
        print(f"\n=== Key Info ===\n{key_info}")

    # Current progress
    current = ssh_exec(client,
        f"tail -5 {OUTPUT_DIR}/eval.log 2>/dev/null")
    if current:
        print(f"\n=== Current ===\n{current}")

    # Output files
    files = ssh_exec(client,
        f"ls -la {OUTPUT_DIR}/eval/ 2>/dev/null")
    if files:
        print(f"\n=== Output Files ===\n{files}")

    client.close()


if __name__ == "__main__":
    main()
