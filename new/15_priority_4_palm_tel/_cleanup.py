"""Clean up Priority 4 processes on server."""

import paramiko

SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT,
                   username=SERVER_USER, password=SERVER_PASS, timeout=15)

    # Kill training
    _, stdout, _ = client.exec_command(
        "pkill -f 'train_controlnet_btc_palm_tel' 2>/dev/null; "
        "pkill -f 'evaluate_palm_tel' 2>/dev/null; "
        "echo 'Killed P4 processes'")
    print(stdout.read().decode().strip())

    # Verify
    _, stdout, _ = client.exec_command(
        "ps aux | grep -E 'palm_tel|btc_palm' | grep -v grep | wc -l")
    count = stdout.read().decode().strip()
    print(f"Remaining P4 processes: {count}")

    client.close()


if __name__ == "__main__":
    main()
