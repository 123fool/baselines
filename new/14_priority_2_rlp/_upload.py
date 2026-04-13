"""Quick upload script to transfer Priority 2 RLP files to server."""
import os
import paramiko
import stat

SERVER = "10.96.27.109"
PORT = 2638
USER = "wangchong"
PASS = "123456"

LOCAL_DIR = r"c:\Users\PC\Desktop\baselines\BrLP-main\new\14_priority_2_rlp"
REMOTE_DIR = "/home/wangchong/data/fwz/code/priority_2_rlp"

# Also upload the brlp src for imports
BRLP_SRC = r"c:\Users\PC\Desktop\baselines\BrLP-main\src\brlp"
REMOTE_BRLP_SRC = "/home/wangchong/data/fwz/code/priority_2_rlp/brlp_src/brlp"

# Innovation 2 src (for bidirectional_temporal.py import)
INNOV2_SRC = r"c:\Users\PC\Desktop\baselines\BrLP-main\new\12_innovation_2\src"
REMOTE_INNOV2 = "/home/wangchong/data/fwz/code/priority_2_rlp/innov2_src"


def mkdir_p(sftp, remote_dir):
    """Recursively create remote directories."""
    dirs = []
    while remote_dir not in ('/', ''):
        try:
            sftp.stat(remote_dir)
            break
        except FileNotFoundError:
            dirs.append(remote_dir)
            remote_dir = os.path.dirname(remote_dir)
    for d in reversed(dirs):
        try:
            sftp.mkdir(d)
            print(f"  mkdir {d}")
        except Exception:
            pass


def upload_dir(sftp, local_dir, remote_dir, exclude=None):
    """Upload a local directory to remote."""
    exclude = exclude or ['__pycache__', '.pyc']
    mkdir_p(sftp, remote_dir)
    for item in os.listdir(local_dir):
        if any(ex in item for ex in exclude):
            continue
        local_path = os.path.join(local_dir, item)
        remote_path = f"{remote_dir}/{item}"
        if os.path.isdir(local_path):
            upload_dir(sftp, local_path, remote_path, exclude)
        else:
            sftp.put(local_path, remote_path)
            print(f"  uploaded {remote_path}")


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {SERVER}:{PORT}...")
    client.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=15)
    sftp = client.open_sftp()

    print(f"\n=== Uploading Priority 2 RLP code to {REMOTE_DIR} ===")
    upload_dir(sftp, LOCAL_DIR, REMOTE_DIR)

    print(f"\n=== Uploading BrLP src to {REMOTE_BRLP_SRC} ===")
    upload_dir(sftp, BRLP_SRC, REMOTE_BRLP_SRC)

    print(f"\n=== Uploading Innovation 2 src to {REMOTE_INNOV2} ===")
    upload_dir(sftp, INNOV2_SRC, REMOTE_INNOV2)

    # Make train.sh executable
    print("\nMaking train.sh executable...")
    _, stdout, _ = client.exec_command(f"chmod +x {REMOTE_DIR}/train.sh")
    stdout.read()

    # Verify upload
    print("\n=== Verifying upload ===")
    _, stdout, _ = client.exec_command(f"find {REMOTE_DIR} -type f | sort")
    files = stdout.read().decode().strip()
    print(files)

    sftp.close()
    client.close()
    print("\nUpload complete!")


if __name__ == '__main__':
    main()
