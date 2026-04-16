"""
Upload verification experiment code to server.
Organize files into /home/wangchong/data/fwz/code/verification/
"""
import os
import sys
import paramiko
from pathlib import Path

SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"
REMOTE_BASE = "/home/wangchong/data/fwz/code/verification"
LOCAL_BASE = os.path.join(os.path.dirname(__file__), "scripts")


def upload():
    print(f"Connecting to {SERVER_HOST}:{SERVER_PORT}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT,
                   username=SERVER_USER, password=SERVER_PASS, timeout=15)

    sftp = client.open_sftp()

    # Create remote directories
    for d in ["/home/wangchong/data/fwz/code",
              REMOTE_BASE,
              f"{REMOTE_BASE}/scripts"]:
        try:
            sftp.mkdir(d)
            print(f"  Created: {d}")
        except IOError:
            pass  # already exists

    # Upload scripts
    scripts_dir = Path(LOCAL_BASE)
    uploaded = 0
    for f in scripts_dir.glob("*.py"):
        remote_path = f"{REMOTE_BASE}/scripts/{f.name}"
        print(f"  Uploading: {f.name} -> {remote_path}")
        sftp.put(str(f), remote_path)
        uploaded += 1

    # Also upload src/brlp changes if needed (sampling.py etc.)
    brlp_src = Path(LOCAL_BASE).parent.parent.parent / "src" / "brlp"
    brlp_remote = "/home/wangchong/data/fwz/code/verification/src/brlp"
    for d in [f"{REMOTE_BASE}/src",
              f"{REMOTE_BASE}/src/brlp"]:
        try:
            sftp.mkdir(d)
        except IOError:
            pass

    for f in brlp_src.glob("*.py"):
        remote_path = f"{brlp_remote}/{f.name}"
        print(f"  Uploading brlp: {f.name} -> {remote_path}")
        sftp.put(str(f), remote_path)
        uploaded += 1

    sftp.close()
    print(f"\nUploaded {uploaded} files to {REMOTE_BASE}")

    # Set permissions and verify
    _, stdout, _ = client.exec_command(f"ls -la {REMOTE_BASE}/scripts/")
    print("\nRemote files:")
    print(stdout.read().decode())

    client.close()
    print("Done.")


if __name__ == '__main__':
    upload()
