"""
Upload ET-BoN experiment code to server.

Usage:
  python upload_et_bon.py
"""

import os
import paramiko
from scp import SCPClient

SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

LOCAL_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")
REMOTE_BASE = "/home/wangchong/data/fwz/code/et_bon"
REMOTE_SCRIPTS = f"{REMOTE_BASE}/scripts"

# Also upload the brlp source if needed
LOCAL_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SERVER_HOST, port=SERVER_PORT, username=SERVER_USER, password=SERVER_PASS, timeout=15)

    # Create remote directories
    for d in [REMOTE_BASE, REMOTE_SCRIPTS]:
        ssh.exec_command(f"mkdir -p {d}")
    import time; time.sleep(1)

    scp = SCPClient(ssh.get_transport())

    # Upload ET-BoN scripts
    files_to_upload = [
        (os.path.join(LOCAL_SCRIPTS_DIR, "sampling_et_bon.py"), f"{REMOTE_SCRIPTS}/sampling_et_bon.py"),
        (os.path.join(LOCAL_SCRIPTS_DIR, "run_et_bon_experiment.py"), f"{REMOTE_SCRIPTS}/run_et_bon_experiment.py"),
    ]

    for local, remote in files_to_upload:
        if os.path.exists(local):
            print(f"Uploading {os.path.basename(local)} -> {remote}")
            scp.put(local, remote)
        else:
            print(f"SKIP (not found): {local}")

    # Verify
    _, stdout, _ = ssh.exec_command(f"ls -la {REMOTE_SCRIPTS}/")
    print("\nRemote files:")
    print(stdout.read().decode())

    # Create output directory
    ssh.exec_command("mkdir -p /home/wangchong/data/fwz/output/verification/et_bon")

    # Check if brlp src exists on server
    _, stdout, _ = ssh.exec_command("ls /home/wangchong/data/fwz/code/et_bon/../../src/brlp/sampling.py 2>/dev/null || echo 'NOT_FOUND'")
    result = stdout.read().decode().strip()
    if "NOT_FOUND" in result:
        # Need symlink or upload src
        print("Creating symlink to brlp src...")
        ssh.exec_command(f"ln -sf /home/wangchong/data/fwz/code/src {REMOTE_BASE}/../../src 2>/dev/null || true")

    # Quick import test
    print("\nTesting imports...")
    test_cmd = (
        f"cd {REMOTE_SCRIPTS} && "
        "source /home/wangchong/anaconda3/etc/profile.d/conda.sh && "
        "conda activate fwz && "
        "python -c \"import sys; sys.path.insert(0, '.'); "
        "from sampling_et_bon import sample_et_bon_weighted; "
        "print('OK: sampling_et_bon imported successfully')\""
    )
    _, stdout, stderr = ssh.exec_command(test_cmd, timeout=30)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f"stdout: {out}")
    if err:
        print(f"stderr: {err}")

    scp.close()
    ssh.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
