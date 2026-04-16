"""Upload fullscale BoN files to server with proper structure."""
import paramiko, os

SSH_HOST = "10.96.27.109"
SSH_PORT = 2638
SSH_USER = "wangchong"
SSH_PASS = "123456"
REMOTE_BASE = "/home/wangchong/data/fwz/code/verification"

# Files to upload: (local_path, remote_relative_path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.join(SCRIPT_DIR, '..', '..', 'src', 'brlp')

FILES = [
    # New fullscale scripts
    (os.path.join(SCRIPT_DIR, 'scripts', 'run_bon_fullscale.py'), 'scripts/run_bon_fullscale.py'),
    (os.path.join(SCRIPT_DIR, 'scripts', 'sampling_bon_integrated.py'), 'scripts/sampling_bon_integrated.py'),
    # BrLP source (in case they changed)
    (os.path.join(BRLP_SRC, 'sampling.py'), 'src/brlp/sampling.py'),
    (os.path.join(BRLP_SRC, 'networks.py'), 'src/brlp/networks.py'),
    (os.path.join(BRLP_SRC, 'utils.py'), 'src/brlp/utils.py'),
    (os.path.join(BRLP_SRC, 'const.py'), 'src/brlp/const.py'),
    (os.path.join(BRLP_SRC, 'data.py'), 'src/brlp/data.py'),
    (os.path.join(BRLP_SRC, '__init__.py'), 'src/brlp/__init__.py'),
]

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS)
    sftp = ssh.open_sftp()

    # Ensure directories
    for d in ['scripts', 'src', 'src/brlp']:
        remote_d = f"{REMOTE_BASE}/{d}"
        try:
            sftp.stat(remote_d)
        except FileNotFoundError:
            ssh.exec_command(f"mkdir -p {remote_d}")
            import time; time.sleep(0.3)

    # Also ensure output directory
    ssh.exec_command("mkdir -p /home/wangchong/data/fwz/output/verification/fullscale_50")

    uploaded = 0
    for local, rel in FILES:
        local = os.path.abspath(local)
        if not os.path.exists(local):
            print(f"  SKIP (not found): {local}")
            continue
        remote = f"{REMOTE_BASE}/{rel}"
        print(f"  {rel} <- {os.path.basename(local)}")
        sftp.put(local, remote)
        uploaded += 1

    sftp.close()
    print(f"\nUploaded {uploaded} files to {REMOTE_BASE}/")

    # Verify imports
    print("\nVerifying imports...")
    cmd = (
        "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
        "conda activate fwz && "
        f"cd {REMOTE_BASE}/scripts && "
        "python -c 'import sys; sys.path.insert(0,\"../src\"); "
        "from brlp import const, utils, networks; "
        "from sampling_bon_integrated import sample_bon_weighted; "
        "print(\"OK: all imports successful\")'"
    )
    stdin, stdout, stderr = ssh.exec_command(cmd)
    print("stdout:", stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("stderr:", err[-500:])

    ssh.close()


if __name__ == '__main__':
    main()
