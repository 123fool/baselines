"""
Upload hippocampus improvement scripts to server and launch experiment.

Usage:
    python upload_and_run.py                    # Quick test: methods A,B,D, 5 pairs
    python upload_and_run.py --full             # Full: methods A,B,C,D,E,F, 50 pairs
    python upload_and_run.py --methods A,B --max-pairs 10
"""
import paramiko
import os
import sys
import time
import argparse

# Server config
SERVER = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'

# Paths
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_SCRIPT = os.path.join(LOCAL_DIR, "scripts", "hippocampus_improvement.py")

REMOTE_BASE = "/home/wangchong/data/fwz/code/33_hippocampus"
REMOTE_SCRIPT = f"{REMOTE_BASE}/scripts/hippocampus_improvement.py"
REMOTE_OUTPUT = "/home/wangchong/data/fwz/output/33_hippocampus"

PYTHON = "/home/wangchong/miniconda3/envs/fwz/bin/python"


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=15)
    return ssh


def upload_files(ssh):
    sftp = ssh.open_sftp()

    # Create directories
    for d in [REMOTE_BASE, f"{REMOTE_BASE}/scripts", REMOTE_OUTPUT]:
        try:
            sftp.mkdir(d)
        except IOError:
            pass

    # Upload script
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
    print(f"✅ Uploaded {LOCAL_SCRIPT} -> {REMOTE_SCRIPT}")

    sftp.close()


def check_running(ssh):
    _, o, _ = ssh.exec_command("ps aux | grep 'hippocampus_improvement' | grep -v grep | wc -l")
    return int(o.read().decode().strip())


def launch(ssh, methods, max_pairs, gpu=2):
    cmd = (
        f"cd {REMOTE_BASE}/scripts && "
        f"source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate fwz && "
        f"export CUDA_VISIBLE_DEVICES={gpu} && "
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"nohup {PYTHON} hippocampus_improvement.py "
        f"--methods {methods} "
        f"--max-pairs {max_pairs} "
        f"--gpu 0 "
        f"--output-dir {REMOTE_OUTPUT} "
        f"> {REMOTE_OUTPUT}/run.log 2>&1 &"
    )
    print(f"\nLaunch command:\n{cmd}\n")
    _, o, e = ssh.exec_command(cmd)
    time.sleep(3)

    # Verify
    n = check_running(ssh)
    if n > 0:
        print(f"✅ Process running ({n} instances)")
    else:
        print("⚠️  Process may not have started. Check logs:")
        print(f"   ssh -p {PORT} {USER}@{SERVER}")
        print(f"   tail -f {REMOTE_OUTPUT}/run.log")

    # Show first few lines of log
    _, o, _ = ssh.exec_command(f"tail -20 {REMOTE_OUTPUT}/run.log 2>/dev/null")
    log_out = o.read().decode().strip()
    if log_out:
        print(f"\n--- Log output ---\n{log_out}\n")


def check_status(ssh):
    n = check_running(ssh)
    print(f"Running instances: {n}")

    _, o, _ = ssh.exec_command(f"tail -30 {REMOTE_OUTPUT}/run.log 2>/dev/null")
    log_out = o.read().decode().strip()
    if log_out:
        print(f"\n--- Latest log ---\n{log_out}\n")

    _, o, _ = ssh.exec_command(f"cat {REMOTE_OUTPUT}/experiment_summary.json 2>/dev/null")
    summary = o.read().decode().strip()
    if summary:
        print(f"\n--- Summary ---\n{summary}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Full experiment: all methods, 50 pairs')
    parser.add_argument('--methods', default=None, type=str)
    parser.add_argument('--max-pairs', default=None, type=int)
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument('--status', action='store_true', help='Check running status')
    parser.add_argument('--upload-only', action='store_true')
    args = parser.parse_args()

    ssh = ssh_connect()
    print(f"Connected to {SERVER}:{PORT}")

    if args.status:
        check_status(ssh)
        ssh.close()
        sys.exit(0)

    # Upload
    upload_files(ssh)

    if args.upload_only:
        ssh.close()
        sys.exit(0)

    # Determine methods and pairs
    if args.full:
        methods = args.methods or 'A,B,C,D,E,F'
        max_pairs = args.max_pairs or 50
    else:
        methods = args.methods or 'A,B,D'
        max_pairs = args.max_pairs or 5

    # Check if already running
    n = check_running(ssh)
    if n > 0:
        print(f"⚠️ Already running ({n} instances). Check status first.")
        resp = input("Continue anyway? (y/n): ").strip().lower()
        if resp != 'y':
            ssh.close()
            sys.exit(0)

    launch(ssh, methods, max_pairs, args.gpu)
    ssh.close()
