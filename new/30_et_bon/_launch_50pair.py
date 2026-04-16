"""Upload updated run_et_bon_experiment.py to server and launch 50-pair expanded test."""
import paramiko
import os

LOCAL_SCRIPT = os.path.join(os.path.dirname(__file__), "scripts", "run_et_bon_experiment.py")
REMOTE_SCRIPT = "/home/wangchong/data/fwz/code/et_bon/scripts/run_et_bon_experiment.py"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=10)

# Upload updated script
sftp = ssh.open_sftp()
sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
sftp.close()
print("Uploaded run_et_bon_experiment.py")

# Check if quick experiment is still running
_, o, _ = ssh.exec_command("ps aux | grep 'run_et_bon_experiment' | grep -v grep | wc -l")
n_running = int(o.read().decode().strip())
print(f"Running instances: {n_running}")

if n_running > 0:
    print("Quick experiment still running. NOT launching 50-pair yet.")
    print("Run this script again after the quick experiment finishes.")
else:
    # Launch 50-pair expanded test
    launch_cmd = (
        "cd /home/wangchong/data/fwz/code/et_bon/scripts && "
        "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
        "conda activate fwz && "
        "export CUDA_VISIBLE_DEVICES=2 && "
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        "nohup python run_et_bon_experiment.py --max-pairs 50 --configs best --gpu 0 "
        "--output-dir /home/wangchong/data/fwz/output/verification/et_bon_50pair "
        "> /home/wangchong/data/fwz/output/verification/et_bon_50pair/et_bon_50pair.log 2>&1 &"
    )
    # Create output dir first
    ssh.exec_command("mkdir -p /home/wangchong/data/fwz/output/verification/et_bon_50pair")
    import time; time.sleep(0.5)
    
    _, o, e = ssh.exec_command(launch_cmd)
    print("Launch command sent.")
    import time; time.sleep(2)
    
    # Verify
    _, o, _ = ssh.exec_command("ps aux | grep 'run_et_bon_experiment' | grep -v grep")
    print(o.read().decode().strip())

ssh.close()
