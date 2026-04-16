"""Launch quick test: 3 pairs, methods A,B,D on GPU 1."""
import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Upload latest script
import os
LOCAL = os.path.join(os.path.dirname(__file__), "scripts", "hippocampus_improvement.py")
REMOTE = "/home/wangchong/data/fwz/code/33_hippocampus/scripts/hippocampus_improvement.py"
sftp = ssh.open_sftp()
sftp.put(LOCAL, REMOTE)
sftp.close()
print("Script uploaded.")

# Create output directory
ssh.exec_command("mkdir -p /home/wangchong/data/fwz/output/33_hippocampus")
time.sleep(1)

# Launch quick test
cmd = (
    "cd /home/wangchong/data/fwz/code/33_hippocampus/scripts && "
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "export CUDA_VISIBLE_DEVICES=1 && "
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
    "nohup python hippocampus_improvement.py "
    "--methods A,B,D "
    "--max-pairs 3 "
    "--gpu 0 "
    "--output-dir /home/wangchong/data/fwz/output/33_hippocampus "
    "> /home/wangchong/data/fwz/output/33_hippocampus/run.log 2>&1 &"
)
print(f"Launching: {cmd[:100]}...")
_, o, e = ssh.exec_command(cmd)
time.sleep(3)

# Verify
_, o, _ = ssh.exec_command("ps aux | grep 'hippocampus_improvement' | grep -v grep")
ps = o.read().decode().strip()
if ps:
    print(f"✅ Running:\n{ps}")
else:
    print("⚠️ Not running. Checking log...")
    _, o, _ = ssh.exec_command("tail -30 /home/wangchong/data/fwz/output/33_hippocampus/run.log")
    print(o.read().decode())

ssh.close()
