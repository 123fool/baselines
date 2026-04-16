"""Upload V2 and launch quick test: A,B,G,I,O with 5 pairs.
A = baseline
B = Best-of-16 hippo scoring
G = Best-of-16 overall scoring
I = Weighted fusion top-5
O = Oracle (GT scoring, upper bound)
"""
import paramiko, os, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Upload V2 script
LOCAL = os.path.join(os.path.dirname(__file__), "scripts", "hippocampus_improvement_v2.py")
REMOTE_DIR = "/home/wangchong/data/fwz/code/33_hippocampus/scripts"
REMOTE = f"{REMOTE_DIR}/hippocampus_improvement_v2.py"
OUT_DIR = "/home/wangchong/data/fwz/output/33_hippocampus_v2"

sftp = ssh.open_sftp()
sftp.put(LOCAL, REMOTE)
sftp.close()
print("V2 script uploaded.")

ssh.exec_command(f"mkdir -p {OUT_DIR}")
time.sleep(0.5)

# Launch on GPU 1
cmd = (
    f"cd {REMOTE_DIR} && "
    f"source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    f"conda activate fwz && "
    f"export CUDA_VISIBLE_DEVICES=1 && "
    f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
    f"nohup python hippocampus_improvement_v2.py "
    f"--methods A,B,G,I,O "
    f"--max-pairs 5 "
    f"--gpu 0 "
    f"--output-dir {OUT_DIR} "
    f"> {OUT_DIR}/run.log 2>&1 &"
)
print(f"Launching: methods A,B,G,I,O, 5 pairs, GPU 1")
ssh.exec_command(cmd)
time.sleep(3)

_, o, _ = ssh.exec_command("ps aux | grep 'hippocampus_improvement_v2' | grep -v grep")
ps = o.read().decode().strip()
if ps:
    print(f"✅ Running:\n{ps}")
else:
    print("⚠️ Not running. Check log:")
    _, o, _ = ssh.exec_command(f"tail -20 {OUT_DIR}/run.log")
    print(o.read().decode())

ssh.close()
