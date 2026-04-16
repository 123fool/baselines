"""Launch ET-BoN experiment on server."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.96.27.109", port=2638, username="wangchong", password="123456", timeout=15)

# Launch experiment with nohup
cmd = (
    "cd /home/wangchong/data/fwz/code/et_bon/scripts && "
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "export CUDA_VISIBLE_DEVICES=2 && "
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
    "nohup python run_et_bon_experiment.py "
    "--max-pairs 10 --configs quick --gpu 0 "
    "> /home/wangchong/data/fwz/output/verification/et_bon/et_bon_experiment.log 2>&1 &"
)
# Note: --gpu 0 because CUDA_VISIBLE_DEVICES=2 maps physical GPU 2 to logical GPU 0

_, so, se = ssh.exec_command(cmd, timeout=30)
import time
time.sleep(3)

# Check if running
_, so, _ = ssh.exec_command("ps aux | grep run_et_bon | grep -v grep")
out = so.read().decode().strip()
if out:
    print("RUNNING!")
    print(out)
else:
    print("NOT RUNNING - checking log for errors...")
    _, so, _ = ssh.exec_command("tail -20 /home/wangchong/data/fwz/output/verification/et_bon/et_bon_experiment.log")
    print(so.read().decode())

ssh.close()
