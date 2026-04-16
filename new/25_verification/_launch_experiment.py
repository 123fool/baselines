"""
Launch verification experiments on server via SSH.
Runs the quick_compare experiment first (5 pairs, small N) as a smoke test.
"""
import paramiko
import time

SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVER_HOST, port=SERVER_PORT, username=SERVER_USER,
          password=SERVER_PASS, timeout=15)

# First, run the quick smoke test (5 pairs, N=5, methods LAS + single + BoN best1)
# This should take ~10-15 min
launch_cmd = """
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd /home/wangchong/data/fwz/code/verification/scripts

export PYTHONPATH="/home/wangchong/data/fwz/code/verification/src:$PYTHONPATH"

mkdir -p /home/wangchong/data/fwz/output/verification/quick_compare

nohup python evaluate_verification.py \
  --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv \
  --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
  --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
  --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth \
  --output_dir /home/wangchong/data/fwz/output/verification/quick_compare \
  --max_pairs 5 \
  --n_candidates 5 \
  --las_m 3 \
  --methods "las,single,bon_best1,bon_topk" \
  --model_name quick_compare \
  > /home/wangchong/data/fwz/output/verification/runner.log 2>&1 &

echo "PID=$!"
"""

print("Launching quick_compare experiment on server...")
_, stdout, stderr = c.exec_command(launch_cmd, timeout=30)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(f"OUT: {out}")
if err:
    print(f"ERR: {err}")

# Wait a bit and check if process started
time.sleep(3)
_, stdout, _ = c.exec_command(
    "ps aux | grep evaluate_verification | grep -v grep | head -3",
    timeout=10
)
ps_out = stdout.read().decode().strip()
print(f"\nProcess check:\n{ps_out}")

# Check initial log output
_, stdout, _ = c.exec_command(
    "cat /home/wangchong/data/fwz/output/verification/runner.log 2>/dev/null | head -20",
    timeout=10
)
log_out = stdout.read().decode().strip()
print(f"\nInitial log:\n{log_out}")

c.close()
print("\nExperiment launched. Monitor via dashboard or check runner.log")
