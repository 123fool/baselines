import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Kill the old process
print("Killing old process...")
_, stdout, _ = c.exec_command("pkill -f evaluate_verification", timeout=10)
import time; time.sleep(2)

# Upload the fixed file
sftp = c.open_sftp()
sftp.put(
    r"c:\Users\PC\Desktop\baselines\BrLP-main\new\25_verification\scripts\evaluate_verification.py",
    "/home/wangchong/data/fwz/code/verification/scripts/evaluate_verification.py"
)
sftp.close()
print("Fixed file uploaded.")

# Clear old log and restart
launch_cmd = """
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd /home/wangchong/data/fwz/code/verification/scripts

export PYTHONPATH="/home/wangchong/data/fwz/code/verification/src:$PYTHONPATH"

rm -f /home/wangchong/data/fwz/output/verification/runner.log
rm -f /home/wangchong/data/fwz/output/verification/quick_compare/eval_verification.log
rm -f /home/wangchong/data/fwz/output/verification/quick_compare/summary_quick_compare.json

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

_, stdout, stderr = c.exec_command(launch_cmd, timeout=30)
out = stdout.read().decode().strip()
print(f"Restarted: {out}")

time.sleep(3)
_, stdout, _ = c.exec_command("ps aux | grep evaluate_verification | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
print(f"Process: {'RUNNING' if ps else 'NOT FOUND'}")

c.close()
