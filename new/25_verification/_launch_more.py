import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

conda = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz"
code_dir = "/home/wangchong/data/fwz/code/verification"
base = "/home/wangchong/data/fwz"

# Experiment 2: Add bon_weighted + roundtrip_bon (5 pairs, N=5)
exp2_name = "weighted_roundtrip"
exp2_dir = f"{base}/output/verification/{exp2_name}"
exp2_cmd = f"""
{conda} && cd {code_dir} && mkdir -p {exp2_dir} && nohup python evaluate_verification.py \\
    --dataset_csv {base}/output/innovation_5/prepared/B_mci.csv \\
    --aekl_ckpt {base}/output/innovation_5/ae/autoencoder-ep-2.pth \\
    --diff_ckpt {base}/brlp-train/pretrained/latentdiffusion.pth \\
    --cnet_ckpt {base}/output/innovation_2/controlnet/cnet-btr-ep-1.pth \\
    --output_dir {exp2_dir} \\
    --n_candidates 5 --las_m 3 --max_pairs 5 \\
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \\
    > {exp2_dir}/eval_verification.log 2>&1 &
echo "PID2=$!"
"""

_, stdout, stderr = c.exec_command(exp2_cmd, timeout=15)
out2 = stdout.read().decode().strip()
err2 = stderr.read().decode().strip()
print(f"Exp2 ({exp2_name}): {out2}")
if err2:
    print(f"  stderr: {err2[:200]}")

# Experiment 3: Larger N=8 experiment with all methods (10 pairs)
exp3_name = "bon_n8_full"
exp3_dir = f"{base}/output/verification/{exp3_name}"
exp3_cmd = f"""
{conda} && cd {code_dir} && mkdir -p {exp3_dir} && nohup python evaluate_verification.py \\
    --dataset_csv {base}/output/innovation_5/prepared/B_mci.csv \\
    --aekl_ckpt {base}/output/innovation_5/ae/autoencoder-ep-2.pth \\
    --diff_ckpt {base}/brlp-train/pretrained/latentdiffusion.pth \\
    --cnet_ckpt {base}/output/innovation_2/controlnet/cnet-btr-ep-1.pth \\
    --output_dir {exp3_dir} \\
    --n_candidates 8 --las_m 3 --max_pairs 10 \\
    --methods "las,single,bon_best1,bon_topk,bon_weighted" \\
    > {exp3_dir}/eval_verification.log 2>&1 &
echo "PID3=$!"
"""

_, stdout, stderr = c.exec_command(exp3_cmd, timeout=15)
out3 = stdout.read().decode().strip()
err3 = stderr.read().decode().strip()
print(f"Exp3 ({exp3_name}): {out3}")
if err3:
    print(f"  stderr: {err3[:200]}")

# Verify processes started
import time
time.sleep(2)
_, stdout, _ = c.exec_command("ps aux | grep evaluate_verification | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
print(f"\nRunning processes:")
for line in ps.split('\n'):
    if line.strip():
        parts = line.split()
        pid = parts[1]
        # find --output_dir to identify experiment
        cmd = ' '.join(parts[10:])
        if 'weighted_roundtrip' in cmd:
            print(f"  PID {pid}: weighted_roundtrip")
        elif 'bon_n8_full' in cmd:
            print(f"  PID {pid}: bon_n8_full")
        else:
            print(f"  PID {pid}: (other)")

c.close()
