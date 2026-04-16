"""Check previous eval SSIM scores and data splits."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=10)

cmds = [
    # Check split distribution
    "head -1 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv",
    "awk -F',' '{print $3}' /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | sort | uniq -c",
    "awk -F',' '{print $3}' /home/wangchong/data/fwz/output/innovation_5/prepared/A_mci.csv | sort | uniq -c",
    # Baseline eval headers and SSIM
    "head -3 /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline_v2.csv",
    "awk -F',' 'NR>1{sum+=$2; n++} END{print \"baseline_v2 avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline_v2.csv",
    # Innovation 5 eval
    "head -3 /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5_v2.csv",
    "awk -F',' 'NR>1{sum+=$2; n++} END{print \"innovation_5_v2 avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5_v2.csv",
    # Innovation 5 raw eval
    "head -3 /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5_raw.csv",
    "awk -F',' 'NR>1{sum+=$2; n++} END{print \"innovation_5_raw avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5_raw.csv",
    # Baseline raw
    "head -3 /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline_raw.csv",
    "awk -F',' 'NR>1{sum+=$2; n++} END{print \"baseline_raw avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline_raw.csv",
    # Baseline  
    "head -3 /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline.csv",
    "awk -F',' 'NR>1{sum+=$2; n++} END{print \"baseline avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline.csv",
    # Innovation 5
    "head -3 /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5.csv",
    "awk -F',' 'NR>1{sum+=$2; n++} END{print \"innovation_5 avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5.csv",
    # Check ET-BoN experiment progress
    "tail -30 /home/wangchong/data/fwz/output/verification/et_bon/et_bon_experiment.log",
    # Check if process is still running
    "ps aux | grep run_et_bon | grep -v grep",
]

for cmd in cmds:
    print(f">>> {cmd[:90]}")
    _, o, e = ssh.exec_command(cmd)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    if out: print(out)
    if err: print(f"ERR: {err}")
    print()

ssh.close()
