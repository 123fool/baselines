"""Get actual SSIM averages from eval files (column 1 = overall_ssim)."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=10)

cmds = [
    # Correct awk - $1 is overall_ssim (first column)
    "awk -F',' 'NR>1{sum+=$1; n++} END{print \"baseline_v2 avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline_v2.csv",
    "awk -F',' 'NR>1{sum+=$1; n++} END{print \"innovation_5_v2 avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5_v2.csv",
    "awk -F',' 'NR>1{sum+=$1; n++} END{print \"baseline avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_baseline.csv",
    "awk -F',' 'NR>1{sum+=$1; n++} END{print \"innovation_5 avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/output/innovation_5/eval/eval_innovation_5.csv",
    # OASIS eval
    "head -3 /home/wangchong/data/fwz/oasis-eval-v2/eval_results.csv 2>/dev/null || echo 'file not found'",
    "awk -F',' 'NR>1{sum+=$1; n++} END{print \"oasis avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/oasis-eval-v2/eval_results.csv 2>/dev/null || echo 'N/A'",
    # ADNI eval 
    "head -3 /home/wangchong/data/fwz/adni-eval/run_20260404_123352/eval/eval_results.csv 2>/dev/null || echo 'file not found'",
    "awk -F',' 'NR>1{sum+=$1; n++} END{print \"adni avg SSIM:\", sum/n, \"n=\"n}' /home/wangchong/data/fwz/adni-eval/run_20260404_123352/eval/eval_results.csv 2>/dev/null || echo 'N/A'",
    # Check number of unique subjects in B_mci test split
    "awk -F',' '$3==\"test\"{print $1}' /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | sort -u | wc -l",
    # Our verification used which subjects?
    "head -3 /home/wangchong/data/fwz/output/verification/fullscale_50/comparison.csv 2>/dev/null || echo 'N/A'",
    # Check latest ET-BoN log
    "tail -15 /home/wangchong/data/fwz/output/verification/et_bon/et_bon_experiment.log",
]

for cmd in cmds:
    print(f">>> {cmd[:90]}")
    _, o, e = ssh.exec_command(cmd)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    if out: print(out)
    if err and 'deprecated' not in err: print(f"ERR: {err}")
    print()

ssh.close()
