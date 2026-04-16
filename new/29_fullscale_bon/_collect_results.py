"""
Collect fullscale experiment results from server.
Reads the JSON summary and formats for analysis.md.
"""
import paramiko
import json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Check if done
stdin, stdout, stderr = ssh.exec_command(
    "grep -c 'done (' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>/dev/null || echo 0"
)
n_done = int(stdout.read().decode().strip())
print(f"Pairs done: {n_done}/50")

# Check if summary JSON exists
stdin, stdout, stderr = ssh.exec_command(
    "ls -la /home/wangchong/data/fwz/output/verification/fullscale_50/summary_verification_eval.json 2>/dev/null"
)
json_info = stdout.read().decode().strip()

if not json_info:
    # Check for alternative name
    stdin, stdout, stderr = ssh.exec_command(
        "ls /home/wangchong/data/fwz/output/verification/fullscale_50/*.json 2>/dev/null || echo 'no json'"
    )
    json_files = stdout.read().decode().strip()
    print(f"JSON files: {json_files}")

    if n_done < 50:
        # Show current wins
        stdin, stdout, stderr = ssh.exec_command(
            "grep 'Winner:' /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log | "
            "sort | uniq -c | sort -rn"
        )
        print(f"Win counts:\n{stdout.read().decode()}")

        # Show last lines
        stdin, stdout, stderr = ssh.exec_command(
            "tail -5 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log"
        )
        print(f"Last log:\n{stdout.read().decode()}")

        # Average SSIMs so far
        stdin, stdout, stderr = ssh.exec_command(
            "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && "
            "python3 -c \""
            "import re; "
            "lines = open('/home/wangchong/data/fwz/output/verification/fullscale_50/eval.log').readlines(); "
            "las_ssims = [float(re.search(r'SSIM=([0-9.]+)', l).group(1)) for l in lines if 'las SSIM' in l]; "
            "bon_ssims = [float(re.search(r'SSIM=([0-9.]+)', l).group(1)) for l in lines if 'bon_w SSIM' in l]; "
            "n = min(len(las_ssims), len(bon_ssims)); "
            "print(f'Pairs: {n}'); "
            "print(f'LAS  avg SSIM: {sum(las_ssims[:n])/n:.4f}'); "
            "print(f'BoN  avg SSIM: {sum(bon_ssims[:n])/n:.4f}'); "
            "print(f'Diff: {(sum(bon_ssims[:n])-sum(las_ssims[:n]))/n:.4f}'); "
            "print(f'BoN wins: {sum(1 for a,b in zip(bon_ssims[:n],las_ssims[:n]) if a>b)}/{n}')"
            "\""
        )
        print(f"\nInterim stats:\n{stdout.read().decode()}")
        print("Experiment still running. Re-run this script later.")
    ssh.close()
    exit(0)

# Read JSON summary
print(f"\nJSON found: {json_info}")
stdin, stdout, stderr = ssh.exec_command(
    "cat /home/wangchong/data/fwz/output/verification/fullscale_50/summary_verification_eval.json"
)
summary_raw = stdout.read().decode()
summary = json.loads(summary_raw)

# Pretty print results
print("\n" + "=" * 60)
print("FULLSCALE EXPERIMENT RESULTS")
print("=" * 60)

config = summary.get('config', {})
print(f"\nConfig: N_candidates={config.get('n_candidates')}, LAS_m={config.get('las_m')}, "
      f"Pairs={config.get('max_pairs')}, Scale={config.get('scale_factor')}")

for method, stats in summary.get('summary', {}).items():
    print(f"\n--- {method} ---")
    print(f"  SSIM:  {stats['overall_ssim']:.4f} ± {stats['overall_ssim_std']:.4f}")
    print(f"  PSNR:  {stats['overall_psnr']:.2f} ± {stats['overall_psnr_std']:.2f}")
    print(f"  MAE:   {stats['overall_mae']:.4f} ± {stats['overall_mae_std']:.4f}")
    if 'roi_ssim' in stats:
        print(f"  ROI SSIM: {stats['roi_ssim']:.4f}")
    if 'hippocampus_ssim' in stats:
        print(f"  Hipp SSIM: {stats['hippocampus_ssim']:.4f}")
    print(f"  Time:  {stats['time_sec']:.1f}s/pair")

# Wins
wins = summary.get('wins', {})
print(f"\nWins (SSIM): bon_weighted={wins.get('bon_weighted_ssim')}, las={wins.get('las_ssim')}")

# Statistical test
ttest = summary.get('paired_ttest', {})
print(f"Paired t-test: t={ttest.get('t_stat', 'N/A'):.4f}, p={ttest.get('p_value', 'N/A'):.6f}")
sig = ttest.get('significant', False)
print(f"Statistical significance: {'YES (p<0.05)' if sig else 'NO (p>=0.05)'}")

# Save locally
with open(r'c:\Users\PC\Desktop\baselines\BrLP-main\new\29_fullscale_bon\fullscale_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved locally to fullscale_results.json")

ssh.close()
