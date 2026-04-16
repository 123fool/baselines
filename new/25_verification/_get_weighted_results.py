import paramiko, json
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Get weighted_compare summary
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/weighted_compare/summary_verification_eval.json 2>/dev/null", timeout=10)
raw = stdout.read().decode().strip()
if raw:
    data = json.loads(raw)
    print("=== weighted_compare SUMMARY ===")
    for method, stats in data['summary'].items():
        ssim = stats['overall_ssim']
        ssim_std = stats['overall_ssim_std']
        psnr = stats['overall_psnr']
        mae = stats['overall_mae']
        roi_ssim = stats['roi_ssim']
        hipp_ssim = stats['hippocampus_ssim']
        time_s = stats['time_sec']
        print(f"  {method:12s}: SSIM={ssim:.4f}±{ssim_std:.4f} PSNR={psnr:.2f} MAE={mae:.4f} ROI_SSIM={roi_ssim:.4f} Hipp_SSIM={hipp_ssim:.4f} Time={time_s:.1f}s")

# Get full eval log
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/weighted_compare/eval_verification.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
# Extract per-pair results
for line in log.split('\n'):
    if 'Pair' in line and 'SSIM' in line:
        print(line.strip())

c.close()
