import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Get roundtrip summary JSON
stdin, stdout, stderr = ssh.exec_command('cat /home/wangchong/data/fwz/output/verification/roundtrip_test/summary_verification_eval.json')
rt = json.loads(stdout.read().decode())

print("=== RT SUMMARY ===")
for method, data in rt['summary'].items():
    ssim = data['overall_ssim']
    ssim_std = data['overall_ssim_std']
    psnr = data['overall_psnr']
    mae = data['overall_mae']
    roi_ssim = data['roi_ssim']
    hipp_ssim = data['hippocampus_ssim']
    t = data['time_sec']
    print(f"  {method:15s}: SSIM={ssim:.4f}+/-{ssim_std:.4f}  PSNR={psnr:.2f}  MAE={mae:.4f}  ROI_SSIM={roi_ssim:.4f}  Hipp_SSIM={hipp_ssim:.4f}  Time={t:.1f}s")

print("\n=== RT PER-PAIR ===")
for p in rt['per_pair']:
    subj = p['subject_id']
    for method in ['las', 'bon_weighted', 'roundtrip_bon']:
        if method in p:
            d = p[method]
            ssim = d['overall_ssim']
            psnr = d['overall_psnr']
            mae = d['overall_mae']
            roi = d['roi_ssim']
            hipp = d['hippocampus_ssim']
            t = d['time_sec']
            print(f"  {subj} | {method:15s} | SSIM={ssim:.4f} PSNR={psnr:.2f} MAE={mae:.4f} ROI={roi:.4f} Hipp={hipp:.4f} T={t:.1f}s")
    print()

# Also get N8 per-pair summary
stdin, stdout, stderr = ssh.exec_command('cat /home/wangchong/data/fwz/output/verification/bon_n8_full/summary_verification_eval.json')
n8 = json.loads(stdout.read().decode())

print("=== N8 PER-PAIR ===")
methods = ['las', 'single', 'bon_best1', 'bon_topk', 'bon_weighted']
for p in n8['per_pair']:
    subj = p['subject_id']
    for method in methods:
        if method in p:
            d = p[method]
            ssim = d['overall_ssim']
            roi = d['roi_ssim']
            hipp = d['hippocampus_ssim']
            print(f"  {subj} | {method:12s} | SSIM={ssim:.4f} ROI={roi:.4f} Hipp={hipp:.4f}")
    print()

# Count wins
print("\n=== N8 WINS (SSIM) ===")
for pi, p in enumerate(n8['per_pair']):
    best_method = None
    best_ssim = -1
    for method in methods:
        if method in p:
            s = p[method]['overall_ssim']
            if s > best_ssim:
                best_ssim = s
                best_method = method
    print(f"  Pair {pi}: {best_method} ({best_ssim:.4f})")

print("\n=== N8 WINS (ROI SSIM) ===")
for pi, p in enumerate(n8['per_pair']):
    best_method = None
    best_ssim = -1
    for method in methods:
        if method in p:
            s = p[method]['roi_ssim']
            if s > best_ssim:
                best_ssim = s
                best_method = method
    print(f"  Pair {pi}: {best_method} ({best_ssim:.4f})")

ssh.close()
