"""
Wait for resumed experiment (pairs 27-49) to complete, then merge with first 27 pairs.
"""
import paramiko
import json
import time
import re

SERVER = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'
LOG_ORIG = '/home/wangchong/data/fwz/output/verification/fullscale_50/eval_pairs_0_26.log'
LOG_RESUME = '/home/wangchong/data/fwz/output/verification/fullscale_50/eval_resume.log'
JSON_PATH = '/home/wangchong/data/fwz/output/verification/fullscale_50/summary_verification_eval.json'
POLL_INTERVAL = 45

def check():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=10)
    
    # Check JSON
    _, o, _ = ssh.exec_command(f'cat {JSON_PATH} 2>/dev/null')
    j = o.read().decode().strip()
    if j:
        ssh.close()
        return True, json.loads(j), 50
    
    # Check if resume log has errors
    _, o, _ = ssh.exec_command(f'grep -cE "Error|Traceback" {LOG_RESUME} 2>/dev/null || echo 0')
    errs = int(o.read().decode().strip().split('\n')[0])
    
    # Count done pairs in resume log  
    _, o, _ = ssh.exec_command(f'grep -c "done (" {LOG_RESUME} 2>/dev/null || echo 0')
    n_resume = int(o.read().decode().strip())
    
    # Get wins from resume log
    _, o, _ = ssh.exec_command(
        f"grep done {LOG_RESUME} 2>/dev/null | grep -o 'Winner:.*' | sort | uniq -c"
    )
    resume_wins = o.read().decode().strip()
    
    # Also get original log wins (27 pairs)
    _, o, _ = ssh.exec_command(
        f"grep done {LOG_ORIG} 2>/dev/null | grep -o 'Winner:.*' | sort | uniq -c"
    )
    orig_wins = o.read().decode().strip()
    
    # Get SSIMs from both logs combined
    _, o, _ = ssh.exec_command(
        f"source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && "
        f"python3 -c \""
        f"import re; "
        f"lines1 = open('{LOG_ORIG}').readlines(); "
        f"lines2 = open('{LOG_RESUME}').readlines(); "
        f"lines = lines1 + lines2; "
        f"las = [float(re.search(r'SSIM=([0-9.]+)', l).group(1)) for l in lines if 'las SSIM' in l]; "
        f"bon = [float(re.search(r'SSIM=([0-9.]+)', l).group(1)) for l in lines if 'bon_w SSIM' in l]; "
        f"n = min(len(las), len(bon)); "
        f"print(f'{{n}} {{sum(las[:n])/n:.4f}} {{sum(bon[:n])/n:.4f}} {{sum(1 for a,b in zip(bon[:n],las[:n]) if a>b)}}')"
        f"\""
    )
    stats = o.read().decode().strip()
    
    # Last line
    _, o, _ = ssh.exec_command(f'tail -1 {LOG_RESUME}')
    last = o.read().decode().strip()
    
    ssh.close()
    total = 27 + n_resume
    return False, {
        'total': total, 'resume': n_resume, 'errs': errs,
        'orig_wins': orig_wins, 'resume_wins': resume_wins,
        'stats': stats, 'last': last
    }, total

print("Monitoring resumed experiment (pairs 27-49)...")
print("=" * 60)

while True:
    try:
        done, data, n = check()
        ts = time.strftime('%H:%M:%S')
        
        if done:
            print(f"\n[{ts}] EXPERIMENT COMPLETE!")
            print("=" * 60)
            config = data.get('config', {})
            print(f"Config: N={config.get('n_candidates')}, m={config.get('las_m')}")
            for method, stats in data.get('summary', {}).items():
                print(f"\n--- {method} ---")
                print(f"  SSIM:  {stats['overall_ssim']:.4f} +/- {stats['overall_ssim_std']:.4f}")
                print(f"  PSNR:  {stats['overall_psnr']:.2f} +/- {stats['overall_psnr_std']:.2f}")
                print(f"  MAE:   {stats['overall_mae']:.4f} +/- {stats['overall_mae_std']:.4f}")
                if 'roi_ssim' in stats:
                    print(f"  ROI SSIM: {stats['roi_ssim']:.4f}")
                if 'hippocampus_ssim' in stats:
                    print(f"  Hipp SSIM: {stats['hippocampus_ssim']:.4f}")
                print(f"  Time:  {stats['time_sec']:.1f}s/pair")
            with open(r'c:\Users\PC\Desktop\baselines\BrLP-main\new\29_fullscale_bon\fullscale_results.json', 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("\nSaved to fullscale_results.json")
            break
        else:
            d = data
            print(f"[{ts}] {d['total']}/50 (resume: {d['resume']}/23) | "
                  f"Combined stats: {d['stats']} | {d['last'][:70]}")
            if d['errs'] > 0:
                print(f"  WARNING: {d['errs']} errors found in log!")
        
        if n >= 50:
            break
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
    
    time.sleep(POLL_INTERVAL)

print("\nDone.")
