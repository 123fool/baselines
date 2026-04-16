"""
Wait for experiment to complete, then collect results.
Polls every 60 seconds to avoid excessive SSH connections.
"""
import paramiko
import json
import time
import re

SERVER = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'
LOG_PATH = '/home/wangchong/data/fwz/output/verification/fullscale_50/eval.log'
JSON_PATH = '/home/wangchong/data/fwz/output/verification/fullscale_50/summary_verification_eval.json'
POLL_INTERVAL = 45  # seconds between checks (about 1 pair per check)

def check_progress():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=10)
    
    # Check if JSON exists (experiment complete)
    _, o, _ = ssh.exec_command(f'cat {JSON_PATH} 2>/dev/null')
    json_data = o.read().decode().strip()
    
    if json_data:
        ssh.close()
        return True, json.loads(json_data), 50
    
    # Get progress
    _, o, _ = ssh.exec_command(f'grep -c done {LOG_PATH} 2>/dev/null || echo 0')
    n_done = int(o.read().decode().strip())
    
    # Get win counts
    _, o, _ = ssh.exec_command(
        f"grep done {LOG_PATH} | grep -o 'Winner:.*' | sort | uniq -c"
    )
    wins = o.read().decode().strip()
    
    # Get last line
    _, o, _ = ssh.exec_command(f'tail -1 {LOG_PATH}')
    last = o.read().decode().strip()
    
    # Get avg SSIMs
    _, o, _ = ssh.exec_command(
        f"source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && "
        f"python3 -c \""
        f"import re; "
        f"lines = open('{LOG_PATH}').readlines(); "
        f"las = [float(re.search(r'SSIM=([0-9.]+)', l).group(1)) for l in lines if 'las SSIM' in l]; "
        f"bon = [float(re.search(r'SSIM=([0-9.]+)', l).group(1)) for l in lines if 'bon_w SSIM' in l]; "
        f"n = min(len(las), len(bon)); "
        f"print(f'{{sum(las[:n])/n:.4f}} {{sum(bon[:n])/n:.4f}} {{sum(1 for a,b in zip(bon[:n],las[:n]) if a>b)}}/{{n}}')"
        f"\""
    )
    stats = o.read().decode().strip()
    
    ssh.close()
    return False, {'n_done': n_done, 'wins': wins, 'last': last, 'stats': stats}, n_done

print("Waiting for experiment to complete...")
print("=" * 60)

while True:
    try:
        done, data, n = check_progress()
        ts = time.strftime('%H:%M:%S')
        
        if done:
            print(f"\n[{ts}] EXPERIMENT COMPLETE!")
            print("=" * 60)
            
            # Pretty print results
            config = data.get('config', {})
            print(f"Config: N={config.get('n_candidates')}, m={config.get('las_m')}, pairs={config.get('max_pairs')}")
            
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
            
            wins = data.get('wins', {})
            print(f"\nWins: bon={wins.get('bon_weighted_ssim')}, las={wins.get('las_ssim')}")
            
            ttest = data.get('paired_ttest', {})
            print(f"t-test: t={ttest.get('t_stat', 0):.4f}, p={ttest.get('p_value', 1):.6f}")
            
            # Save locally
            with open(r'c:\Users\PC\Desktop\baselines\BrLP-main\new\29_fullscale_bon\fullscale_results.json', 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("\nSaved to fullscale_results.json")
            break
        else:
            print(f"[{ts}] {data['n_done']}/50 | {data['wins'].strip()} | SSIM(las/bon): {data['stats']} | {data['last'][:60]}")
            
        if n >= 50:
            break
            
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
    
    time.sleep(POLL_INTERVAL)

print("\nDone.")
