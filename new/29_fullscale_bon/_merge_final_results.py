"""
Merge results from both log files (eval_pairs_0_26.log + eval_resume.log)
and compute full 50-pair statistics with paired t-test.
Run via SSH on server, or parse logs fetched locally.
"""
import subprocess, re, json, sys
from collections import defaultdict

SSH = "ssh -p 2638 wangchong@10.96.27.109"
LOG_DIR = "/home/wangchong/data/fwz/output/verification/fullscale_50"

def ssh_cmd(cmd):
    full = f'{SSH} "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && {cmd}"'
    r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=30)
    return r.stdout.strip()

def parse_log(text):
    """Extract per-pair metrics from log text."""
    pairs = {}
    # Pattern: Pair N: method SSIM=X MAE=Y
    for m in re.finditer(r'Pair\s+(\d+):\s+(las|bon_weighted)\s+SSIM=([\d.]+)\s+MAE=([\d.]+)', text):
        pid, method, ssim, mae = int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        if pid not in pairs:
            pairs[pid] = {}
        pairs[pid][method] = {'ssim': ssim, 'mae': mae}
    return pairs

def main():
    print("=== Fetching logs from server ===")
    
    # Fetch both logs
    log1 = ssh_cmd(f"cat {LOG_DIR}/eval_pairs_0_26.log 2>/dev/null || echo ''")
    log2 = ssh_cmd(f"cat {LOG_DIR}/eval_resume.log 2>/dev/null || echo ''")
    
    if not log1 and not log2:
        print("ERROR: No log files found!")
        sys.exit(1)
    
    # Check if experiment is complete
    total_lines_bon = len(re.findall(r'bon_weighted\s+SSIM=', log1 + '\n' + log2))
    total_lines_las = len(re.findall(r'las\s+SSIM=', log1 + '\n' + log2))
    print(f"Found {total_lines_bon} BoN results, {total_lines_las} LAS results")
    
    # Parse both logs
    pairs1 = parse_log(log1)
    pairs2 = parse_log(log2)
    
    # Merge (pairs2 overwrites pairs1 for any duplicates)
    all_pairs = {**pairs1, **pairs2}
    
    # Collect metrics
    bon_ssims, las_ssims = [], []
    bon_maes, las_maes = [], []
    bon_wins = 0
    
    complete_pairs = []
    for pid in sorted(all_pairs.keys()):
        p = all_pairs[pid]
        if 'las' in p and 'bon_weighted' in p:
            complete_pairs.append(pid)
            bon_ssims.append(p['bon_weighted']['ssim'])
            las_ssims.append(p['las']['ssim'])
            bon_maes.append(p['bon_weighted']['mae'])
            las_maes.append(p['las']['mae'])
            if p['bon_weighted']['ssim'] > p['las']['ssim']:
                bon_wins += 1
    
    n = len(complete_pairs)
    print(f"\n=== Final Results ({n} complete pairs) ===")
    
    if n == 0:
        print("No complete pairs found!")
        sys.exit(1)
    
    # Average metrics
    avg_bon_ssim = sum(bon_ssims) / n
    avg_las_ssim = sum(las_ssims) / n
    avg_bon_mae = sum(bon_maes) / n
    avg_las_mae = sum(las_maes) / n
    
    print(f"\nMethod       | SSIM (avg)  | MAE (avg)   | Wins")
    print(f"-------------|-------------|-------------|------")
    print(f"LAS (M=3)    | {avg_las_ssim:.6f}    | {avg_las_mae:.6f}    | {n - bon_wins}")
    print(f"BoN (N=8)    | {avg_bon_ssim:.6f}    | {avg_bon_mae:.6f}    | {bon_wins}")
    print(f"Diff (BoN-LAS)| {avg_bon_ssim - avg_las_ssim:+.6f}   | {avg_bon_mae - avg_las_mae:+.6f}   |")
    print(f"BoN win rate | {bon_wins}/{n} = {100*bon_wins/n:.1f}%")
    
    # Paired t-test (manual, no scipy needed)
    ssim_diffs = [b - l for b, l in zip(bon_ssims, las_ssims)]
    mae_diffs = [b - l for b, l in zip(bon_maes, las_maes)]
    
    def paired_ttest(diffs):
        n = len(diffs)
        if n < 2:
            return 0, 1.0
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d)**2 for d in diffs) / (n - 1)
        se = (var_d / n) ** 0.5
        if se == 0:
            return float('inf'), 0.0
        t_stat = mean_d / se
        # Approximate p-value using normal distribution for n>=30
        import math
        z = abs(t_stat)
        # Two-tailed p-value approximation
        p_approx = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        return t_stat, p_approx
    
    t_ssim, p_ssim = paired_ttest(ssim_diffs)
    t_mae, p_mae = paired_ttest(mae_diffs)
    
    print(f"\n=== Statistical Tests (paired t-test, two-tailed) ===")
    print(f"SSIM: t={t_ssim:.4f}, p={p_ssim:.6f} {'*' if p_ssim < 0.05 else 'ns'}")
    print(f"MAE:  t={t_mae:.4f}, p={p_mae:.6f} {'*' if p_mae < 0.05 else 'ns'}")
    
    # Per-pair detail table
    print(f"\n=== Per-Pair Details ===")
    print(f"Pair | LAS_SSIM  | BoN_SSIM  | LAS_MAE   | BoN_MAE   | Winner")
    print(f"-----|-----------|-----------|-----------|-----------|-------")
    for i, pid in enumerate(complete_pairs):
        p = all_pairs[pid]
        w = "BoN" if p['bon_weighted']['ssim'] > p['las']['ssim'] else "LAS"
        print(f"{pid:4d} | {p['las']['ssim']:.6f}  | {p['bon_weighted']['ssim']:.6f}  | {p['las']['mae']:.6f}  | {p['bon_weighted']['mae']:.6f}  | {w}")
    
    # Save JSON summary
    summary = {
        'n_pairs': n,
        'pair_ids': complete_pairs,
        'avg_bon_ssim': avg_bon_ssim,
        'avg_las_ssim': avg_las_ssim,
        'avg_bon_mae': avg_bon_mae,
        'avg_las_mae': avg_las_mae,
        'bon_wins': bon_wins,
        'las_wins': n - bon_wins,
        'bon_win_rate': bon_wins / n,
        'ssim_diff': avg_bon_ssim - avg_las_ssim,
        'mae_diff': avg_bon_mae - avg_las_mae,
        't_ssim': t_ssim,
        'p_ssim': p_ssim,
        't_mae': t_mae,
        'p_mae': p_mae,
        'per_pair': {str(pid): all_pairs[pid] for pid in complete_pairs}
    }
    
    out_path = r"c:\Users\PC\Desktop\baselines\BrLP-main\new\29_fullscale_bon\merged_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved merged results to {out_path}")
    
    return summary

if __name__ == '__main__':
    main()
