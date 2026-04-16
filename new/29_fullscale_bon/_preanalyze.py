"""Pre-analyze results from both log files. Run locally."""
import re

# Paste log data inline for quick analysis
LOG1 = """
[15:43:11]   Pair 0: las SSIM=0.9393 MAE=0.0190 time=5.5s
[15:43:51]   Pair 0: bon_w SSIM=0.9477 MAE=0.0172 time=39.4s
[15:43:57]   Pair 1: las SSIM=0.9499 MAE=0.0188 time=5.5s
[15:44:37]   Pair 1: bon_w SSIM=0.9497 MAE=0.0190 time=39.3s
[15:44:43]   Pair 2: las SSIM=0.9470 MAE=0.0192 time=5.4s
[15:45:23]   Pair 2: bon_w SSIM=0.9426 MAE=0.0286 time=39.5s
[15:45:28]   Pair 3: las SSIM=0.9408 MAE=0.0236 time=5.3s
[15:46:08]   Pair 3: bon_w SSIM=0.9460 MAE=0.0231 time=39.4s
[15:46:14]   Pair 4: las SSIM=0.9515 MAE=0.0141 time=5.3s
[15:46:54]   Pair 4: bon_w SSIM=0.9497 MAE=0.0202 time=39.3s
[15:47:00]   Pair 5: las SSIM=0.9506 MAE=0.0152 time=5.3s
[15:47:39]   Pair 5: bon_w SSIM=0.9484 MAE=0.0192 time=39.5s
[15:47:45]   Pair 6: las SSIM=0.9340 MAE=0.0267 time=5.5s
[15:48:25]   Pair 6: bon_w SSIM=0.9459 MAE=0.0248 time=39.3s
[15:48:31]   Pair 7: las SSIM=0.9486 MAE=0.0169 time=5.3s
[15:49:10]   Pair 7: bon_w SSIM=0.9546 MAE=0.0147 time=39.2s
[15:49:16]   Pair 8: las SSIM=0.9376 MAE=0.0296 time=5.3s
[15:49:56]   Pair 8: bon_w SSIM=0.9384 MAE=0.0259 time=39.3s
[15:50:02]   Pair 9: las SSIM=0.9108 MAE=0.0355 time=5.3s
[15:50:41]   Pair 9: bon_w SSIM=0.9263 MAE=0.0298 time=39.2s
[15:50:47]   Pair 10: las SSIM=0.8928 MAE=0.0327 time=5.4s
[15:51:27]   Pair 10: bon_w SSIM=0.8860 MAE=0.0359 time=38.9s
[15:51:32]   Pair 11: las SSIM=0.9099 MAE=0.0286 time=5.2s
[15:52:12]   Pair 11: bon_w SSIM=0.9163 MAE=0.0265 time=39.3s
[15:52:18]   Pair 12: las SSIM=0.9347 MAE=0.0263 time=5.4s
[15:52:58]   Pair 12: bon_w SSIM=0.9375 MAE=0.0263 time=39.4s
[15:53:03]   Pair 13: las SSIM=0.8991 MAE=0.0361 time=5.3s
[15:53:43]   Pair 13: bon_w SSIM=0.9037 MAE=0.0331 time=39.4s
[15:53:49]   Pair 14: las SSIM=0.9221 MAE=0.0326 time=5.5s
[15:54:29]   Pair 14: bon_w SSIM=0.9171 MAE=0.0374 time=39.5s
[15:54:35]   Pair 15: las SSIM=0.9240 MAE=0.0225 time=5.3s
[15:55:15]   Pair 15: bon_w SSIM=0.9234 MAE=0.0218 time=39.3s
[15:55:20]   Pair 16: las SSIM=0.9560 MAE=0.0160 time=5.1s
[15:55:59]   Pair 16: bon_w SSIM=0.9551 MAE=0.0190 time=38.6s
[15:56:05]   Pair 17: las SSIM=0.9452 MAE=0.0213 time=5.2s
[15:56:44]   Pair 17: bon_w SSIM=0.9402 MAE=0.0300 time=38.3s
[15:56:50]   Pair 18: las SSIM=0.9373 MAE=0.0356 time=5.3s
[15:57:28]   Pair 18: bon_w SSIM=0.9338 MAE=0.0335 time=38.4s
[15:57:34]   Pair 19: las SSIM=0.9455 MAE=0.0240 time=5.2s
[15:58:13]   Pair 19: bon_w SSIM=0.9451 MAE=0.0225 time=38.3s
[15:58:19]   Pair 20: las SSIM=0.9308 MAE=0.0273 time=5.2s
[15:58:57]   Pair 20: bon_w SSIM=0.9327 MAE=0.0253 time=38.5s
[15:59:03]   Pair 21: las SSIM=0.9430 MAE=0.0205 time=5.2s
[15:59:42]   Pair 21: bon_w SSIM=0.9482 MAE=0.0195 time=38.4s
[15:59:48]   Pair 22: las SSIM=0.9454 MAE=0.0251 time=5.3s
[16:00:26]   Pair 22: bon_w SSIM=0.9305 MAE=0.0318 time=38.4s
[16:00:32]   Pair 23: las SSIM=0.9369 MAE=0.0267 time=5.2s
[16:01:11]   Pair 23: bon_w SSIM=0.9473 MAE=0.0203 time=38.5s
[16:01:17]   Pair 24: las SSIM=0.9315 MAE=0.0251 time=5.2s
[16:01:56]   Pair 24: bon_w SSIM=0.9336 MAE=0.0240 time=38.6s
[16:02:01]   Pair 25: las SSIM=0.9024 MAE=0.0418 time=5.2s
[16:02:40]   Pair 25: bon_w SSIM=0.9113 MAE=0.0377 time=38.4s
[16:02:46]   Pair 26: las SSIM=0.9215 MAE=0.0375 time=5.4s
[16:03:25]   Pair 26: bon_w SSIM=0.9303 MAE=0.0319 time=38.5s
"""

# Resume log data (pairs 27-49, COMPLETE)
LOG2 = """
[16:07:15]   Pair 27: las SSIM=0.9136 MAE=0.0339 time=5.7s
[16:07:55]   Pair 27: bon_w SSIM=0.9100 MAE=0.0362 time=40.2s
[16:08:01]   Pair 28: las SSIM=0.9463 MAE=0.0185 time=5.4s
[16:08:42]   Pair 28: bon_w SSIM=0.9507 MAE=0.0154 time=40.1s
[16:08:48]   Pair 29: las SSIM=0.9285 MAE=0.0220 time=5.4s
[16:09:29]   Pair 29: bon_w SSIM=0.9365 MAE=0.0188 time=40.1s
[16:09:35]   Pair 30: las SSIM=0.9406 MAE=0.0215 time=5.4s
[16:10:16]   Pair 30: bon_w SSIM=0.9431 MAE=0.0186 time=41.1s
[16:10:22]   Pair 31: las SSIM=0.9381 MAE=0.0276 time=5.4s
[16:11:04]   Pair 31: bon_w SSIM=0.9326 MAE=0.0325 time=40.9s
[16:11:10]   Pair 32: las SSIM=0.9038 MAE=0.0311 time=5.6s
[16:11:53]   Pair 32: bon_w SSIM=0.8958 MAE=0.0376 time=42.7s
[16:12:00]   Pair 33: las SSIM=0.9529 MAE=0.0224 time=5.8s
[16:12:43]   Pair 33: bon_w SSIM=0.9523 MAE=0.0189 time=42.7s
[16:12:49]   Pair 34: las SSIM=0.9489 MAE=0.0200 time=5.7s
[16:13:32]   Pair 34: bon_w SSIM=0.9339 MAE=0.0359 time=42.9s
[16:13:39]   Pair 35: las SSIM=0.9179 MAE=0.0403 time=5.7s
[16:14:22]   Pair 35: bon_w SSIM=0.9010 MAE=0.0409 time=42.9s
[16:14:29]   Pair 36: las SSIM=0.9004 MAE=0.0361 time=5.7s
[16:15:12]   Pair 36: bon_w SSIM=0.8807 MAE=0.0500 time=42.7s
[16:15:19]   Pair 37: las SSIM=0.9509 MAE=0.0185 time=5.7s
[16:16:02]   Pair 37: bon_w SSIM=0.9540 MAE=0.0176 time=42.6s
[16:16:08]   Pair 38: las SSIM=0.9442 MAE=0.0188 time=5.5s
[16:16:56]   Pair 38: bon_w SSIM=0.9474 MAE=0.0216 time=47.0s
[16:17:03]   Pair 39: las SSIM=0.9219 MAE=0.0359 time=6.3s
[16:17:46]   Pair 39: bon_w SSIM=0.9234 MAE=0.0375 time=42.1s
[16:17:52]   Pair 40: las SSIM=0.9240 MAE=0.0264 time=5.9s
[16:18:36]   Pair 40: bon_w SSIM=0.9066 MAE=0.0348 time=43.0s
[16:18:42]   Pair 41: las SSIM=0.9138 MAE=0.0369 time=5.9s
[16:19:26]   Pair 41: bon_w SSIM=0.8830 MAE=0.0540 time=42.9s
[16:19:32]   Pair 42: las SSIM=0.8479 MAE=0.0556 time=5.4s
[16:20:15]   Pair 42: bon_w SSIM=0.8583 MAE=0.0535 time=42.8s
[16:20:22]   Pair 43: las SSIM=0.9126 MAE=0.0489 time=6.0s
[16:21:05]   Pair 43: bon_w SSIM=0.9347 MAE=0.0354 time=42.8s
[16:21:12]   Pair 44: las SSIM=0.9238 MAE=0.0348 time=5.9s
[16:21:56]   Pair 44: bon_w SSIM=0.9291 MAE=0.0341 time=43.9s
[16:22:03]   Pair 45: las SSIM=0.9219 MAE=0.0380 time=5.9s
[16:22:46]   Pair 45: bon_w SSIM=0.9338 MAE=0.0364 time=43.2s
[16:22:53]   Pair 46: las SSIM=0.9583 MAE=0.0193 time=5.6s
[16:23:37]   Pair 46: bon_w SSIM=0.9463 MAE=0.0314 time=43.7s
[16:23:43]   Pair 47: las SSIM=0.9299 MAE=0.0420 time=6.1s
[16:24:28]   Pair 47: bon_w SSIM=0.9469 MAE=0.0265 time=44.6s
[16:24:35]   Pair 48: las SSIM=0.9513 MAE=0.0165 time=6.1s
[16:25:19]   Pair 48: bon_w SSIM=0.9540 MAE=0.0185 time=43.8s
[16:25:26]   Pair 49: las SSIM=0.9358 MAE=0.0262 time=5.7s
[16:26:10]   Pair 49: bon_w SSIM=0.9221 MAE=0.0332 time=43.5s
"""

def parse(text):
    pairs = {}
    for m in re.finditer(r'Pair\s+(\d+):\s+(las|bon_w)\s+SSIM=([\d.]+)\s+MAE=([\d.]+)', text):
        pid, method, ssim, mae = int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        method = 'bon_weighted' if method == 'bon_w' else method
        if pid not in pairs:
            pairs[pid] = {}
        pairs[pid][method] = {'ssim': ssim, 'mae': mae}
    return pairs

p1 = parse(LOG1)
p2 = parse(LOG2)
all_p = {**p1, **p2}

bon_ssims, las_ssims, bon_maes, las_maes = [], [], [], []
bon_wins = 0
for pid in sorted(all_p.keys()):
    p = all_p[pid]
    if 'las' in p and 'bon_weighted' in p:
        bon_ssims.append(p['bon_weighted']['ssim'])
        las_ssims.append(p['las']['ssim'])
        bon_maes.append(p['bon_weighted']['mae'])
        las_maes.append(p['las']['mae'])
        if p['bon_weighted']['ssim'] > p['las']['ssim']:
            bon_wins += 1

n = len(bon_ssims)
print(f"Complete pairs so far: {n}/50")
print(f"LAS avg SSIM: {sum(las_ssims)/n:.6f}  avg MAE: {sum(las_maes)/n:.6f}")
print(f"BoN avg SSIM: {sum(bon_ssims)/n:.6f}  avg MAE: {sum(bon_maes)/n:.6f}")
print(f"BoN wins: {bon_wins}/{n} ({100*bon_wins/n:.1f}%)")
print(f"SSIM diff: {sum(bon_ssims)/n - sum(las_ssims)/n:+.6f}")
print(f"MAE diff:  {sum(bon_maes)/n - sum(las_maes)/n:+.6f}")

# Paired t-test
import math
ssim_diffs = [all_p[pid]['bon_weighted']['ssim'] - all_p[pid]['las']['ssim'] 
              for pid in sorted(all_p.keys()) if 'las' in all_p[pid] and 'bon_weighted' in all_p[pid]]
mae_diffs = [all_p[pid]['bon_weighted']['mae'] - all_p[pid]['las']['mae']
             for pid in sorted(all_p.keys()) if 'las' in all_p[pid] and 'bon_weighted' in all_p[pid]]

def paired_ttest(diffs):
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d)**2 for d in diffs) / (n - 1)
    se = (var_d / n) ** 0.5
    if se == 0: return 0, 1.0
    t_stat = mean_d / se
    z = abs(t_stat)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return t_stat, p

t_ssim, p_ssim = paired_ttest(ssim_diffs)
t_mae, p_mae = paired_ttest(mae_diffs)

print(f"\n=== Paired t-test (two-tailed) ===")
print(f"SSIM: t={t_ssim:.4f}, p={p_ssim:.4f} {'*sig*' if p_ssim < 0.05 else '(not significant)'}")
print(f"MAE:  t={t_mae:.4f}, p={p_mae:.4f} {'*sig*' if p_mae < 0.05 else '(not significant)'}")

# Win analysis by first/second half
first_wins = sum(1 for pid in range(0, 27) if pid in all_p and 'bon_weighted' in all_p[pid] 
                 and all_p[pid]['bon_weighted']['ssim'] > all_p[pid]['las']['ssim'])
second_wins = sum(1 for pid in range(27, 50) if pid in all_p and 'bon_weighted' in all_p[pid]
                  and all_p[pid]['bon_weighted']['ssim'] > all_p[pid]['las']['ssim'])
print(f"\nBoN wins by segment:")
print(f"  First 27 (pairs 0-26): {first_wins}/27 ({100*first_wins/27:.0f}%)")
print(f"  Last 23 (pairs 27-49): {second_wins}/23 ({100*second_wins/23:.0f}%)")

# Absolute SSIM diff distribution
abs_diffs = [abs(d) for d in ssim_diffs]
print(f"\nSSIM |diff| distribution:")
print(f"  Mean: {sum(abs_diffs)/n:.4f}")
print(f"  Max BoN advantage: {max(ssim_diffs):.4f}")
print(f"  Max LAS advantage: {min(ssim_diffs):.4f}")
print(f"  |diff| < 0.005: {sum(1 for d in abs_diffs if d < 0.005)}/50 ({sum(1 for d in abs_diffs if d < 0.005)*2}%)")
print(f"  |diff| < 0.01: {sum(1 for d in abs_diffs if d < 0.01)}/50 ({sum(1 for d in abs_diffs if d < 0.01)*2}%)")

# Show per-pair detail
print(f"\nPair | LAS_SSIM | BoN_SSIM | Diff    | Winner")
for pid in sorted(all_p.keys()):
    p = all_p[pid]
    if 'las' in p and 'bon_weighted' in p:
        d = p['bon_weighted']['ssim'] - p['las']['ssim']
        w = "BoN" if d > 0 else "LAS"
        print(f"  {pid:2d} | {p['las']['ssim']:.4f}   | {p['bon_weighted']['ssim']:.4f}   | {d:+.4f} | {w}")
