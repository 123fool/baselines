#!/usr/bin/env python3
"""拉取所有评估结果的详细指标"""
import paramiko, json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=15)

EVAL = '/home/wangchong/data/fwz/output/37_expanded_validation/eval'

# 获取所有结果JSON
_, out, _ = ssh.exec_command(f'ls {EVAL}/*.json 2>/dev/null | grep -v progress', timeout=10)
files = [f for f in out.read().decode().strip().split('\n') if f]

print("=" * 80)
print("所有评估结果汇总")
print("=" * 80)

for fpath in sorted(files):
    _, out, _ = ssh.exec_command(f'cat {fpath}', timeout=10)
    data = out.read().decode().strip()
    if not data:
        continue
    try:
        r = json.loads(data)
    except:
        continue
    
    label = r.get('label', fpath.split('/')[-1])
    results = r.get('results', {})
    
    print(f"\n{'='*60}")
    print(f"模型: {label}")
    print(f"ref_ckpt: {r.get('ref_ckpt', 'None')}")
    print(f"{'='*60}")
    
    # 关键指标
    keys = ['overall_mean_dice', 'AD_composite_mean', 
            'hippocampus', 'amygdala', 'thalamus', 
            'lateral_ventricle', 'caudate', 'putamen',
            'cerebral_cortex', 'cerebral_wm', 'pallidum']
    
    for k in keys:
        val = results.get(k)
        std_key = k + '_std' if k != 'overall_mean_dice' and k != 'AD_composite_mean' else None
        ci_key = k + '_ci95'
        
        if val is not None:
            std = results.get(f'{k}_std', '')
            ci = results.get(f'{k}_ci95', '')
            std_str = f" ± {std:.4f}" if isinstance(std, float) else ""
            ci_str = f"  CI={ci}" if ci else ""
            print(f"  {k:25s}: {val:.4f}{std_str}{ci_str}")

# 训练历史
print("\n\n" + "=" * 80)
print("训练完成状态")
print("=" * 80)
OUT = '/home/wangchong/data/fwz/output/37_expanded_validation'
for exp in ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']:
    _, out, _ = ssh.exec_command(f'cat {OUT}/{exp}/training_log.json 2>/dev/null', timeout=10)
    data = out.read().decode().strip()
    if data:
        h = json.loads(data)
        n = len(h.get('train_loss', []))
        vl = h.get('val_loss', [])
        print(f"\n{exp}: {n} epochs 完成")
        print(f"  最终 val_loss: {vl[-1]:.4f}")
        print(f"  最佳 val_loss: {min(vl):.4f}")
        best_ep = vl.index(min(vl))
        print(f"  最佳 epoch: {best_ep}")
        # 检查early stopping
        if h.get('early_stopped'):
            print(f"  ⚠ Early stopped at epoch {n}")

# 检查最新checkpoint信息
print("\n\n" + "=" * 80)
print("Checkpoint 状态")
print("=" * 80)
for exp in ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']:
    _, out, _ = ssh.exec_command(f'ls -la {OUT}/{exp}/*.pth 2>/dev/null', timeout=10)
    print(f"\n{exp}:")
    print(out.read().decode().strip())

ssh.close()
