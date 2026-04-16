import paramiko, json
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=30)

names = ['RefA_H1a30', 'RefB_H1a30', 'RefC_H1a30', 'RefB_BTR', 'RefB_H1a30_LAS5', 'S35best_H1a30_noref']
keys = ['overall_mean','ad_composite_mean','hippocampus_mean','amygdala_mean','thalamus_mean',
        'lateral_ventricle_mean','caudate_mean','putamen_mean','cerebral_cortex_mean','cerebral_wm_mean','pallidum_mean']

results = {}
for name in names:
    _, o, _ = ssh.exec_command(f'cat /home/wangchong/data/fwz/output/36_refinement/eval/{name}.json')
    raw = o.read().decode().strip()
    if raw:
        try:
            results[name] = json.loads(raw)
        except:
            pass

ssh.close()

# Print comparison table
print(f"{'Region':<22}", end='')
for name in results:
    print(f" {name:>18}", end='')
print()
print("-" * (22 + 19 * len(results)))

for key in keys:
    label = key.replace('_mean', '')
    print(f"{label:<22}", end='')
    for name in results:
        val = results[name].get(key, 0)
        print(f" {val:>18.4f}", end='')
    print()

# Also dump raw JSON for each
print("\n\n=== Raw JSON Summary ===")
for name, data in results.items():
    print(f"\n{name}:")
    for key in keys:
        val = data.get(key, 0)
        std_key = key.replace('_mean', '_std')
        std = data.get(std_key, 0)
        label = key.replace('_mean', '')
        print(f"  {label:<22}: {val:.4f} +/- {std:.4f}")
