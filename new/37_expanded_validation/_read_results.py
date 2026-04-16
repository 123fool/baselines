import paramiko, json
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

eval_dir = '/home/wangchong/data/fwz/output/37_expanded_validation/eval'

for name in ['S36_RefC_H1a30_50subj', 'S35best_noref_50subj']:
    _, out, _ = ssh.exec_command(f'cat {eval_dir}/{name}.json 2>/dev/null', timeout=30)
    raw = out.read().decode().strip()
    if not raw:
        print(f'\n{name}: not yet available')
        continue
    data = json.loads(raw)
    s = data['summary']
    print(f'\n{name} (n={data["n_test"]}):')
    regions = ['overall', 'ad_composite', 'hippocampus', 'amygdala', 'thalamus',
               'lateral_ventricle', 'caudate', 'putamen', 'cerebral_cortex',
               'cerebral_wm', 'pallidum']
    for key in regions:
        if key in s:
            m = s[key]
            print(f'  {key:25s}: {m["mean"]:.4f} +/- {m["std"]:.4f}  '
                  f'95%CI=[{m["ci95_low"]:.4f}, {m["ci95_high"]:.4f}]')

ssh.close()
