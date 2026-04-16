import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=30)

# GPU status
_, o, _ = ssh.exec_command('nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits')
print('GPU:', o.read().decode().strip())

# Process count
_, o, _ = ssh.exec_command('ps -eo etime,cmd | grep train_refinement | grep -v grep')
procs = o.read().decode().strip()
print('Processes:')
for l in procs.split('\n')[:3]:
    parts = l.strip().split()
    if parts:
        print(f'  elapsed={parts[0]}')

# Checkpoints
_, o, _ = ssh.exec_command('find /home/wangchong/data/fwz/output/36_refinement/ -name "*.pth" | sort')
ckpts = o.read().decode().strip()
print('Checkpoints:', ckpts)

# Log tail (use tail -c to get bytes not lines, avoids buffering issue)
for name in ['RefA', 'RefB', 'RefC']:
    _, o, _ = ssh.exec_command(f'tail -c 5000 /home/wangchong/data/fwz/output/36_refinement/{name}_train.log 2>/dev/null')
    raw = o.read().decode()
    lines = [l for l in raw.split('\n') if ('Epoch' in l or 'DONE' in l or 'loss=' in l) and l.strip()]
    print(f'\n{name}:')
    for l in lines[-5:]:
        print(f'  {l.strip()[:120]}')

# Eval progress
print('\n=== Eval ===')
_, o, _ = ssh.exec_command('ls /home/wangchong/data/fwz/output/36_refinement/eval/*.json 2>/dev/null')
jsons = o.read().decode().strip()
if jsons:
    for jf in jsons.split('\n'):
        print(f'  JSON: {jf.strip().split("/")[-1]}')
else:
    print('  No eval JSONs yet')

_, o, _ = ssh.exec_command('tail -c 3000 /home/wangchong/data/fwz/output/36_refinement/eval.log 2>/dev/null')
elog = o.read().decode().strip()
elines = [l for l in elog.split('\n') if l.strip() and ('Eval:' in l or 'SSIM' in l or 'Error' in l or 'error' in l or 'saved' in l or 'Sample' in l or 'Traceback' in l or 'AD-Composite' in l)]
if elines:
    print('  Log:')
    for l in elines[-8:]:
        print(f'    {l.strip()[:120]}')
else:
    _, o2, _ = ssh.exec_command('tail -5 /home/wangchong/data/fwz/output/36_refinement/eval.log 2>/dev/null')
    raw2 = o2.read().decode().strip()
    if raw2:
        print(f'  Log tail: {raw2[-300:]}')
    else:
        print('  No eval log yet')

ssh.close()
