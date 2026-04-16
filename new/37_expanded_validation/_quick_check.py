import paramiko, json
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

for exp in ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']:
    _, out, _ = ssh.exec_command(
        f'cat /home/wangchong/data/fwz/output/37_expanded_validation/{exp}/training_log.json 2>/dev/null',
        timeout=10)
    data = out.read().decode().strip()
    if data:
        hist = json.loads(data)
        n = len(hist.get('train_loss', []))
        tl = [f'{x:.4f}' for x in hist['train_loss']]
        vl = [f'{x:.4f}' for x in hist['val_loss']]
        print(f'{exp}: {n} epochs')
        print(f'  train_loss: {tl}')
        print(f'  val_loss:   {vl}')
    else:
        print(f'{exp}: no training_log.json yet')

# Check processes
_, out, _ = ssh.exec_command('ps aux | grep train_refinement_v2 | grep -v grep', timeout=10)
procs = out.read().decode().strip()
print(f'\nTraining processes:\n{procs}')

# Check eval progress
_, out, _ = ssh.exec_command('ps aux | grep evaluate_refinement_v2 | grep -v grep', timeout=10)
eprocs = out.read().decode().strip()
print(f'\nEval processes:\n{eprocs}')

ssh.close()
