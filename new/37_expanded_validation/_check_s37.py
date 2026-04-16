"""
Section 37: Monitoring script for training and evaluation progress.
Checks GPU usage, training logs, checkpoints, and eval progress files.
"""
import paramiko, json, time, sys, os

HOST = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'
OUTPUT_DIR = '/home/wangchong/data/fwz/output/37_expanded_validation'
EVAL_DIR = f'{OUTPUT_DIR}/eval'

EXPS = ['RefC_v2_cont', 'RefC_v2_fresh', 'RefD_v2_highnoise']
EVAL_CONFIGS = ['S36_RefC_H1a30_50subj', 'S35best_noref_50subj',
                'RefC_v2_cont_50subj', 'RefC_v2_fresh_50subj', 'RefD_v2_highnoise_50subj']


def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, timeout=30)
    return ssh


def run(ssh, cmd, timeout=30):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode().strip()


def check_training(ssh):
    print('\n' + '='*60)
    print('TRAINING STATUS')
    print('='*60)

    # GPU usage
    gpu = run(ssh, 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits')
    print(f'\nGPU Status:\n{gpu}')

    # Check each experiment
    for exp in EXPS:
        print(f'\n--- {exp} ---')
        # Check training log
        log = run(ssh, f'tail -3 {OUTPUT_DIR}/{exp}_train.log 2>/dev/null || echo "no log"')
        print(f'  Log: {log}')

        # Check training_log.json for history
        json_str = run(ssh, f'cat {OUTPUT_DIR}/{exp}/training_log.json 2>/dev/null || echo "none"')
        if json_str != 'none':
            try:
                hist = json.loads(json_str)
                n_ep = len(hist.get('train_loss', []))
                if n_ep > 0:
                    tl = hist['train_loss'][-1]
                    vl = hist['val_loss'][-1]
                    print(f'  Epochs: {n_ep}, train_loss={tl:.4f}, val_loss={vl:.4f}')
                    if n_ep >= 2:
                        print(f'  History: train=[{",".join(f"{x:.4f}" for x in hist["train_loss"])}]')
                        print(f'           valid=[{",".join(f"{x:.4f}" for x in hist["val_loss"])}]')
            except:
                pass

        # Check checkpoints
        ckpts = run(ssh, f'ls -la {OUTPUT_DIR}/{exp}/refnet-*.pth 2>/dev/null | wc -l')
        best = run(ssh, f'ls -la {OUTPUT_DIR}/{exp}/refnet-*-best.pth 2>/dev/null || echo "no best yet"')
        print(f'  Checkpoints: {ckpts}, Best: {best}')


def check_evaluation(ssh):
    print('\n' + '='*60)
    print('EVALUATION STATUS')
    print('='*60)

    for cfg in EVAL_CONFIGS:
        # Check progress file
        prog_str = run(ssh, f'cat {EVAL_DIR}/{cfg}_progress.json 2>/dev/null || echo "none"')
        json_path = f'{EVAL_DIR}/{cfg}.json'
        json_exists = run(ssh, f'test -f {json_path} && echo "yes" || echo "no"')

        if json_exists == 'yes':
            # Completed - read summary
            result = run(ssh, f'cat {json_path} 2>/dev/null')
            try:
                data = json.loads(result)
                s = data.get('summary', {})
                overall = s.get('overall', {}).get('mean', '?')
                ad_comp = s.get('ad_composite', {}).get('mean', '?')
                hippo = s.get('hippocampus', {}).get('mean', '?')
                n = data.get('n_test', '?')
                print(f'\n{cfg}: ✅ DONE (n={n})')
                print(f'  Overall={overall:.4f}, AD-Comp={ad_comp:.4f}, Hippo={hippo:.4f}')
            except:
                print(f'\n{cfg}: ✅ JSON exists but parse error')
        elif prog_str != 'none':
            try:
                prog = json.loads(prog_str)
                done = prog.get('completed', 0)
                total = prog.get('total', '?')
                eta = prog.get('eta_minutes', '?')
                print(f'\n{cfg}: 🔄 {done}/{total} (ETA: {eta}min)')
            except:
                print(f'\n{cfg}: 🔄 In progress (cannot parse)')
        else:
            print(f'\n{cfg}: ⏳ Not started')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    ssh = get_ssh()

    if mode in ('train', 'all'):
        check_training(ssh)
    if mode in ('eval', 'all'):
        check_evaluation(ssh)

    ssh.close()
