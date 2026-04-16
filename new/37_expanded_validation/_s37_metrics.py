#!/usr/bin/env python3
"""
Extract all eval summaries from server, display, and optionally launch V2 evals.
"""
import paramiko, json, sys, time, io

HOST, PORT, USER, PASS = '10.96.27.109', 2638, 'wangchong', '123456'
EVAL_DIR  = '/home/wangchong/data/fwz/output/37_expanded_validation/eval'
OUTPUT_DIR= '/home/wangchong/data/fwz/output/37_expanded_validation'
CSV       = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
AE_CKPT   = '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'
DIFF_CKPT = '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'
CNET_CKPT = '/home/wangchong/data/fwz/output/34_hippo_training/H1_a30/cnet-hippo-H1_a30-ep2.pth'
CODE_DIR  = '/home/wangchong/data/fwz/code/37_expanded_validation/scripts'
PYTHON    = '/home/wangchong/miniconda3/envs/fwz/bin/python'

REGIONS = ['overall', 'ad_composite', 'hippocampus', 'amygdala', 'thalamus',
           'lateral_ventricle', 'caudate', 'putamen', 'cerebral_cortex', 'cerebral_wm', 'pallidum']

def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, timeout=15)
    return ssh

def run_cmd(ssh, cmd, wait=True):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
    if wait:
        return stdout.read().decode().strip()
    time.sleep(1)
    return ''

def read_json_via_sftp(ssh, filepath):
    """Read JSON file via SFTP"""
    sftp = ssh.open_sftp()
    try:
        with sftp.open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None
    finally:
        sftp.close()

def extract_summaries(ssh):
    """Read all eval JSONs via SFTP and print summary table"""
    out = run_cmd(ssh, f'ls {EVAL_DIR}/*.json 2>/dev/null | grep -v progress')
    if not out:
        print("No evaluation JSON files found!")
        return {}
    
    json_files = [f.strip() for f in out.split('\n') if f.strip()]
    print(f"Found {len(json_files)} evaluation result files\n")
    
    all_data = {}
    sftp = ssh.open_sftp()
    
    for jf in sorted(json_files):
        try:
            with sftp.open(jf, 'r') as f:
                # Read only summary part - skip per_subject
                raw = f.read().decode()
                d = json.loads(raw)
            
            label = d.get('label', jf.split('/')[-1])
            summary = d.get('summary', {})
            n = d.get('n_test', '?')
            ref = d.get('ref_ckpt', 'None')
            
            all_data[label] = summary
            
            print(f"=== {label} (n={n}) ===")
            if ref and ref != 'None':
                print(f"  ref_ckpt: ...{ref[-40:]}")
            else:
                print(f"  ref_ckpt: None (baseline)")
            
            for k in REGIONS:
                if k in summary:
                    m = summary[k]
                    mean = m['mean']
                    std = m['std']
                    ci_lo = m.get('ci95_low', 0)
                    ci_hi = m.get('ci95_high', 0)
                    print(f"  {k:20s}: {mean:.4f} ± {std:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
            print()
        except Exception as e:
            print(f"Error reading {jf}: {e}")
    
    sftp.close()
    
    # Print condensed table
    print("=" * 140)
    print("CONDENSED COMPARISON TABLE (mean SSIM)")
    print("=" * 140)
    
    header = f"{'Model':30s}"
    for r in REGIONS:
        header += f"{r[:12]:>13s}"
    print(header)
    print("-" * 140)
    
    for label in sorted(all_data.keys()):
        s = all_data[label]
        row = f"{label[:29]:30s}"
        for r in REGIONS:
            if r in s:
                row += f"{s[r]['mean']:.4f}".rjust(13)
            else:
                row += "---".rjust(13)
        print(row)
    
    # Print improvement comparison
    if 'S35best_noref_50subj' in all_data and 'S36_RefC_H1a30_50subj' in all_data:
        print("\n--- Improvement: S36 RefC vs S35 Baseline (test set) ---")
        s35 = all_data['S35best_noref_50subj']
        s36 = all_data['S36_RefC_H1a30_50subj']
        for r in REGIONS:
            if r in s35 and r in s36:
                diff = s36[r]['mean'] - s35[r]['mean']
                pct = diff / s35[r]['mean'] * 100
                print(f"  {r:20s}: {diff:+.4f} ({pct:+.1f}%)")
    
    if 'S35best_noref_valid44' in all_data and 'S36_RefC_H1a30_valid44' in all_data:
        print("\n--- Improvement: S36 RefC vs S35 Baseline (valid set) ---")
        s35v = all_data['S35best_noref_valid44']
        s36v = all_data['S36_RefC_H1a30_valid44']
        for r in REGIONS:
            if r in s35v and r in s36v:
                diff = s36v[r]['mean'] - s35v[r]['mean']
                pct = diff / s35v[r]['mean'] * 100
                print(f"  {r:20s}: {diff:+.4f} ({pct:+.1f}%)")
    
    return all_data


def launch_v2_test_evals(ssh):
    """Launch V2 model evaluations on test set"""
    evals = [
        {'label': 'RefC_v2_cont_50subj',      'ref': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',           'split': 'test',  'n': 50, 'gpu': 0},
        {'label': 'RefC_v2_fresh_50subj',      'ref': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',         'split': 'test',  'n': 50, 'gpu': 1},
        {'label': 'RefD_v2_highnoise_50subj',  'ref': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth', 'split': 'test',  'n': 50, 'gpu': 2},
    ]
    
    # Check if already running
    procs = run_cmd(ssh, 'ps aux | grep evaluate_refinement | grep -v grep')
    if procs:
        print(f"WARNING: Eval processes already running:\n{procs}")
        return
    
    # Check checkpoints exist
    for ev in evals:
        exists = run_cmd(ssh, f'test -f {ev["ref"]} && echo YES || echo NO')
        print(f"  Checkpoint {ev['label']}: {exists}")
    
    print("\n--- Launching V2 TEST set evaluations (3 GPUs) ---")
    for ev in evals:
        cmd = (
            f"cd {CODE_DIR} && "
            f"nohup {PYTHON} evaluate_refinement_v2.py "
            f"  --csv {CSV} "
            f"  --aekl_ckpt {AE_CKPT} "
            f"  --diff_ckpt {DIFF_CKPT} "
            f"  --cnet_ckpt {CNET_CKPT} "
            f"  --ref_ckpt {ev['ref']} "
            f"  --eval_split {ev['split']} "
            f"  --n_test {ev['n']} "
            f"  --m_las 3 "
            f"  --output_json {EVAL_DIR}/{ev['label']}.json "
            f"  --progress_file {EVAL_DIR}/{ev['label']}_progress.json "
            f"  --label {ev['label']} "
            f"  --gpu {ev['gpu']} "
            f"> {EVAL_DIR}/{ev['label']}.log 2>&1 &"
        )
        run_cmd(ssh, cmd, wait=False)
        print(f"  Launched: {ev['label']} on GPU{ev['gpu']}")
        time.sleep(2)
    
    print("\n3 V2 test-set evaluations launched!")
    print("Each takes ~7-8 minutes (50 subjects × ~9s each)")


def launch_v2_valid_evals(ssh):
    """Launch V2 model evaluations on valid set"""
    evals = [
        {'label': 'RefC_v2_cont_valid44',      'ref': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',           'split': 'valid', 'n': 44, 'gpu': 0},
        {'label': 'RefC_v2_fresh_valid44',     'ref': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',         'split': 'valid', 'n': 44, 'gpu': 1},
        {'label': 'RefD_v2_highnoise_valid44', 'ref': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth', 'split': 'valid', 'n': 44, 'gpu': 2},
    ]
    
    procs = run_cmd(ssh, 'ps aux | grep evaluate_refinement | grep -v grep')
    if procs:
        print(f"WARNING: Eval processes still running:\n{procs}")
        return
    
    print("--- Launching V2 VALID set evaluations (3 GPUs) ---")
    for ev in evals:
        cmd = (
            f"cd {CODE_DIR} && "
            f"nohup {PYTHON} evaluate_refinement_v2.py "
            f"  --csv {CSV} "
            f"  --aekl_ckpt {AE_CKPT} "
            f"  --diff_ckpt {DIFF_CKPT} "
            f"  --cnet_ckpt {CNET_CKPT} "
            f"  --ref_ckpt {ev['ref']} "
            f"  --eval_split {ev['split']} "
            f"  --n_test {ev['n']} "
            f"  --m_las 3 "
            f"  --output_json {EVAL_DIR}/{ev['label']}.json "
            f"  --progress_file {EVAL_DIR}/{ev['label']}_progress.json "
            f"  --label {ev['label']} "
            f"  --gpu {ev['gpu']} "
            f"> {EVAL_DIR}/{ev['label']}.log 2>&1 &"
        )
        run_cmd(ssh, cmd, wait=False)
        print(f"  Launched: {ev['label']} on GPU{ev['gpu']}")
        time.sleep(2)
    
    print("\n3 V2 valid-set evaluations launched!")


def check_progress(ssh):
    """Check all evaluation progress"""
    out = run_cmd(ssh, f'ls {EVAL_DIR}/*progress*.json 2>/dev/null')
    if not out:
        print("No progress files found.")
        return
    
    sftp = ssh.open_sftp()
    for pf in sorted(out.split('\n')):
        if pf.strip():
            try:
                with sftp.open(pf.strip(), 'r') as f:
                    pg = json.load(f)
                print(f"  {pf.split('/')[-1]:45s}: {pg.get('completed', '?')}/{pg.get('total', '?')}")
            except:
                pass
    sftp.close()
    
    procs = run_cmd(ssh, 'ps aux | grep evaluate_refinement | grep -v grep | wc -l')
    print(f"\nRunning eval processes: {procs}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'extract'
    
    ssh = get_ssh()
    
    if mode == 'extract':
        extract_summaries(ssh)
    elif mode == 'launch_test':
        launch_v2_test_evals(ssh)
    elif mode == 'launch_valid':
        launch_v2_valid_evals(ssh)
    elif mode == 'progress':
        check_progress(ssh)
    elif mode == 'launch_all':
        launch_v2_test_evals(ssh)
    else:
        print("""Usage: python _s37_metrics.py <mode>
  extract      - Extract existing eval summaries
  launch_test  - Launch V2 test-set evals (3 GPUs)
  launch_valid - Launch V2 valid-set evals (3 GPUs)
  progress     - Check evaluation progress
  launch_all   - Launch V2 test-set evals
""")
    
    ssh.close()
