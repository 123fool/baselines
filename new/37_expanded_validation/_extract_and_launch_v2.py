#!/usr/bin/env python3
"""
Step 1: Extract all existing eval summaries cleanly
Step 2: Launch V2 model evaluations (3 models × 2 splits = 6 runs)
"""
import paramiko, json, sys, time

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
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    if wait:
        return stdout.read().decode().strip()
    time.sleep(1)
    return ''

def extract_summaries(ssh):
    """Read all eval JSONs and print summary table"""
    # List eval JSON files
    out = run_cmd(ssh, f'ls {EVAL_DIR}/*.json 2>/dev/null | grep -v progress')
    if not out:
        print("No evaluation JSON files found!")
        return
    
    json_files = [f for f in out.split('\n') if f.strip()]
    print(f"Found {len(json_files)} evaluation result files\n")
    
    all_summaries = {}
    for jf in sorted(json_files):
        # Read JSON via python one-liner to extract summary only
        cmd = f"""{PYTHON} -c "
import json
with open('{jf}') as f:
    d = json.load(f)
label = d.get('label', '{jf.split('/')[-1]}')
s = d.get('summary', {{}})
n = d.get('n_test', '?')
print(f'LABEL: {{label}} (n={{n}})')
for k in ['overall', 'ad_composite', 'hippocampus', 'amygdala', 'thalamus', 'lateral_ventricle', 'caudate', 'putamen', 'cerebral_cortex', 'cerebral_wm', 'pallidum']:
    if k in s:
        m = s[k]
        print(f'  {{k}}: mean={{m[\"mean\"]:.4f}} std={{m[\"std\"]:.4f}} ci95=[{{m[\"ci95_low\"]:.4f}}, {{m[\"ci95_high\"]:.4f}}]')
    elif k == 'overall' and 'overall' not in s:
        # Try overall_ssim
        om = s.get('overall_ssim', None)
        if om:
            print(f'  overall: mean={{om[\"mean\"]:.4f}} std={{om[\"std\"]:.4f}}')
"
"""
        result = run_cmd(ssh, cmd)
        if result:
            print(result)
            print()
    
    # Also print a condensed comparison table
    print("=" * 100)
    print("CONDENSED COMPARISON TABLE (mean SSIM)")
    print("=" * 100)
    
    # Re-extract just means for table
    cmd = f"""{PYTHON} -c "
import json, os, glob

files = sorted(glob.glob('{EVAL_DIR}/*.json'))
files = [f for f in files if 'progress' not in f]

regions = {REGIONS}
header = 'Model'.ljust(30) + ''.join(r[:12].rjust(13) for r in regions)
print(header)
print('-' * len(header))

for jf in files:
    with open(jf) as f:
        d = json.load(f)
    label = d.get('label', os.path.basename(jf))
    s = d.get('summary', {{}})
    row = label[:29].ljust(30)
    for r in regions:
        m = s.get(r, {{}})
        if 'mean' in m:
            row += f'{{m[\"mean\"]:.4f}}'.rjust(13)
        else:
            row += '---'.rjust(13)
    print(row)
"
"""
    result = run_cmd(ssh, cmd)
    if result:
        print(result)


def launch_v2_evals(ssh):
    """Launch V2 model evaluations on test + valid sets"""
    # Check if any GPUs are busy
    gpu_check = run_cmd(ssh, 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader')
    print(f"GPU utilization: {gpu_check}")
    
    # Check which V2 eval JSONs already exist
    existing = run_cmd(ssh, f'ls {EVAL_DIR}/*v2*.json 2>/dev/null | grep -v progress')
    if existing:
        print(f"\nExisting V2 evals: {existing}")
        resp = input("V2 eval files already exist. Skip launch? [y/N]: ").strip().lower()
        if resp == 'y':
            return
    
    evals = [
        # Test set (50 subjects)
        {'label': 'RefC_v2_cont_50subj',      'ref': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',             'split': 'test',  'n': 50, 'gpu': 0},
        {'label': 'RefC_v2_fresh_50subj',      'ref': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',           'split': 'test',  'n': 50, 'gpu': 1},
        {'label': 'RefD_v2_highnoise_50subj',  'ref': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth',   'split': 'test',  'n': 50, 'gpu': 2},
        # Valid set (44 subjects) 
        {'label': 'RefC_v2_cont_valid44',      'ref': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',             'split': 'valid', 'n': 44, 'gpu': 0},
        {'label': 'RefC_v2_fresh_valid44',     'ref': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',           'split': 'valid', 'n': 44, 'gpu': 1},
        {'label': 'RefD_v2_highnoise_valid44', 'ref': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth',   'split': 'valid', 'n': 44, 'gpu': 2},
    ]
    
    # Launch test set evals first (3 GPUs parallel)
    print("\n--- Launching TEST set V2 evaluations ---")
    for ev in evals[:3]:
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
        print(f"  ✓ Launched: {ev['label']} on GPU{ev['gpu']}")
        time.sleep(2)
    
    print("\nTest set evals launched on 3 GPUs.")
    print("Valid set evals will need to wait until test set completes (~7-8 min).")
    print("Run this script again with 'launch_valid' after test set completes.")
    
    # Store valid set commands for later
    return evals[3:]


def launch_valid_only(ssh):
    """Launch just the valid set V2 evaluations"""
    evals = [
        {'label': 'RefC_v2_cont_valid44',      'ref': f'{OUTPUT_DIR}/RefC_v2_cont/refnet-RefC_v2_cont-best.pth',             'split': 'valid', 'n': 44, 'gpu': 0},
        {'label': 'RefC_v2_fresh_valid44',     'ref': f'{OUTPUT_DIR}/RefC_v2_fresh/refnet-RefC_v2_fresh-best.pth',           'split': 'valid', 'n': 44, 'gpu': 1},
        {'label': 'RefD_v2_highnoise_valid44', 'ref': f'{OUTPUT_DIR}/RefD_v2_highnoise/refnet-RefD_v2_highnoise-best.pth',   'split': 'valid', 'n': 44, 'gpu': 2},
    ]
    
    print("\n--- Launching VALID set V2 evaluations ---")
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
        print(f"  ✓ Launched: {ev['label']} on GPU{ev['gpu']}")
        time.sleep(2)
    print("\nValid set V2 evals launched!")


def check_progress(ssh):
    """Check V2 evaluation progress"""
    out = run_cmd(ssh, f"""ls {EVAL_DIR}/*v2*progress*.json 2>/dev/null""")
    if not out:
        print("No V2 progress files found yet.")
        return
    
    for pf in out.split('\n'):
        if pf.strip():
            content = run_cmd(ssh, f'cat {pf.strip()}')
            try:
                pg = json.loads(content)
                print(f"  {pf.split('/')[-1]}: {pg.get('completed', '?')}/{pg.get('total', '?')}")
            except:
                print(f"  {pf.split('/')[-1]}: {content[:100]}")
    
    # Check running eval processes
    procs = run_cmd(ssh, 'ps aux | grep evaluate_refinement | grep -v grep')
    if procs:
        print(f"\nRunning eval processes: {len(procs.split(chr(10)))}")
    else:
        print("\nNo evaluation processes running.")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'extract'
    
    ssh = get_ssh()
    
    if mode == 'extract':
        extract_summaries(ssh)
    elif mode == 'launch':
        extract_summaries(ssh)
        print("\n" + "=" * 80)
        launch_v2_evals(ssh)
    elif mode == 'launch_valid':
        launch_valid_only(ssh)
    elif mode == 'progress':
        check_progress(ssh)
    elif mode == 'all':
        extract_summaries(ssh)
        print("\n" + "=" * 80)
        launch_v2_evals(ssh)
    else:
        print(f"""Usage: python _extract_and_launch_v2.py <mode>
Modes:
  extract      - Extract and display all existing eval summaries
  launch       - Extract summaries then launch V2 test-set evals
  launch_valid - Launch V2 valid-set evals (after test completes)
  progress     - Check V2 eval progress
  all          - Extract + launch all
""")
    
    ssh.close()
