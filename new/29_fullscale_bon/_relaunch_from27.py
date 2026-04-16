"""
Upload fixed run_bon_fullscale.py and relaunch from pair 27.
Also copies existing log results for merging later.
"""
import paramiko
import os

LOCAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            'scripts', 'run_bon_fullscale.py')
REMOTE_SCRIPT = '/home/wangchong/data/fwz/code/verification/scripts/run_bon_fullscale.py'
LOG_DIR = '/home/wangchong/data/fwz/output/verification/fullscale_50'
LOG_FILE = f'{LOG_DIR}/eval.log'
LOG_RESUME = f'{LOG_DIR}/eval_pairs_0_26.log'  # backup first 27 pairs

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# 1. Kill any existing process
print("Killing old process...")
ssh.exec_command('pkill -f run_bon_fullscale.py')

# 2. Backup existing log (first 27 pairs)
print("Backing up first 27 pairs log...")
ssh.exec_command(f'cp {LOG_FILE} {LOG_RESUME}')

# 3. Upload fixed script
print("Uploading fixed script...")
sftp = ssh.open_sftp()
sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
sftp.close()

# 4. Verify syntax
print("Checking syntax...")
_, o, e = ssh.exec_command(
    f'source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && '
    f'python -c "import ast; ast.parse(open(\'{REMOTE_SCRIPT}\').read()); print(\'OK\')"'
)
result = o.read().decode().strip()
err = e.read().decode().strip()
print(f"  Syntax: {result}")
if err:
    print(f"  Error: {err}")
    ssh.close()
    exit(1)

# 5. Launch with memory optimization env var
print("Launching from pair 27 with memory optimization...")
launch_cmd = (
    f'source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && '
    f'cd /home/wangchong/data/fwz/code/verification/scripts && '
    f'PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 '
    f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '
    f'nohup python run_bon_fullscale.py '
    f'> {LOG_DIR}/eval_resume.log 2>&1 &'
)
ssh.exec_command(launch_cmd)

# 6. Check started
import time
time.sleep(3)
_, o, _ = ssh.exec_command('ps aux | grep run_bon_fullscale | grep -v grep')
ps = o.read().decode().strip()
print(f"Process: {ps[:120] if ps else 'NOT FOUND'}")

_, o, _ = ssh.exec_command(f'head -3 {LOG_DIR}/eval_resume.log 2>/dev/null')
head = o.read().decode().strip()
print(f"Log start: {head}")

ssh.close()
print("\nDone! Monitor with: python _wait_and_collect_resume.py")
