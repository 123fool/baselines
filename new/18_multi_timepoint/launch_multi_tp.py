"""Upload evaluate_multi_timepoint.py to server and launch evaluation."""
import paramiko
import os

SERVER = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'

LOCAL_SCRIPT = os.path.join(os.path.dirname(__file__), 'evaluate_multi_timepoint.py')
REMOTE_DIR = '/home/wangchong/data/fwz/code/multi_timepoint'
REMOTE_SCRIPT = f'{REMOTE_DIR}/evaluate_multi_timepoint.py'

# Model paths
AEKL = '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'
DIFF = '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'
CNET = '/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth'
TPN  = '/home/wangchong/data/fwz/output/tpn_v3b/tpn_best.pth'
CSV  = '/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'
OUTPUT = '/home/wangchong/data/fwz/output/multi_timepoint'

GPU = 1  # Use GPU 1

print("Connecting to server...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(SERVER, port=PORT, username=USER, password=PASS)

# Create remote directory
print(f"Creating remote dir: {REMOTE_DIR}")
ssh.exec_command(f'mkdir -p {REMOTE_DIR}')
ssh.exec_command(f'mkdir -p {OUTPUT}')

# Upload script
print(f"Uploading {LOCAL_SCRIPT} -> {REMOTE_SCRIPT}")
sftp = ssh.open_sftp()
sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
sftp.close()
print("Upload complete")

# Launch evaluation
CMD = (
    f'cd {REMOTE_DIR} && '
    f'nohup /home/wangchong/miniconda3/envs/fwz/bin/python {REMOTE_SCRIPT} '
    f'--dataset_csv {CSV} '
    f'--aekl_ckpt {AEKL} '
    f'--diff_ckpt {DIFF} '
    f'--cnet_ckpt {CNET} '
    f'--tpn_ckpt {TPN} '
    f'--output_dir {OUTPUT} '
    f'--min_visits 3 '
    f'--splits test,valid '
    f'--methods Direct-Skip,Direct-Linear,Direct-TPN,Auto-Linear '
    f'--gpu {GPU} '
    f'> {OUTPUT}/run.log 2>&1 &'
)

print(f"\nLaunching: {CMD}\n")
stdin, stdout, stderr = ssh.exec_command(CMD)
out = stdout.read().decode()
err = stderr.read().decode()
if out.strip():
    print("STDOUT:", out)
if err.strip():
    print("STDERR:", err)

# Verify process started
import time
time.sleep(2)
stdin2, stdout2, stderr2 = ssh.exec_command("ps aux | grep 'evaluate_multi_timepoint' | grep -v grep")
procs = stdout2.read().decode().strip()
if procs:
    print(f"Process running:\n{procs}")
else:
    print("WARNING: Process not found! Check run.log for errors.")
    stdin3, stdout3, _ = ssh.exec_command(f"tail -30 {OUTPUT}/run.log 2>/dev/null")
    print("Last log lines:")
    print(stdout3.read().decode())

ssh.close()
print("\nDone. Monitor via dashboard or check logs at:")
print(f"  {OUTPUT}/eval_multi_tp.log")
print(f"  {OUTPUT}/run.log")
