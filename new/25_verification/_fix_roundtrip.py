import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

scripts = "/home/wangchong/data/fwz/code/verification/scripts"

# Upload fixed sampling_roundtrip.py
sftp = c.open_sftp()
local_path = r"c:\Users\PC\Desktop\baselines\BrLP-main\new\25_verification\scripts\sampling_roundtrip.py"
remote_path = f"{scripts}/sampling_roundtrip.py"
sftp.put(local_path, remote_path)
print("Uploaded fixed sampling_roundtrip.py")

# Delete cached .pyc
try:
    sftp.remove(f"{scripts}/__pycache__/sampling_roundtrip.cpython-39.pyc")
    print("Deleted cached .pyc")
except:
    print("No .pyc to delete")
sftp.close()

# Kill old roundtrip process
_, stdout, _ = c.exec_command("ps aux | grep 'roundtrip_test' | grep -v grep | awk '{print $2}'", timeout=10)
pids = stdout.read().decode().strip().split('\n')
for pid in pids:
    if pid.strip():
        c.exec_command(f"kill {pid.strip()}", timeout=5)
        print(f"Killed PID {pid.strip()}")

# Also kill any evaluate_verification on GPU 0
_, stdout, _ = c.exec_command("ps aux | grep evaluate_verification | grep -v grep | awk '{print $2}'", timeout=10)
pids = stdout.read().decode().strip().split('\n')
for pid in pids:
    if pid.strip():
        # Check if it's the N8 process (which we want to keep)
        _, out, _ = c.exec_command(f"ls -la /proc/{pid.strip()}/fd/1 2>/dev/null | grep bon_n8", timeout=5)
        is_n8 = out.read().decode().strip()
        if not is_n8:
            # Check if it's the roundtrip one
            _, out2, _ = c.exec_command(f"cat /proc/{pid.strip()}/cmdline 2>/dev/null | tr '\\0' ' '", timeout=5)
            cmdline = out2.read().decode().strip()
            if 'roundtrip' in cmdline or 'bon_n8' not in cmdline:
                pass  # Don't kill indiscriminately
c.close()

# Relaunch roundtrip on GPU 0
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

launcher = f"""#!/bin/bash
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd {scripts}
export CUDA_VISIBLE_DEVICES=0
rm -rf /home/wangchong/data/fwz/output/verification/roundtrip_test/*
mkdir -p /home/wangchong/data/fwz/output/verification/roundtrip_test
echo "Starting roundtrip_test v2 on GPU 0 at $(date)"
python evaluate_verification.py --dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth --cnet_ckpt /home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth --output_dir /home/wangchong/data/fwz/output/verification/roundtrip_test --n_candidates 5 --las_m 3 --max_pairs 5 --methods "las,bon_weighted,roundtrip_bon" > /home/wangchong/data/fwz/output/verification/roundtrip_test/eval_verification.log 2>&1
echo "roundtrip_test DONE at $(date)"
"""

sftp2 = c2.open_sftp()
with sftp2.open(f'{scripts}/run_roundtrip_v2.sh', 'w') as f:
    f.write(launcher)
sftp2.close()

transport = c2.get_transport()
channel = transport.open_session()
channel.exec_command(f"nohup bash {scripts}/run_roundtrip_v2.sh > /home/wangchong/data/fwz/output/verification/roundtrip_runner.log 2>&1 &")

import time
time.sleep(3)

c3 = paramiko.SSHClient()
c3.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c3.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
_, stdout, _ = c3.exec_command("tail -2 /home/wangchong/data/fwz/output/verification/roundtrip_runner.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
print(f"Runner: {log}")

c2.close()
c3.close()
print("Done")
