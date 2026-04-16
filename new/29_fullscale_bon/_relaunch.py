"""Upload fixed run_bon_fullscale.py and relaunch."""
import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

# Upload fixed script
sftp = ssh.open_sftp()
sftp.put(
    r"c:\Users\PC\Desktop\baselines\BrLP-main\new\29_fullscale_bon\scripts\run_bon_fullscale.py",
    "/home/wangchong/data/fwz/code/verification/scripts/run_bon_fullscale.py"
)
print("Uploaded fixed run_bon_fullscale.py")
sftp.close()

# Kill any old process
ssh.exec_command("pkill -f run_bon_fullscale || true")
time.sleep(2)

# Quick syntax check
stdin, stdout, stderr = ssh.exec_command(
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "cd /home/wangchong/data/fwz/code/verification/scripts && "
    "python -c \"import py_compile; py_compile.compile('run_bon_fullscale.py', doraise=True); print('SYNTAX OK')\""
)
print(stdout.read().decode().strip())
err = stderr.read().decode()
if 'Error' in err:
    print(f"Syntax errors:\n{err}")
    ssh.close()
    exit(1)

# Launch
stdin, stdout, stderr = ssh.exec_command(
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "cd /home/wangchong/data/fwz/code/verification/scripts && "
    "mkdir -p /home/wangchong/data/fwz/output/verification/fullscale_50 && "
    "nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 "
    "python run_bon_fullscale.py "
    "> /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>&1 &"
)
stdout.read()
print("Launched on GPU 2...")
time.sleep(8)

# Check status
stdin, stdout, stderr = ssh.exec_command("ps aux | grep run_bon_fullscale | grep -v grep")
proc = stdout.read().decode().strip()
if proc:
    print("RUNNING!")
else:
    print("NOT RUNNING")

# Show log tail
stdin, stdout, stderr = ssh.exec_command(
    "tail -20 /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log"
)
print(f"Log:\n{stdout.read().decode()}")

ssh.close()
