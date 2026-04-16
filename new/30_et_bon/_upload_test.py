"""Upload ET-BoN scripts and test import."""
import paramiko
from pathlib import Path

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.96.27.109", port=2638, username="wangchong", password="123456", timeout=15)

sftp = ssh.open_sftp()
for f in ["sampling_et_bon.py", "run_et_bon_experiment.py"]:
    local = str(Path(__file__).parent / "scripts" / f)
    remote = f"/home/wangchong/data/fwz/code/et_bon/scripts/{f}"
    sftp.put(local, remote)
    print(f"Uploaded {f}")
sftp.close()

# Test import
cmd = (
    "cd /home/wangchong/data/fwz/code/et_bon/scripts && "
    "source /home/wangchong/anaconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    'python -c "from sampling_et_bon import sample_et_bon_weighted; print(\'OK: ET-BoN imported\')"'
)
_, so, se = ssh.exec_command(cmd, timeout=60)
print("stdout:", so.read().decode().strip())
err = se.read().decode().strip()
if err:
    print("stderr:", err)

ssh.close()
print("Done")
