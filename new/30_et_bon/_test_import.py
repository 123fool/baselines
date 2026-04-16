"""Test ET-BoN import on server."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.96.27.109", port=2638, username="wangchong", password="123456", timeout=15)

cmd = (
    "cd /home/wangchong/data/fwz/code/et_bon/scripts && "
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "python -c 'from sampling_et_bon import sample_et_bon_weighted; print(\"IMPORT_OK\")'"
)
_, so, se = ssh.exec_command(cmd, timeout=60)
out = so.read().decode().strip()
err = se.read().decode().strip()
print(f"stdout: {out}")
if err:
    print(f"stderr: {err}")

# Also test runner import
cmd2 = (
    "cd /home/wangchong/data/fwz/code/et_bon/scripts && "
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "python -c 'from run_et_bon_experiment import run_experiment; print(\"RUNNER_OK\")'"
)
_, so, se = ssh.exec_command(cmd2, timeout=60)
out = so.read().decode().strip()
err = se.read().decode().strip()
print(f"stdout: {out}")
if err:
    print(f"stderr: {err}")

# Check CSV exists
_, so, _ = ssh.exec_command("wc -l /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv")
print(f"CSV lines: {so.read().decode().strip()}")

ssh.close()
print("All tests done.")
