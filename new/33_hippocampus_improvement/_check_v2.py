"""Check V2 experiment status."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

_, o, _ = ssh.exec_command("ps aux | grep 'hippocampus_improvement_v2' | grep -v grep | wc -l")
n = int(o.read().decode().strip())
print(f"Running: {n}")

_, o, _ = ssh.exec_command("tail -60 /home/wangchong/data/fwz/output/33_hippocampus_v2/run.log 2>/dev/null")
log = o.read().decode()
# Filter out warnings, show only our log lines
lines = [l for l in log.split('\n') if l.strip() and not any(x in l for x in ['Warning', 'deprecated', 'tensorflow', 'TF-TRT', 'pkg_resources', 'data_array'])]
print('\n'.join(lines[-40:]))

_, o, _ = ssh.exec_command("cat /home/wangchong/data/fwz/output/33_hippocampus_v2/summary.json 2>/dev/null")
s = o.read().decode().strip()
if s:
    print(f"\n=== SUMMARY ===\n{s}")

ssh.close()
