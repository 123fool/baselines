"""Investigate 0.93 SSIM - check data files and potential overlap"""
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=10)

commands = [
    'echo "=== CSV files in prepared ===" && ls -la /home/wangchong/data/fwz/output/innovation_5/prepared/*.csv',
    'echo "=== Line counts ===" && wc -l /home/wangchong/data/fwz/output/innovation_5/prepared/*.csv',
    'echo "=== B_mci first subject ===" && head -2 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | cut -d, -f1-5',
    'echo "=== Other CSV files ===" && find /home/wangchong/data/fwz/ -name "*.csv" -type f 2>/dev/null | head -30',
    'echo "=== Training data ===" && ls -la /home/wangchong/data/fwz/brlp-train/ 2>/dev/null || echo "no brlp-train"',
    'echo "=== Check output dirs ===" && ls /home/wangchong/data/fwz/output/innovation_5/prepared/',
]

for cmd in commands:
    _, o, e = ssh.exec_command(cmd)
    out = o.read().decode()
    err = e.read().decode()
    if out.strip():
        print(out)
    if err.strip():
        print(f"ERR: {err}")

ssh.close()
