import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

def r(cmd):
    _, o, e = ssh.exec_command(cmd, timeout=30)
    return o.read().decode().strip()

print("=== B_mci.csv columns ===")
print(r("head -1 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv")[:600])
print()

print("=== 002_S_1070 directories ===")
print(r("ls /home/wangchong/data/fwz/data/mci_longitudinal/002_S_1070/"))
print()

first = r("ls /home/wangchong/data/fwz/data/mci_longitudinal/002_S_1070/ | head -1")
print(f"=== {first} contents ===")
print(r(f"ls /home/wangchong/data/fwz/data/mci_longitudinal/002_S_1070/{first}/"))
print()

print("=== B_mci sample for 002_S_1070 ===")
print(r("grep 002_S_1070 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv | head -2")[:500])
print()

print("=== volumes_3class.csv ===")
print(r("ls -la /home/wangchong/data/fwz/output/classification_animation/volumes_3class.csv 2>/dev/null || echo NOT_FOUND"))
print()

# Check age info availability
print("=== B_adni_from_processed.csv columns ===")
print(r("head -1 /home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv 2>/dev/null || echo NOT_FOUND")[:500])

ssh.close()
