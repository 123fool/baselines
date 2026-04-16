import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

base = "/home/wangchong/data/fwz/output/verification"

for exp in ["bon_n8_full", "roundtrip_test"]:
    d = f"{base}/{exp}"
    _, stdout, _ = c.exec_command(f"wc -l {d}/eval_verification.log 2>/dev/null", timeout=10)
    wc = stdout.read().decode().strip()
    _, stdout, _ = c.exec_command(f"tail -5 {d}/eval_verification.log 2>/dev/null", timeout=10)
    tail = stdout.read().decode().strip()
    _, stdout, _ = c.exec_command(f"ls {d}/summary_*.json 2>/dev/null", timeout=10)
    summary = stdout.read().decode().strip()
    print(f"=== {exp} ===")
    print(f"  Lines: {wc}")
    print(f"  Summary: {summary if summary else 'not yet'}")
    print(f"  Tail: {tail[-200:]}")
    print()

_, stdout, _ = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=10)
gpu = stdout.read().decode().strip()
print(f"GPU: {gpu}")

c.close()
