import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Full roundtrip log
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/roundtrip_test/eval_verification.log 2>/dev/null", timeout=10)
log = stdout.read().decode().strip()
lines = log.split('\n')
print(f"=== ROUNDTRIP LOG ({len(lines)} lines) ===")
for line in lines:
    if any(kw in line for kw in ['Error', 'error', 'Traceback', 'RuntimeError', 'Pair', 'SSIM', 'Scale', 'Loading', 'roundtrip', 'Exception']):
        print(f"  {line.strip()}")

# Full N8 log
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/bon_n8_full/eval_verification.log 2>/dev/null", timeout=10)
log8 = stdout.read().decode().strip()
lines8 = log8.split('\n')
print(f"\n=== N8 LOG ({len(lines8)} lines) ===")
for line in lines8:
    if 'Pair' in line and 'SSIM' in line:
        print(f"  {line.strip()}")

c.close()
