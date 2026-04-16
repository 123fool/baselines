import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Check roundtrip full log
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/roundtrip_test/eval_verification.log 2>/dev/null | tail -30", timeout=10)
log = stdout.read().decode().strip()
print("=== ROUNDTRIP FULL LOG (last 30 lines) ===")
print(log)

# Check if roundtrip process is still running
_, stdout, _ = c.exec_command("ps aux | grep 'roundtrip' | grep -v grep", timeout=10)
ps = stdout.read().decode().strip()
print(f"\nRoundtrip process: {'RUNNING' if ps else 'FINISHED/ERRORED'}")

# Check roundtrip summary
_, stdout, _ = c.exec_command("ls -la /home/wangchong/data/fwz/output/verification/roundtrip_test/summary_*.json 2>/dev/null", timeout=10)
summary = stdout.read().decode().strip()
print(f"Summary: {summary if summary else 'not found'}")

# Check runner log
_, stdout, _ = c.exec_command("cat /home/wangchong/data/fwz/output/verification/roundtrip_runner.log 2>/dev/null", timeout=10)
runner = stdout.read().decode().strip()
print(f"\nRunner log: {runner}")

c.close()
