"""Detailed crash investigation."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

cmds = [
    "ps aux | grep 'multi_timepoint' | grep -v grep",
    "ls -la /home/wangchong/data/fwz/output/multi_timepoint/",
    "wc -l /home/wangchong/data/fwz/output/multi_timepoint/run.log",
    "tail -5 /home/wangchong/data/fwz/output/multi_timepoint/run.log",
    "cat /home/wangchong/data/fwz/code/multi_timepoint/nohup.out 2>/dev/null | tail -30",
    # Maybe stdout isn't flushed - check if the process wrote to stderr in run.log 
    "grep -i 'error\\|exception\\|traceback\\|failed' /home/wangchong/data/fwz/output/multi_timepoint/run.log 2>/dev/null | tail -10",
    # Check all python processes running
    "ps aux | grep python | grep -v grep | head -10",
]
for cmd in cmds:
    print(f"\n$ {cmd}")
    _, stdout, _ = ssh.exec_command(cmd, timeout=10)
    print(stdout.read().decode().strip()[:2000])
ssh.close()
