"""Check server process and logs."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

cmds = [
    "ps aux | grep 'evaluate_multi_timepoint' | grep -v grep",
    "tail -40 /home/wangchong/data/fwz/output/multi_timepoint/run.log 2>/dev/null",
    "cat /home/wangchong/data/fwz/output/multi_timepoint/eval_multi_tp.log 2>/dev/null | tail -10",
]
for cmd in cmds:
    print(f"\n$ {cmd}")
    _, stdout, _ = ssh.exec_command(cmd)
    print(stdout.read().decode().strip())
ssh.close()
