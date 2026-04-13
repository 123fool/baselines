"""Helper script to manage the no-aux evaluation on server."""
import paramiko
import sys
import time

def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')
    return ssh

def run_cmd(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    return stdout.read().decode('utf-8', errors='replace'), stderr.read().decode('utf-8', errors='replace')

def start():
    ssh = get_ssh()
    out, err = run_cmd(ssh, 
        'cd /home/wangchong/data/fwz/code/no_aux_model && '
        'nohup bash run_eval.sh > /home/wangchong/data/fwz/output/no_aux_model/eval_no_aux.log 2>&1 & '
        'echo $!')
    print(f"Started PID: {out.strip()}")
    time.sleep(3)
    out2, _ = run_cmd(ssh, 'ps aux | grep evaluate_no_aux | grep -v grep')
    print(f"Process: {out2.strip()[:200] or 'NOT FOUND'}")
    out3, _ = run_cmd(ssh, 'tail -10 /home/wangchong/data/fwz/output/no_aux_model/eval_no_aux.log')
    print(f"Log:\n{out3[:500]}")
    ssh.close()

def check():
    ssh = get_ssh()
    out, _ = run_cmd(ssh, 'ps aux | grep evaluate_no_aux | grep -v grep')
    print(f"Process: {out.strip()[:200] or 'NOT RUNNING'}")
    out2, _ = run_cmd(ssh, 'tail -30 /home/wangchong/data/fwz/output/no_aux_model/eval_no_aux.log')
    print(f"Log:\n{out2[:2000]}")
    # Also check for method lines
    out3, _ = run_cmd(ssh, 'grep NO_AUX /home/wangchong/data/fwz/output/no_aux_model/eval_no_aux.log | tail -10')
    print(f"Progress:\n{out3[:1000]}")
    ssh.close()

def results():
    ssh = get_ssh()
    out, _ = run_cmd(ssh, 'cat /home/wangchong/data/fwz/output/no_aux_model/summary_no_aux.json')
    print(out)
    ssh.close()

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if cmd == 'start':
        start()
    elif cmd == 'check':
        check()
    elif cmd == 'results':
        results()
    else:
        print(f"Usage: python {sys.argv[0]} [start|check|results]")
