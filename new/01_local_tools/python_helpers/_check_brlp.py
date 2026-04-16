import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    # Check where brlp is actually installed
    'find /home/wangchong/data/fwz/ -path "*/brlp/losses.py" 2>/dev/null',
    # Check what's in brlp-train vs brlp-code
    'ls /home/wangchong/data/fwz/brlp-train/src/brlp/ 2>/dev/null',
    'ls /home/wangchong/data/fwz/brlp-code/src/brlp/ 2>/dev/null',
    # Check the losses module content
    'grep "class\|def " /home/wangchong/data/fwz/brlp-train/src/brlp/losses.py 2>/dev/null',
    'grep "class\|def " /home/wangchong/data/fwz/brlp-code/src/brlp/losses.py 2>/dev/null',
    # Check which brlp Python imports
    'PYTHONPATH="/home/wangchong/data/fwz/brlp-train/src:$PYTHONPATH" /home/wangchong/miniconda3/envs/fwz/bin/python -c "import brlp; print(brlp.__file__)"',
    # Check if brlp is installed as a package
    '/home/wangchong/miniconda3/envs/fwz/bin/pip show brlp 2>/dev/null',
]

for cmd in cmds:
    print(f'=== {cmd[:80]} ===')
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print(out if out.strip() else "(empty)")
    if err.strip() and 'Warning' not in err:
        print(f"ERR: {err[:200]}")

ssh.close()
