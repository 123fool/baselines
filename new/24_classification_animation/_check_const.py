import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)
def r(cmd):
    _, o, e = ssh.exec_command(cmd, timeout=30)
    return o.read().decode().strip()

# Check const attributes
cmd = """source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 -c "
import sys; sys.path.insert(0, '/home/wangchong/data/fwz/brlp-code/src')
from brlp import const
for attr in sorted(dir(const)):
    if not attr.startswith('_'):
        val = getattr(const, attr)
        if not callable(val):
            print(f'{attr} = {val}')
"
"""
print(r(cmd))
print()

# Check networks attributes
cmd2 = """source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python3 -c "
import sys; sys.path.insert(0, '/home/wangchong/data/fwz/brlp-code/src')
from brlp import networks
for attr in sorted(dir(networks)):
    if not attr.startswith('_') and callable(getattr(networks, attr)):
        print(attr)
" 2>&1 | grep -E 'init_|create_|^[a-z]'
"""
print("networks functions:", r(cmd2))

ssh.close()
