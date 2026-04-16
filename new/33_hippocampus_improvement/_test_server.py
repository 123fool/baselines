"""Quick server test: verify imports and CSV structure."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Test 1: Check imports
test_script = r"""
import sys
sys.path.insert(0, '/home/wangchong/data/fwz/code/src')
from brlp import const, utils, networks
from brlp.sampling import sample_using_controlnet_and_z
print('IMPORTS OK')
print('LATENT_SHAPE_DM:', const.LATENT_SHAPE_DM)

import pandas as pd
df = pd.read_csv('/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv')
print('CSV columns:', list(df.columns))
print('CSV len:', len(df))
if 'split' in df.columns:
    print('Test split:', len(df[df.split=='test']))
    print('Valid split:', len(df[df.split=='valid']))
segm_cols = [c for c in df.columns if 'segm' in c.lower() or 'image' in c.lower() or 'latent' in c.lower()]
print('Image/segm/latent cols:', segm_cols)

# Check first row's segm paths
row = df.iloc[0]
for c in segm_cols:
    import os
    val = row[c]
    exists = os.path.exists(str(val)) if pd.notna(val) else False
    print(f'  {c}: {val} (exists={exists})')
"""

# Write test script to server
sftp = ssh.open_sftp()
with sftp.open('/tmp/test_hippo.py', 'w') as f:
    f.write(test_script)
sftp.close()

cmd = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && python /tmp/test_hippo.py"
_, o, e = ssh.exec_command(cmd)
print(o.read().decode())
err = e.read().decode()
if err:
    print('STDERR:', err[-1000:])

ssh.close()
