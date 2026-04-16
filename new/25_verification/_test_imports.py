import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# Test import chain in the fwz conda environment
test_script = '''
source /home/wangchong/miniconda3/etc/profile.d/conda.sh
conda activate fwz
cd /home/wangchong/data/fwz/code/verification/scripts

# Test 1: Basic import check
python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/home/wangchong/data/fwz/code/verification/src')
from quality_metrics import composite_score
print('quality_metrics: OK')
from brlp import const, networks, utils
print('brlp imports: OK')
from brlp import sample_using_controlnet_and_z
print('sampling import: OK')
from sampling_bon import sample_best_of_n
print('sampling_bon: OK')
import pandas as pd
import numpy as np
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')

# Test reading dataset
df = pd.read_csv('/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv')
test_df = df[df.split == 'test']
print(f'Dataset: {len(df)} rows, {len(test_df)} test pairs')
print(f'Columns needed: starting_latent={\"starting_latent\" in df.columns}, '
      f'followup_image={\"followup_image\" in df.columns}, '
      f'starting_age={\"starting_age\" in df.columns}')

# Check checkpoint files exist
import os
for name, path in [
    ('AE', '/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth'),
    ('Diff', '/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth'),
    ('CNet', '/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth'),
]:
    exists = os.path.exists(path)
    size = os.path.getsize(path) / 1e6 if exists else 0
    print(f'{name}: {\"EXISTS\" if exists else \"MISSING\"} ({size:.0f}MB)')

print('ALL CHECKS PASSED')
" 2>&1
'''

_, stdout, stderr = c.exec_command(test_script, timeout=60, get_pty=True)
out = stdout.read().decode().strip()
print(out)
c.close()
