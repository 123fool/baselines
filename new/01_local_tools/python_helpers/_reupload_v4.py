import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
sftp = ssh.open_sftp()

# Re-upload fixed train_ae_v4.py
local_path = r'c:\Users\PC\Desktop\baselines\BrLP-main\new\07_innovation_4\train_ae_v4.py'
remote_path = '/home/wangchong/data/fwz/code/innovation_4_v4/scripts/train_ae_v4.py'

print(f"Re-uploading {remote_path}...")
sftp.put(local_path, remote_path)
print("Done!")

# Test imports again
print("\n=== Testing imports ===")
test_cmd = """cd /home/wangchong/data/fwz/code/innovation_4_v4 && \
PYTHONPATH="/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH" \
/home/wangchong/miniconda3/envs/fwz/bin/python -c "
import sys; sys.path.insert(0, 'src')
import torch
from brlp import const, utils, networks
from brlp.data import get_dataset_from_pd
from brlp.gradacc import GradientAccumulation
from brlp.losses import KLDivergenceLoss
from medicalnet_perceptual_v2 import MedicalNet3DPerceptualLoss, LaplacianPyramidLoss
from monai.losses import PatchAdversarialLoss
from generative.losses import PerceptualLoss
print('All imports OK!')
"
"""
stdin, stdout, stderr = ssh.exec_command(test_cmd, timeout=60)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(out)
if 'Traceback' in err:
    print("ERRORS:", err[-2000:])
elif 'All imports OK' not in out:
    print("STDERR:", err[-500:])

sftp.close()
ssh.close()
