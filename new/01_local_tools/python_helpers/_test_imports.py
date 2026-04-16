import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

# First, test imports to make sure everything works
print("=== Testing imports ===")
test_cmd = """cd /home/wangchong/data/fwz/code/innovation_4_v4 && \
PYTHONPATH="/home/wangchong/data/fwz/brlp-train/src:/home/wangchong/data/fwz/code/innovation_4_v4/src:$PYTHONPATH" \
/home/wangchong/miniconda3/envs/fwz/bin/python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
from brlp import const, utils, networks
from brlp.data import get_dataset_from_pd
from brlp.gradacc import GradientAccumulation
from brlp.losses import KLDivergenceLoss, L1Loss
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
if 'Error' in err or 'Traceback' in err:
    print("ERRORS:", err[-2000:])
else:
    print("(warnings suppressed)")

# Check if any GPU is busy
print("\n=== GPU Status ===")
stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader', timeout=10)
print(stdout.read().decode())

ssh.close()
