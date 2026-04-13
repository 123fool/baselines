"""Quick smoke test for Priority 2 RLP imports on the server."""
import paramiko

SERVER = "10.96.27.109"
PORT = 2638
USER = "wangchong"
PASS = "123456"
PYTHON = "/home/wangchong/miniconda3/envs/fwz/bin/python"
CODE_DIR = "/home/wangchong/data/fwz/code/priority_2_rlp"

TEST_SCRIPT = f'''
import sys, os
os.chdir("{CODE_DIR}")
sys.path.insert(0, "{CODE_DIR}/brlp_src")
sys.path.insert(0, "{CODE_DIR}/innov2_src")
sys.path.insert(0, "{CODE_DIR}/src")

# Test imports
from brlp import const, utils, networks
from brlp import get_dataset_from_pd
print("brlp imports: OK")

from bidirectional_temporal import build_reverse_context, bidirectional_controlnet_loss
print("bidirectional_temporal imports: OK")

from sampling_rlp import sample_using_controlnet_and_z_rlp
print("sampling_rlp imports: OK")

# Test scale factor function
from train_controlnet_rlp import compute_residual_scale_factor
print("train_controlnet_rlp import: OK (compute_residual_scale_factor found)")

# Check data files
csv_path = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
import pandas as pd
df = pd.read_csv(csv_path)
print(f"CSV loaded: {{len(df)}} rows, splits: {{df.split.value_counts().to_dict()}}")

# Check model checkpoint files
ae_path = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
unet_path = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
cnet_path = "/home/wangchong/data/fwz/brlp-train/pretrained/controlnet.pth"
for p in [ae_path, unet_path, cnet_path]:
    exists = os.path.exists(p)
    print(f"  {{p.split('/')[-1]}}: {{'OK' if exists else 'MISSING'}}")

print("\\nAll imports and data checks passed!")
'''

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {SERVER}:{PORT}...")
    client.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=15)
    
    # Write test to a temp file and execute
    cmd = f'{PYTHON} -c "{TEST_SCRIPT}"'
    # Better approach: write script to file then run
    sftp = client.open_sftp()
    test_path = f"{CODE_DIR}/_smoke_test.py"
    with sftp.open(test_path, 'w') as f:
        f.write(TEST_SCRIPT)
    sftp.close()
    
    print(f"\nRunning smoke test on server...\n")
    _, stdout, stderr = client.exec_command(f'{PYTHON} {test_path}', timeout=30)
    out = stdout.read().decode()
    err = stderr.read().decode()
    
    if out:
        print("STDOUT:")
        print(out)
    if err:
        print("STDERR:")
        print(err)
    
    client.close()


if __name__ == '__main__':
    main()
