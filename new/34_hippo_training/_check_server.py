"""Check server data for hippocampus training."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

script = """
import pandas as pd, os, nibabel as nib, numpy as np
df = pd.read_csv('/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv')
print('=== Splits ===')
print(df['split'].value_counts().to_string())
train_df = df[df.split=='train']
print('Train size:', len(train_df))
count = 0
for _, row in train_df.head(5).iterrows():
    p = str(row.get('followup_segm',''))
    exists = os.path.exists(p) if p and p != 'nan' else False
    if exists: count += 1
    print('  ', os.path.basename(p), ':', exists)
print('  %d/5 segm files exist' % count)
seg_path = str(train_df.iloc[0]['followup_segm'])
seg = nib.load(seg_path)
data = seg.get_fdata()
print('Segm shape:', seg.shape)
print('Segm voxel size:', seg.header.get_zooms())
unique = np.unique(data)
print('Unique labels (%d):' % len(unique), unique[:20])
hippo = np.isin(data, [17, 53])
print('Hippocampus voxels: %d / %d (%.2f%%)' % (hippo.sum(), data.size, 100*hippo.sum()/data.size))
latent_path = str(train_df.iloc[0]['followup_latent'])
lat = np.load(latent_path)
print('Latent keys:', list(lat.keys()))
print('Latent shape:', lat['data'].shape)
"""

sftp = ssh.open_sftp()
with sftp.open('/tmp/_check_data.py', 'w') as f:
    f.write(script)
sftp.close()

_, o, e = ssh.exec_command('python3 /tmp/_check_data.py')
print(o.read().decode())
err = e.read().decode()
if err:
    for line in err.split('\n'):
        if line.strip() and 'Warning' not in line and 'deprecated' not in line:
            print('ERR:', line)
ssh.close()
