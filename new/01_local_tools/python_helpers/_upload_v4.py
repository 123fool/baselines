import paramiko
import os
import stat

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)
sftp = ssh.open_sftp()

# Target directory structure
base_dir = '/home/wangchong/data/fwz/code/innovation_4_v4'
dirs_to_create = [
    base_dir,
    f'{base_dir}/scripts',
    f'{base_dir}/src',
]

for d in dirs_to_create:
    try:
        sftp.stat(d)
        print(f"  EXISTS: {d}")
    except FileNotFoundError:
        sftp.mkdir(d)
        print(f"  CREATED: {d}")

# Files to upload
local_base = r'c:\Users\PC\Desktop\baselines\BrLP-main\new\07_innovation_4'
uploads = [
    (f'{local_base}\\train_ae_v4.py',              f'{base_dir}/scripts/train_ae_v4.py'),
    (f'{local_base}\\run_v4.sh',                   f'{base_dir}/run_v4.sh'),
    (f'{local_base}\\src\\medicalnet_perceptual_v2.py', f'{base_dir}/src/medicalnet_perceptual_v2.py'),
    (f'{local_base}\\src\\frequency_losses.py',    f'{base_dir}/src/frequency_losses.py'),
]

# Also copy evaluate script from v2
eval_src = '/home/wangchong/data/fwz/code/innovation_4_v2/scripts/evaluate_innovation4.py'
eval_dst = f'{base_dir}/scripts/evaluate_innovation4.py'

for local_path, remote_path in uploads:
    print(f"  UPLOADING: {os.path.basename(local_path)} -> {remote_path}")
    sftp.put(local_path, remote_path)

# Copy evaluate script from existing v2
print(f"  COPYING: evaluate_innovation4.py from innovation_4_v2")
stdin, stdout, stderr = ssh.exec_command(f'cp {eval_src} {eval_dst}')
stdout.read()

# Create __init__.py for src
sftp.open(f'{base_dir}/src/__init__.py', 'w').close()
print(f"  CREATED: src/__init__.py")

# Make run script executable
ssh.exec_command(f'chmod +x {base_dir}/run_v4.sh')

# Verify upload
print("\n=== Verification ===")
stdin, stdout, stderr = ssh.exec_command(f'find {base_dir} -type f | sort')
print(stdout.read().decode())

sftp.close()
ssh.close()
print("Upload complete!")
