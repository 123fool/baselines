"""Upload infer_bon_multitime.py to server."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

sftp = ssh.open_sftp()
sftp.put(
    r"c:\Users\PC\Desktop\baselines\BrLP-main\new\29_fullscale_bon\scripts\infer_bon_multitime.py",
    "/home/wangchong/data/fwz/code/verification/scripts/infer_bon_multitime.py"
)
print("Uploaded infer_bon_multitime.py")
sftp.close()
ssh.close()
