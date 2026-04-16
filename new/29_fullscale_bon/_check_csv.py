"""Check CSV columns on server."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456')

cmd = (
    "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && "
    "conda activate fwz && "
    "python -c \""
    "import pandas as pd; "
    "df = pd.read_csv('/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv'); "
    "print('Columns:', list(df.columns)); "
    "print('Shape:', df.shape); "
    "print('First row:'); "
    "print(df.iloc[0].to_dict())"
    "\""
)

stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode())
err = stderr.read().decode()
if 'Error' in err or 'Traceback' in err:
    print("ERRORS:", err)

ssh.close()
