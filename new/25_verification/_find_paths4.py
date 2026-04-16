import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=15)

cmds = [
    # Check the B_mci.csv which is the typical dataset CSV with split column
    'head -2 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv 2>/dev/null',
    # Check the Innovation 1 CSV
    'head -2 /home/wangchong/data/fwz/output/innovation_1/prepared/B_mci_inn1.csv 2>/dev/null',
    # Check the MCI dataset 
    'head -2 /home/wangchong/data/fwz/data/diagnosis_categorized/mci_brlp_innovation.csv 2>/dev/null',
    # Count test rows in B_mci.csv
    'grep -c "test" /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv 2>/dev/null',
    # Check what BTR eval actually used (look at eval logs in innovation_2)
    'find /home/wangchong/data/fwz/output/innovation_2 -name "*.log" -type f 2>/dev/null',
    'ls -la /home/wangchong/data/fwz/output/innovation_2/ 2>/dev/null',
    # Check ADNI eval CSV
    'head -2 /home/wangchong/data/fwz/adni-eval/run_20260404_123352/B_adni_from_processed.csv 2>/dev/null',
]
for cmd in cmds:
    _, stdout, stderr = c.exec_command(cmd, timeout=30, get_pty=True)
    out = stdout.read().decode().strip()
    print(f'CMD: {cmd[:120]}')
    print(f'  OUT: {out[:800]}')
    print()
c.close()
