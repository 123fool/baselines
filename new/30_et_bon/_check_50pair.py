"""Quick status check on 50-pair experiment."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.96.27.109', port=2638, username='wangchong', password='123456', timeout=10)

# Get all ET-BoN pair lines
cmd = "grep 'Δ vs LAS' /home/wangchong/data/fwz/output/verification/et_bon_50pair/et_bon_50pair.log"
_, o, _ = ssh.exec_command(cmd)
lines = o.read().decode().strip().split('\n')

wins = 0
losses = 0
for line in lines:
    if 'Δ vs LAS=+' in line:
        wins += 1
    elif 'Δ vs LAS=-' in line:
        losses += 1

total = wins + losses
print(f"ET-BoN 50-pair progress: {total}/50 pairs done")
print(f"Wins: {wins}, Losses: {losses}")
print(f"Win rate: {wins/total*100:.1f}%" if total > 0 else "N/A")
print(f"\nLast 5 lines:")
for l in lines[-5:]:
    # Extract just the key info
    import re
    m = re.search(r'Pair (\d+).*SSIM=([\d.]+).*Δ vs LAS=([+-][\d.]+)', l)
    if m:
        print(f"  Pair {m.group(1)}: SSIM={m.group(2)}, Δ={m.group(3)}")
    else:
        print(f"  {l.strip()[-80:]}")

ssh.close()
