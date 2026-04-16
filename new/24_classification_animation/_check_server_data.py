"""Check server for preprocessed data availability for MCI→AD converter patients."""
import paramiko
import json

# Server config
HOST = "10.96.27.109"
PORT = 2638
USER = "wangchong"
PASS = "123456"

# Top MCI→AD converters from E:\ADNI
CONVERTERS = [
    "002_S_1070", "023_S_0331", "023_S_0388", "023_S_0604",
    "027_S_0835", "037_S_0588", "053_S_0507", "116_S_0649",
    "136_S_0429", "016_S_1121", "016_S_1326", "023_S_1247",
    "027_S_1387", "033_S_0725", "133_S_0638",
]

def ssh_exec(client, cmd, timeout=15):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace").strip()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

print("=== Checking preprocessed data on server ===\n")

# 1. Check B_mci.csv for these subjects
print("--- B_mci.csv check ---")
bmci = ssh_exec(client, "head -1 /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv")
print(f"B_mci columns: {bmci[:200]}")
for ptid in CONVERTERS[:5]:
    found = ssh_exec(client, f"grep -c '{ptid}' /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv 2>/dev/null || echo 0")
    print(f"  {ptid} in B_mci.csv: {found} rows")

# 2. Check ad_brlp_innovation.csv 
print("\n--- ad_brlp_innovation.csv check ---")
for ptid in CONVERTERS[:5]:
    found = ssh_exec(client, f"grep -c '{ptid}' /home/wangchong/data/fwz/data/diagnosis_categorized/ad_brlp_innovation.csv 2>/dev/null || echo 0")
    print(f"  {ptid} in ad_brlp: {found} rows")

# 3. Check for preprocessed images in brlp-data
print("\n--- brlp-data raw images check ---")
for ptid in CONVERTERS[:5]:
    found = ssh_exec(client, f"find /home/wangchong/data/fwz/brlp-data/ -path '*{ptid}*' -name '*.nii.gz' 2>/dev/null | head -5")
    if found:
        print(f"  {ptid}: FOUND in brlp-data")
        for f in found.split('\n')[:3]:
            print(f"    {f}")
    else:
        print(f"  {ptid}: NOT in brlp-data")

# 4. Check for preprocessed images in data/mci_longitudinal or data/ad_longitudinal
print("\n--- data directories check ---")
for ptid in CONVERTERS[:5]:
    found_mci = ssh_exec(client, f"find /home/wangchong/data/fwz/data/mci_longitudinal/ -path '*{ptid}*' -name 't1w_final.nii.gz' 2>/dev/null | head -5")
    found_ad = ssh_exec(client, f"find /home/wangchong/data/fwz/data/ad_longitudinal/ -path '*{ptid}*' -name 't1w_final.nii.gz' 2>/dev/null | head -5")
    if found_mci:
        print(f"  {ptid}: FOUND in mci_longitudinal")
        for f in found_mci.split('\n')[:3]:
            print(f"    {f}")
    elif found_ad:
        print(f"  {ptid}: FOUND in ad_longitudinal")
        for f in found_ad.split('\n')[:3]:
            print(f"    {f}")
    else:
        print(f"  {ptid}: NOT in mci_long or ad_long")

# 5. Check ALL converter subjects
print("\n--- Full check for all 15 converters ---")
for ptid in CONVERTERS:
    mci_count = ssh_exec(client, f"find /home/wangchong/data/fwz/data/mci_longitudinal/{ptid} -name 't1w_final.nii.gz' 2>/dev/null | wc -l").strip()
    ad_count = ssh_exec(client, f"find /home/wangchong/data/fwz/data/ad_longitudinal/{ptid} -name 't1w_final.nii.gz' 2>/dev/null | wc -l").strip()
    brlp_count = ssh_exec(client, f"grep -c '{ptid}' /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv 2>/dev/null || echo 0").strip()
    segm_mci = ssh_exec(client, f"find /home/wangchong/data/fwz/data/mci_longitudinal/{ptid} -name 'synthseg.nii.gz' 2>/dev/null | wc -l").strip()
    segm_ad = ssh_exec(client, f"find /home/wangchong/data/fwz/data/ad_longitudinal/{ptid} -name 'synthseg.nii.gz' 2>/dev/null | wc -l").strip()
    latent_count = ssh_exec(client, f"find /home/wangchong/data/fwz/data/ -path '*{ptid}*' -name 'latent.npz' 2>/dev/null | wc -l").strip()
    total_img = int(mci_count or 0) + int(ad_count or 0)
    total_segm = int(segm_mci or 0) + int(segm_ad or 0)
    status = "✓" if total_img > 0 else "✗"
    print(f"  {status} {ptid}: images={total_img} (mci:{mci_count}, ad:{ad_count}) segm={total_segm} latents={latent_count} B_mci={brlp_count}")

client.close()
print("\nDone.")
