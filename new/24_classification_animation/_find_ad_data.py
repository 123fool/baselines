"""查找服务器上所有可能包含 AD 患者的数据源"""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.96.27.109', 2638, 'wangchong', '123456', timeout=30)

def run(cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode().strip()

# 1. 检查 data 目录结构
print("="*60)
print("1. 数据目录结构:")
print("="*60)
out = run("ls -la /home/wangchong/data/fwz/data/")
print(out)

# 2. 检查有没有其他 CSV 文件
print("\n" + "="*60)
print("2. 所有 prepared CSV 文件:")
print("="*60)
out = run("find /home/wangchong/data/fwz/output -name '*.csv' -type f 2>/dev/null | head -30")
print(out)

# 3. 检查 B_ 开头的其他 CSV
print("\n" + "="*60)
print("3. B_ 开头的 CSV 文件:")
print("="*60)
out = run("find /home/wangchong/data/fwz -name 'B_*.csv' -type f 2>/dev/null")
print(out)

# 4. 检查有没有 AD 相关目录/文件
print("\n" + "="*60)
print("4. AD 相关目录和文件:")
print("="*60)
out = run("ls -la /home/wangchong/data/fwz/data/ | grep -i 'ad\|dementia\|alzheimer'")
print(out if out else "(no match)")
out = run("ls /home/wangchong/data/fwz/data/")
print("Data subdirs:", out)

# 5. 检查 BrLP prepared 目录
print("\n" + "="*60)
print("5. Prepared 目录内容:")
print("="*60)
out = run("ls -la /home/wangchong/data/fwz/output/innovation_5/prepared/")
print(out)

# 6. 看其他 CSV 有没有不同诊断值
print("\n" + "="*60)
print("6. 检查所有 CSV 中的诊断列:")
print("="*60)
csvs = run("find /home/wangchong/data/fwz/output -name '*.csv' -type f 2>/dev/null").split('\n')
for csv_file in csvs[:10]:
    csv_file = csv_file.strip()
    if not csv_file:
        continue
    # 检查是否有 diagnosis 列
    header = run(f"head -1 '{csv_file}'")
    if 'diagnosis' in header.lower():
        # 统计诊断值
        cols = header.split(',')
        for i, col in enumerate(cols):
            if 'diagnosis' in col.lower() and 'last' not in col.lower():
                # 用 python 来提取唯一值
                break
        fname = csv_file.split('/')[-1]
        nrows = run(f"wc -l < '{csv_file}'")
        print(f"\n  {fname} ({nrows} rows): has diagnosis")

# 7. 检查 BrLP 代码中引用的数据文件
print("\n" + "="*60)
print("7. 查找 data 目录中的 longitudinal 子目录:")
print("="*60)
out = run("ls /home/wangchong/data/fwz/data/ 2>/dev/null")
for d in out.split('\n'):
    d = d.strip()
    if d:
        count = run(f"ls /home/wangchong/data/fwz/data/{d}/ 2>/dev/null | wc -l")
        print(f"  {d}/: {count} items")

client.close()
