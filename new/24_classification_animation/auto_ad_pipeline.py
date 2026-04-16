#!/usr/bin/env python3
"""
自动化 AD Pipeline: 通过 paramiko SSH 完成全部操作
1) 上传脚本到服务器 (有组织的目录结构)
2) 查找 AD 转化患者
3) 运行 pipeline
4) 下载结果到本地
"""

import os
import sys
import time
import paramiko
from pathlib import Path
from scp import SCPClient

# ── 配置 ──
SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"

# 服务器目录 (按用户要求: /home/wangchong/data/fwz/code/ 下有结构的组织)
REMOTE_CODE_DIR = "/home/wangchong/data/fwz/code/24_classification_animation"
REMOTE_OUTPUT_DIR = "/home/wangchong/data/fwz/output/classification_animation"
CONDA_ACTIVATE = "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz"

# 本地目录
LOCAL_DIR = Path(__file__).resolve().parent
LOCAL_RESULTS = LOCAL_DIR / "results_ad"

# 要上传的脚本
UPLOAD_FILES = [
    "run_pipeline.py",
    "find_and_run_ad.py",
    "find_ad_subjects.py",
    "extract_volumes_for_classification.py",
]

# Pipeline 参数
GPU = 1
AVG_N = 3


def create_ssh_client():
    """创建并返回 SSH 连接"""
    print(f"[SSH] 连接到 {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=SERVER_HOST,
        port=SERVER_PORT,
        username=SERVER_USER,
        password=SERVER_PASS,
        timeout=30,
    )
    print("[SSH] 连接成功!")
    return client


def run_cmd(client, cmd, timeout=600, print_output=True):
    """执行远程命令并返回输出"""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    exit_code = stdout.channel.recv_exit_status()
    if print_output and out.strip():
        print(out.strip())
    if print_output and err.strip():
        # 过滤常见的 conda/python warnings
        for line in err.strip().split('\n'):
            if 'UserWarning' not in line and 'FutureWarning' not in line:
                print(f"  [stderr] {line}")
    return out, err, exit_code


def upload_scripts(client):
    """上传脚本到服务器的有组织目录"""
    print(f"\n{'='*60}")
    print(f"[STEP 1] 上传脚本到 {REMOTE_CODE_DIR}")
    print(f"{'='*60}")
    
    # 创建远程目录
    run_cmd(client, f"mkdir -p {REMOTE_CODE_DIR}", print_output=False)
    
    # 上传文件
    with SCPClient(client.get_transport()) as scp:
        for fname in UPLOAD_FILES:
            local_path = str(LOCAL_DIR / fname)
            if os.path.exists(local_path):
                remote_path = f"{REMOTE_CODE_DIR}/{fname}"
                scp.put(local_path, remote_path)
                print(f"  ✓ {fname} -> {remote_path}")
            else:
                print(f"  ✗ {fname} 不存在，跳过")
    
    # 验证
    out, _, _ = run_cmd(client, f"ls -la {REMOTE_CODE_DIR}/", print_output=False)
    print(f"\n  远程目录结构:")
    for line in out.strip().split('\n'):
        print(f"    {line}")


def find_ad_candidates(client):
    """查找 AD 转化患者候选人"""
    print(f"\n{'='*60}")
    print(f"[STEP 2] 查找 AD 转化患者")
    print(f"{'='*60}")
    
    cmd = f"{CONDA_ACTIVATE} && cd {REMOTE_CODE_DIR} && python find_and_run_ad.py --list-only"
    out, err, code = run_cmd(client, cmd, timeout=120)
    
    if code != 0:
        print(f"  [ERROR] 查找失败 (exit code {code})")
        print(f"  stderr: {err}")
        return None
    
    # 解析输出，找到第一个候选人
    lines = out.strip().split('\n')
    candidate = None
    for line in lines:
        line = line.strip()
        # 找数字开头的行 (候选人列表)
        if line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 2:
                # 第一个字段是序号，第二个是 subject_id
                candidate = parts[1]
                break
    
    if candidate:
        print(f"\n  选定候选人: {candidate}")
    return candidate


def run_pipeline(client, subject_id):
    """在服务器上运行 pipeline"""
    print(f"\n{'='*60}")
    print(f"[STEP 3] 运行 Pipeline: subject={subject_id}, GPU={GPU}")
    print(f"{'='*60}")
    
    # 确保输出目录存在
    run_cmd(client, f"mkdir -p {REMOTE_OUTPUT_DIR}", print_output=False)
    
    # 需要加项目根目录到 PYTHONPATH (因为 run_pipeline.py 中 import brlp)
    # run_pipeline.py 中的 PROJECT_ROOT 会自动 resolve 到 REMOTE_CODE_DIR/../../
    # 所以需要确保 src/brlp 在正确位置
    
    # 查找 brlp 包的位置
    out, _, _ = run_cmd(client, f"find /home/wangchong/data/fwz/code -name 'const.py' -path '*/brlp/*' 2>/dev/null | head -3", print_output=False)
    brlp_paths = [p.strip() for p in out.strip().split('\n') if p.strip()]
    print(f"  brlp 包位置: {brlp_paths}")
    
    # run_pipeline.py 用 PROJECT_ROOT = SCRIPT_DIR.parent.parent
    # SCRIPT_DIR = REMOTE_CODE_DIR
    # PROJECT_ROOT = /home/wangchong/data/fwz/code (因为 24_classification_animation 是直接目录)
    # 所以需要 /home/wangchong/data/fwz/code/src/brlp/ 存在
    # 
    # 但实际 brlp 代码可能在其他位置。让我们先检查路径。
    
    # 先检查 sys.path 插入后找不找得到 brlp
    check_cmd = f"""{CONDA_ACTIVATE} && python3 -c "
import sys
sys.path.insert(0, '/home/wangchong/data/fwz/code/brlp_src/src')
try:
    from brlp import const
    print('brlp found at:', const.__file__)
except ImportError as e:
    print('brlp NOT found:', e)
    # 搜索
    import subprocess
    r = subprocess.run(['find', '/home/wangchong/data/fwz', '-name', 'const.py', '-path', '*/brlp/*'], capture_output=True, text=True)
    print('Possible locations:', r.stdout)
"
"""
    out, err, _ = run_cmd(client, check_cmd, timeout=30)
    
    # 根据 run_pipeline.py 的逻辑:
    # SCRIPT_DIR = Path(__file__).resolve().parent  => REMOTE_CODE_DIR
    # PROJECT_ROOT = SCRIPT_DIR.parent.parent => /home/wangchong/data/fwz/code
    # sys.path.insert(0, str(PROJECT_ROOT / "src"))
    # 所以需要 /home/wangchong/data/fwz/code/src/brlp/ 存在
    
    # 创建符号链接确保 src/brlp 可用
    src_parent = str(Path(REMOTE_CODE_DIR).parent.parent)  # /home/wangchong/data/fwz/code
    link_cmd = f"""
if [ ! -d "{src_parent}/src/brlp" ]; then
    # 查找现有的 brlp src 目录
    BRLP_SRC=$(find /home/wangchong/data/fwz -maxdepth 5 -name 'brlp' -type d -path '*/src/brlp' 2>/dev/null | head -1)
    if [ -n "$BRLP_SRC" ]; then
        mkdir -p "{src_parent}/src"
        ln -sf "$BRLP_SRC" "{src_parent}/src/brlp"
        echo "Created symlink: {src_parent}/src/brlp -> $BRLP_SRC"
    else
        echo "ERROR: Cannot find brlp source directory"
    fi
else
    echo "src/brlp already exists at {src_parent}/src/brlp"
fi
ls -la {src_parent}/src/ 2>/dev/null
"""
    run_cmd(client, link_cmd)
    
    # 运行 pipeline
    pipeline_cmd = (
        f"{CONDA_ACTIVATE} && "
        f"cd {REMOTE_CODE_DIR} && "
        f"python run_pipeline.py "
        f"--gpu {GPU} --subject {subject_id} --avg_n {AVG_N} "
        f"--output_dir {REMOTE_OUTPUT_DIR}"
    )
    
    print(f"\n  执行命令: python run_pipeline.py --gpu {GPU} --subject {subject_id} --avg_n {AVG_N}")
    print(f"  这可能需要几分钟，请耐心等待...\n")
    
    out, err, code = run_cmd(client, pipeline_cmd, timeout=1800)  # 30 min timeout
    
    if code != 0:
        print(f"\n  [ERROR] Pipeline 失败 (exit code {code})")
        if err:
            print(f"  Error details:\n{err[-2000:]}")  # 最后 2000 字符
        return False
    
    print(f"\n  ✓ Pipeline 完成!")
    return True


def download_results(client, subject_id):
    """下载结果到本地"""
    print(f"\n{'='*60}")
    print(f"[STEP 4] 下载结果到本地: {LOCAL_RESULTS}")
    print(f"{'='*60}")
    
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    
    # 列出远程输出文件
    out, _, _ = run_cmd(client, f"ls -la {REMOTE_OUTPUT_DIR}/{subject_id}_* 2>/dev/null", print_output=False)
    if not out.strip():
        print("  [WARN] 没有找到结果文件!")
        # 列出整个目录看看
        out2, _, _ = run_cmd(client, f"ls -la {REMOTE_OUTPUT_DIR}/ 2>/dev/null", print_output=False)
        print(f"  远程目录内容:\n{out2}")
        return
    
    print(f"  远程文件:")
    for line in out.strip().split('\n'):
        print(f"    {line}")
    
    # 获取需要下载的文件列表
    out, _, _ = run_cmd(
        client, 
        f"find {REMOTE_OUTPUT_DIR} -name '{subject_id}_*' -type f",
        print_output=False
    )
    remote_files = [f.strip() for f in out.strip().split('\n') if f.strip()]
    
    # 下载
    with SCPClient(client.get_transport()) as scp:
        for remote_path in remote_files:
            fname = os.path.basename(remote_path)
            local_path = str(LOCAL_RESULTS / fname)
            try:
                scp.get(remote_path, local_path)
                fsize = os.path.getsize(local_path) / 1024
                print(f"  ✓ {fname} ({fsize:.1f} KB)")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")
    
    # 列出下载的文件
    print(f"\n  本地结果目录: {LOCAL_RESULTS}")
    for f in sorted(LOCAL_RESULTS.iterdir()):
        print(f"    {f.name} ({f.stat().st_size / 1024:.1f} KB)")


def main():
    t_start = time.time()
    
    print("=" * 60)
    print("  BrLP AD Pipeline 自动化脚本")
    print("=" * 60)
    
    # 1. SSH 连接
    client = create_ssh_client()
    
    try:
        # 2. 上传脚本
        upload_scripts(client)
        
        # 3. 查找 AD 候选人
        subject_id = find_ad_candidates(client)
        if not subject_id:
            print("\n[ERROR] 未找到合适的 AD 转化候选人")
            return
        
        # 4. 运行 Pipeline
        success = run_pipeline(client, subject_id)
        if not success:
            print("\n[ERROR] Pipeline 运行失败")
            return
        
        # 5. 下载结果
        download_results(client, subject_id)
        
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"  全部完成! 耗时 {elapsed/60:.1f} 分钟")
        print(f"  结果目录: {LOCAL_RESULTS}")
        print(f"{'='*60}")
        
    finally:
        client.close()
        print("\n[SSH] 连接已关闭")


if __name__ == '__main__':
    main()
