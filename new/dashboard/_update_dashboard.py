"""
BrLP 项目可视化面板 — 数据更新脚本
====================================
用法: python _update_dashboard.py [--status JSON_STRING]
功能: SSH获取服务器信息 + 刷新dashboard.html
"""
import paramiko, json, time, sys, os, re
from datetime import datetime

# ── 配置 ──
SERVER = {"host": "10.96.27.109", "port": 2638, "user": "wangchong", "pass": "123456"}
DASH_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(DASH_DIR, "dashboard_state.json")
HTML_FILE = os.path.join(DASH_DIR, "dashboard.html")

def ssh_exec(client, cmd, timeout=8):
    try:
        _, stdout, _ = client.exec_command(cmd, timeout=timeout)
        return stdout.read().decode("utf-8", errors="replace").strip()
    except:
        return ""

def fetch_server(client):
    gpu = ssh_exec(client, "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null")
    mem = ssh_exec(client, "free -h | head -3")
    disk = ssh_exec(client, "df -h /home/wangchong/data 2>/dev/null | tail -1")
    procs = ssh_exec(client, "ps aux --sort=-%mem | grep -E 'python|train|eval' | grep -v grep | head -10")
    load = ssh_exec(client, "cat /proc/loadavg")
    return {"gpu_raw": gpu, "mem_raw": mem, "disk_raw": disk, "proc_raw": procs, "load": load}

def parse_gpus(raw):
    gpus = []
    if not raw: return gpus
    for line in raw.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            try:
                u, m, t = int(parts[2]), int(parts[3]), int(parts[4])
                gpus.append({"idx": parts[0], "name": parts[1], "util": u,
                             "mem_used": m, "mem_total": t, "mem_pct": int(m/t*100) if t else 0,
                             "temp": parts[5]})
            except: pass
    return gpus

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ai_ops": [], "tasks": [], "thinking": "", "current_phase": "初始化", "experiments": []}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def generate_html(server, state):
    gpus = parse_gpus(server.get("gpu_raw", ""))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu_cards = ""
    for g in gpus:
        color = "#4caf50" if g["util"] < 50 else "#ff9800" if g["util"] < 80 else "#f44336"
        mem_color = "#4caf50" if g["mem_pct"] < 50 else "#ff9800" if g["mem_pct"] < 80 else "#f44336"
        gpu_cards += f"""
        <div class="gpu-card">
          <div class="gpu-header">GPU {g['idx']}: {g['name']}</div>
          <div class="metric-row">
            <span>利用率</span>
            <div class="bar"><div class="bar-fill" style="width:{g['util']}%;background:{color}">{g['util']}%</div></div>
          </div>
          <div class="metric-row">
            <span>显存</span>
            <div class="bar"><div class="bar-fill" style="width:{g['mem_pct']}%;background:{mem_color}">{g['mem_used']}/{g['mem_total']} MiB</div></div>
          </div>
          <div class="metric-row"><span>温度: {g['temp']}°C</span></div>
        </div>"""

    if not gpu_cards:
        gpu_cards = '<div class="gpu-card"><div class="gpu-header">无法获取 GPU 信息</div></div>'

    # Memory & Disk
    mem_html = f"<pre>{server.get('mem_raw', 'N/A')}</pre>"
    disk_html = f"<pre>{server.get('disk_raw', 'N/A')}</pre>"
    load_html = f"<span>Load: {server.get('load', 'N/A')}</span>"

    # Processes
    procs_html = ""
    for line in server.get("proc_raw", "").split("\n"):
        if line.strip():
            parts = line.split(None, 10)
            if len(parts) >= 11:
                procs_html += f"<tr><td>{parts[1]}</td><td>{parts[2]}%</td><td>{parts[3]}%</td><td class='cmd'>{parts[10][:100]}</td></tr>"

    # AI Operations
    ops_html = ""
    for op in reversed(state.get("ai_ops", [])[-30:]):
        icon = {"think": "🧠", "code": "💻", "test": "🧪", "result": "📊", "info": "ℹ️"}.get(op.get("type", "info"), "📝")
        ops_html += f"""<div class="op-item op-{op.get('type','info')}">
          <div class="op-time">{op.get('time','')}</div>
          <div class="op-text">{icon} {op.get('text','')}</div>
        </div>"""

    if not ops_html:
        ops_html = '<div class="op-item op-info"><div class="op-text">ℹ️ 暂无操作记录</div></div>'

    # Tasks / Training
    tasks_html = ""
    for t in state.get("tasks", []):
        pct = t.get("percent", 0)
        eta = t.get("eta", "")
        status_class = {"running": "task-running", "completed": "task-done", "queued": "task-queued"}.get(t.get("status", ""), "task-queued")
        status_icon = {"running": "🔄", "completed": "✅", "queued": "⏳", "failed": "❌"}.get(t.get("status", ""), "⏳")
        tasks_html += f"""<div class="task-item {status_class}">
          <div class="task-name">{status_icon} {t.get('name','')}</div>
          <div class="bar task-bar"><div class="bar-fill" style="width:{pct}%">{pct}%</div></div>
          <div class="task-eta">{eta}</div>
        </div>"""

    if not tasks_html:
        tasks_html = '<div class="task-item task-queued"><div class="task-name">⏳ 暂无活跃任务</div></div>'

    # Current Thinking
    thinking = state.get("thinking", "等待操作...")

    # Experiments summary
    exp_html = ""
    for e in state.get("experiments", []):
        exp_html += f"<tr><td>{e.get('name','')}</td><td>{e.get('metric','')}</td><td>{e.get('value','')}</td><td>{e.get('status','')}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>BrLP MCI→AD 研究监控面板</title>
<style>
:root {{
  --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9;
  --text2: #8b949e; --accent: #58a6ff; --green: #3fb950; --orange: #d29922;
  --red: #f85149; --purple: #bc8cff;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:14px; }}
.header {{ background:linear-gradient(135deg,#1a1f36,#0d1117); padding:16px 24px; border-bottom:1px solid var(--border);
           display:flex; align-items:center; justify-content:space-between; }}
.header h1 {{ font-size:18px; color:var(--accent); font-weight:600; }}
.header .meta {{ font-size:12px; color:var(--text2); }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; padding:12px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px; }}
.card-title {{ font-size:13px; font-weight:600; color:var(--accent); margin-bottom:10px; text-transform:uppercase; letter-spacing:0.5px; }}
.full {{ grid-column:1/-1; }}
.gpu-card {{ background:#1c2333; border-radius:6px; padding:10px; margin-bottom:8px; }}
.gpu-header {{ font-size:13px; font-weight:600; color:var(--green); margin-bottom:6px; }}
.metric-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:12px; }}
.metric-row span {{ min-width:50px; color:var(--text2); }}
.bar {{ flex:1; background:#21262d; border-radius:4px; height:18px; overflow:hidden; position:relative; }}
.bar-fill {{ height:100%; border-radius:4px; font-size:11px; line-height:18px; padding-left:6px; color:#fff;
             min-width:fit-content; white-space:nowrap; transition:width 0.3s; }}
pre {{ font-size:11px; color:var(--text2); white-space:pre-wrap; word-break:break-all; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); color:var(--accent); font-weight:600; }}
td {{ padding:5px 8px; border-bottom:1px solid #21262d; }}
td.cmd {{ font-family:monospace; font-size:11px; color:var(--text2); max-width:400px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.thinking-box {{ background:#1c2333; border-left:3px solid var(--purple); padding:10px 14px; border-radius:4px;
                 font-size:13px; color:var(--text); line-height:1.6; min-height:40px; max-height:150px; overflow-y:auto; }}
.op-item {{ padding:6px 10px; border-left:3px solid var(--border); margin-bottom:4px; border-radius:0 4px 4px 0; }}
.op-think {{ border-left-color:var(--purple); background:rgba(188,140,255,0.05); }}
.op-code {{ border-left-color:var(--accent); background:rgba(88,166,255,0.05); }}
.op-test {{ border-left-color:var(--orange); background:rgba(210,153,34,0.05); }}
.op-result {{ border-left-color:var(--green); background:rgba(63,185,80,0.05); }}
.op-info {{ border-left-color:var(--text2); }}
.op-time {{ font-size:10px; color:var(--text2); }}
.op-text {{ font-size:12px; line-height:1.5; }}
.ops-container {{ max-height:400px; overflow-y:auto; }}
.task-item {{ display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid #21262d; }}
.task-name {{ min-width:200px; font-size:13px; }}
.task-bar {{ flex:1; }}
.task-bar .bar-fill {{ background:var(--accent); }}
.task-done .task-bar .bar-fill {{ background:var(--green); }}
.task-running .task-bar .bar-fill {{ background:var(--orange); }}
.task-eta {{ font-size:11px; color:var(--text2); min-width:80px; text-align:right; }}
.phase-badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;
                background:rgba(88,166,255,0.15); color:var(--accent); }}
</style>
</head>
<body>
<div class="header">
  <h1>🧬 BrLP MCI→AD 研究项目 — 实验监控面板</h1>
  <div class="meta">
    <span class="phase-badge">{state.get('current_phase','')}</span>
    &nbsp; 更新: {now} &nbsp; 服务器: {SERVER['host']}:{SERVER['port']}
  </div>
</div>

<div class="grid">
  <!-- 当前思考 -->
  <div class="card full">
    <div class="card-title">🧠 当前思考与方向</div>
    <div class="thinking-box">{thinking}</div>
  </div>

  <!-- GPU -->
  <div class="card">
    <div class="card-title">🖥️ GPU 状态</div>
    {gpu_cards}
  </div>

  <!-- 内存 & 磁盘 -->
  <div class="card">
    <div class="card-title">💾 内存 & 磁盘 &nbsp; {load_html}</div>
    {mem_html}
    <div style="margin-top:8px;font-size:12px;color:var(--text2);">磁盘:</div>
    {disk_html}
  </div>

  <!-- 任务进度 -->
  <div class="card full">
    <div class="card-title">📋 训练 / 测试进度</div>
    {tasks_html}
  </div>

  <!-- AI 操作日志 -->
  <div class="card full">
    <div class="card-title">📝 AI 操作日志</div>
    <div class="ops-container">{ops_html}</div>
  </div>

  <!-- 进程列表 -->
  <div class="card">
    <div class="card-title">⚙️ 运行中的进程</div>
    <table>
      <tr><th>PID</th><th>CPU</th><th>MEM</th><th>命令</th></tr>
      {procs_html if procs_html else '<tr><td colspan="4" style="color:var(--text2)">无活跃 Python 进程</td></tr>'}
    </table>
  </div>

  <!-- 实验结果汇总 -->
  <div class="card">
    <div class="card-title">📊 实验结果汇总</div>
    <table>
      <tr><th>实验</th><th>指标</th><th>值</th><th>状态</th></tr>
      {exp_html if exp_html else '<tr><td colspan="4" style="color:var(--text2)">暂无实验数据</td></tr>'}
    </table>
  </div>
</div>

</body>
</html>"""
    return html

def update_dashboard(new_ops=None, thinking=None, phase=None, tasks=None, experiments=None):
    """主更新函数: 获取服务器数据 + 合并状态 + 生成HTML"""
    state = load_state()

    if new_ops:
        for op in new_ops:
            op["time"] = datetime.now().strftime("%H:%M:%S")
            state["ai_ops"].append(op)
    if thinking is not None:
        state["thinking"] = thinking
    if phase is not None:
        state["current_phase"] = phase
    if tasks is not None:
        state["tasks"] = tasks
    if experiments is not None:
        state["experiments"] = experiments

    save_state(state)

    # SSH 获取服务器信息
    server = {}
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(SERVER["host"], port=SERVER["port"], username=SERVER["user"], password=SERVER["pass"], timeout=15)
        server = fetch_server(client)
        client.close()
    except Exception as e:
        server = {"gpu_raw": "", "mem_raw": f"连接失败: {e}", "disk_raw": "", "proc_raw": "", "load": "N/A"}

    html = generate_html(server, state)
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard updated → {HTML_FILE}")

if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--refresh-only", action="store_true",
                   help="仅根据现有 state 刷新 HTML，不重置状态")
    _args = p.parse_args()

    if _args.refresh_only:
        # 保留现有状态，仅刷新服务器数据和 HTML
        update_dashboard()
    else:
        # 初始更新（全部重置）
        update_dashboard(
            thinking="正在初始化面板...",
            phase="初始化",
            new_ops=[{"type": "info", "text": "面板初始化完成"}],
            tasks=[],
            experiments=[
                {"name": "Baseline BTR", "metric": "H-SSIM", "value": "0.8006±0.0182", "status": "✅ 完成"},
                {"name": "AE_dec+H1_a30 (最佳)", "metric": "H-SSIM", "value": "0.8127±0.0093", "status": "✅ 完成"},
                {"name": "AE 重建天花板", "metric": "H-SSIM", "value": "0.8288±0.0054", "status": "📏 参考"},
            ]
        )
