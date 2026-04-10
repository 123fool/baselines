"""
BrLP 研究项目 — 可视化监控面板
=================================
功能:
  1. 服务器实时信息 (CPU / 内存 / GPU / 运行进程)
  2. 代码修改记录 (修改内容 / 原因 / 结果)
  3. 实验指标对比 (Baseline vs Innovation 4 vs Innovation 5)

运行:
  python server_monitor.py            # 默认 http://127.0.0.1:8080
  python server_monitor.py --port 9090
"""

import os
import sys
import json
import time
import re
import threading
import argparse
from datetime import datetime

import paramiko
from flask import Flask, render_template_string, jsonify, request

# ─── 配置 ────────────────────────────────────────────────────────
SERVER_HOST = "10.96.27.109"
SERVER_PORT = 2638
SERVER_USER = "wangchong"
SERVER_PASS = "123456"
CODE_DIR    = "/home/wangchong/data/fwz/code/"
TRAIN_DIR   = "/home/wangchong/data/fwz/brlp-train/"
TRAIN_LOG   = "/home/wangchong/data/fwz/output/innovation_4_v4/train_v4.log"
AUTO_EVAL_LOG = "/home/wangchong/data/fwz/output/innovation_4_v4/auto_eval_monitor.log"
EVAL_SUMMARY = "/home/wangchong/data/fwz/output/innovation_4_v4/eval/summary_innovation_4_v4.json"

# 缓存
_cache = {
    "server_info": None,
    "gpu_info": None,
    "processes": None,
  "task_progress": None,
    "last_update": None,
    "error": None,
}
_cache_lock = threading.Lock()

app = Flask(__name__)

# ─── SSH 工具 ────────────────────────────────────────────────────

def ssh_exec(cmd, timeout=10):
    """通过 SSH 在服务器执行命令并返回 stdout。"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(SERVER_HOST, port=SERVER_PORT,
                       username=SERVER_USER, password=SERVER_PASS,
                       timeout=timeout)
        _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace").strip()
        return out
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        client.close()


def fetch_server_info():
    """从服务器采集一次完整信息。"""
    info = {}
    try:
        # CPU
        cpu_raw = ssh_exec(
            "top -bn1 | head -5; echo '---CPUCOUNT---'; nproc; "
            "echo '---LOADAVG---'; cat /proc/loadavg"
        )
        info["cpu_raw"] = cpu_raw

        # Memory
        mem_raw = ssh_exec("free -h | head -3")
        info["mem_raw"] = mem_raw

        # GPU (nvidia-smi)
        gpu_raw = ssh_exec(
            "nvidia-smi --query-gpu=index,name,utilization.gpu,"
            "memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits 2>/dev/null || echo 'NO_GPU'"
        )
        info["gpu_raw"] = gpu_raw

        # 正在运行的训练/Python 进程
        proc_raw = ssh_exec(
            "ps aux --sort=-%mem | "
            "grep -E 'python|train|eval' | "
            "grep -v grep | head -15"
        )
        info["proc_raw"] = proc_raw

        # 磁盘
        disk_raw = ssh_exec("df -h /home/wangchong/data 2>/dev/null | tail -1")
        info["disk_raw"] = disk_raw

        info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info["status"] = "connected"
    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)

    return info


def parse_gpu(raw):
    """解析 nvidia-smi CSV 输出为结构化列表。"""
    if not raw or "NO_GPU" in raw or "ERROR" in raw:
        return []
    gpus = []
    for line in raw.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "util": parts[2] + "%",
                "mem_used": parts[3] + " MiB",
                "mem_total": parts[4] + " MiB",
                "temp": parts[5] + "°C",
            })
    return gpus


def parse_processes(raw):
    """解析 ps aux 输出。"""
    procs = []
    if not raw or "ERROR" in raw:
        return procs
    for line in raw.strip().split("\n"):
        parts = line.split(None, 10)
        if len(parts) >= 11:
            procs.append({
                "user": parts[0],
                "pid": parts[1],
                "cpu": parts[2] + "%",
                "mem": parts[3] + "%",
                "command": parts[10][:120],
            })
    return procs


def fetch_task_progress():
    """采集 Innovation 4 v4 的训练与评估进度。"""
    progress = {
        "train": {
            "state": "unknown",
            "state_text": "未知",
            "epoch_current": 0,
            "epoch_total": 10,
            "step_current": 0,
            "step_total": 0,
            "percent": 0,
            "latest_val": [],
        },
        "eval": {
            "state": "unknown",
            "state_text": "未知",
            "pair_current": 0,
            "pair_total": 50,
            "percent": 0,
            "summary_metrics": {},
        },
        "pipeline_percent": 0,
    }

    train_proc = ssh_exec("ps aux | grep 'scripts/train_ae_v4.py' | grep -v grep")
    auto_eval_proc = ssh_exec("ps aux | grep 'auto_eval.sh' | grep -v grep")
    train_tail = ssh_exec(f"tail -180 {TRAIN_LOG} 2>/dev/null")
    train_vals = ssh_exec(f"grep -E '\\[Epoch [0-9]+\\] val_' {TRAIN_LOG} | tail -5 2>/dev/null")
    auto_eval_tail = ssh_exec(f"tail -220 {AUTO_EVAL_LOG} 2>/dev/null")
    summary_raw = ssh_exec(f"cat {EVAL_SUMMARY} 2>/dev/null")

    train_completed = "Training complete" in train_tail
    epoch_matches = re.findall(r"Epoch\s+(\d+):\s*(\d+)%\|.*?(\d+)/(\d+)", train_tail)
    if epoch_matches:
        ep, pct, cur, total = epoch_matches[-1]
        progress["train"].update({
            "epoch_current": int(ep),
            "step_current": int(cur),
            "step_total": int(total),
            "percent": int(pct),
        })

    if train_completed:
        progress["train"]["state"] = "completed"
        progress["train"]["state_text"] = "训练完成"
        progress["train"]["epoch_current"] = progress["train"]["epoch_total"]
        progress["train"]["percent"] = 100
    elif train_proc and "ERROR" not in train_proc:
        progress["train"]["state"] = "running"
        progress["train"]["state_text"] = "训练中"
    else:
        progress["train"]["state"] = "idle"
        progress["train"]["state_text"] = "未运行"

    if train_vals and "ERROR" not in train_vals:
        progress["train"]["latest_val"] = [x.strip() for x in train_vals.splitlines() if x.strip()]

    eval_completed = "Evaluation complete" in auto_eval_tail
    eval_matches = re.findall(r"Evaluating pairs:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)", auto_eval_tail)
    if eval_matches:
        pct, cur, total = eval_matches[-1]
        progress["eval"].update({
            "pair_current": int(cur),
            "pair_total": int(total),
            "percent": int(pct),
        })

    if summary_raw and "ERROR" not in summary_raw:
        try:
            summary_obj = json.loads(summary_raw)
            progress["eval"]["summary_metrics"] = summary_obj.get("metrics", {})
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
            progress["eval"]["percent"] = 100
        except Exception:
            pass

    if progress["eval"]["state"] != "completed":
        if eval_completed:
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
            progress["eval"]["percent"] = 100
        elif "Starting evaluation" in auto_eval_tail or "Evaluating pairs" in auto_eval_tail:
            progress["eval"]["state"] = "running"
            progress["eval"]["state_text"] = "评估中"
        elif auto_eval_proc and "ERROR" not in auto_eval_proc:
            progress["eval"]["state"] = "waiting"
            progress["eval"]["state_text"] = "等待评估启动"
        else:
            progress["eval"]["state"] = "idle"
            progress["eval"]["state_text"] = "未运行"

    done_steps = 0
    if progress["train"]["state"] == "completed":
        done_steps += 1
    if progress["eval"]["state"] in ("running", "completed"):
        done_steps += 1
    if progress["eval"]["state"] == "completed":
        done_steps += 1
    progress["pipeline_percent"] = int(done_steps / 3 * 100)

    return progress


def background_refresh():
    """后台线程定时刷新服务器数据。"""
    while True:
        try:
            info = fetch_server_info()
            task_progress = fetch_task_progress()
            with _cache_lock:
                _cache["server_info"] = info
                _cache["gpu_info"] = parse_gpu(info.get("gpu_raw", ""))
                _cache["processes"] = parse_processes(info.get("proc_raw", ""))
                _cache["task_progress"] = task_progress
                _cache["last_update"] = info.get("timestamp")
                _cache["error"] = None
        except Exception as e:
            with _cache_lock:
                _cache["error"] = str(e)
        time.sleep(15)


# ─── 实验数据 ────────────────────────────────────────────────────

REFERENCE_METRICS = {
    "baseline_v2": {
        "overall_ssim": 0.9015, "overall_psnr": 25.9243, "overall_mae": 0.0288,
        "hippocampus_ssim": 0.8199, "hippocampus_mae": 0.0604,
        "roi_ssim": 0.7983, "roi_mae": 0.0625,
    },
    "innovation_4_v1": {
        "overall_ssim": 0.9081, "overall_psnr": 26.0283, "overall_mae": 0.0308,
        "hippocampus_ssim": 0.8301, "hippocampus_mae": 0.0656,
        "roi_ssim": 0.8184, "roi_mae": 0.0670,
    },
    "innovation_5_v2": {
        "overall_ssim": 0.9145, "overall_psnr": 26.2282, "overall_mae": 0.0289,
        "hippocampus_ssim": 0.8319, "hippocampus_mae": 0.0723,
        "roi_ssim": 0.8141, "roi_mae": 0.0755,
    },
}

# 代码修改历史记录
CODE_CHANGES = [
    {
        "time": "2026-04-09 22:00",
        "file": "train_ae_v3.py",
        "change": "冻结 Encoder，仅微调 Decoder + post_quant_conv",
        "reason": "保持潜空间不变，确保 Diffusion/ControlNet 兼容",
        "result": "SSIM ↑0.73%, MAE ↓6.9% (待改进)",
    },
    {
        "time": "2026-04-09 23:30",
        "file": "medicalnet_perceptual_v2.py",
        "change": "多尺度特征提取 (layers 1-4)，改用 L1 距离，下采样 80×96×80",
        "reason": "v1 仅用最终层特征且 L2 距离不够鲁棒",
        "result": "roi_ssim ↑2.52% vs baseline",
    },
    {
        "time": "2026-04-10 (进行中)",
        "file": "train_ae_v4.py",
        "change": "降低 freq_weight、增加 epoch、加入 MAE 约束",
        "reason": "v3 的频率损失权重过高导致 MAE 退步",
        "result": "测试中...",
    },
]

# ─── HTML 模板 ────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>BrLP MCI 研究项目 — 监控面板</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg: #0f172a; --card: #1e293b; --border: #334155;
          --text: #e2e8f0; --dim: #94a3b8; --blue: #38bdf8;
          --green: #4ade80; --red: #f87171; --yellow: #fbbf24; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: var(--text); }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 16px; }

  /* Header */
  header { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
  header h1 { font-size: 1.5em; color: var(--blue); }
  .conn-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .conn-ok { background: #064e3b; color: var(--green); }
  .conn-err { background: #7f1d1d; color: var(--red); }
  .ts { color: var(--dim); font-size: 0.8em; margin-left: auto; }

  /* Grid */
  .grid { display: grid; gap: 14px; }
  .g2 { grid-template-columns: 1fr 1fr; }
  .g3 { grid-template-columns: 1fr 1fr 1fr; }
  @media(max-width:900px){ .g2,.g3 { grid-template-columns: 1fr; } }

  /* Card */
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 14px; }
  .card h2 { font-size: 1em; color: var(--blue); margin-bottom: 10px;
             border-bottom: 1px solid var(--border); padding-bottom: 6px; }

  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--dim); font-weight: 600; }
  .up { color: var(--green); font-weight: 600; }
  .down { color: var(--red); font-weight: 600; }
  .same { color: var(--dim); }

  /* GPU bars */
  .bar-outer { background: #334155; border-radius: 4px; height: 18px; position: relative; }
  .bar-inner { border-radius: 4px; height: 100%; }
  .bar-label { position: absolute; top: 0; left: 6px; font-size: 0.75em;
               line-height: 18px; color: #fff; font-weight: 600; }

  /* Process list */
  .proc { font-family: 'Cascadia Code', Consolas, monospace; font-size: 0.8em;
          color: var(--dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Change log */
  .change { background: var(--bg); border-radius: 6px; padding: 10px 12px;
            margin-bottom: 8px; font-size: 0.85em; border-left: 3px solid var(--blue); }
  .change .meta { color: var(--dim); font-size: 0.8em; margin-bottom: 4px; }
  .change .reason { color: var(--yellow); }
  .change .result { color: var(--green); }

  pre { background: var(--bg); padding: 10px; border-radius: 6px;
        font-size: 0.8em; overflow-x: auto; color: var(--dim);
        white-space: pre-wrap; word-break: break-all; }

  .refresh-btn { background: var(--border); color: var(--text); border: none;
                 padding: 6px 14px; border-radius: 6px; cursor: pointer; }
  .refresh-btn:hover { background: #475569; }

  .tabs { display: flex; gap: 4px; margin-bottom: 12px; }
  .tab { padding: 6px 16px; border-radius: 6px 6px 0 0; cursor: pointer;
         background: var(--border); color: var(--dim); font-size: 0.9em; }
  .tab.active { background: var(--card); color: var(--blue); font-weight: 600; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>BrLP MCI 研究监控面板</h1>
  <span class="conn-badge {{ 'conn-ok' if connected else 'conn-err' }}">
    {{ '已连接' if connected else '连接失败' }} · {{ server_host }}:{{ server_port }}
  </span>
  <span class="ts" id="update-time">{{ last_update or '---' }}</span>
  <button class="refresh-btn" onclick="location.reload()">刷新</button>
</header>

<!-- ===== 任务进度（实时） ===== -->
<div class="card" style="margin-bottom:14px;">
  <h2>任务进度 (Innovation 4 v4)</h2>
  <div style="margin-bottom:10px; font-size:0.9em; color:var(--dim);">
    流程进度: <span id="pipeline-percent">{{ task_progress.pipeline_percent if task_progress else 0 }}</span>%
  </div>

  <div style="margin-bottom:8px;"><strong>训练状态:</strong> <span id="train-state">{{ task_progress.train.state_text if task_progress else '未知' }}</span></div>
  <div class="bar-outer" style="margin-bottom:8px;">
    <div id="train-bar" class="bar-inner" style="width:{{ task_progress.train.percent if task_progress else 0 }}%; background:var(--blue);"></div>
    <span id="train-label" class="bar-label">
      Epoch {{ task_progress.train.epoch_current if task_progress else 0 }}/{{ task_progress.train.epoch_total if task_progress else 10 }} · {{ task_progress.train.percent if task_progress else 0 }}%
    </span>
  </div>

  <div style="margin-bottom:8px;"><strong>评估状态:</strong> <span id="eval-state">{{ task_progress.eval.state_text if task_progress else '未知' }}</span></div>
  <div class="bar-outer" style="margin-bottom:8px;">
    <div id="eval-bar" class="bar-inner" style="width:{{ task_progress.eval.percent if task_progress else 0 }}%; background:var(--green);"></div>
    <span id="eval-label" class="bar-label">
      Pairs {{ task_progress.eval.pair_current if task_progress else 0 }}/{{ task_progress.eval.pair_total if task_progress else 50 }} · {{ task_progress.eval.percent if task_progress else 0 }}%
    </span>
  </div>

  <div style="font-size:0.85em; color:var(--dim); margin-top:8px;">最近验证指标:</div>
  <pre id="latest-val-box">{% if task_progress and task_progress.train.latest_val %}{{ task_progress.train.latest_val|join('\n') }}{% else %}暂无{% endif %}</pre>
</div>

<!-- ===== 服务器状态 ===== -->
<div class="grid g3" style="margin-bottom:14px;">

  <div class="card">
    <h2>CPU / 负载</h2>
    <pre id="cpu-info">{{ cpu_raw or '加载中...' }}</pre>
  </div>

  <div class="card">
    <h2>内存</h2>
    <pre id="mem-info">{{ mem_raw or '加载中...' }}</pre>
    <div style="margin-top:8px;">
      <strong style="font-size:0.85em;">磁盘 (data):</strong>
      <span id="disk-info" style="font-size:0.85em; color:var(--dim);">{{ disk_raw or 'N/A' }}</span>
    </div>
  </div>

  <div class="card">
    <h2>GPU</h2>
    <div id="gpu-box">
      {% if gpus %}
        {% for g in gpus %}
        <div style="margin-bottom:8px;">
          <div style="font-size:0.85em; margin-bottom:2px;">
            <strong>GPU {{ g.index }}</strong>: {{ g.name }} · {{ g.temp }}
          </div>
          <div class="bar-outer">
            <div class="bar-inner" style="width:{{ g.util }}; background: {% if g.util|replace('%','')|int > 80 %}var(--red){% elif g.util|replace('%','')|int > 40 %}var(--yellow){% else %}var(--green){% endif %};"></div>
            <span class="bar-label">利用率 {{ g.util }}</span>
          </div>
          <div style="font-size:0.8em; color:var(--dim); margin-top:2px;">
            显存: {{ g.mem_used }} / {{ g.mem_total }}
          </div>
        </div>
        {% endfor %}
      {% else %}
        <pre>{{ gpu_raw or '无 GPU 信息' }}</pre>
      {% endif %}
    </div>
  </div>

</div>

<!-- ===== 运行进程 ===== -->
<div class="card" style="margin-bottom:14px;">
  <h2>正在运行的程序 (Python / train / eval)</h2>
  {% if processes %}
  <table>
    <thead><tr><th>PID</th><th>CPU</th><th>MEM</th><th>命令</th></tr></thead>
    <tbody id="process-body">
    {% for p in processes %}
    <tr>
      <td>{{ p.pid }}</td>
      <td>{{ p.cpu }}</td>
      <td>{{ p.mem }}</td>
      <td class="proc" title="{{ p.command }}">{{ p.command }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
    <p style="color:var(--dim); font-size:0.9em;">当前没有检测到训练/评估进程</p>
  {% endif %}
</div>

<!-- ===== 标签页: 实验结果 / 代码修改 ===== -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('metrics')">实验指标对比</div>
  <div class="tab" onclick="switchTab('changes')">代码修改记录</div>
</div>

<div id="tab-metrics" class="tab-content active">
  <div class="card">
    <h2>公平对比表 (MCI 纵向预测, test=20)</h2>
    <table>
      <thead>
        <tr>
          <th>指标</th><th>Baseline</th><th>Innov4 v1</th>
          <th>Innov4 v2 (最新)</th><th>Innov5</th><th>v2 vs Baseline</th>
        </tr>
      </thead>
      <tbody>
      {% for m in metrics_table %}
        <tr>
          <td><strong>{{ m.name }}</strong></td>
          <td>{{ "%.4f"|format(m.bl) }}</td>
          <td>{{ "%.4f"|format(m.v1) }}</td>
          <td>{{ "%.4f"|format(m.v2) if m.v2 else 'Pending' }}</td>
          <td>{{ "%.4f"|format(m.i5) }}</td>
          <td class="{{ m.cls }}">{{ m.delta }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<div id="tab-changes" class="tab-content">
  <div class="card">
    <h2>代码修改历史</h2>
    {% for c in changes %}
    <div class="change">
      <div class="meta">{{ c.time }} · {{ c.file }}</div>
      <div><strong>修改:</strong> {{ c.change }}</div>
      <div class="reason"><strong>原因:</strong> {{ c.reason }}</div>
      <div class="result"><strong>结果:</strong> {{ c.result }}</div>
    </div>
    {% endfor %}
  </div>
</div>

</div><!-- .wrap -->

<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function renderGpu(gpus, gpuRaw) {
  const box = document.getElementById('gpu-box');
  if (!box) return;
  if (!gpus || gpus.length === 0) {
    box.innerHTML = '<pre>' + (gpuRaw || '无 GPU 信息') + '</pre>';
    return;
  }

  const html = gpus.map(g => {
    const utilNum = parseInt(String(g.util).replace('%', ''), 10) || 0;
    const color = utilNum > 80 ? 'var(--red)' : (utilNum > 40 ? 'var(--yellow)' : 'var(--green)');
    return `
      <div style="margin-bottom:8px;">
        <div style="font-size:0.85em; margin-bottom:2px;"><strong>GPU ${g.index}</strong>: ${g.name} · ${g.temp}</div>
        <div class="bar-outer">
          <div class="bar-inner" style="width:${g.util}; background:${color};"></div>
          <span class="bar-label">利用率 ${g.util}</span>
        </div>
        <div style="font-size:0.8em; color:var(--dim); margin-top:2px;">显存: ${g.mem_used} / ${g.mem_total}</div>
      </div>
    `;
  }).join('');

  box.innerHTML = html;
}

function renderProcesses(processes) {
  const tbody = document.getElementById('process-body');
  if (!tbody) return;
  if (!processes || processes.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="color:var(--dim);">当前没有检测到训练/评估进程</td></tr>';
    return;
  }
  tbody.innerHTML = processes.map(p =>
    `<tr><td>${p.pid}</td><td>${p.cpu}</td><td>${p.mem}</td><td class="proc" title="${p.command}">${p.command}</td></tr>`
  ).join('');
}

function renderTaskProgress(task) {
  if (!task) return;
  const train = task.train || {};
  const evalp = task.eval || {};

  const pipeline = document.getElementById('pipeline-percent');
  if (pipeline) pipeline.textContent = task.pipeline_percent || 0;

  const trainState = document.getElementById('train-state');
  if (trainState) trainState.textContent = train.state_text || '未知';
  const trainBar = document.getElementById('train-bar');
  if (trainBar) trainBar.style.width = String(train.percent || 0) + '%';
  const trainLabel = document.getElementById('train-label');
  if (trainLabel) {
    trainLabel.textContent = `Epoch ${train.epoch_current || 0}/${train.epoch_total || 10} · ${train.percent || 0}%`;
  }

  const evalState = document.getElementById('eval-state');
  if (evalState) evalState.textContent = evalp.state_text || '未知';
  const evalBar = document.getElementById('eval-bar');
  if (evalBar) evalBar.style.width = String(evalp.percent || 0) + '%';
  const evalLabel = document.getElementById('eval-label');
  if (evalLabel) {
    evalLabel.textContent = `Pairs ${evalp.pair_current || 0}/${evalp.pair_total || 50} · ${evalp.percent || 0}%`;
  }

  const latestVal = document.getElementById('latest-val-box');
  if (latestVal) {
    const lines = (train.latest_val && train.latest_val.length) ? train.latest_val.join('\n') : '暂无';
    latestVal.textContent = lines;
  }
}

function tickRefresh() {
  fetch('/api/refresh')
    .then(r => r.json())
    .then(d => {
      if (d.cpu_raw) document.getElementById('cpu-info').textContent = d.cpu_raw;
      if (d.mem_raw) document.getElementById('mem-info').textContent = d.mem_raw;
      if (d.last_update) document.getElementById('update-time').textContent = d.last_update;
      if (d.disk_raw) document.getElementById('disk-info').textContent = d.disk_raw;

      renderGpu(d.gpus || [], d.gpu_raw || '');
      renderProcesses(d.processes || []);
      renderTaskProgress(d.task_progress || null);
    })
    .catch(() => {});
}

tickRefresh();
setInterval(tickRefresh, 8000);
</script>
</body>
</html>
"""

# ─── 路由 ─────────────────────────────────────────────────────────

def build_metrics_table():
    bl = REFERENCE_METRICS["baseline_v2"]
    v1 = REFERENCE_METRICS["innovation_4_v1"]
    i5 = REFERENCE_METRICS["innovation_5_v2"]
    keys = [
        ("overall_ssim",     "Overall SSIM ↑",      True),
        ("overall_psnr",     "Overall PSNR ↑",      True),
        ("overall_mae",      "Overall MAE ↓",       False),
        ("hippocampus_ssim", "Hippocampus SSIM ↑",  True),
        ("hippocampus_mae",  "Hippocampus MAE ↓",   False),
        ("roi_ssim",         "ROI SSIM ↑",          True),
        ("roi_mae",          "ROI MAE ↓",           False),
    ]
    rows = []
    for key, name, higher_better in keys:
        b, v, i = bl[key], v1[key], i5[key]
        diff = v - b
        if higher_better:
            cls = "up" if diff > 0 else "down"
            sign = "+" if diff > 0 else ""
        else:
            cls = "up" if diff < 0 else "down"
            sign = "" if diff < 0 else "+"
        pct = abs(diff / b * 100) if b else 0
        delta_str = f"{sign}{diff:.4f} ({pct:.2f}%)"
        rows.append({"name": name, "bl": b, "v1": v, "v2": v, "i5": i,
                      "cls": cls, "delta": delta_str})
    return rows


@app.route("/")
def index():
    with _cache_lock:
        info = _cache["server_info"] or {}
    task_progress = _cache["task_progress"] or {}
    connected = info.get("status") == "connected"
    gpus = parse_gpu(info.get("gpu_raw", ""))
    procs = parse_processes(info.get("proc_raw", ""))

    return render_template_string(
        HTML,
        connected=connected,
        server_host=SERVER_HOST,
        server_port=SERVER_PORT,
        last_update=info.get("timestamp"),
        cpu_raw=info.get("cpu_raw", "加载中..."),
        mem_raw=info.get("mem_raw", "加载中..."),
        gpu_raw=info.get("gpu_raw", ""),
        disk_raw=info.get("disk_raw", "N/A"),
        gpus=gpus,
        processes=procs,
        task_progress=task_progress,
        metrics_table=build_metrics_table(),
        changes=CODE_CHANGES,
    )


@app.route("/api/refresh")
def api_refresh():
    with _cache_lock:
        info = _cache["server_info"] or {}
    task_progress = _cache["task_progress"] or {}
    return jsonify({
        "cpu_raw": info.get("cpu_raw", ""),
        "mem_raw": info.get("mem_raw", ""),
        "gpu_raw": info.get("gpu_raw", ""),
    "gpus": parse_gpu(info.get("gpu_raw", "")),
        "disk_raw": info.get("disk_raw", ""),
        "last_update": info.get("timestamp", ""),
        "processes": parse_processes(info.get("proc_raw", "")),
    "task_progress": task_progress,
    })


@app.route("/api/server_info")
def api_server_info():
    """完整的服务器信息 JSON。"""
    with _cache_lock:
        info = dict(_cache["server_info"] or {})
    info["gpus"] = parse_gpu(info.get("gpu_raw", ""))
    info["processes"] = parse_processes(info.get("proc_raw", ""))
    return jsonify(info)


# ─── 入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # 启动后台刷新线程
    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()

    # 先做一次立即刷新
    print(f"[Dashboard] 正在连接服务器 {SERVER_HOST}:{SERVER_PORT} ...")
    try:
        info = fetch_server_info()
        task_progress = fetch_task_progress()
        with _cache_lock:
            _cache["server_info"] = info
            _cache["gpu_info"] = parse_gpu(info.get("gpu_raw", ""))
            _cache["processes"] = parse_processes(info.get("proc_raw", ""))
            _cache["task_progress"] = task_progress
            _cache["last_update"] = info.get("timestamp")
        print(f"[Dashboard] 服务器连接成功")
    except Exception as e:
        print(f"[Dashboard] 服务器连接失败: {e}")

    print(f"[Dashboard] 启动中: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
