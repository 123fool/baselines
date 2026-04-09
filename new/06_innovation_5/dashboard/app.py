"""
Real-time Monitoring Dashboard for Innovation 5.

Features:
  - Server CPU / Memory / GPU usage (real-time)
  - Running processes
  - Code modification changelog
  - Training progress
  
Usage:
    python app.py --port 8501 --changelog /path/to/changelog.json
    
Then open http://<server-ip>:8501 in browser.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

CHANGELOG_PATH = None
TRAINING_LOG_DIR = None

# ============================================================
# HTML Template (embedded for single-file deployment)
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Innovation 5 - 实时监控面板</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: #0f1923;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a2a3a 0%, #0d1b2a 100%);
            padding: 20px 30px;
            border-bottom: 2px solid #00d4ff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 1.5em;
            color: #00d4ff;
        }
        .header .status {
            font-size: 0.9em;
            color: #8ab4f8;
        }
        .header .status .dot {
            display: inline-block;
            width: 8px; height: 8px;
            background: #4caf50;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .panel {
            background: #1a2a3a;
            border: 1px solid #2a3a4a;
            border-radius: 8px;
            padding: 20px;
        }
        .panel h2 {
            font-size: 1.1em;
            color: #00d4ff;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a3a4a;
        }
        .panel.full-width {
            grid-column: 1 / -1;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #1e2e3e;
        }
        .metric-label { color: #8ab4f8; font-size: 0.9em; }
        .metric-value { font-weight: bold; font-size: 1.1em; }
        .progress-bar {
            width: 200px;
            height: 8px;
            background: #0d1b2a;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 4px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .progress-fill.cpu { background: linear-gradient(90deg, #4caf50, #ff9800); }
        .progress-fill.mem { background: linear-gradient(90deg, #2196f3, #e91e63); }
        .progress-fill.gpu { background: linear-gradient(90deg, #9c27b0, #ff5722); }
        
        .process-table {
            width: 100%;
            font-size: 0.85em;
        }
        .process-table th {
            text-align: left;
            color: #8ab4f8;
            padding: 6px 8px;
            border-bottom: 1px solid #2a3a4a;
        }
        .process-table td {
            padding: 5px 8px;
            border-bottom: 1px solid #1e2e3e;
        }
        
        .changelog-item {
            padding: 12px;
            margin-bottom: 10px;
            background: #0d1b2a;
            border-radius: 6px;
            border-left: 3px solid #00d4ff;
        }
        .changelog-item .time {
            font-size: 0.8em;
            color: #607d8b;
            margin-bottom: 4px;
        }
        .changelog-item .action {
            font-size: 0.9em;
            color: #4caf50;
            font-weight: bold;
        }
        .changelog-item .desc {
            font-size: 0.85em;
            color: #b0bec5;
            margin-top: 4px;
        }
        .changelog-item .params {
            font-size: 0.8em;
            color: #78909c;
            margin-top: 6px;
            font-family: monospace;
            background: #152535;
            padding: 6px;
            border-radius: 4px;
        }
        
        .gpu-card {
            background: #0d1b2a;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
        }
        .gpu-card .name {
            color: #9c27b0;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .refresh-info {
            text-align: center;
            color: #546e7a;
            font-size: 0.8em;
            padding: 10px;
        }
        
        .tab-bar {
            display: flex;
            gap: 2px;
            margin-bottom: 15px;
        }
        .tab {
            padding: 8px 16px;
            background: #0d1b2a;
            color: #8ab4f8;
            border: none;
            cursor: pointer;
            border-radius: 4px 4px 0 0;
            font-size: 0.9em;
        }
        .tab.active {
            background: #1a2a3a;
            color: #00d4ff;
            border-bottom: 2px solid #00d4ff;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .code-diff {
            background: #0d1b2a;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Cascadia Code', 'Fira Code', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            overflow-x: auto;
            line-height: 1.6;
        }
        .code-diff .added { color: #4caf50; }
        .code-diff .removed { color: #f44336; }
        .code-diff .file { color: #ff9800; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Innovation 5: 海马体区域注意力加权 — 实时监控</h1>
        <div class="status">
            <span class="dot"></span>
            <span id="status-text">连接中...</span>
            <span style="margin-left:15px" id="clock"></span>
        </div>
    </div>

    <div class="container">
        <!-- System Metrics -->
        <div class="panel">
            <h2>📊 系统资源</h2>
            <div id="system-metrics">
                <div class="metric-row">
                    <span class="metric-label">CPU 使用率</span>
                    <div>
                        <span class="metric-value" id="cpu-pct">--</span>
                        <div class="progress-bar"><div class="progress-fill cpu" id="cpu-bar" style="width:0%"></div></div>
                    </div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">内存使用</span>
                    <div>
                        <span class="metric-value" id="mem-pct">--</span>
                        <div class="progress-bar"><div class="progress-fill mem" id="mem-bar" style="width:0%"></div></div>
                    </div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">内存详情</span>
                    <span class="metric-value" id="mem-detail" style="font-size:0.85em">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">磁盘使用</span>
                    <span class="metric-value" id="disk-info" style="font-size:0.85em">--</span>
                </div>
            </div>
        </div>

        <!-- GPU Metrics -->
        <div class="panel">
            <h2>🎮 GPU 状态</h2>
            <div id="gpu-metrics">
                <p style="color:#546e7a">加载中...</p>
            </div>
        </div>

        <!-- Running Processes -->
        <div class="panel full-width">
            <h2>⚙️ 运行中的训练/Python 进程</h2>
            <table class="process-table">
                <thead>
                    <tr>
                        <th>PID</th>
                        <th>用户</th>
                        <th>CPU%</th>
                        <th>MEM%</th>
                        <th>运行时间</th>
                        <th>命令</th>
                    </tr>
                </thead>
                <tbody id="process-list">
                    <tr><td colspan="6" style="color:#546e7a">加载中...</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Changelog -->
        <div class="panel full-width">
            <h2>📝 代码修改日志</h2>
            <div class="tab-bar">
                <button class="tab active" onclick="switchTab('changelog')">修改记录</button>
                <button class="tab" onclick="switchTab('changes-detail')">修改详情</button>
                <button class="tab" onclick="switchTab('results')">评估结果</button>
            </div>
            
            <div id="changelog" class="tab-content active">
                <div id="changelog-list">
                    <p style="color:#546e7a">加载中...</p>
                </div>
            </div>
            
            <div id="changes-detail" class="tab-content">
                <div class="code-diff">
<span class="file">## Innovation 5: 海马体区域注意力加权</span>

<span class="file">--- 修改文件 1: train_autoencoder.py</span>
<span class="removed">- rec_loss = l1_loss_fn(reconstruction.float(), images.float())</span>
<span class="added">+ # 使用区域加权 L1 损失，海马体+杏仁核 3x 权重</span>
<span class="added">+ rec_loss = region_loss_fn(reconstruction.float(), images.float(), weight_batch)</span>

<span class="file">--- 修改文件 2: train_controlnet.py</span>
<span class="removed">- loss = F.mse_loss(noise_pred.float(), noise.float())</span>
<span class="added">+ # 潜空间区域加权 MSE，海马体区域权重更高</span>
<span class="added">+ loss = region_mse_fn(noise_pred.float(), noise.float(), weight_batch)</span>

<span class="file">--- 新增文件: region_weights.py</span>
<span class="added">+ 从 SynthSeg 分割生成区域权重图</span>
<span class="added">+ 支持图像空间 (AE) 和潜空间 (ControlNet) 两种模式</span>
<span class="added">+ 海马体(17,53) + 杏仁核(18,54) 区域 ROI 权重</span>

<span class="file">--- 新增文件: weighted_losses.py</span>
<span class="added">+ RegionWeightedL1Loss: 空间加权 L1</span>
<span class="added">+ RegionWeightedMSELoss: 空间加权 MSE</span>
<span class="added">+ CombinedRegionLoss: alpha 混合均匀损失 + 区域损失</span>

<span class="file">--- 新增文件: evaluate_regional.py</span>
<span class="added">+ 海马体区域 SSIM / MAE</span>
<span class="added">+ 杏仁核区域指标</span>
<span class="added">+ 体积误差评估</span>
                </div>
                
                <h3 style="color:#00d4ff;margin:15px 0 10px">修改原因</h3>
                <div style="padding:10px;background:#0d1b2a;border-radius:6px;font-size:0.9em;line-height:1.8">
                    <p><strong>问题：</strong>BrLP 的损失函数对所有脑区一视同仁，但 MCI→AD 转化的关键标志物是海马体萎缩。</p>
                    <p><strong>方案：</strong>利用 SynthSeg 已有的分割结果，构建区域权重图，使模型训练时更关注海马体+杏仁核区域的重建/预测质量。</p>
                    <p><strong>预期：</strong>海马体区域的 MAE 下降 10-20%，对 MCI 纵向预测的临床意义更大。</p>
                    <p><strong>风险：</strong>极低 — 不改变网络结构，仅修改损失计算逻辑，与原模型完全兼容。</p>
                </div>
            </div>
            
            <div id="results" class="tab-content">
                <div id="results-content">
                    <p style="color:#546e7a">等待评估完成后显示结果...</p>
                </div>
            </div>
        </div>
    </div>

    <div class="refresh-info">每 3 秒自动刷新 | Innovation 5 监控面板</div>

    <script>
        // Tab switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }

        // Clock
        function updateClock() {
            const now = new Date();
            document.getElementById('clock').textContent = now.toLocaleString('zh-CN');
        }
        setInterval(updateClock, 1000);
        updateClock();

        // Fetch system metrics
        async function fetchMetrics() {
            try {
                const resp = await fetch('/api/metrics');
                const data = await resp.json();
                
                document.getElementById('status-text').textContent = '在线';

                // CPU
                document.getElementById('cpu-pct').textContent = data.cpu_percent + '%';
                document.getElementById('cpu-bar').style.width = data.cpu_percent + '%';

                // Memory
                document.getElementById('mem-pct').textContent = data.memory.percent + '%';
                document.getElementById('mem-bar').style.width = data.memory.percent + '%';
                document.getElementById('mem-detail').textContent = 
                    `${data.memory.used_gb} / ${data.memory.total_gb} GB`;

                // Disk
                if (data.disk) {
                    document.getElementById('disk-info').textContent =
                        `${data.disk.used_gb} / ${data.disk.total_gb} GB (${data.disk.percent}%)`;
                }

                // GPU
                const gpuDiv = document.getElementById('gpu-metrics');
                if (data.gpus && data.gpus.length > 0) {
                    gpuDiv.innerHTML = data.gpus.map((g, i) => `
                        <div class="gpu-card">
                            <div class="name">GPU ${i}: ${g.name}</div>
                            <div class="metric-row">
                                <span class="metric-label">显存</span>
                                <div>
                                    <span class="metric-value">${g.memory_used_mb} / ${g.memory_total_mb} MB</span>
                                    <div class="progress-bar"><div class="progress-fill gpu" style="width:${g.memory_percent}%"></div></div>
                                </div>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">GPU 利用率</span>
                                <span class="metric-value">${g.utilization}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">温度</span>
                                <span class="metric-value">${g.temperature}°C</span>
                            </div>
                        </div>
                    `).join('');
                } else {
                    gpuDiv.innerHTML = '<p style="color:#546e7a">未检测到 GPU 或 nvidia-smi 不可用</p>';
                }

                // Processes
                const tbody = document.getElementById('process-list');
                if (data.processes && data.processes.length > 0) {
                    tbody.innerHTML = data.processes.map(p => `
                        <tr>
                            <td>${p.pid}</td>
                            <td>${p.user}</td>
                            <td>${p.cpu_percent}%</td>
                            <td>${p.mem_percent}%</td>
                            <td>${p.elapsed}</td>
                            <td style="max-width:500px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.command}</td>
                        </tr>
                    `).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="6" style="color:#546e7a">无 Python 训练进程</td></tr>';
                }

            } catch (e) {
                document.getElementById('status-text').textContent = '连接失败';
            }
        }

        // Fetch changelog
        async function fetchChangelog() {
            try {
                const resp = await fetch('/api/changelog');
                const data = await resp.json();
                
                const div = document.getElementById('changelog-list');
                if (data.length === 0) {
                    div.innerHTML = '<p style="color:#546e7a">暂无修改记录</p>';
                    return;
                }
                
                div.innerHTML = data.slice().reverse().map(item => `
                    <div class="changelog-item">
                        <div class="time">${item.timestamp || ''}</div>
                        <div class="action">${item.action || ''}</div>
                        <div class="desc">${item.description || item.result || ''}</div>
                        ${item.params ? '<div class="params">' + JSON.stringify(item.params, null, 2) + '</div>' : ''}
                        ${item.metrics_summary ? '<div class="params">' + JSON.stringify(item.metrics_summary, null, 2) + '</div>' : ''}
                    </div>
                `).join('');
            } catch (e) {
                console.error('Changelog fetch failed:', e);
            }
        }

        // Fetch evaluation results
        async function fetchResults() {
            try {
                const resp = await fetch('/api/results');
                const data = await resp.json();
                
                const div = document.getElementById('results-content');
                if (!data || Object.keys(data).length === 0) {
                    div.innerHTML = '<p style="color:#546e7a">等待评估完成后显示结果...</p>';
                    return;
                }
                
                let html = '';
                for (const [model, metrics] of Object.entries(data)) {
                    html += `<h3 style="color:#00d4ff;margin:10px 0">${model}</h3>`;
                    html += '<table class="process-table"><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>';
                    for (const [k, v] of Object.entries(metrics)) {
                        html += `<tr><td>${k}</td><td>${v}</td></tr>`;
                    }
                    html += '</tbody></table>';
                }
                div.innerHTML = html;
            } catch (e) {
                console.error('Results fetch failed:', e);
            }
        }

        // Auto-refresh
        setInterval(fetchMetrics, 3000);
        setInterval(fetchChangelog, 10000);
        setInterval(fetchResults, 15000);
        fetchMetrics();
        fetchChangelog();
        fetchResults();
    </script>
</body>
</html>
"""


# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/metrics')
def api_metrics():
    """Return system metrics as JSON."""
    import psutil
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.5)
    
    # Memory
    mem = psutil.virtual_memory()
    memory_info = {
        'percent': mem.percent,
        'total_gb': f'{mem.total / (1024**3):.1f}',
        'used_gb': f'{mem.used / (1024**3):.1f}',
        'available_gb': f'{mem.available / (1024**3):.1f}',
    }
    
    # Disk
    try:
        disk = psutil.disk_usage('/')
        disk_info = {
            'percent': disk.percent,
            'total_gb': f'{disk.total / (1024**3):.0f}',
            'used_gb': f'{disk.used / (1024**3):.0f}',
        }
    except Exception:
        disk_info = None
    
    # GPU (via nvidia-smi)
    gpus = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    gpus.append({
                        'name': parts[0],
                        'memory_used_mb': mem_used,
                        'memory_total_mb': mem_total,
                        'memory_percent': round(mem_used / max(mem_total, 1) * 100, 1),
                        'utilization': int(parts[3]),
                        'temperature': int(parts[4]),
                    })
    except Exception:
        pass
    
    # Python/training processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'cmdline', 'create_time']):
        try:
            info = proc.info
            cmdline = info.get('cmdline', [])
            if not cmdline:
                continue
            cmd_str = ' '.join(cmdline)
            # Show python processes and training-related ones
            if 'python' in info.get('name', '').lower() or 'train' in cmd_str.lower():
                elapsed = time.time() - info.get('create_time', time.time())
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                processes.append({
                    'pid': info['pid'],
                    'user': info.get('username', 'N/A'),
                    'cpu_percent': round(info.get('cpu_percent', 0), 1),
                    'mem_percent': round(info.get('memory_percent', 0), 1),
                    'elapsed': f'{hours}h{minutes:02d}m',
                    'command': cmd_str[:200],
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Sort by CPU usage
    processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
    
    return jsonify({
        'cpu_percent': cpu_percent,
        'memory': memory_info,
        'disk': disk_info,
        'gpus': gpus,
        'processes': processes[:20],
    })


@app.route('/api/changelog')
def api_changelog():
    """Return changelog entries."""
    if CHANGELOG_PATH and os.path.exists(CHANGELOG_PATH):
        with open(CHANGELOG_PATH, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])


@app.route('/api/results')
def api_results():
    """Return evaluation results summary."""
    results = {}
    if TRAINING_LOG_DIR:
        for fname in os.listdir(TRAINING_LOG_DIR) if os.path.exists(TRAINING_LOG_DIR) else []:
            if fname.startswith('summary_') and fname.endswith('.json'):
                fpath = os.path.join(TRAINING_LOG_DIR, fname)
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    results[data.get('model', fname)] = data.get('metrics', {})
    return jsonify(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Innovation 5 Monitoring Dashboard')
    parser.add_argument('--port', default=8501, type=int)
    parser.add_argument('--host', default='0.0.0.0', type=str)
    parser.add_argument('--changelog', default=None, type=str,
                        help='Path to changelog.json')
    parser.add_argument('--results_dir', default=None, type=str,
                        help='Directory containing evaluation result JSONs')
    args = parser.parse_args()
    
    CHANGELOG_PATH = args.changelog
    TRAINING_LOG_DIR = args.results_dir
    
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    print(f"Changelog: {args.changelog}")
    print(f"Results dir: {args.results_dir}")
    
    app.run(host=args.host, port=args.port, debug=False)
