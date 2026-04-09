"""
Real-time Monitoring Dashboard for Innovation 4: 3D Perceptual Loss + Frequency Constraint.

Features:
  - Server CPU / Memory / GPU usage (real-time)
  - Running processes
  - Code modification changelog
  - Training progress with loss curves
  
Usage:
    python app.py --port 8502 --changelog /path/to/changelog.json

Then open http://<server-ip>:8502 in browser.
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

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Innovation 4 - 3D感知损失+频域约束 监控面板</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: #0a1628;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a3e 0%, #0a1628 100%);
            padding: 20px 30px;
            border-bottom: 2px solid #7c4dff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.5em; color: #7c4dff; }
        .header .status { font-size: 0.9em; color: #b388ff; }
        .header .status .dot {
            display: inline-block; width: 8px; height: 8px;
            background: #4caf50; border-radius: 50%; margin-right: 5px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        .container {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 20px; padding: 20px;
            max-width: 1400px; margin: 0 auto;
        }
        .panel {
            background: #141e33; border: 1px solid #1e2e4a;
            border-radius: 8px; padding: 20px;
        }
        .panel h2 {
            font-size: 1.1em; color: #7c4dff; margin-bottom: 15px;
            padding-bottom: 8px; border-bottom: 1px solid #1e2e4a;
        }
        .panel.full-width { grid-column: 1 / -1; }
        .metric-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 0; border-bottom: 1px solid #0f1a2e;
        }
        .metric-label { color: #b388ff; font-size: 0.9em; }
        .metric-value { font-weight: bold; font-size: 1.1em; }
        .progress-bar {
            width: 200px; height: 8px; background: #0a1628;
            border-radius: 4px; overflow: hidden; margin-top: 4px;
        }
        .progress-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
        .progress-fill.cpu { background: linear-gradient(90deg, #4caf50, #ff9800); }
        .progress-fill.mem { background: linear-gradient(90deg, #2196f3, #e91e63); }
        .progress-fill.gpu { background: linear-gradient(90deg, #7c4dff, #ff5722); }
        .process-table { width: 100%; font-size: 0.85em; }
        .process-table th {
            text-align: left; color: #b388ff; padding: 6px 8px;
            border-bottom: 1px solid #1e2e4a;
        }
        .process-table td { padding: 5px 8px; border-bottom: 1px solid #0f1a2e; }
        .changelog-item {
            padding: 12px; margin-bottom: 10px;
            background: #0a1628; border-radius: 6px;
            border-left: 3px solid #7c4dff;
        }
        .changelog-item .time { font-size: 0.8em; color: #607d8b; margin-bottom: 4px; }
        .changelog-item .action { font-size: 0.9em; color: #4caf50; font-weight: bold; }
        .changelog-item .desc { font-size: 0.85em; color: #b0bec5; margin-top: 4px; }
        .changelog-item .params {
            font-size: 0.8em; color: #78909c; margin-top: 6px;
            font-family: monospace; background: #0f1a2e; padding: 6px; border-radius: 4px;
        }
        .gpu-card {
            background: #0a1628; border-radius: 6px; padding: 12px; margin-bottom: 8px;
        }
        .gpu-card .name { color: #7c4dff; font-weight: bold; font-size: 0.9em; }
        .refresh-info { text-align: center; color: #546e7a; font-size: 0.8em; padding: 10px; }
        .tab-bar { display: flex; gap: 2px; margin-bottom: 15px; }
        .tab {
            padding: 8px 16px; background: #0a1628; color: #b388ff;
            border: none; cursor: pointer; border-radius: 4px 4px 0 0; font-size: 0.9em;
        }
        .tab.active { background: #141e33; color: #7c4dff; border-bottom: 2px solid #7c4dff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .code-diff {
            background: #0a1628; border-radius: 6px; padding: 15px; margin: 10px 0;
            font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.85em;
            white-space: pre-wrap; overflow-x: auto; line-height: 1.6;
        }
        .code-diff .added { color: #4caf50; }
        .code-diff .removed { color: #f44336; }
        .code-diff .file { color: #ff9800; font-weight: bold; }
        .innovation-badge {
            display: inline-block; padding: 2px 8px; border-radius: 3px;
            background: #7c4dff; color: white; font-size: 0.75em; margin-left: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Innovation 4: 3D 感知损失 + 频域约束 — 实时监控</h1>
        <div class="status">
            <span class="dot"></span>
            <span id="status-text">连接中...</span>
            <span style="margin-left:15px" id="clock"></span>
        </div>
    </div>
    <div class="container">
        <div class="panel">
            <h2>系统资源</h2>
            <div id="system-metrics">
                <div class="metric-row">
                    <span class="metric-label">CPU 使用率</span>
                    <div><span class="metric-value" id="cpu-pct">--</span>
                    <div class="progress-bar"><div class="progress-fill cpu" id="cpu-bar" style="width:0%"></div></div></div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">内存使用</span>
                    <div><span class="metric-value" id="mem-pct">--</span>
                    <div class="progress-bar"><div class="progress-fill mem" id="mem-bar" style="width:0%"></div></div></div>
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
        <div class="panel">
            <h2>GPU 状态</h2>
            <div id="gpu-metrics"><p style="color:#546e7a">加载中...</p></div>
        </div>
        <div class="panel full-width">
            <h2>运行中的训练/Python 进程</h2>
            <table class="process-table">
                <thead><tr><th>PID</th><th>用户</th><th>CPU%</th><th>MEM%</th><th>运行时间</th><th>命令</th></tr></thead>
                <tbody id="process-list"><tr><td colspan="6" style="color:#546e7a">加载中...</td></tr></tbody>
            </table>
        </div>
        <div class="panel full-width">
            <h2>Innovation 4 日志</h2>
            <div class="tab-bar">
                <button class="tab active" onclick="switchTab('changelog')">修改记录</button>
                <button class="tab" onclick="switchTab('changes-detail')">修改详情</button>
                <button class="tab" onclick="switchTab('results')">评估结果</button>
            </div>
            <div id="changelog" class="tab-content active">
                <div id="changelog-list"><p style="color:#546e7a">加载中...</p></div>
            </div>
            <div id="changes-detail" class="tab-content">
                <div class="code-diff">
<span class="file">## Innovation 4: 3D 感知损失替换 + 频域约束</span>

<span class="file">--- 修改文件 1: train_autoencoder.py (AE 训练脚本)</span>
<span class="removed">- perc_loss_fn = PerceptualLoss(spatial_dims=3, network_type="squeeze",</span>
<span class="removed">-                              is_fake_3d=True, fake_3d_ratio=0.2)</span>
<span class="removed">- # 2D VGG squeeze net + 随机 20% 切片采样 → 对 3D 结构连续性约束有限</span>
<span class="added">+ perc_loss_fn = MedicalNet3DPerceptualLoss(</span>
<span class="added">+     pretrained_path="resnet_10_23dataset.pth"</span>
<span class="added">+ )  # 真 3D ResNet-10 特征匹配 (23 个医学影像数据集预训练)</span>
<span class="added">+ freq_loss_fn = LaplacianPyramidLoss(num_levels=3)</span>
<span class="added">+ # 多尺度频域约束: 分离高频/低频, 强制匹配脑沟回等高频纹理</span>

<span class="file">--- 新增文件: medicalnet_perceptual.py</span>
<span class="added">+ 3D ResNet-10 感知损失 (MedicalNet 预训练)</span>
<span class="added">+ 支持单通道输入，无尺寸限制</span>
<span class="added">+ 自动 z-score 归一化 + 特征 L2 归一化</span>

<span class="file">--- 新增文件: frequency_losses.py</span>
<span class="added">+ LaplacianPyramidLoss: 3D 拉普拉斯金字塔多尺度损失</span>
<span class="added">+ FFTFrequencyLoss: 3D FFT 频域一致性约束</span>
<span class="added">+ 缓解 AE 重建的过平滑问题，保留边界和高频纹理</span>
                </div>
                <h3 style="color:#7c4dff;margin:15px 0 10px">修改原因</h3>
                <div style="padding:10px;background:#0a1628;border-radius:6px;font-size:0.9em;line-height:1.8">
                    <p><strong>问题：</strong>BrLP 的 AE 使用 fake3D 感知损失 (2D VGG squeeze + 20% 切片采样)，对 3D 空间连续性约束不足，导致重建图像过度平滑，丢失脑沟回等高频细节。</p>
                    <p><strong>方案：</strong>替换为 MedicalNet ResNet-10 的真 3D 感知损失 (23 个医学影像数据集预训练)，并增加拉普拉斯金字塔频域约束强制保留高频纹理。</p>
                    <p><strong>预期：</strong>AE 重建 SSIM 提升 0.005-0.01，脑沟回清晰度显著改善。</p>
                    <p><strong>来源：</strong>3D MedDiffusion (IEEE TMI 2025) 的 MedicalNetPerceptual 模块。</p>
                </div>
            </div>
            <div id="results" class="tab-content">
                <div id="results-content"><p style="color:#546e7a">等待评估完成后显示结果...</p></div>
            </div>
        </div>
    </div>
    <div class="refresh-info">每 3 秒自动刷新 | Innovation 4 监控面板</div>
    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
        function updateClock() {
            document.getElementById('clock').textContent = new Date().toLocaleString('zh-CN');
        }
        setInterval(updateClock, 1000); updateClock();
        async function fetchMetrics() {
            try {
                const resp = await fetch('/api/metrics');
                const data = await resp.json();
                document.getElementById('status-text').textContent = '在线';
                document.getElementById('cpu-pct').textContent = data.cpu_percent + '%';
                document.getElementById('cpu-bar').style.width = data.cpu_percent + '%';
                document.getElementById('mem-pct').textContent = data.memory.percent + '%';
                document.getElementById('mem-bar').style.width = data.memory.percent + '%';
                document.getElementById('mem-detail').textContent = data.memory.used_gb+' / '+data.memory.total_gb+' GB';
                if(data.disk) document.getElementById('disk-info').textContent = data.disk.used_gb+' / '+data.disk.total_gb+' GB ('+data.disk.percent+'%)';
                const gpuDiv = document.getElementById('gpu-metrics');
                if(data.gpus && data.gpus.length>0) {
                    gpuDiv.innerHTML = data.gpus.map((g,i)=>`
                        <div class="gpu-card"><div class="name">GPU ${i}: ${g.name}</div>
                        <div class="metric-row"><span class="metric-label">显存</span>
                        <div><span class="metric-value">${g.memory_used_mb} / ${g.memory_total_mb} MB</span>
                        <div class="progress-bar"><div class="progress-fill gpu" style="width:${g.memory_percent}%"></div></div></div></div>
                        <div class="metric-row"><span class="metric-label">利用率</span><span class="metric-value">${g.utilization}%</span></div>
                        <div class="metric-row"><span class="metric-label">温度</span><span class="metric-value">${g.temperature}°C</span></div></div>
                    `).join('');
                } else { gpuDiv.innerHTML='<p style="color:#546e7a">未检测到 GPU</p>'; }
                const tbody = document.getElementById('process-list');
                if(data.processes && data.processes.length>0) {
                    tbody.innerHTML = data.processes.map(p=>`<tr><td>${p.pid}</td><td>${p.user}</td><td>${p.cpu_percent}%</td><td>${p.mem_percent}%</td><td>${p.elapsed}</td><td style="max-width:500px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.command}</td></tr>`).join('');
                } else { tbody.innerHTML='<tr><td colspan="6" style="color:#546e7a">无 Python 训练进程</td></tr>'; }
            } catch(e) { document.getElementById('status-text').textContent='连接失败'; }
        }
        async function fetchChangelog() {
            try {
                const resp = await fetch('/api/changelog');
                const data = await resp.json();
                const div = document.getElementById('changelog-list');
                if(!data.length){div.innerHTML='<p style="color:#546e7a">暂无修改记录</p>';return;}
                div.innerHTML = data.slice().reverse().map(item=>`
                    <div class="changelog-item"><div class="time">${item.timestamp||''}</div>
                    <div class="action">${item.action||''}</div>
                    <div class="desc">${item.description||item.result||''}</div>
                    ${item.params?'<div class="params">'+JSON.stringify(item.params,null,2)+'</div>':''}
                    ${item.metrics_summary?'<div class="params">'+JSON.stringify(item.metrics_summary,null,2)+'</div>':''}</div>
                `).join('');
            } catch(e){}
        }
        async function fetchResults() {
            try {
                const resp = await fetch('/api/results');
                const data = await resp.json();
                const div = document.getElementById('results-content');
                if(!data||!Object.keys(data).length){div.innerHTML='<p style="color:#546e7a">等待评估完成后显示结果...</p>';return;}
                let html='';
                for(const[model,metrics] of Object.entries(data)){
                    html+=`<h3 style="color:#7c4dff;margin:10px 0">${model}</h3>`;
                    html+='<table class="process-table"><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>';
                    for(const[k,v] of Object.entries(metrics)){html+=`<tr><td>${k}</td><td>${v}</td></tr>`;}
                    html+='</tbody></table>';
                }
                div.innerHTML=html;
            } catch(e){}
        }
        setInterval(fetchMetrics, 3000);
        setInterval(fetchChangelog, 10000);
        setInterval(fetchResults, 15000);
        fetchMetrics(); fetchChangelog(); fetchResults();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/metrics')
def api_metrics():
    import psutil
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    memory_info = {
        'percent': mem.percent,
        'total_gb': f'{mem.total/(1024**3):.1f}',
        'used_gb': f'{mem.used/(1024**3):.1f}',
        'available_gb': f'{mem.available/(1024**3):.1f}',
    }
    try:
        disk = psutil.disk_usage('/')
        disk_info = {'percent': disk.percent, 'total_gb': f'{disk.total/(1024**3):.0f}', 'used_gb': f'{disk.used/(1024**3):.0f}'}
    except Exception:
        disk_info = None
    gpus = []
    try:
        result = subprocess.run(
            ['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts)>=5:
                    mu,mt=int(parts[1]),int(parts[2])
                    gpus.append({'name':parts[0],'memory_used_mb':mu,'memory_total_mb':mt,
                                 'memory_percent':round(mu/max(mt,1)*100,1),'utilization':int(parts[3]),'temperature':int(parts[4])})
    except Exception: pass
    processes = []
    for proc in psutil.process_iter(['pid','name','username','cpu_percent','memory_percent','cmdline','create_time']):
        try:
            info = proc.info; cmdline = info.get('cmdline',[])
            if not cmdline: continue
            cmd_str = ' '.join(cmdline)
            if 'python' in info.get('name','').lower() or 'train' in cmd_str.lower():
                elapsed = time.time()-info.get('create_time',time.time())
                h,m=int(elapsed//3600),int((elapsed%3600)//60)
                processes.append({'pid':info['pid'],'user':info.get('username','N/A'),
                    'cpu_percent':round(info.get('cpu_percent',0),1),'mem_percent':round(info.get('memory_percent',0),1),
                    'elapsed':f'{h}h{m:02d}m','command':cmd_str[:200]})
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    processes.sort(key=lambda x:x['cpu_percent'],reverse=True)
    return jsonify({'cpu_percent':cpu_percent,'memory':memory_info,'disk':disk_info,'gpus':gpus,'processes':processes[:20]})

@app.route('/api/changelog')
def api_changelog():
    if CHANGELOG_PATH and os.path.exists(CHANGELOG_PATH):
        with open(CHANGELOG_PATH,'r') as f: return jsonify(json.load(f))
    return jsonify([])

@app.route('/api/results')
def api_results():
    results = {}
    if TRAINING_LOG_DIR:
        for fname in (os.listdir(TRAINING_LOG_DIR) if os.path.exists(TRAINING_LOG_DIR) else []):
            if fname.startswith('summary_') and fname.endswith('.json'):
                fpath = os.path.join(TRAINING_LOG_DIR, fname)
                with open(fpath,'r') as f:
                    data = json.load(f)
                    results[data.get('model',fname)] = data.get('metrics',{})
    return jsonify(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Innovation 4 Dashboard')
    parser.add_argument('--port', default=8502, type=int)
    parser.add_argument('--host', default='0.0.0.0', type=str)
    parser.add_argument('--changelog', default=None, type=str)
    parser.add_argument('--results_dir', default=None, type=str)
    args = parser.parse_args()
    CHANGELOG_PATH = args.changelog
    TRAINING_LOG_DIR = args.results_dir
    print(f"Starting Innovation 4 dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
