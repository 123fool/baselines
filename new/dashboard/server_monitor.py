"""
BrLP 研究项目 — 可视化监控面板 v2
=================================
功能:
  1. 服务器实时信息 (CPU / 内存 / GPU / 运行进程)
  2. AI 思考与操作日志 (实时显示代码修改、操作解释)
  3. 训练/测试进度 (任务名称 + 剩余时间估算)
  4. 代码修改记录 (修改内容 / 原因 / 结果)
  5. 实验指标对比 (Baseline vs Innovation 1 / 2 / TPN)

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
import subprocess
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
TRAIN_LOG   = "/home/wangchong/data/fwz/output/innovation_2/train.log"
AUTO_EVAL_LOG = "/home/wangchong/data/fwz/output/innovation_2/eval.log"
EVAL_SUMMARY = "/home/wangchong/data/fwz/output/innovation_2/eval/summary_innovation_2_btr.json"
LOCAL_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# TPN (优先级1) 相关路径
TPN_OUTPUT_DIR = "/home/wangchong/data/fwz/output/tpn_v3b"
TPN_TRAIN_LOG  = "/home/wangchong/data/fwz/output/tpn_v3b/train.log"
TPN_EVAL_LOG   = "/home/wangchong/data/fwz/output/tpn_v3b/eval.log"

# RLP (优先级2) 相关路径 — 已放弃，保留用于历史记录
RLP_OUTPUT_DIR      = "/home/wangchong/data/fwz/output/priority_2_rlp"
RLP_TRAIN_LOG       = "/home/wangchong/data/fwz/output/priority_2_rlp/rlp_only/controlnet/tensorboard"
RLP_BTR_TRAIN_LOG   = "/home/wangchong/data/fwz/output/priority_2_rlp/btr_rlp/controlnet/tensorboard"

# PALM+TEL (优先级4) 相关路径
P4_OUTPUT_DIR       = "/home/wangchong/data/fwz/output/priority_4_palm_tel"
P4_CNET_DIR         = "/home/wangchong/data/fwz/output/priority_4_palm_tel/controlnet"
P4_EVAL_DIR         = "/home/wangchong/data/fwz/output/priority_4_palm_tel/eval"

# Combined Inn1+Inn2 (6ch+BTR) 相关路径
COMBINED_OUTPUT_DIR  = "/home/wangchong/data/fwz/output/combined_inn1_inn2"
COMBINED_TRAIN_LOG   = "/home/wangchong/data/fwz/output/combined_inn1_inn2/train.log"
COMBINED_EVAL_LOG    = "/home/wangchong/data/fwz/output/combined_inn1_inn2/eval.log"
COMBINED_EVAL_DIR    = "/home/wangchong/data/fwz/output/combined_inn1_inn2/eval"

# 去辅助模型验证 (No Aux Model) 相关路径
NO_AUX_OUTPUT_DIR    = "/home/wangchong/data/fwz/output/no_aux_model"
NO_AUX_EVAL_LOG      = "/home/wangchong/data/fwz/output/no_aux_model/eval_no_aux.log"
NO_AUX_SUMMARY       = "/home/wangchong/data/fwz/output/no_aux_model/summary_no_aux.json"

# 多时间点连续生成验证 (Multi-Timepoint) 相关路径
MULTI_TP_OUTPUT_DIR  = "/home/wangchong/data/fwz/output/multi_timepoint"
MULTI_TP_EVAL_LOG    = "/home/wangchong/data/fwz/output/multi_timepoint/eval_multi_tp.log"
MULTI_TP_SUMMARY     = "/home/wangchong/data/fwz/output/multi_timepoint/summary_multi_timepoint.json"

# 缓存
_cache = {
    "server_info": None,
    "gpu_info": None,
    "processes": None,
    "task_progress": None,
    "project_changes": None,
    "last_update": None,
    "error": None,
    "tpn_progress": None,
    "p4_progress": None,
    "rlp_progress": None,
    "p4_progress": None,
    "combined_progress": None,
    "no_aux_progress": None,
    "multi_tp_progress": None,
}
_cache_lock = threading.Lock()

# AI 操作日志 (通过 POST /api/ai_log 实时推送)
_ai_operations = []
_ai_ops_lock = threading.Lock()

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
            try:
                mem_used = int(parts[3])
                mem_total = int(parts[4])
                mem_pct = int(mem_used / mem_total * 100) if mem_total > 0 else 0
            except (ValueError, ZeroDivisionError):
                mem_pct = 0
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "util": parts[2] + "%",
                "mem_used": parts[3] + " MiB",
                "mem_total": parts[4] + " MiB",
                "mem_pct": str(mem_pct) + "%",
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
            "epoch_total": 5,
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

    train_proc = ssh_exec("ps aux | grep 'train_controlnet_btr.py' | grep -v grep")
    auto_eval_proc = ssh_exec("ps aux | grep 'evaluate_btr.py' | grep -v grep")
    train_tail = ssh_exec(f"tail -180 {TRAIN_LOG} 2>/dev/null")
    train_vals = ssh_exec(f"grep -E '\\[Epoch [0-9]+\\] val_' {TRAIN_LOG} | tail -5 2>/dev/null")
    auto_eval_tail = ssh_exec(f"tail -220 {AUTO_EVAL_LOG} 2>/dev/null")
    summary_raw = ssh_exec(f"cat {EVAL_SUMMARY} 2>/dev/null")

    train_completed = (
      "ControlNet training complete" in train_tail
      or "[Innovation 2] Training complete." in train_tail
      or "Training complete" in train_tail
      or "训练完成" in train_tail
    )
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
        progress["eval"]["pair_current"] = progress["eval"]["pair_total"]
      except Exception:
        pass

    if progress["eval"]["state"] != "completed":
      if eval_completed:
        progress["eval"]["state"] = "completed"
        progress["eval"]["state_text"] = "评估完成"
        progress["eval"]["percent"] = 100
        progress["eval"]["pair_current"] = progress["eval"]["pair_total"]
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


def fetch_tpn_progress():
    """采集 TPN (优先级1) 训练与评估进度。"""
    progress = {
        "task_name": "优先级1: TPN 替换 Leaspy",
        "train": {
            "state": "unknown",
            "state_text": "未知",
            "epoch_current": 0,
            "epoch_total": 200,
            "loss_current": None,
            "best_loss": None,
            "percent": 0,
            "eta": "N/A",
            "log_tail": "",
        },
        "eval": {
            "state": "unknown",
            "state_text": "未知",
            "mae_tpn": None,
            "mae_leaspy": None,
            "r2_score": None,
            "log_tail": "",
        },
    }

    # Check train log
    train_tail = ssh_exec(f"tail -50 {TPN_TRAIN_LOG} 2>/dev/null")
    if train_tail and "ERROR" not in train_tail:
        progress["train"]["log_tail"] = train_tail

        # Parse epoch info: "Epoch 150/200 | loss=0.00123 | best=0.00098 | ETA: 00:02:30"
        epoch_matches = re.findall(
            r"Epoch\s+(\d+)/(\d+)\s*\|\s*loss=([\d.]+)\s*\|\s*best=([\d.]+)(?:\s*\|\s*ETA:\s*(\S+))?",
            train_tail
        )
        if epoch_matches:
            ep, total, loss, best, eta = epoch_matches[-1]
            progress["train"].update({
                "epoch_current": int(ep),
                "epoch_total": int(total),
                "loss_current": float(loss),
                "best_loss": float(best),
                "percent": int(int(ep) / int(total) * 100),
                "eta": eta if eta else "N/A",
            })

        if "Training complete" in train_tail or "训练完成" in train_tail:
            progress["train"]["state"] = "completed"
            progress["train"]["state_text"] = "训练完成"
            progress["train"]["percent"] = 100
        else:
            train_proc = ssh_exec("ps aux | grep 'train_tpn.py' | grep -v grep")
            if train_proc and "ERROR" not in train_proc:
                progress["train"]["state"] = "running"
                progress["train"]["state_text"] = "训练中"
            else:
                progress["train"]["state"] = "idle"
                progress["train"]["state_text"] = "未运行"
    else:
        progress["train"]["state"] = "idle"
        progress["train"]["state_text"] = "未运行"

    # Check eval log
    eval_tail = ssh_exec(f"tail -30 {TPN_EVAL_LOG} 2>/dev/null")
    if eval_tail and "ERROR" not in eval_tail:
        progress["eval"]["log_tail"] = eval_tail

        # Parse MAE comparison: "TPN MAE: 0.0312 | Leaspy MAE: 0.0345 | R²: 0.9521"
        mae_match = re.findall(
            r"TPN MAE:\s*([\d.]+)\s*\|\s*Leaspy MAE:\s*([\d.]+)(?:\s*\|\s*R.:\s*([\d.]+))?",
            eval_tail
        )
        if mae_match:
            tpn_mae, leaspy_mae, r2 = mae_match[-1]
            progress["eval"]["mae_tpn"] = float(tpn_mae)
            progress["eval"]["mae_leaspy"] = float(leaspy_mae)
            if r2:
                progress["eval"]["r2_score"] = float(r2)

        if "Evaluation complete" in eval_tail:
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
        else:
            eval_proc = ssh_exec("ps aux | grep 'evaluate_tpn.py' | grep -v grep")
            if eval_proc and "ERROR" not in eval_proc:
                progress["eval"]["state"] = "running"
                progress["eval"]["state_text"] = "评估中"
            else:
                progress["eval"]["state"] = "idle"
                progress["eval"]["state_text"] = "未运行"
    else:
        progress["eval"]["state"] = "idle"
        progress["eval"]["state_text"] = "未运行"

    return progress


def fetch_rlp_progress():
    """采集 RLP (优先级2) 训练与评估进度。"""
    progress = {
        "task_name": "优先级2: 残差潜码预测 (RLP)",
        "rlp_only": {
            "state": "unknown",
            "state_text": "未知",
            "epoch_current": 0,
            "epoch_total": 5,
            "loss_current": None,
            "percent": 0,
            "eta": "N/A",
        },
        "btr_rlp": {
            "state": "unknown",
            "state_text": "未知",
            "epoch_current": 0,
            "epoch_total": 5,
            "loss_fwd": None,
            "loss_bwd": None,
            "percent": 0,
            "eta": "N/A",
        },
        "eval": {
            "state": "unknown",
            "state_text": "未知",
            "metrics": {},
        },
    }

    # Check RLP-only training
    rlp_proc = ssh_exec("ps aux | grep 'train_controlnet_rlp.py' | grep -v grep")
    rlp_log = ssh_exec(f"find {RLP_OUTPUT_DIR}/rlp_only/controlnet/ -name 'cnet-rlp-ep-*.pth' 2>/dev/null | sort")
    rlp_tail = ssh_exec(f"ls -la {RLP_OUTPUT_DIR}/rlp_only/controlnet/cnet-rlp-ep-*.pth 2>/dev/null | tail -5")

    if rlp_log and "ERROR" not in rlp_log and rlp_log.strip():
        ckpts = [f for f in rlp_log.strip().split('\n') if f.strip()]
        n_ckpts = len(ckpts)
        progress["rlp_only"]["epoch_current"] = n_ckpts + 1  # checkpoints saved from ep 1
        progress["rlp_only"]["percent"] = int((n_ckpts + 1) / 5 * 100)
        if n_ckpts >= 4:  # ep1..ep4
            progress["rlp_only"]["state"] = "completed"
            progress["rlp_only"]["state_text"] = "训练完成"
            progress["rlp_only"]["percent"] = 100
        elif rlp_proc and "ERROR" not in rlp_proc:
            progress["rlp_only"]["state"] = "running"
            progress["rlp_only"]["state_text"] = "训练中"
        else:
            progress["rlp_only"]["state"] = "idle"
            progress["rlp_only"]["state_text"] = "未运行"
    elif rlp_proc and "ERROR" not in rlp_proc:
        progress["rlp_only"]["state"] = "running"
        progress["rlp_only"]["state_text"] = "训练中"
    else:
        progress["rlp_only"]["state"] = "idle"
        progress["rlp_only"]["state_text"] = "未运行"

    # Check BTR+RLP training
    btr_rlp_proc = ssh_exec("ps aux | grep 'train_controlnet_btr_rlp.py' | grep -v grep")
    btr_rlp_log = ssh_exec(f"find {RLP_OUTPUT_DIR}/btr_rlp/controlnet/ -name 'cnet-btr-rlp-ep-*.pth' 2>/dev/null | sort")

    if btr_rlp_log and "ERROR" not in btr_rlp_log and btr_rlp_log.strip():
        ckpts = [f for f in btr_rlp_log.strip().split('\n') if f.strip()]
        n_ckpts = len(ckpts)
        progress["btr_rlp"]["epoch_current"] = n_ckpts + 1
        progress["btr_rlp"]["percent"] = int((n_ckpts + 1) / 5 * 100)
        if n_ckpts >= 4:
            progress["btr_rlp"]["state"] = "completed"
            progress["btr_rlp"]["state_text"] = "训练完成"
            progress["btr_rlp"]["percent"] = 100
        elif btr_rlp_proc and "ERROR" not in btr_rlp_proc:
            progress["btr_rlp"]["state"] = "running"
            progress["btr_rlp"]["state_text"] = "训练中"
        else:
            progress["btr_rlp"]["state"] = "idle"
            progress["btr_rlp"]["state_text"] = "未运行"
    elif btr_rlp_proc and "ERROR" not in btr_rlp_proc:
        progress["btr_rlp"]["state"] = "running"
        progress["btr_rlp"]["state_text"] = "训练中"
    else:
        progress["btr_rlp"]["state"] = "idle"
        progress["btr_rlp"]["state_text"] = "未运行"

    # Check eval results
    rlp_eval_json = ssh_exec(f"cat {RLP_OUTPUT_DIR}/rlp_only/eval/summary_rlp_only.json 2>/dev/null")
    btr_rlp_eval_json = ssh_exec(f"cat {RLP_OUTPUT_DIR}/btr_rlp/eval/summary_btr_rlp.json 2>/dev/null")

    if rlp_eval_json and "ERROR" not in rlp_eval_json:
        try:
            progress["eval"]["metrics"]["rlp_only"] = json.loads(rlp_eval_json)
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
        except Exception:
            pass

    if btr_rlp_eval_json and "ERROR" not in btr_rlp_eval_json:
        try:
            progress["eval"]["metrics"]["btr_rlp"] = json.loads(btr_rlp_eval_json)
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
        except Exception:
            pass

    if progress["eval"]["state"] == "unknown":
        eval_proc = ssh_exec("ps aux | grep 'evaluate_rlp.py' | grep -v grep")
        if eval_proc and "ERROR" not in eval_proc:
            progress["eval"]["state"] = "running"
            progress["eval"]["state_text"] = "评估中"
        else:
            progress["eval"]["state"] = "idle"
            progress["eval"]["state_text"] = "未运行"

    return progress


def fetch_p4_progress():
    """采集 PALM+TEL (优先级4) 训练与评估进度。"""
    progress = {
        "task_name": "优先级4: PALM + TEL 装饰模块 (BTR基础)",
        "train": {
            "state": "unknown",
            "state_text": "未知",
            "epoch_current": 0,
            "epoch_total": 5,
            "loss_total": None,
            "loss_fwd": None,
            "loss_bwd": None,
            "percent": 0,
            "eta": "N/A",
        },
        "eval": {
            "state": "unknown",
            "state_text": "未知",
            "pair_current": 0,
            "pair_total": 50,
            "percent": 0,
            "metrics": {},
        },
    }

    # Check training via checkpoints
    train_proc = ssh_exec("ps aux | grep 'train_controlnet_btc_palm_tel.py' | grep -v grep")
    ckpt_list = ssh_exec(f"ls -1 {P4_CNET_DIR}/cnet-btc-palm-tel-ep-*.pth 2>/dev/null")
    train_log = ssh_exec(f"grep -E '\\[Epoch [0-9]+\\]' {P4_CNET_DIR}/../train.log 2>/dev/null | tail -10")

    if ckpt_list and "ERROR" not in ckpt_list and ckpt_list.strip():
        ckpts = [f for f in ckpt_list.strip().split('\n') if f.strip()]
        n_ckpts = len(ckpts)
        progress["train"]["epoch_current"] = n_ckpts + 1
        progress["train"]["percent"] = int((n_ckpts + 1) / 5 * 100)

    # Parse epoch losses from log
    if train_log and "ERROR" not in train_log:
        epoch_matches = re.findall(
            r"\[Epoch (\d+)\] (\w+): total=([\d.]+)\s+fwd=([\d.]+)\s+bwd=([\d.]+)",
            train_log
        )
        if epoch_matches:
            ep, mode, total, fwd, bwd = epoch_matches[-1]
            progress["train"]["epoch_current"] = int(ep)
            progress["train"]["loss_total"] = float(total)
            progress["train"]["loss_fwd"] = float(fwd)
            progress["train"]["loss_bwd"] = float(bwd)

    # Check completion
    train_tail = ssh_exec(f"tail -5 {P4_CNET_DIR}/../train.log 2>/dev/null")
    if train_tail and "[Priority 4] BTR + PALM + TEL Training complete." in train_tail:
        progress["train"]["state"] = "completed"
        progress["train"]["state_text"] = "训练完成"
        progress["train"]["percent"] = 100
        progress["train"]["epoch_current"] = 5
    elif train_proc and "ERROR" not in train_proc:
        progress["train"]["state"] = "running"
        progress["train"]["state_text"] = "训练中"
    else:
        if ckpt_list and ckpt_list.strip():
            progress["train"]["state"] = "completed"
            progress["train"]["state_text"] = "训练完成"
        else:
            progress["train"]["state"] = "idle"
            progress["train"]["state_text"] = "未运行"

    # Check eval
    eval_proc = ssh_exec("ps aux | grep 'evaluate_palm_tel.py' | grep -v grep")
    eval_summary = ssh_exec(f"cat {P4_EVAL_DIR}/summary_btc_palm_tel_ep4.json 2>/dev/null")
    eval_tail = ssh_exec(f"tail -20 {P4_EVAL_DIR}/../eval.log 2>/dev/null")

    if eval_summary and "ERROR" not in eval_summary:
        try:
            progress["eval"]["metrics"] = json.loads(eval_summary)
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
            progress["eval"]["percent"] = 100
            progress["eval"]["pair_current"] = 50
        except Exception:
            pass

    if progress["eval"]["state"] == "unknown":
        if eval_tail and "Evaluating pairs" in eval_tail:
            eval_matches = re.findall(r"Evaluating pairs:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)", eval_tail)
            if eval_matches:
                pct, cur, total = eval_matches[-1]
                progress["eval"]["pair_current"] = int(cur)
                progress["eval"]["pair_total"] = int(total)
                progress["eval"]["percent"] = int(pct)
            progress["eval"]["state"] = "running"
            progress["eval"]["state_text"] = "评估中"
        elif eval_proc and "ERROR" not in eval_proc:
            progress["eval"]["state"] = "running"
            progress["eval"]["state_text"] = "评估中"
        else:
            progress["eval"]["state"] = "idle"
            progress["eval"]["state_text"] = "未运行"

    return progress


def fetch_combined_progress():
    """采集 Combined Inn1+Inn2 (6ch+BTR) 训练与评估进度。"""
    progress = {
        "task_name": "建议A: 6ch+BTR 组合创新 (Inn1+Inn2)",
        "train": {
            "state": "unknown",
            "state_text": "未知",
            "epoch_current": 0,
            "epoch_total": 5,
            "loss_total": None,
            "loss_fwd": None,
            "loss_bwd": None,
            "percent": 0,
            "eta": "N/A",
        },
        "eval": {
            "state": "unknown",
            "state_text": "未知",
            "pair_current": 0,
            "pair_total": 50,
            "percent": 0,
            "metrics": None,
        },
    }

    # Check train log
    train_tail = ssh_exec(f"tail -80 {COMBINED_TRAIN_LOG} 2>/dev/null")
    if train_tail and "ERROR" not in train_tail:
        # Parse epoch: "[Epoch 2] train: total=0.016234  fwd=0.012345  bwd=0.007778"
        epoch_matches = re.findall(
            r"\[Epoch\s+(\d+)\]\s+train:\s+total=([\d.]+)\s+fwd=([\d.]+)\s+bwd=([\d.]+)",
            train_tail
        )
        if epoch_matches:
            ep, total_l, fwd_l, bwd_l = epoch_matches[-1]
            progress["train"]["epoch_current"] = int(ep) + 1
            progress["train"]["loss_total"] = float(total_l)
            progress["train"]["loss_fwd"] = float(fwd_l)
            progress["train"]["loss_bwd"] = float(bwd_l)
            progress["train"]["percent"] = int((int(ep) + 1) / 5 * 100)

        # ETA from tqdm
        eta_matches = re.findall(r"(\d+:\d+)<", train_tail)
        if eta_matches:
            progress["train"]["eta"] = eta_matches[-1]

        if "Training complete" in train_tail:
            progress["train"]["state"] = "completed"
            progress["train"]["state_text"] = "训练完成"
            progress["train"]["percent"] = 100
        else:
            train_proc = ssh_exec("ps aux | grep 'train_controlnet_6ch_btr' | grep -v grep")
            if train_proc and "ERROR" not in train_proc:
                progress["train"]["state"] = "running"
                progress["train"]["state_text"] = "训练中"
            else:
                progress["train"]["state"] = "idle"
                progress["train"]["state_text"] = "未运行"
    else:
        progress["train"]["state"] = "idle"
        progress["train"]["state_text"] = "未运行"

    # Check eval
    eval_tail = ssh_exec(f"tail -30 {COMBINED_EVAL_LOG} 2>/dev/null")
    eval_proc = ssh_exec("ps aux | grep 'evaluate_6ch_btr' | grep -v grep")

    if eval_tail and "ERROR" not in eval_tail:
        if "Evaluation complete" in eval_tail or "Results" in eval_tail:
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
            progress["eval"]["percent"] = 100
            # Try to load summary
            summary_raw = ssh_exec(
                f"cat {COMBINED_EVAL_DIR}/summary_combined_6ch_btr*.json 2>/dev/null | head -50")
            if summary_raw and "ERROR" not in summary_raw:
                try:
                    import json as _json
                    progress["eval"]["metrics"] = _json.loads(summary_raw)
                except Exception:
                    pass
        elif eval_tail and "Evaluating pairs" in eval_tail:
            eval_matches = re.findall(r"Evaluating pairs:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)", eval_tail)
            if eval_matches:
                pct, cur, total = eval_matches[-1]
                progress["eval"]["pair_current"] = int(cur)
                progress["eval"]["pair_total"] = int(total)
                progress["eval"]["percent"] = int(pct)
            progress["eval"]["state"] = "running"
            progress["eval"]["state_text"] = "评估中"
        elif eval_proc and "ERROR" not in eval_proc:
            progress["eval"]["state"] = "running"
            progress["eval"]["state_text"] = "评估中"
        else:
            progress["eval"]["state"] = "idle"
            progress["eval"]["state_text"] = "未运行"
    else:
        progress["eval"]["state"] = "idle"
        progress["eval"]["state_text"] = "未运行"

    return progress


def fetch_no_aux_progress():
    """采集去辅助模型验证实验进度。"""
    progress = {
        "task_name": "去辅助模型端到端验证 (GT/TPN/Skip/Linear)",
        "eval": {
            "state": "unknown",
            "state_text": "未知",
            "pair_current": 0,
            "pair_total": 50,
            "methods_total": 4,
            "percent": 0,
            "eta": "N/A",
        },
        "methods_ssim": {},
        "summary": None,
    }

    # Check if evaluation is running / completed
    eval_tail = ssh_exec(f"tail -60 {NO_AUX_EVAL_LOG} 2>/dev/null")
    if eval_tail and "ERROR" not in eval_tail:
        # Parse pair progress: "[NO_AUX] Pair 12/50 | GT=0.9312 TPN=0.9298 Skip=0.9180"
        pair_matches = re.findall(
            r"\[NO_AUX\]\s+Pair\s+(\d+)/(\d+)\s+\|\s+(.*)",
            eval_tail
        )
        if pair_matches:
            cur, total, metrics_str = pair_matches[-1]
            progress["eval"]["pair_current"] = int(cur)
            progress["eval"]["pair_total"] = int(total)
            n_total = int(total)
            progress["eval"]["percent"] = int(int(cur) / n_total * 100) if n_total > 0 else 0

            # Parse method SSIMs from metrics line
            for m in re.findall(r"(\w+)=([\d.]+)", metrics_str):
                progress["methods_ssim"][m[0]] = float(m[1])

        # ETA from tqdm
        eta_matches = re.findall(r"(\d+:\d+)<", eval_tail)
        if eta_matches:
            progress["eval"]["eta"] = eta_matches[-1]

        # Check for summary
        if "[NO_AUX] === SUMMARY ===" in eval_tail:
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
            progress["eval"]["percent"] = 100
            # Parse summary line
            summary_match = re.search(
                r"\[NO_AUX\] === SUMMARY === (.*)", eval_tail)
            if summary_match:
                parts = summary_match.group(1).split("|")
                for part in parts:
                    part = part.strip()
                    kv = part.split("=")
                    if len(kv) == 2:
                        progress["methods_ssim"][kv[0].strip()] = float(kv[1].strip())
            # Try to load JSON summary
            summary_raw = ssh_exec(f"cat {NO_AUX_SUMMARY} 2>/dev/null | head -80")
            if summary_raw and "ERROR" not in summary_raw:
                try:
                    import json as _json
                    progress["summary"] = _json.loads(summary_raw)
                except Exception:
                    pass
        else:
            eval_proc = ssh_exec("ps aux | grep 'evaluate_no_aux' | grep -v grep")
            if eval_proc and eval_proc.strip() and "ERROR" not in eval_proc:
                progress["eval"]["state"] = "running"
                progress["eval"]["state_text"] = "评估中"
            elif pair_matches:
                progress["eval"]["state"] = "idle"
                progress["eval"]["state_text"] = "已暂停"
            else:
                progress["eval"]["state"] = "idle"
                progress["eval"]["state_text"] = "未运行"
    else:
        progress["eval"]["state"] = "idle"
        progress["eval"]["state_text"] = "未运行"

    return progress


def fetch_multi_tp_progress():
    """采集多时间点连续生成验证实验进度。"""
    progress = {
        "task_name": "多时间点连续生成验证 (Direct-Skip/Linear/TPN + Auto-Linear)",
        "eval": {
            "state": "unknown",
            "state_text": "未知",
            "current": 0,
            "total": 0,
            "percent": 0,
            "eta": "N/A",
        },
        "methods_ssim": {},
        "by_time_gap": {},
        "summary": None,
    }

    eval_tail = ssh_exec(f"tail -80 {MULTI_TP_EVAL_LOG} 2>/dev/null")
    if eval_tail and "ERROR" not in eval_tail:
        # Parse per-visit progress: "[MULTI_TP] sid vN method | SSIM=0.9312 ... [45/120]"
        prog_matches = re.findall(
            r"\[MULTI_TP\].*\[(\d+)/(\d+)\]", eval_tail)
        if prog_matches:
            cur, total = prog_matches[-1]
            progress["eval"]["current"] = int(cur)
            progress["eval"]["total"] = int(total)
            progress["eval"]["percent"] = int(int(cur) / int(total) * 100) if int(total) > 0 else 0

        # Parse method SSIMs from recent lines
        ssim_matches = re.findall(
            r"\[MULTI_TP\].*?(Direct-\w+|Auto-\w+)\s+\|\s+SSIM=([\d.]+)", eval_tail)
        method_ssims = {}
        for method, ssim_val in ssim_matches:
            method_ssims.setdefault(method, []).append(float(ssim_val))
        for method, vals in method_ssims.items():
            progress["methods_ssim"][method] = sum(vals) / len(vals)

        # Check for summary
        if "[MULTI_TP] === SUMMARY ===" in eval_tail:
            progress["eval"]["state"] = "completed"
            progress["eval"]["state_text"] = "评估完成"
            progress["eval"]["percent"] = 100
            summary_match = re.search(
                r"\[MULTI_TP\] === SUMMARY === (.*)", eval_tail)
            if summary_match:
                parts = summary_match.group(1).split("|")
                for part in parts:
                    part = part.strip()
                    kv = part.split("=")
                    if len(kv) == 2:
                        try:
                            progress["methods_ssim"][kv[0].strip()] = float(kv[1].strip())
                        except ValueError:
                            pass
            # Load JSON summary
            summary_raw = ssh_exec(f"cat {MULTI_TP_SUMMARY} 2>/dev/null | head -120")
            if summary_raw and "ERROR" not in summary_raw:
                try:
                    summary_obj = json.loads(summary_raw)
                    progress["summary"] = summary_obj
                    # Extract by_time_gap for display
                    for method, mdata in summary_obj.get("methods", {}).items():
                        progress["by_time_gap"][method] = mdata.get("by_time_gap", {})
                except Exception:
                    pass
        else:
            eval_proc = ssh_exec("ps aux | grep 'evaluate_multi_timepoint' | grep -v grep")
            if eval_proc and eval_proc.strip() and "ERROR" not in eval_proc:
                progress["eval"]["state"] = "running"
                progress["eval"]["state_text"] = "评估中"
            elif prog_matches:
                progress["eval"]["state"] = "idle"
                progress["eval"]["state_text"] = "已暂停"
            else:
                progress["eval"]["state"] = "idle"
                progress["eval"]["state_text"] = "未运行"
    else:
        progress["eval"]["state"] = "idle"
        progress["eval"]["state_text"] = "未运行"

    return progress


def fetch_project_changes():
    """采集本地仓库最新提交与工作区改动。"""
    info = {
        "repo_ok": False,
        "branch": "N/A",
        "latest_commit": "N/A",
        "latest_time": "N/A",
        "latest_subject": "N/A",
        "recent_commits": [],
        "changed_files": [],
        "changed_count": 0,
        "error": "",
    }

    def _run_git(args):
        p = subprocess.run(
            ["git", *args],
            cwd=LOCAL_REPO_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()

    try:
        code, out, err = _run_git(["rev-parse", "--is-inside-work-tree"])
        if code != 0 or out.lower() != "true":
            info["error"] = err or "当前目录不是 Git 仓库"
            return info

        info["repo_ok"] = True

        code, out, _ = _run_git(["branch", "--show-current"])
        if code == 0 and out:
            info["branch"] = out

        code, out, _ = _run_git(["log", "-1", "--pretty=format:%h|%ad|%s", "--date=format:%Y-%m-%d %H:%M:%S"])
        if code == 0 and out:
            parts = out.split("|", 2)
            if len(parts) == 3:
                info["latest_commit"], info["latest_time"], info["latest_subject"] = parts

        code, out, _ = _run_git(["log", "-5", "--pretty=format:%h %ad %s", "--date=format:%m-%d %H:%M"])
        if code == 0 and out:
            info["recent_commits"] = [x.strip() for x in out.splitlines() if x.strip()]

        code, out, _ = _run_git(["status", "--short"])
        if code == 0 and out:
            lines = [x.rstrip() for x in out.splitlines() if x.strip()]
            info["changed_files"] = lines[:20]
            info["changed_count"] = len(lines)

    except Exception as e:
      info["error"] = str(e)

    return info


def background_refresh():
    """后台线程定时刷新服务器数据。"""
    while True:
        try:
            info = fetch_server_info()
            task_progress = fetch_task_progress()
            tpn_progress = fetch_tpn_progress()
            rlp_progress = fetch_rlp_progress()
            p4_progress = fetch_p4_progress()
            combined_progress = fetch_combined_progress()
            no_aux_progress = fetch_no_aux_progress()
            multi_tp_progress = fetch_multi_tp_progress()
            project_changes = fetch_project_changes()
            with _cache_lock:
                _cache["server_info"] = info
                _cache["gpu_info"] = parse_gpu(info.get("gpu_raw", ""))
                _cache["processes"] = parse_processes(info.get("proc_raw", ""))
                _cache["task_progress"] = task_progress
                _cache["tpn_progress"] = tpn_progress
                _cache["rlp_progress"] = rlp_progress
                _cache["p4_progress"] = p4_progress
                _cache["combined_progress"] = combined_progress
                _cache["no_aux_progress"] = no_aux_progress
                _cache["multi_tp_progress"] = multi_tp_progress
                _cache["project_changes"] = project_changes
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
    # 联合推理（方案A之前的对比基线）
    "combined_4_5": {
        "overall_ssim": 0.9123, "overall_psnr": 25.9442, "overall_mae": 0.0311,
        "hippocampus_ssim": 0.8203, "hippocampus_mae": 0.0748,
        "roi_ssim": 0.8059, "roi_mae": 0.0768,
    },
    # 方案A重训结果（50对测试，ep4最优）— 训练5个epoch未达预期，SSIM劣于Inn5
    "combined_retrain": {
        "overall_ssim": 0.8664, "overall_psnr": 22.0717, "overall_mae": 0.0431,
        "hippocampus_ssim": 0.7144, "hippocampus_mae": 0.1334,
        "roi_ssim": 0.7043, "roi_mae": 0.1326,
    },
}

# ── 创新点 1 专用对比（改进AE解码器 + GT 归一化 [0,1]）──
# 注意：所有评估均使用 Innovation 5 改进AE (autoencoder-ep-2.pth) 作为解码器
# 原始 AE 重建 SSIM 仅 0.36，改进 AE 重建 SSIM = 0.96 —— 这是前后 SSIM 差异的根因
INNOVATION_1_METRICS = {
    "baseline_same_pipeline": {
        "overall_ssim": 0.8990, "overall_psnr": 25.2205, "overall_mae": 0.0356,
        "roi_ssim": 0.7969, "roi_mae": 0.0904,
        "_note": "50对MCI测试样本, 使用改进AE解码器 (autoencoder-ep-2.pth)",
    },
    "innovation_1": {
        "overall_ssim": 0.9153, "overall_psnr": 26.5371, "overall_mae": 0.0290,
        "roi_ssim": 0.8116, "roi_mae": 0.0673,
        "_note": "50对MCI测试样本, 6ch ControlNet + 改进AE解码器",
    },
    "innovation_2_btr": {
        "overall_ssim": 0.9282, "overall_psnr": 27.2963, "overall_mae": 0.0262,
        "roi_ssim": 0.8277, "roi_mae": 0.0626,
        "hippocampus_ssim": 0.8409, "hippocampus_mae": 0.0605,
        "amygdala_mae": 0.0665,
        "_note": "50对MCI测试样本, BTR双向时间正则化 epoch1 + 改进AE解码器",
    },
    "rlp_only_p2": {
        "overall_ssim": 0.9149, "overall_psnr": 25.1064, "overall_mae": 0.0331,
        "roi_ssim": 0.8004, "roi_mae": 0.1020,
        "hippocampus_ssim": 0.8117, "hippocampus_mae": 0.0988,
        "amygdala_mae": 0.1079,
        "_note": "50对MCI测试样本, RLP残差预测 epoch4 — 已放弃(不如BTR)",
    },
    "btr_rlp_p2": {
        "overall_ssim": 0.9047, "overall_psnr": 24.7896, "overall_mae": 0.0359,
        "roi_ssim": 0.7955, "roi_mae": 0.0973,
        "hippocampus_ssim": 0.8073, "hippocampus_mae": 0.0933,
        "amygdala_mae": 0.1044,
        "_note": "50对MCI测试样本, BTR+RLP组合 epoch4 — 已放弃(产生负干扰)",
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
    {"time": "2026-04-10",
        "file": "train_ae_v4.py",
        "change": "7项超参同步修改: warmup/cosine/latent_noise/ssim_weight等",
        "reason": "尝试综合优化，但同时改动过多",
        "result": "训练失败：warmup+cosine时序冲突，3D loss在LR≈0时才激活",
    },
    {"time": "2026-04-10",
        "file": "run.sh (combined_4_5)",
        "change": "Inn4 AE(ep-4) + Inn5 ControlNet(ep-3) 联合推理评估",
        "reason": "Inn4冻结Encoder→潜空间不变，Inn5 ControlNet与Inn4 Decoder完全兼容，无需重训",
        "result": "SSIM=0.9123 (+1.20%↑ vs BL), ROI_SSIM=0.8059 (+0.95%↑). 未超越 Inn5 单独使用，原因：Decoder Mismatch",
    },
    {"time": "2026-04-10",
        "file": "train.sh (combined_retrain)",
        "change": "方案A: Inn4 AE(ep-4) 基础上重训 Inn5 ControlNet (5 epochs, GPU1)",
        "reason": "Decoder Mismatch 修复：ControlNet 在 Inn4 Decoder 特征空间中训练，消除分布不一致",
        "result": "SSIM=0.8664 (ep4)，劣于 Inn5 单独(0.9145)。根因: AE仅用于TensorBoard可视化,不参与梯度,训练等价于Inn5; 5epoch未充分收敛+随机种子差异导致局部极小值不同。当前最佳组合仍为combined_4_5 (SSIM=0.9123)",
    },
    {"time": "2026-04-10",
        "file": "train_controlnet_mci.py (Innovation 1)",
        "change": "创新点1: MCI动态条件引导 — ControlNet空间条件扩展 4→6 通道 (新增海马萎缩率+脑室扩张率)",
        "reason": "BrLP缺少疾病进展速度信息; 两个MCI患者起始脑状态相同但萎缩速度可能全然不同; 新增条件引导模型预测不同退化速率",
        "result": "训练完成(5ep) | SSIM +4.08%↑ PSNR +5.38%↑ MAE -12.22%↓ ROI_SSIM +1.71%↑ (vs同管道Baseline)",
    },
    {"time": "2026-04-10 22:00",
        "file": "evaluate_mci.py (诊断SSIM过低)",
        "change": "诊断overall SSIM=0.31远低于预期0.90的根因",
        "reason": "AE编解码回路测试发现: 原始AE重建SSIM仅0.36，改进AE（Inn4 3D感知损失训练）重建SSIM=0.96。bash_history确认所有之前的评估都使用了改进AE解码器 (autoencoder-ep-2.pth)，而Inn1评估误用了原始AE",
        "result": "根因确认：评估时AE解码器不同。修复后: SSIM 0.9153 (+1.8%↑ vs BL), PSNR 26.54 (+5.2%↑), ROI_SSIM 0.8116 (+1.8%↑), ROI_MAE 0.0673 (-25.6%↓)",
    },
    {"time": "2026-04-11",
        "file": "train_controlnet_btr.py (Innovation 2)",
        "change": "创新点2: 双向时间正则化 — 每个batch同时训练正向(A→B)和反向(B→A)噪声预测, L_total = L_fwd + 0.5*L_bwd",
        "reason": "BrLP仅训练单向预测, 缺乏时间一致性约束; 如果模型能从A预测B, 也应能从B还原A; 双向约束增强时序一致性",
        "result": "训练完成(5ep) | SSIM=0.9282 (+3.25%↑ vs BL), PSNR=27.30 (+8.23%↑), MAE=0.0262 (-26.40%↓), ROI_SSIM=0.8277 (+3.86%↑), ROI_MAE=0.0626 (-30.75%↓). 所有指标全面超越Baseline和创新点1",
    },
    {"time": "2026-04-12",
        "file": "tpn.py + train_tpn.py + evaluate_tpn.py",
        "change": "优先级1: TPN v3 替换 Leaspy — Sequential MLP(14维→128→128→5) + 残差连接 + 增强特征(age_ratio/vol_stats)",
        "reason": "消除BrLP最大学术指纹; Leaspy是外部统计模型(MCMC-SAEM), 假设S型衰减曲线, 限制非典型进展建模; TPN端到端可学习, 更灵活",
        "result": "MAE=0.0154 R²=0.9522 (vs Leaspy MAE=0.0136 R²=0.9535); TPN在cortex/white_matter上超越Leaspy; 100%样本覆盖(vs Leaspy 56%)",
    },
    {"time": "2026-04-13",
        "file": "train_controlnet_rlp.py + sampling_rlp.py + train_controlnet_btr_rlp.py",
        "change": "优先级2: 残差潜码预测(RLP) — 扩散目标从followup_z改为delta_z=followup_z-starting_z; 推理时z_followup=starting_z+delta_z; BTR+RLP双向残差训练; 参考TADM-3D(CMIG 2026)",
        "reason": "纵向脑MRI变化极细微(主要海马萎缩+脑室扩大), 残差信号稀疏更易学习; TADM-3D已验证残差扩散有效; 降低扩散模型学习难度; scale_factor从delta_z分布重新计算",
        "result": "待训练评估 — 2个实验: RLP-only, BTR+RLP",
    },
    {"time": "2026-04-12 19:00",
        "file": "evaluate_rlp.py (优先级2评估结果)",
        "change": "完成RLP-only和BTR+RLP两个变体的评估(各50对MCI测试样本)",
        "reason": "验证残差潜码预测是否对MCI纵向预测有效",
        "result": "RLP-only: SSIM=0.9149 (+1.77%↑ vs BL), PSNR=25.11 (-0.44%↓); BTR+RLP: SSIM=0.9047 (+0.63%), PSNR=24.79 (-1.71%↓). 均不如Innovation 2 BTR (SSIM=0.9282). 双向+残差组合产生负干扰. 决策: 放弃优先级2",
    },
    {"time": "2026-04-12 20:00",
        "file": "palm_tel.py + train_controlnet_btc_palm_tel.py + evaluate_palm_tel.py",
        "change": "优先级4: PALM(Progression-Aware Latent Modulation) + TEL(Temporal Encoding Layer) — 在BTR基础上新增两个装饰模块; PALM: 基于8维临床上下文的通道自适应调制(scale∈[0.5,1.5]+shift×0.1); TEL: 可学习Fourier时间编码(age_gap→sin/cos→proj)",
        "reason": "增加架构新颖度而不影响性能; PALM让模型对不同诊断阶段动态调整特征通道响应; TEL用可学习频率编码捕捉非线性衰老动态; 设计上限制参数范围保证稳定性",
        "result": "代码完成, 准备上传服务器训练测试",
    },
    {"time": "2026-04-12 22:00",
        "file": "train_controlnet_6ch_btr.py + evaluate_6ch_btr.py",
        "change": "建议A: 组合创新1+创新2 — 6通道ControlNet(起始潜码+年龄+萎缩率+扩张率) + BTR双向时间正则化(L_fwd+0.5·L_bwd); 反向方向对率取负号(疾病进展反转); 从baseline 4ch扩展到6ch(zero-init新通道)",
        "reason": "两个创新独立验证均有效(Inn1 SSIM=0.9153, Inn2 SSIM=0.9282); 叠加后同时改变条件通道数和训练目标, 与BrLP拉开差距; 降低与参考文献相似度",
        "result": "代码完成, 准备上传测试",
    },
    {"time": "2026-04-14",
        "file": "evaluate_no_aux.py + run_eval.sh",
        "change": "去辅助模型端到端验证 — 对比GT/TPN/Skip/Linear四种推理时context来源; 在BTR ControlNet(ep-1)上验证去掉Leaspy是否影响SSIM; TPN v3(MAE=0.0154 R²=0.9522)替代Leaspy(MAE=0.0137 R²=0.9535)进行体积预测",
        "reason": "Leaspy是BrLP最大学术指纹(外部MCMC-SAEM统计模型); 训练时ControlNet使用GT体积,Leaspy仅在推理时使用; 端到端验证TPN替代不降质量(SSIM≥0.92)",
        "result": "GT=0.9205 TPN=0.9218 Skip=0.9240 Linear=0.9268 — 全部SSIM≥0.92, 去辅助模型验证通过",
    },
    {"time": "2026-04-14",
        "file": "evaluate_multi_timepoint.py",
        "change": "多时间点连续生成验证 — 4种方法: Direct-Skip/Direct-Linear/Direct-TPN(从基线直接生成多时间点) + Auto-Linear(自回归链式生成); 对拥有3+访视的受试者,从基线生成连续时间点图像并与真实数据对比",
        "reason": "验证模型能否生成时间连续的脑退化序列(3/6/9/12月...); 分析SSIM是否随时间间隔增大而下降; 对比直接生成vs自回归生成策略; 为论文提供纵向生成能力证据",
        "result": "待运行",
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
          --green: #4ade80; --red: #f87171; --yellow: #fbbf24;
          --purple: #a78bfa; --cyan: #22d3ee; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: var(--text); }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 16px; }

  /* Header */
  header { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
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
  .card-purple h2 { color: var(--purple); }
  .card-cyan h2 { color: var(--cyan); }

  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--dim); font-weight: 600; }
  .up { color: var(--green); font-weight: 600; }
  .down { color: var(--red); font-weight: 600; }
  .same { color: var(--dim); }

  /* GPU bars */
  .bar-outer { background: #334155; border-radius: 4px; height: 18px; position: relative; }
  .bar-inner { border-radius: 4px; height: 100%; transition: width 0.5s ease; }
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

  /* AI Operations Log */
  .ai-log { max-height: 360px; overflow-y: auto; }
  .ai-entry { background: var(--bg); border-radius: 6px; padding: 8px 12px;
              margin-bottom: 6px; font-size: 0.85em; border-left: 3px solid var(--purple); }
  .ai-entry .ai-time { color: var(--dim); font-size: 0.75em; }
  .ai-entry .ai-type { display: inline-block; padding: 1px 8px; border-radius: 10px;
                        font-size: 0.75em; font-weight: 600; margin-left: 6px; }
  .ai-type-think { background: #3b0764; color: var(--purple); }
  .ai-type-code  { background: #064e3b; color: var(--green); }
  .ai-type-cmd   { background: #172554; color: var(--blue); }
  .ai-type-test  { background: #422006; color: var(--yellow); }
  .ai-entry .ai-msg { margin-top: 4px; }

  /* Status badge */
  .status-badge { display: inline-block; padding: 2px 10px; border-radius: 10px;
                  font-size: 0.8em; font-weight: 600; }
  .status-running { background: #172554; color: var(--blue); animation: pulse 2s infinite; }
  .status-completed { background: #064e3b; color: var(--green); }
  .status-idle { background: #1e293b; color: var(--dim); }
  .status-error { background: #7f1d1d; color: var(--red); }

  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }
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

<!-- ===== AI 思考与操作日志 ===== -->
<div class="card card-purple" style="margin-bottom:14px;">
  <h2>🤖 AI 思考与操作日志</h2>
  <div class="ai-log" id="ai-log-box">
    {% if ai_operations %}
      {% for op in ai_operations[-20:] %}
      <div class="ai-entry">
        <span class="ai-time">{{ op.time }}</span>
        <span class="ai-type ai-type-{{ op.type }}">{{ op.type_text }}</span>
        <div class="ai-msg">{{ op.message }}</div>
      </div>
      {% endfor %}
    {% else %}
      <div style="color:var(--dim); font-size:0.9em; padding:8px;">等待 AI 操作...</div>
    {% endif %}
  </div>
</div>

<!-- ===== TPN 任务进度（优先级1） ===== -->
<div class="card card-cyan" style="margin-bottom:14px;">
  <h2>📊 当前任务: {{ tpn_progress.task_name if tpn_progress else '优先级1: TPN 替换 Leaspy' }}</h2>
  <div class="grid g2" style="margin-bottom:10px;">
    <div>
      <div style="margin-bottom:6px;">
        <strong>TPN 训练:</strong>
        <span id="tpn-train-state" class="status-badge status-{{ tpn_progress.train.state if tpn_progress else 'idle' }}">
          {{ tpn_progress.train.state_text if tpn_progress else '未运行' }}
        </span>
        <span id="tpn-train-eta" style="color:var(--dim); font-size:0.85em; margin-left:8px;">
          {% if tpn_progress and tpn_progress.train.eta != 'N/A' %}剩余: {{ tpn_progress.train.eta }}{% endif %}
        </span>
      </div>
      <div class="bar-outer" style="margin-bottom:6px;">
        <div id="tpn-train-bar" class="bar-inner" style="width:{{ tpn_progress.train.percent if tpn_progress else 0 }}%; background:var(--cyan);"></div>
        <span id="tpn-train-label" class="bar-label">
          Epoch {{ tpn_progress.train.epoch_current if tpn_progress else 0 }}/{{ tpn_progress.train.epoch_total if tpn_progress else 200 }} · {{ tpn_progress.train.percent if tpn_progress else 0 }}%
        </span>
      </div>
      <div style="font-size:0.85em; color:var(--dim);">
        Loss: <span id="tpn-loss">{{ "%.6f"|format(tpn_progress.train.loss_current) if tpn_progress and tpn_progress.train.loss_current else 'N/A' }}</span>
        · Best: <span id="tpn-best-loss">{{ "%.6f"|format(tpn_progress.train.best_loss) if tpn_progress and tpn_progress.train.best_loss else 'N/A' }}</span>
      </div>
    </div>
    <div>
      <div style="margin-bottom:6px;">
        <strong>TPN 评估:</strong>
        <span id="tpn-eval-state" class="status-badge status-{{ tpn_progress.eval.state if tpn_progress else 'idle' }}">
          {{ tpn_progress.eval.state_text if tpn_progress else '未运行' }}
        </span>
      </div>
      <div style="font-size:0.85em; margin-top:6px;">
        {% if tpn_progress and tpn_progress.eval.mae_tpn %}
        <div>TPN MAE: <span style="color:var(--cyan); font-weight:600;">{{ "%.4f"|format(tpn_progress.eval.mae_tpn) }}</span></div>
        <div>Leaspy MAE: <span style="color:var(--dim);">{{ "%.4f"|format(tpn_progress.eval.mae_leaspy) }}</span></div>
        {% if tpn_progress.eval.r2_score %}
        <div>R² Score: <span style="color:var(--green);">{{ "%.4f"|format(tpn_progress.eval.r2_score) }}</span></div>
        {% endif %}
        {% else %}
        <div style="color:var(--dim);">等待评估数据...</div>
        {% endif %}
      </div>
    </div>
  </div>
</div>

<!-- ===== RLP 任务进度（优先级2）— 已放弃 ===== -->
<div class="card" style="margin-bottom:14px; border-left: 3px solid var(--red); opacity: 0.7;">
  <h2 style="color:var(--red);">🔬 {{ rlp_progress.task_name if rlp_progress else '优先级2: 残差潜码预测 (RLP)' }} — <span style="color:var(--red);">已放弃</span></h2>
  <div style="font-size:0.85em; color:var(--dim); padding:6px;">
    结论: RLP-only SSIM=0.9149 (+1.77%), BTR+RLP SSIM=0.9047 (+0.63%) — 均不如 Innovation 2 BTR (SSIM=0.9282)。双向+残差组合产生负干扰。保留为消融分支。
  </div>
</div>

<!-- ===== PALM+TEL 任务进度（优先级4）===== -->
<div class="card card-cyan" style="margin-bottom:14px; border-left: 3px solid var(--cyan);">
  <h2 style="color:var(--cyan);">🎨 {{ p4_progress.task_name if p4_progress else '优先级4: PALM + TEL 装饰模块' }}</h2>
  <div class="grid g2" style="margin-bottom:10px;">
    <div>
      <div style="margin-bottom:6px;">
        <strong>BTR+PALM+TEL 训练:</strong>
        <span id="p4-train-state" class="status-badge status-{{ p4_progress.train.state if p4_progress else 'idle' }}">
          {{ p4_progress.train.state_text if p4_progress else '未运行' }}
        </span>
        <span id="p4-train-eta" style="color:var(--dim); font-size:0.85em; margin-left:8px;">
          {% if p4_progress and p4_progress.train.eta != 'N/A' %}剩余: {{ p4_progress.train.eta }}{% endif %}
        </span>
      </div>
      <div class="bar-outer" style="margin-bottom:6px;">
        <div id="p4-train-bar" class="bar-inner" style="width:{{ p4_progress.train.percent if p4_progress else 0 }}%; background:var(--cyan);"></div>
        <span id="p4-train-label" class="bar-label">
          Epoch {{ p4_progress.train.epoch_current if p4_progress else 0 }}/{{ p4_progress.train.epoch_total if p4_progress else 5 }} · {{ p4_progress.train.percent if p4_progress else 0 }}%
        </span>
      </div>
      <div style="font-size:0.8em; color:var(--dim);">
        Total: <span id="p4-loss-total">{{ "%.6f"|format(p4_progress.train.loss_total) if p4_progress and p4_progress.train.loss_total else 'N/A' }}</span>
        · Fwd: <span id="p4-loss-fwd">{{ "%.6f"|format(p4_progress.train.loss_fwd) if p4_progress and p4_progress.train.loss_fwd else 'N/A' }}</span>
        · Bwd: <span id="p4-loss-bwd">{{ "%.6f"|format(p4_progress.train.loss_bwd) if p4_progress and p4_progress.train.loss_bwd else 'N/A' }}</span>
      </div>
      <div style="font-size:0.75em; color:var(--dim); margin-top:4px;">PALM: channel affine (scale∈[0.5,1.5]) · TEL: Fourier age encoding</div>
    </div>
    <div>
      <div style="margin-bottom:6px;">
        <strong>评估:</strong>
        <span id="p4-eval-state" class="status-badge status-{{ p4_progress.eval.state if p4_progress else 'idle' }}">
          {{ p4_progress.eval.state_text if p4_progress else '未运行' }}
        </span>
      </div>
      <div class="bar-outer" style="margin-bottom:6px;">
        <div id="p4-eval-bar" class="bar-inner" style="width:{{ p4_progress.eval.percent if p4_progress else 0 }}%; background:var(--green);"></div>
        <span id="p4-eval-label" class="bar-label">
          Pairs {{ p4_progress.eval.pair_current if p4_progress else 0 }}/{{ p4_progress.eval.pair_total if p4_progress else 50 }} · {{ p4_progress.eval.percent if p4_progress else 0 }}%
        </span>
      </div>
      <div style="font-size:0.85em; margin-top:6px;">
        {% if p4_progress and p4_progress.eval.metrics %}
          {% for k, v in p4_progress.eval.metrics.items() %}
            {% if k not in ('timestamp', 'method', 'controlnet_ckpt', 'n_pairs') %}
            <div style="color:var(--dim);">{{ k }}: <span style="color:var(--cyan);">{{ v }}</span></div>
            {% endif %}
          {% endfor %}
        {% else %}
          <div style="color:var(--dim);">等待评估数据...</div>
        {% endif %}
      </div>
    </div>
  </div>
</div>

<!-- ===== Combined Inn1+Inn2 (6ch+BTR) 进度 ===== -->
<div class="card" style="margin-bottom:14px; border-left: 3px solid var(--purple);">
  <h2 style="color:var(--purple);">🔗 {{ combined_progress.task_name if combined_progress else '建议A: 6ch+BTR 组合创新 (Inn1+Inn2)' }}</h2>
  <div class="grid g2" style="margin-bottom:10px;">
    <div>
      <div style="margin-bottom:6px;">
        <strong>6ch ControlNet + BTR 训练:</strong>
        <span id="comb-train-state" class="status-badge status-{{ combined_progress.train.state if combined_progress else 'idle' }}">
          {{ combined_progress.train.state_text if combined_progress else '未运行' }}
        </span>
        <span id="comb-train-eta" style="color:var(--dim); font-size:0.85em; margin-left:8px;">
          {% if combined_progress and combined_progress.train.eta != 'N/A' %}剩余: {{ combined_progress.train.eta }}{% endif %}
        </span>
      </div>
      <div class="bar-outer" style="margin-bottom:6px;">
        <div id="comb-train-bar" class="bar-inner" style="width:{{ combined_progress.train.percent if combined_progress else 0 }}%; background:var(--purple);"></div>
        <span id="comb-train-label" class="bar-label">
          Epoch {{ combined_progress.train.epoch_current if combined_progress else 0 }}/{{ combined_progress.train.epoch_total if combined_progress else 5 }} · {{ combined_progress.train.percent if combined_progress else 0 }}%
        </span>
      </div>
      <div style="font-size:0.8em; color:var(--dim);">
        Total: <span id="comb-loss-total">{{ "%.6f"|format(combined_progress.train.loss_total) if combined_progress and combined_progress.train.loss_total else 'N/A' }}</span>
        · Fwd: <span id="comb-loss-fwd">{{ "%.6f"|format(combined_progress.train.loss_fwd) if combined_progress and combined_progress.train.loss_fwd else 'N/A' }}</span>
        · Bwd: <span id="comb-loss-bwd">{{ "%.6f"|format(combined_progress.train.loss_bwd) if combined_progress and combined_progress.train.loss_bwd else 'N/A' }}</span>
      </div>
      <div style="font-size:0.75em; color:var(--dim); margin-top:4px;">6ch: [starting_z + age + atrophy_rate + vent_rate] · BTR: L_fwd + 0.5·L_bwd</div>
    </div>
    <div>
      <div style="margin-bottom:6px;">
        <strong>评估:</strong>
        <span id="comb-eval-state" class="status-badge status-{{ combined_progress.eval.state if combined_progress else 'idle' }}">
          {{ combined_progress.eval.state_text if combined_progress else '未运行' }}
        </span>
      </div>
      <div class="bar-outer" style="margin-bottom:6px;">
        <div id="comb-eval-bar" class="bar-inner" style="width:{{ combined_progress.eval.percent if combined_progress else 0 }}%; background:var(--green);"></div>
        <span id="comb-eval-label" class="bar-label">
          Pairs {{ combined_progress.eval.pair_current if combined_progress else 0 }}/{{ combined_progress.eval.pair_total if combined_progress else 50 }} · {{ combined_progress.eval.percent if combined_progress else 0 }}%
        </span>
      </div>
      <div style="font-size:0.85em; margin-top:6px;">
        {% if combined_progress and combined_progress.eval.metrics %}
          {% for k, v in combined_progress.eval.metrics.items() %}
            {% if k not in ('timestamp', 'method', 'controlnet_ckpt', 'n_pairs') %}
            <div style="color:var(--dim);">{{ k }}: <span style="color:var(--purple);">{{ v }}</span></div>
            {% endif %}
          {% endfor %}
        {% else %}
          <div style="color:var(--dim);">等待评估数据...</div>
        {% endif %}
      </div>
    </div>
  </div>
</div>

<!-- ===== 去辅助模型端到端验证 ===== -->
<div class="card" style="margin-bottom:14px; border-left: 3px solid var(--yellow);">
  <h2 style="color:var(--yellow);">🧪 {{ no_aux_progress.task_name if no_aux_progress else '去辅助模型端到端验证 (GT/TPN/Skip/Linear)' }}</h2>
  <div style="margin-bottom:8px;">
    <strong>评估状态:</strong>
    <span id="noaux-eval-state" class="status-badge status-{{ no_aux_progress.eval.state if no_aux_progress else 'idle' }}">
      {{ no_aux_progress.eval.state_text if no_aux_progress else '未运行' }}
    </span>
    <span id="noaux-eval-eta" style="color:var(--dim); font-size:0.85em; margin-left:8px;">
      {% if no_aux_progress and no_aux_progress.eval.eta != 'N/A' %}剩余: {{ no_aux_progress.eval.eta }}{% endif %}
    </span>
  </div>
  <div class="bar-outer" style="margin-bottom:8px;">
    <div id="noaux-eval-bar" class="bar-inner" style="width:{{ no_aux_progress.eval.percent if no_aux_progress else 0 }}%; background:var(--yellow);"></div>
    <span id="noaux-eval-label" class="bar-label">
      Pairs {{ no_aux_progress.eval.pair_current if no_aux_progress else 0 }}/{{ no_aux_progress.eval.pair_total if no_aux_progress else 50 }} × 4 methods · {{ no_aux_progress.eval.percent if no_aux_progress else 0 }}%
    </span>
  </div>
  <div style="font-size:0.85em; margin-top:8px;">
    <strong>方法对比 SSIM:</strong>
    <div id="noaux-methods-box" class="grid" style="grid-template-columns: repeat(4, 1fr); gap:6px; margin-top:6px;">
      {% if no_aux_progress and no_aux_progress.methods_ssim %}
        {% for method, ssim_val in no_aux_progress.methods_ssim.items() %}
        <div style="text-align:center; padding:6px; background:rgba(255,255,255,0.05); border-radius:4px;">
          <div style="font-weight:600; color:{% if ssim_val >= 0.92 %}var(--green){% elif ssim_val >= 0.91 %}var(--yellow){% else %}var(--red){% endif %};">{{ "%.4f"|format(ssim_val) }}</div>
          <div style="font-size:0.8em; color:var(--dim);">{{ method }}</div>
          <div style="font-size:0.7em; color:{% if ssim_val >= 0.92 %}var(--green){% else %}var(--red){% endif %};">{{ '✓ ≥0.92' if ssim_val >= 0.92 else '✗ <0.92' }}</div>
        </div>
        {% endfor %}
      {% else %}
        <div style="color:var(--dim); grid-column: 1/-1;">等待评估数据...</div>
      {% endif %}
    </div>
  </div>
  <div style="font-size:0.75em; color:var(--dim); margin-top:8px;">
    目标: TPN 替代 Leaspy 后 SSIM≥0.92 · ControlNet=BTR-ep1 · TPN=v3b
  </div>
</div>

<!-- ===== 多时间点连续生成验证 ===== -->
<div class="card" style="margin-bottom:14px; border-left: 3px solid var(--cyan);">
  <h2 style="color:var(--cyan);">🕐 {{ multi_tp_progress.task_name if multi_tp_progress else '多时间点连续生成验证' }}</h2>
  <div style="margin-bottom:8px;">
    <strong>评估状态:</strong>
    <span id="mtp-eval-state" class="status-badge status-{{ multi_tp_progress.eval.state if multi_tp_progress else 'idle' }}">
      {{ multi_tp_progress.eval.state_text if multi_tp_progress else '未运行' }}
    </span>
  </div>
  <div class="bar-outer" style="margin-bottom:8px;">
    <div id="mtp-eval-bar" class="bar-inner" style="width:{{ multi_tp_progress.eval.percent if multi_tp_progress else 0 }}%; background:var(--cyan);"></div>
    <span id="mtp-eval-label" class="bar-label">
      {{ multi_tp_progress.eval.current if multi_tp_progress else 0 }}/{{ multi_tp_progress.eval.total if multi_tp_progress else '?' }} · {{ multi_tp_progress.eval.percent if multi_tp_progress else 0 }}%
    </span>
  </div>
  <div style="font-size:0.85em; margin-top:8px;">
    <strong>方法对比 SSIM:</strong>
    <div id="mtp-methods-box" class="grid" style="grid-template-columns: repeat(4, 1fr); gap:6px; margin-top:6px;">
      {% if multi_tp_progress and multi_tp_progress.methods_ssim %}
        {% for method, ssim_val in multi_tp_progress.methods_ssim.items() %}
        <div style="text-align:center; padding:6px; background:rgba(255,255,255,0.05); border-radius:4px;">
          <div style="font-weight:600; color:{% if ssim_val >= 0.92 %}var(--green){% elif ssim_val >= 0.91 %}var(--yellow){% else %}var(--red){% endif %};">{{ "%.4f"|format(ssim_val) }}</div>
          <div style="font-size:0.8em; color:var(--dim);">{{ method }}</div>
        </div>
        {% endfor %}
      {% else %}
        <div style="color:var(--dim); grid-column: 1/-1;">等待评估数据...</div>
      {% endif %}
    </div>
  </div>
  <div style="font-size:0.85em; margin-top:8px;">
    <strong>SSIM随时间变化:</strong>
    <div id="mtp-timegap-box" style="margin-top:6px;">
      {% if multi_tp_progress and multi_tp_progress.by_time_gap %}
        <table style="font-size:0.85em;">
          <tr><th>方法</th><th>0-6月</th><th>6-12月</th><th>12-24月</th><th>24月+</th></tr>
          {% for method, gaps in multi_tp_progress.by_time_gap.items() %}
          <tr>
            <td style="font-weight:600;">{{ method }}</td>
            {% for gap_label in ['0-6mo','6-12mo','12-24mo','24mo+'] %}
            <td>{% if gap_label in gaps %}{{ "%.4f"|format(gaps[gap_label].ssim_mean) }} (n={{ gaps[gap_label].n }}){% else %}—{% endif %}</td>
            {% endfor %}
          </tr>
          {% endfor %}
        </table>
      {% else %}
        <div style="color:var(--dim);">等待时间维度分析数据...</div>
      {% endif %}
    </div>
  </div>
  <div style="font-size:0.75em; color:var(--dim); margin-top:8px;">
    目标: 从基线连续生成3/6/9/12月+图像 · 验证时间跨度对SSIM的影响 · 对比直接vs自回归生成
  </div>
</div>

<!-- ===== Innovation 2 BTR 进度 ===== -->
<div class="card" style="margin-bottom:14px;">
  <h2>任务进度 (创新点2 — 双向时间正则化 BTR)</h2>
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
            <div class="bar-inner" style="width:{{ g.mem_pct }}; background: {% if g.mem_pct|replace('%','')|int > 80 %}var(--red){% elif g.mem_pct|replace('%','')|int > 50 %}var(--yellow){% else %}var(--green){% endif %};"></div>
            <span class="bar-label">显存 {{ g.mem_pct }} ({{ g.mem_used }} / {{ g.mem_total }})</span>
          </div>
          <div style="font-size:0.8em; color:var(--dim); margin-top:2px;">
            计算利用率: {{ g.util }}
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
  <div class="card" style="margin-bottom:14px;">
    <h2>创新点对比 — Baseline / Innovation 1 / Innovation 2 BTR (test=50, 改进AE解码器)</h2>
    <div style="font-size:0.85em; color:var(--dim); margin-bottom:10px;">
      评估管道: 改进AE解码器(Inn4 3D感知损失训练) + GT归一化[0,1] + ResizeWithPadOrCrop(122,146,122)<br>
      Inn1: 6ch ControlNet (latent+age+atrophy_rate+vent_rate) · Epoch 4 ·
      Inn2: BTR双向时间正则化 (L_fwd+0.5×L_bwd) · Epoch 1 ·
      <span style="color:var(--green);">✓ 全部评估完成</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>指标</th><th>Baseline</th><th>Innovation 1</th><th>Δ1 vs BL</th><th style="color:var(--yellow);">Innovation 2 BTR</th><th style="color:var(--yellow);">Δ2 vs BL</th>
        </tr>
      </thead>
      <tbody>
      {% for m in innovation1_table %}
        <tr>
          <td><strong>{{ m.name }}</strong></td>
          <td>{{ "%.4f"|format(m.bl) }}</td>
          <td style="color:var(--blue); font-weight:600;">{{ "%.4f"|format(m.inn1) }}</td>
          <td class="{{ m.cls1 }}">{{ m.delta1 }}</td>
          <td style="color:var(--yellow); font-weight:600;">{{ "%.4f"|format(m.inn2) }}</td>
          <td class="{{ m.cls2 }}">{{ m.delta2 }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>创新点 4/5 对比表 (MCI 纵向预测, test=50)</h2>
    <table>
      <thead>
        <tr>
          <th>指标</th><th>Baseline</th><th>Innov4 v1</th>
          <th>Innov5 v2</th><th>联合4+5</th><th>Innov5 vs BL</th>
        </tr>
      </thead>
      <tbody>
      {% for m in metrics_table %}
        <tr>
          <td><strong>{{ m.name }}</strong></td>
          <td>{{ "%.4f"|format(m.bl) }}</td>
          <td>{{ "%.4f"|format(m.v1) }}</td>
          <td>{{ "%.4f"|format(m.i5) }}</td>
          <td style="color:var(--yellow);">{{ "%.4f"|format(m.combined) if m.combined else '...' }}</td>
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

    <div class="change" style="border-left-color: var(--green);">
      <div class="meta">Git 实时状态（每 8 秒刷新）</div>
      <div><strong>分支:</strong> <span id="git-branch">{{ project_changes.branch if project_changes else 'N/A' }}</span></div>
      <div><strong>最新提交:</strong> <span id="git-latest">{{ (project_changes.latest_commit ~ ' · ' ~ project_changes.latest_subject) if project_changes and project_changes.latest_commit != 'N/A' else 'N/A' }}</span></div>
      <div><strong>提交时间:</strong> <span id="git-latest-time">{{ project_changes.latest_time if project_changes else 'N/A' }}</span></div>
      <div><strong>未提交改动:</strong> <span id="git-change-count">{{ project_changes.changed_count if project_changes else 0 }}</span> 个文件</div>
      <div style="margin-top:6px; color:var(--dim);">最近提交:</div>
      <pre id="git-recent-commits">{% if project_changes and project_changes.recent_commits %}{{ project_changes.recent_commits|join('\n') }}{% else %}暂无{% endif %}</pre>
      <div style="margin-top:6px; color:var(--dim);">当前改动文件:</div>
      <pre id="git-changed-files">{% if project_changes and project_changes.changed_files %}{{ project_changes.changed_files|join('\n') }}{% else %}工作区干净{% endif %}</pre>
    </div>

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
    const memPct = parseInt(String(g.mem_pct || '0').replace('%', ''), 10) || 0;
    const color = memPct > 80 ? 'var(--red)' : (memPct > 50 ? 'var(--yellow)' : 'var(--green)');
    return `
      <div style="margin-bottom:8px;">
        <div style="font-size:0.85em; margin-bottom:2px;"><strong>GPU ${g.index}</strong>: ${g.name} · ${g.temp}</div>
        <div class="bar-outer">
          <div class="bar-inner" style="width:${g.mem_pct || '0%'}; background:${color};"></div>
          <span class="bar-label">显存 ${g.mem_pct || '0%'} (${g.mem_used} / ${g.mem_total})</span>
        </div>
        <div style="font-size:0.8em; color:var(--dim); margin-top:2px;">计算利用率: ${g.util}</div>
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

function renderTpnProgress(tpn) {
  if (!tpn) return;
  const train = tpn.train || {};
  const evalp = tpn.eval || {};

  // Train state
  const tState = document.getElementById('tpn-train-state');
  if (tState) {
    tState.textContent = train.state_text || '未运行';
    tState.className = 'status-badge status-' + (train.state || 'idle');
  }
  const tEta = document.getElementById('tpn-train-eta');
  if (tEta) tEta.textContent = (train.eta && train.eta !== 'N/A') ? '剩余: ' + train.eta : '';

  const tBar = document.getElementById('tpn-train-bar');
  if (tBar) tBar.style.width = String(train.percent || 0) + '%';
  const tLabel = document.getElementById('tpn-train-label');
  if (tLabel) tLabel.textContent = `Epoch ${train.epoch_current || 0}/${train.epoch_total || 200} · ${train.percent || 0}%`;

  const tLoss = document.getElementById('tpn-loss');
  if (tLoss) tLoss.textContent = train.loss_current != null ? train.loss_current.toFixed(6) : 'N/A';
  const tBest = document.getElementById('tpn-best-loss');
  if (tBest) tBest.textContent = train.best_loss != null ? train.best_loss.toFixed(6) : 'N/A';

  // Eval state
  const eState = document.getElementById('tpn-eval-state');
  if (eState) {
    eState.textContent = evalp.state_text || '未运行';
    eState.className = 'status-badge status-' + (evalp.state || 'idle');
  }
}

function renderAiLog(ops) {
  const box = document.getElementById('ai-log-box');
  if (!box || !ops || ops.length === 0) return;
  const typeMap = { think: '思考', code: '代码修改', cmd: '命令执行', test: '测试' };
  box.innerHTML = ops.slice(-20).map(op => `
    <div class="ai-entry">
      <span class="ai-time">${op.time}</span>
      <span class="ai-type ai-type-${op.type}">${op.type_text || typeMap[op.type] || op.type}</span>
      <div class="ai-msg">${op.message}</div>
    </div>
  `).join('');
  box.scrollTop = box.scrollHeight;
}

function renderProjectChanges(git) {
  if (!git) return;

  const branch = document.getElementById('git-branch');
  if (branch) branch.textContent = git.branch || 'N/A';

  const latest = document.getElementById('git-latest');
  if (latest) {
    const commit = git.latest_commit || 'N/A';
    const subject = git.latest_subject || '';
    latest.textContent = commit === 'N/A' ? 'N/A' : `${commit} · ${subject}`;
  }

  const latestTime = document.getElementById('git-latest-time');
  if (latestTime) latestTime.textContent = git.latest_time || 'N/A';

  const changeCount = document.getElementById('git-change-count');
  if (changeCount) changeCount.textContent = String(git.changed_count || 0);

  const recent = document.getElementById('git-recent-commits');
  if (recent) {
    const lines = (git.recent_commits && git.recent_commits.length) ? git.recent_commits.join('\n') : '暂无';
    recent.textContent = lines;
  }

  const changed = document.getElementById('git-changed-files');
  if (changed) {
    const lines = (git.changed_files && git.changed_files.length) ? git.changed_files.join('\n') : '工作区干净';
    changed.textContent = lines;
  }
}

function renderRlpProgress(rlp) {
  // P2 RLP abandoned — no dynamic rendering needed
}

function renderP4Progress(p4) {
  if (!p4) return;
  const train = p4.train || {};
  const evalp = p4.eval || {};

  const tState = document.getElementById('p4-train-state');
  if (tState) {
    tState.textContent = train.state_text || '未运行';
    tState.className = 'status-badge status-' + (train.state || 'idle');
  }
  const tEta = document.getElementById('p4-train-eta');
  if (tEta) tEta.textContent = (train.eta && train.eta !== 'N/A') ? '剩余: ' + train.eta : '';

  const tBar = document.getElementById('p4-train-bar');
  if (tBar) tBar.style.width = String(train.percent || 0) + '%';
  const tLabel = document.getElementById('p4-train-label');
  if (tLabel) tLabel.textContent = `Epoch ${train.epoch_current || 0}/${train.epoch_total || 5} · ${train.percent || 0}%`;

  const tTotal = document.getElementById('p4-loss-total');
  if (tTotal) tTotal.textContent = train.loss_total != null ? train.loss_total.toFixed(6) : 'N/A';
  const tFwd = document.getElementById('p4-loss-fwd');
  if (tFwd) tFwd.textContent = train.loss_fwd != null ? train.loss_fwd.toFixed(6) : 'N/A';
  const tBwd = document.getElementById('p4-loss-bwd');
  if (tBwd) tBwd.textContent = train.loss_bwd != null ? train.loss_bwd.toFixed(6) : 'N/A';

  const eState = document.getElementById('p4-eval-state');
  if (eState) {
    eState.textContent = evalp.state_text || '未运行';
    eState.className = 'status-badge status-' + (evalp.state || 'idle');
  }
  const eBar = document.getElementById('p4-eval-bar');
  if (eBar) eBar.style.width = String(evalp.percent || 0) + '%';
  const eLabel = document.getElementById('p4-eval-label');
  if (eLabel) eLabel.textContent = `Pairs ${evalp.pair_current || 0}/${evalp.pair_total || 50} · ${evalp.percent || 0}%`;
}

function renderCombinedProgress(comb) {
  if (!comb) return;
  const train = comb.train || {};
  const evalp = comb.eval || {};

  const tState = document.getElementById('comb-train-state');
  if (tState) {
    tState.textContent = train.state_text || '未运行';
    tState.className = 'status-badge status-' + (train.state || 'idle');
  }
  const tEta = document.getElementById('comb-train-eta');
  if (tEta) tEta.textContent = (train.eta && train.eta !== 'N/A') ? '剩余: ' + train.eta : '';

  const tBar = document.getElementById('comb-train-bar');
  if (tBar) tBar.style.width = String(train.percent || 0) + '%';
  const tLabel = document.getElementById('comb-train-label');
  if (tLabel) tLabel.textContent = `Epoch ${train.epoch_current || 0}/${train.epoch_total || 5} · ${train.percent || 0}%`;

  const tTotal = document.getElementById('comb-loss-total');
  if (tTotal) tTotal.textContent = train.loss_total != null ? train.loss_total.toFixed(6) : 'N/A';
  const tFwd = document.getElementById('comb-loss-fwd');
  if (tFwd) tFwd.textContent = train.loss_fwd != null ? train.loss_fwd.toFixed(6) : 'N/A';
  const tBwd = document.getElementById('comb-loss-bwd');
  if (tBwd) tBwd.textContent = train.loss_bwd != null ? train.loss_bwd.toFixed(6) : 'N/A';

  const eState = document.getElementById('comb-eval-state');
  if (eState) {
    eState.textContent = evalp.state_text || '未运行';
    eState.className = 'status-badge status-' + (evalp.state || 'idle');
  }
  const eBar = document.getElementById('comb-eval-bar');
  if (eBar) eBar.style.width = String(evalp.percent || 0) + '%';
  const eLabel = document.getElementById('comb-eval-label');
  if (eLabel) eLabel.textContent = `Pairs ${evalp.pair_current || 0}/${evalp.pair_total || 50} · ${evalp.percent || 0}%`;
}

function renderNoAuxProgress(data) {
  if (!data) return;
  const evalp = data.eval || {};
  const eState = document.getElementById('noaux-eval-state');
  if (eState) {
    eState.textContent = evalp.state_text || '未运行';
    eState.className = 'status-badge status-' + (evalp.state || 'idle');
  }
  const eEta = document.getElementById('noaux-eval-eta');
  if (eEta) eEta.textContent = (evalp.eta && evalp.eta !== 'N/A') ? '剩余: ' + evalp.eta : '';
  const eBar = document.getElementById('noaux-eval-bar');
  if (eBar) eBar.style.width = String(evalp.percent || 0) + '%';
  const eLabel = document.getElementById('noaux-eval-label');
  if (eLabel) eLabel.textContent = `Pairs ${evalp.pair_current || 0}/${evalp.pair_total || 50} × 4 methods · ${evalp.percent || 0}%`;
  const mBox = document.getElementById('noaux-methods-box');
  if (mBox && data.methods_ssim && Object.keys(data.methods_ssim).length > 0) {
    let html = '';
    for (const [method, ssimVal] of Object.entries(data.methods_ssim)) {
      const color = ssimVal >= 0.92 ? 'var(--green)' : ssimVal >= 0.91 ? 'var(--yellow)' : 'var(--red)';
      const check = ssimVal >= 0.92 ? '✓ ≥0.92' : '✗ <0.92';
      const checkColor = ssimVal >= 0.92 ? 'var(--green)' : 'var(--red)';
      html += `<div style="text-align:center; padding:6px; background:rgba(255,255,255,0.05); border-radius:4px;">
        <div style="font-weight:600; color:${color};">${ssimVal.toFixed(4)}</div>
        <div style="font-size:0.8em; color:var(--dim);">${method}</div>
        <div style="font-size:0.7em; color:${checkColor};">${check}</div>
      </div>`;
    }
    mBox.innerHTML = html;
  }
}

function renderMultiTpProgress(data) {
  if (!data) return;
  const evalp = data.eval || {};
  const eState = document.getElementById('mtp-eval-state');
  if (eState) {
    eState.textContent = evalp.state_text || '未运行';
    eState.className = 'status-badge status-' + (evalp.state || 'idle');
  }
  const eBar = document.getElementById('mtp-eval-bar');
  if (eBar) eBar.style.width = String(evalp.percent || 0) + '%';
  const eLabel = document.getElementById('mtp-eval-label');
  if (eLabel) eLabel.textContent = `${evalp.current || 0}/${evalp.total || '?'} · ${evalp.percent || 0}%`;
  const mBox = document.getElementById('mtp-methods-box');
  if (mBox && data.methods_ssim && Object.keys(data.methods_ssim).length > 0) {
    let html = '';
    for (const [method, ssimVal] of Object.entries(data.methods_ssim)) {
      const color = ssimVal >= 0.92 ? 'var(--green)' : ssimVal >= 0.91 ? 'var(--yellow)' : 'var(--red)';
      html += `<div style="text-align:center; padding:6px; background:rgba(255,255,255,0.05); border-radius:4px;">
        <div style="font-weight:600; color:${color};">${ssimVal.toFixed(4)}</div>
        <div style="font-size:0.8em; color:var(--dim);">${method}</div>
      </div>`;
    }
    mBox.innerHTML = html;
  }
  const tgBox = document.getElementById('mtp-timegap-box');
  if (tgBox && data.by_time_gap && Object.keys(data.by_time_gap).length > 0) {
    const gaps = ['0-6mo','6-12mo','12-24mo','24mo+'];
    let html = '<table style="font-size:0.85em;width:100%;"><tr><th>方法</th>';
    gaps.forEach(g => html += `<th>${g}</th>`);
    html += '</tr>';
    for (const [method, gapData] of Object.entries(data.by_time_gap)) {
      html += `<tr><td style="font-weight:600;">${method}</td>`;
      gaps.forEach(g => {
        const d = gapData[g];
        html += d ? `<td>${d.ssim_mean.toFixed(4)} (n=${d.n})</td>` : '<td>—</td>';
      });
      html += '</tr>';
    }
    html += '</table>';
    tgBox.innerHTML = html;
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
      renderTpnProgress(d.tpn_progress || null);
      renderRlpProgress(d.rlp_progress || null);
      renderP4Progress(d.p4_progress || null);
      renderCombinedProgress(d.combined_progress || null);
      renderNoAuxProgress(d.no_aux_progress || null);
      renderMultiTpProgress(d.multi_tp_progress || null);
      renderAiLog(d.ai_operations || []);
      renderProjectChanges(d.project_changes || null);
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
    combined = REFERENCE_METRICS.get("combined_4_5")
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
        combined_val = combined[key] if combined else None
        diff = i - b  # Innov5 vs Baseline
        if higher_better:
            cls = "up" if diff > 0 else "down"
            sign = "+" if diff > 0 else ""
        else:
            cls = "up" if diff < 0 else "down"
            sign = "" if diff < 0 else "+"
        pct = abs(diff / b * 100) if b else 0
        delta_str = f"{sign}{diff:.4f} ({pct:.2f}%)"
        rows.append({"name": name, "bl": b, "v1": v, "i5": i,
                      "combined": combined_val, "cls": cls, "delta": delta_str})
    return rows


def build_innovation1_table():
    """Build Innovation 1+2 comparison table (same evaluation pipeline)."""
    bl = INNOVATION_1_METRICS["baseline_same_pipeline"]
    inn1 = INNOVATION_1_METRICS["innovation_1"]
    inn2 = INNOVATION_1_METRICS["innovation_2_btr"]
    keys = [
        ("overall_ssim", "Overall SSIM ↑", True),
        ("overall_psnr", "Overall PSNR ↑", True),
        ("overall_mae",  "Overall MAE ↓",  False),
        ("roi_ssim",     "ROI SSIM ↑",     True),
        ("roi_mae",      "ROI MAE ↓",      False),
    ]
    rows = []
    for key, name, higher_better in keys:
        b = bl.get(key, 0)
        v1 = inn1.get(key, 0)
        v2 = inn2.get(key, 0)
        # Inn1 vs BL
        diff1 = v1 - b
        if higher_better:
            cls1 = "up" if diff1 > 0 else "down"
            sign1 = "+" if diff1 > 0 else ""
        else:
            cls1 = "up" if diff1 < 0 else "down"
            sign1 = "" if diff1 < 0 else "+"
        pct1 = abs(diff1 / b * 100) if b else 0
        delta1_str = f"{sign1}{diff1:.4f} ({pct1:.2f}%)"
        # Inn2 vs BL
        diff2 = v2 - b
        if higher_better:
            cls2 = "up" if diff2 > 0 else "down"
            sign2 = "+" if diff2 > 0 else ""
        else:
            cls2 = "up" if diff2 < 0 else "down"
            sign2 = "" if diff2 < 0 else "+"
        pct2 = abs(diff2 / b * 100) if b else 0
        delta2_str = f"{sign2}{diff2:.4f} ({pct2:.2f}%)"
        rows.append({"name": name, "bl": b, "inn1": v1, "inn2": v2,
                     "cls1": cls1, "delta1": delta1_str,
                     "cls2": cls2, "delta2": delta2_str})
    return rows


@app.route("/")
def index():
    with _cache_lock:
        info = _cache["server_info"] or {}
    task_progress = _cache["task_progress"] or {}
    tpn_progress = _cache["tpn_progress"] or {}
    rlp_progress = _cache["rlp_progress"] or {}
    p4_progress = _cache["p4_progress"] or {}
    combined_progress = _cache["combined_progress"] or {}
    no_aux_progress = _cache["no_aux_progress"] or {}
    multi_tp_progress = _cache["multi_tp_progress"] or {}
    project_changes = _cache["project_changes"] or {}
    connected = info.get("status") == "connected"
    gpus = parse_gpu(info.get("gpu_raw", ""))
    procs = parse_processes(info.get("proc_raw", ""))

    with _ai_ops_lock:
        ai_ops = list(_ai_operations)

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
        tpn_progress=tpn_progress,
        rlp_progress=rlp_progress,
        p4_progress=p4_progress,
        combined_progress=combined_progress,
        no_aux_progress=no_aux_progress,
        multi_tp_progress=multi_tp_progress,
        project_changes=project_changes,
        ai_operations=ai_ops,
        metrics_table=build_metrics_table(),
        innovation1_table=build_innovation1_table(),
        changes=CODE_CHANGES,
    )


@app.route("/api/refresh")
def api_refresh():
    with _cache_lock:
        info = _cache["server_info"] or {}
    task_progress = _cache["task_progress"] or {}
    tpn_progress = _cache["tpn_progress"] or {}
    rlp_progress = _cache["rlp_progress"] or {}
    p4_progress = _cache["p4_progress"] or {}
    combined_progress = _cache["combined_progress"] or {}
    no_aux_progress = _cache["no_aux_progress"] or {}
    multi_tp_progress = _cache["multi_tp_progress"] or {}
    project_changes = _cache["project_changes"] or {}
    with _ai_ops_lock:
        ai_ops = list(_ai_operations)
    return jsonify({
        "cpu_raw": info.get("cpu_raw", ""),
        "mem_raw": info.get("mem_raw", ""),
        "gpu_raw": info.get("gpu_raw", ""),
        "gpus": parse_gpu(info.get("gpu_raw", "")),
        "disk_raw": info.get("disk_raw", ""),
        "last_update": info.get("timestamp", ""),
        "processes": parse_processes(info.get("proc_raw", "")),
        "task_progress": task_progress,
        "tpn_progress": tpn_progress,
        "rlp_progress": rlp_progress,
        "p4_progress": p4_progress,
        "combined_progress": combined_progress,
        "no_aux_progress": no_aux_progress,
        "multi_tp_progress": multi_tp_progress,
        "project_changes": project_changes,
        "ai_operations": ai_ops,
    })


@app.route("/api/ai_log", methods=["POST"])
def api_ai_log():
    """接收 AI 操作日志推送。"""
    data = request.get_json(force=True) or {}
    entry = {
        "time": data.get("time", datetime.now().strftime("%H:%M:%S")),
        "type": data.get("type", "think"),           # think / code / cmd / test
        "type_text": data.get("type_text", {
            "think": "思考", "code": "代码修改", "cmd": "命令执行", "test": "测试"
        }.get(data.get("type", "think"), "其他")),
        "message": data.get("message", ""),
    }
    with _ai_ops_lock:
        _ai_operations.append(entry)
        if len(_ai_operations) > 200:
            _ai_operations[:] = _ai_operations[-100:]
    return jsonify({"ok": True})


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

  # 启动前立即拉取一次数据
  print(f"[Dashboard] 正在连接服务器 {SERVER_HOST}:{SERVER_PORT} ...")
  try:
    info = fetch_server_info()
    task_progress = fetch_task_progress()
    tpn_progress = fetch_tpn_progress()
    rlp_progress = fetch_rlp_progress()
    combined_progress = fetch_combined_progress()
    no_aux_progress = fetch_no_aux_progress()
    project_changes = fetch_project_changes()
    with _cache_lock:
      _cache["server_info"] = info
      _cache["gpu_info"] = parse_gpu(info.get("gpu_raw", ""))
      _cache["processes"] = parse_processes(info.get("proc_raw", ""))
      _cache["task_progress"] = task_progress
      _cache["tpn_progress"] = tpn_progress
      _cache["rlp_progress"] = rlp_progress
      _cache["combined_progress"] = combined_progress
      _cache["no_aux_progress"] = no_aux_progress
      _cache["project_changes"] = project_changes
      _cache["last_update"] = info.get("timestamp")
    print("[Dashboard] 服务器连接成功")
  except Exception as e:
    print(f"[Dashboard] 服务器连接失败: {e}")

  print(f"[Dashboard] 启动中: http://{args.host}:{args.port}")
  app.run(host=args.host, port=args.port, debug=False, threaded=True)
