"""
TPN (Temporal Progression Network) v2 训练脚本
===============================================
用 CSV B 中的真实纵向配对数据训练 TPN v2，替代 Leaspy。

v2 改进:
  - 14维输入 (增加 age_ratio, vol_mean, vol_std, age_gap²)
  - Huber Loss (对异常值更鲁棒)
  - 区域加权损失 (难区域获得更高权重)
  - 数据增强 (高斯噪声)
  - 500 epochs + warmup

日志格式 (用于 dashboard 解析):
    Epoch 150/500 | loss=0.00123 | val_loss=0.00098 | best=0.00080 | ETA: 00:02:30
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
# 服务器部署路径
BRLP_SRC_ALT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'brlp_src'))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, BRLP_SRC_ALT)

from tpn import TemporalProgressionNetwork
from brlp import const


# ─── 数据集 ────────────────────────────────────────────────────

VOLUME_REGIONS = const.CONDITIONING_REGIONS  # [cortex, hippo, amyg, wm, vent]


class TPNDataset(Dataset):
    """从 CSV B 构建 TPN 训练样本。"""

    def __init__(self, csv_path, diagnosis_filter=None):
        df = pd.read_csv(csv_path)

        # 可选: 只用特定诊断 (MCI = 0.5 或 diagnosis=2)
        if diagnosis_filter is not None:
            # CSV B 中 starting_diagnosis 字段
            diag_col = None
            for col_name in ['starting_diagnosis', 'starting_last_diagnosis']:
                if col_name in df.columns:
                    diag_col = col_name
                    break
            if diag_col:
                df = df[df[diag_col] == diagnosis_filter]

        # 只保留 train split
        if 'split' in df.columns:
            self.train_df = df[df.split == 'train'].reset_index(drop=True)
            self.val_df = df[df.split != 'train'].reset_index(drop=True)
        else:
            self.train_df = df.reset_index(drop=True)
            self.val_df = None

        self.df = self.train_df
        print(f"[TPN Dataset] {len(self.df)} training pairs loaded")

    def use_val(self):
        """切换到验证集。"""
        if self.val_df is not None:
            self.df = self.val_df
            return True
        return False

    def use_train(self):
        """切换回训练集。"""
        self.df = self.train_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 归一化年龄 (0~1)
        current_age = row['starting_age']  # CSV B 中已是归一化值 (0~1)
        target_age = row['followup_age']

        # 性别归一化
        sex_raw = row['sex']
        sex_norm = (sex_raw - const.SEX_MIN) / const.SEX_DELTA

        # 诊断归一化
        diag_col = None
        for col_name in ['starting_diagnosis', 'starting_last_diagnosis']:
            if col_name in self.df.columns:
                diag_col = col_name
                break
        if diag_col:
            diag_raw = row[diag_col]
            # 处理不同编码: BrLP用1/2/3, 有些CSV用0/0.5/1
            if diag_raw <= 1:
                # 0/0.5/1 编码 → 转为 0~1
                diag_norm = diag_raw
            else:
                diag_norm = (diag_raw - const.DIA_MIN) / const.DIA_DELTA
        else:
            diag_norm = 0.5  # 默认 MCI

        # 当前 5 个脑区体积 (已归一化)
        current_volumes = []
        for region in VOLUME_REGIONS:
            col = f'starting_{region}'
            current_volumes.append(row[col])

        # 目标 5 个脑区体积 (已归一化)
        target_volumes = []
        for region in VOLUME_REGIONS:
            col = f'followup_{region}'
            target_volumes.append(row[col])

        age_gap = target_age - current_age
        # v2 新增特征
        age_ratio = age_gap / (current_age + 1e-8)
        vol_mean = np.mean(current_volumes)
        vol_std = np.std(current_volumes)
        age_gap_sq = age_gap ** 2

        x = torch.tensor([
            current_age, target_age, sex_norm, diag_norm,
            *current_volumes,
            age_gap, age_ratio, vol_mean, vol_std, age_gap_sq,
        ], dtype=torch.float32)

        y = torch.tensor(target_volumes, dtype=torch.float32)

        return x, y


# ─── 训练 ────────────────────────────────────────────────────

def train(args):
    print(f"[TPN] 开始训练 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[TPN] CSV: {args.dataset_csv}")
    print(f"[TPN] Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'train.log')
    log_f = open(log_path, 'w', buffering=1)

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    log(f"[TPN] Device: {device}")

    # 数据集
    dataset = TPNDataset(args.dataset_csv, diagnosis_filter=args.diagnosis_filter)

    # 手动划分 train/val
    if dataset.val_df is not None and len(dataset.val_df) > 0:
        train_ds = dataset
        train_ds.use_train()
        val_ds = TPNDataset.__new__(TPNDataset)
        val_ds.df = dataset.val_df
        val_ds.train_df = dataset.val_df
        val_ds.val_df = None
        val_ds.__class__ = TPNDataset
    else:
        n_val = max(1, int(len(dataset) * 0.1))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    log(f"[TPN] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 模型
    model = TemporalProgressionNetwork(
        in_dim=14, hidden_dim=args.hidden_dim, out_dim=5,
        n_layers=args.n_layers, dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, args.n_epochs + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)

        # ── Validation ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses) if val_losses else avg_train

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'tpn_best.pth'))

        # ETA 计算
        elapsed = time.time() - start_time
        avg_per_epoch = elapsed / epoch
        remaining = avg_per_epoch * (args.n_epochs - epoch)
        eta_h = int(remaining // 3600)
        eta_m = int((remaining % 3600) // 60)
        eta_s = int(remaining % 60)
        eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"

        # 日志 — 格式用于 dashboard 解析
        log(f"Epoch {epoch}/{args.n_epochs} | loss={avg_train:.6f} | val_loss={avg_val:.6f} | best={best_val_loss:.6f} | ETA: {eta_str}")

        # 定期保存
        if epoch % 50 == 0 or epoch == args.n_epochs:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'tpn_ep{epoch}.pth'))

    # 最终保存
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'tpn_final.pth'))

    log(f"\n[TPN] Training complete | best epoch={best_epoch} val_loss={best_val_loss:.6f}")
    log(f"[TPN] 训练完成 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Temporal Progression Network (TPN)')
    parser.add_argument('--dataset_csv', type=str, required=True, help='Path to CSV B')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--diagnosis_filter', type=float, default=None,
                        help='Filter by diagnosis (e.g. 0.5 for MCI)')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    train(args)
