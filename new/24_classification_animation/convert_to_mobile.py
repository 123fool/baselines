#!/usr/bin/env python3
"""
为手机适配 - 转换视频格式 + 生成兼容 GIF
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def video_to_universal_gif(video_path, output_gif, max_size=1000, duration=100):
    """
    将视频转换为手机友好的通用 GIF
    
    Args:
        video_path: 视频文件路径
        output_gif: 输出 GIF 路径
        max_size: 最大尺寸（手机屏幕友好）
        duration: 每帧间隔（毫秒）
    """
    print(f"{'='*60}")
    print(f"生成手机友好 GIF")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    print(f"? 读取视频帧...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换色彩空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
        
        if frame_count % 5 == 0:
            print(f"  读取: {frame_count} 帧")
    
    cap.release()
    
    print(f"? 调整尺寸到手机屏幕...")
    # 调整尺寸
    resized_frames = []
    for i, frame in enumerate(frames):
        w, h = frame.size
        ratio = min(max_size / max(w, h), 1.0)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        resized = frame.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_frames.append(resized)
        
        if (i+1) % max(1, len(frames)//3) == 0:
            print(f"  调整: {i+1}/{len(frames)} -> {new_w}x{new_h}")
    
    print(f"? 保存 GIF (每帧 {duration}ms)...")
    resized_frames[0].save(
        output_gif,
        save_all=True,
        append_images=resized_frames[1:],
        duration=duration,
        loop=0,
        optimize=False  # 不优化，保证兼容性
    )
    
    size_mb = os.path.getsize(output_gif) / 1024 / 1024
    print(f"✓ 完成: {size_mb:.2f} MB")
    print(f"{'='*60}\n")

def create_mobile_friendly_videos(input_video):
    """
    创建多个手机友好的视频格式
    """
    video_path = Path(input_video)
    output_dir = video_path.parent
    name = video_path.stem
    
    # 读取原始视频信息
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"\n原始视频: {width}x{height} @ {fps} FPS")
    
    # 方案1: 生成通用兼容 GIF
    print("\n[方案 1] 生成通用兼容 GIF (小)")
    video_to_universal_gif(
        input_video,
        str(output_dir / f"{name}_mobile.gif"),
        max_size=720,  # 手机屏幕宽度
        duration=100
    )
    
    # 方案2: 生成 WebM (VP9 编码，兼容性好)
    print("[方案 2] 转换为 WebM (手机兼容)")
    output_webm = str(output_dir / f"{name}_mobile.webm")
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    cap = cv2.VideoCapture(input_video)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # 调整尺寸
    print(f"? 缩放视频到 720p...")
    scaled_frames = []
    scale_factor = min(720 / max(width, height), 1.0)
    new_w = int(width * scale_factor)
    new_h = int(height * scale_factor)
    
    for frame in frames:
        scaled = cv2.resize(frame, (new_w, new_h))
        scaled_frames.append(scaled)
    
    print(f"? 输出尺寸: {new_w}x{new_h}")
    print(f"? 写入 WebM...")
    out = cv2.VideoWriter(output_webm, fourcc, fps, (new_w, new_h))
    for idx, frame in enumerate(scaled_frames):
        if idx % max(1, len(scaled_frames)//3) == 0:
            print(f"  写入: {idx+1}/{len(scaled_frames)}")
        out.write(frame)
    out.release()
    
    size_mb = os.path.getsize(output_webm) / 1024 / 1024
    print(f"✓ 完成: {size_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    # 方案3: 生成 MP4 (320p 超清晰)
    print("[方案 3] 转换为 MP4 (超清晰版)")
    output_mp4 = str(output_dir / f"{name}_mobile_hq.mp4")
    
    print(f"? 缩放视频到 480p...")
    scale_factor_hq = min(480 / max(width, height), 1.0)
    new_w_hq = int(width * scale_factor_hq)
    new_h_hq = int(height * scale_factor_hq)
    
    scaled_frames_hq = []
    for frame in frames:
        scaled = cv2.resize(frame, (new_w_hq, new_h_hq))
        scaled_frames_hq.append(scaled)
    
    print(f"? 输出尺寸: {new_w_hq}x{new_h_hq}")
    print(f"? 写入 MP4...")
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out_mp4 = cv2.VideoWriter(output_mp4, fourcc_mp4, fps, (new_w_hq, new_h_hq))
    for idx, frame in enumerate(scaled_frames_hq):
        if idx % max(1, len(scaled_frames_hq)//3) == 0:
            print(f"  写入: {idx+1}/{len(scaled_frames_hq)}")
        out_mp4.write(frame)
    out_mp4.release()
    
    size_mb = os.path.getsize(output_mp4) / 1024 / 1024
    print(f"✓ 完成: {size_mb:.2f} MB")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    input_video = r"C:\Users\PC\Desktop\baselines\BrLP-main\new\24_classification_animation\results_v3\005_S_0572_slow_smooth_15fps.mp4"
    
    print("\n" + "="*60)
    print("为手机优化 - 多格式转换")
    print("="*60 + "\n")
    
    create_mobile_friendly_videos(input_video)
    
    print("="*60)
    print("✅ 所有手机友好格式已生成!")
    print("="*60)
    print("\n推荐:")
    print("  📱 手机浏览器:   用 *_mobile.webm 或 *_mobile_hq.mp4")
    print("  📱 微信/QQ:     用 *_mobile.gif")
    print("  📱 通用推荐:    用 *_mobile_hq.mp4 (最兼容)")
