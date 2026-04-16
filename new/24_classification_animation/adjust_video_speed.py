#!/usr/bin/env python3
"""
视频速度调节工具 - 降速播放 / 增加时长
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def slow_down_video(input_path, output_path, fps_out=10, speed_factor=0.5):
    """
    方式1: 降帧率 - 直接降低输出帧率来慢速播放
    
    Args:
        input_path: 输入视频
        output_path: 输出视频
        fps_out: 输出帧率 (越低越慢)
        speed_factor: 速度系数 (0.5 = 一半速度)
    """
    print(f"{'='*60}")
    print(f"方式1: 降帧率方法 (FPS: {fps_out})")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(input_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"? 输入: {fps_in:.1f} FPS, {frame_count} 帧, {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
    
    frame_idx = 0
    saved_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根据 speed_factor 跳帧
        if frame_idx % int(1/speed_factor) == 0:
            out.write(frame)
            saved_frames += 1
            if saved_frames % 5 == 0:
                print(f"  处理: {frame_idx}/{frame_count} -> 保存 {saved_frames} 帧")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    duration_out = saved_frames / fps_out
    print(f"? 输出: {fps_out} FPS, {saved_frames} 帧")
    print(f"? 时长: {duration_out:.2f} 秒 (原来 {frame_count/fps_in:.2f} 秒)")
    print(f"? 文件: {os.path.getsize(output_path)/1024/1024:.2f} MB\n")

def interpolate_and_slow(gif_path, output_path, factor=3, fps=15):
    """
    方式2: 帧插值 + 降帧率 - 更平滑的慢速播放
    
    Args:
        gif_path: GIF或视频路径
        output_path: 输出视频
        factor: 插值倍数 (3 = 原来3倍帧数)
        fps: 输出帧率
    """
    print(f"{'='*60}")
    print(f"方式2: 帧插值方法 (插值 {factor}x, {fps} FPS)")
    print(f"{'='*60}")
    
    # 读取视频帧
    cap = cv2.VideoCapture(gif_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"? 读取 {len(frames)} 帧")
    
    # 帧插值
    interpolated = []
    for i in range(len(frames) - 1):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)
        
        interpolated.append(frames[i])
        
        for j in range(1, factor):
            alpha = j / factor
            # Smoothstep 平滑曲线
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3
            blended = ((1 - smooth_alpha) * frame1 + smooth_alpha * frame2).astype(np.uint8)
            interpolated.append(blended)
    
    interpolated.append(frames[-1])
    
    print(f"? 插值后 {len(interpolated)} 帧")
    
    # 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for idx, frame in enumerate(interpolated):
        if idx % max(1, len(interpolated)//5) == 0:
            print(f"  写入: {idx+1}/{len(interpolated)}")
        out.write(frame)
    
    out.release()
    
    duration = len(interpolated) / fps
    print(f"? 输出: {fps} FPS, {len(interpolated)} 帧")
    print(f"? 时长: {duration:.2f} 秒")
    print(f"? 文件: {os.path.getsize(output_path)/1024/1024:.2f} MB\n")

if __name__ == '__main__':
    input_video = r"C:\Users\PC\Desktop\baselines\BrLP-main\new\24_classification_animation\results_v3\005_S_0572_v2_smoothstep_2x.mp4"
    output_dir = Path(input_video).parent
    
    print("\n" + "="*60)
    print("视频速度调节工具 - 让播放更慢")
    print("="*60 + "\n")
    
    # 方案1: 降帧率 (最简单快速)
    print("[方案 1] 降帧率 - 最简单")
    slow_down_video(
        input_video,
        str(output_dir / "005_S_0572_slow_10fps.mp4"),
        fps_out=10
    )
    
    # 方案2: 降帧率更低 (超级慢)
    print("[方案 2] 超级降速")
    slow_down_video(
        input_video,
        str(output_dir / "005_S_0572_slow_5fps.mp4"),
        fps_out=5
    )
    
    # 方案3: 帧插值 + 降帧率 (最平滑)
    print("[方案 3] 帧插值 + 降帧率 - 最平滑")
    interpolate_and_slow(
        input_video,
        str(output_dir / "005_S_0572_slow_smooth_15fps.mp4"),
        factor=2,
        fps=15
    )
    
    # 方案4: 极限平滑降速
    print("[方案 4] 极限平滑降速")
    interpolate_and_slow(
        input_video,
        str(output_dir / "005_S_0572_slow_smooth_8fps.mp4"),
        factor=3,
        fps=8
    )
    
    print("="*60)
    print("✅ 所有版本已生成!")
    print("="*60)
    print("\n速度对比:")
    print("  ? 10 FPS  - 一般慢速")
    print("  ? 5 FPS   - 超级慢速")
    print("  ? 15 FPS + 插值 - 平滑慢速 ⭐ 推荐")
    print("  ? 8 FPS + 插值 - 极限慢速")
