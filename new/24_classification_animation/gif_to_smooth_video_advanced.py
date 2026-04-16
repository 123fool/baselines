#!/usr/bin/env python3
"""
高级 GIF 转视频工具 - 多种平滑度选项
支持：线性插值、Catmull-Rom 插值、以及 FFmpeg 高质量编码
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageSequence, ImageFilter
from scipy.interpolate import CubicSpline
import tempfile
from pathlib import Path

def get_gif_info(gif_path):
    """获取 GIF 的帧数和时间间隔信息"""
    img = Image.open(gif_path)
    frames = []
    durations = []
    
    try:
        for frame_idx in range(img.n_frames):
            img.seek(frame_idx)
            duration = img.info.get('duration', 100)
            durations.append(duration)
            frames.append(img.copy().convert('RGB'))
    except EOFError:
        pass
    
    return frames, durations

def linear_interpolate(frames, factor=2):
    """线性插值 - 速度快，效果一般"""
    interpolated = []
    
    for i in range(len(frames) - 1):
        frame1 = np.array(frames[i], dtype=np.float32)
        frame2 = np.array(frames[i + 1], dtype=np.float32)
        
        interpolated.append(frames[i])
        
        for j in range(1, factor):
            alpha = j / factor
            blended = (1 - alpha) * frame1 + alpha * frame2
            interpolated.append(Image.fromarray(blended.astype(np.uint8)))
    
    interpolated.append(frames[-1])
    return interpolated

def smooth_step_interpolate(frames, factor=2):
    """
    Smooth step 插值 - 使用余弦平滑
    产生更平缓的过渡
    """
    interpolated = []
    
    for i in range(len(frames) - 1):
        frame1 = np.array(frames[i], dtype=np.float32)
        frame2 = np.array(frames[i + 1], dtype=np.float32)
        
        interpolated.append(frames[i])
        
        for j in range(1, factor):
            t = j / factor
            # 使用 smoothstep 平滑曲线 (3*t^2 - 2*t^3)
            smooth_t = 3 * t**2 - 2 * t**3
            blended = (1 - smooth_t) * frame1 + smooth_t * frame2
            interpolated.append(Image.fromarray(blended.astype(np.uint8)))
    
    interpolated.append(frames[-1])
    return interpolated

def cubic_interpolate(frames, factor=2):
    """
    Catmull-Rom 立方插值 - 效果最好，但计算量大
    """
    interpolated = []
    
    for i in range(len(frames) - 1):
        p0 = np.array(frames[i-1] if i > 0 else frames[0], dtype=np.float32)
        p1 = np.array(frames[i], dtype=np.float32)
        p2 = np.array(frames[i+1], dtype=np.float32)
        p3 = np.array(frames[i+2] if i+2 < len(frames) else frames[-1], dtype=np.float32)
        
        interpolated.append(frames[i])
        
        for j in range(1, factor):
            t = j / factor
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom 基函数
            q = 0.5 * (
                2*p1 +
                (-p0 + p2)*t +
                (2*p0 - 5*p1 + 4*p2 - p3)*t2 +
                (-p0 + 3*p1 - 3*p2 + p3)*t3
            )
            
            frame = np.clip(q, 0, 255).astype(np.uint8)
            interpolated.append(Image.fromarray(frame))
    
    interpolated.append(frames[-1])
    return interpolated

def adaptive_interpolate(frames, factor=2):
    """
    自适应插值 - 根据帧差异程度调整
    """
    interpolated = []
    
    for i in range(len(frames) - 1):
        frame1 = np.array(frames[i], dtype=np.float32)
        frame2 = np.array(frames[i + 1], dtype=np.float32)
        
        # 计算帧间差异
        diff = np.mean(np.abs(frame1 - frame2))
        
        # 根据差异程度调整插值数量
        if diff > 50:
            # 差异大，增加插值帧
            local_factor = factor + 1
        else:
            local_factor = factor
        
        interpolated.append(frames[i])
        
        for j in range(1, local_factor):
            alpha = j / local_factor
            blended = (1 - alpha) * frame1 + alpha * frame2
            interpolated.append(Image.fromarray(blended.astype(np.uint8)))
    
    interpolated.append(frames[-1])
    return interpolated

def gif_to_video_advanced(gif_path, output_video_path, fps=30, interpolate='smoothstep', factor=2):
    """
    高级 GIF 转视频
    
    Args:
        gif_path: GIF 文件路径
        output_video_path: 输出 MP4 文件路径
        fps: 输出视频帧率
        interpolate: 插值方法 ('linear', 'smoothstep', 'cubic', 'adaptive')
        factor: 插值倍数
    """
    print(f"{'='*60}")
    print(f"? 读取 GIF: {Path(gif_path).name}")
    
    frames, durations = get_gif_info(gif_path)
    original_frame_count = len(frames)
    avg_duration_ms = np.mean(durations)
    
    print(f"? 原始帧数: {original_frame_count}, 平均间隔: {avg_duration_ms:.0f}ms")
    
    # 选择插值方法
    print(f"? 使用插值方法: {interpolate} (倍数: {factor}x)")
    
    if interpolate == 'linear':
        frames = linear_interpolate(frames, factor=factor)
    elif interpolate == 'smoothstep':
        frames = smooth_step_interpolate(frames, factor=factor)
    elif interpolate == 'cubic':
        frames = cubic_interpolate(frames, factor=factor)
    elif interpolate == 'adaptive':
        frames = adaptive_interpolate(frames, factor=factor)
    else:
        print(f"? 未知插值方法: {interpolate}，使用 linear")
        frames = linear_interpolate(frames, factor=factor)
    
    print(f"? 插值后帧数: {len(frames)}")
    
    # 获取帧尺寸
    frame_array = np.array(frames[0])
    height, width = frame_array.shape[:2]
    
    print(f"? 输出参数: {width}x{height} @ {fps}FPS")
    print(f"? 总播放时长: {len(frames)/fps:.2f} 秒")
    print(f"? 写入视频...")
    
    # 使用 OpenCV 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for idx, frame in enumerate(frames):
        if idx % max(1, len(frames)//10) == 0:
            progress = int(100 * idx / len(frames))
            print(f"  [{progress:3d}%] {idx+1}/{len(frames)}")
        
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)
    
    out.release()
    
    file_size_mb = os.path.getsize(output_video_path) / 1024 / 1024
    print(f"? 完成! 文件大小: {file_size_mb:.2f} MB")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    gif_path = r"C:\Users\PC\Desktop\baselines\BrLP-main\new\24_classification_animation\results_v3\005_S_0572_progression.gif"
    output_dir = Path(gif_path).parent
    
    print("\n" + "="*60)
    print("? 高级 GIF 转视频工具 - 多个平滑度版本")
    print("="*60 + "\n")
    
    # 方案1: 线性插值 (基础)
    print("[版本 1] 线性插值 2x - 基础平滑")
    gif_to_video_advanced(
        gif_path,
        str(output_dir / "005_S_0572_v1_linear_2x.mp4"),
        fps=30,
        interpolate='linear',
        factor=2
    )
    
    # 方案2: Smoothstep 插值 (平衡)
    print("[版本 2] Smoothstep 插值 2x - 平滑过渡")
    gif_to_video_advanced(
        gif_path,
        str(output_dir / "005_S_0572_v2_smoothstep_2x.mp4"),
        fps=30,
        interpolate='smoothstep',
        factor=2
    )
    
    # 方案3: 立方插值 (高质量)
    print("[版本 3] Catmull-Rom 立方插值 2x - 最平滑")
    gif_to_video_advanced(
        gif_path,
        str(output_dir / "005_S_0572_v3_cubic_2x.mp4"),
        fps=30,
        interpolate='cubic',
        factor=2
    )
    
    # 方案4: 自适应插值 (智能)
    print("[版本 4] 自适应插值 2x - 智能调整")
    gif_to_video_advanced(
        gif_path,
        str(output_dir / "005_S_0572_v4_adaptive_2x.mp4"),
        fps=30,
        interpolate='adaptive',
        factor=2
    )
    
    # 方案5: 高倍插值 (极限平滑)
    print("[版本 5] Smoothstep 插值 3x - 极限平滑")
    gif_to_video_advanced(
        gif_path,
        str(output_dir / "005_S_0572_v5_smoothstep_3x.mp4"),
        fps=30,
        interpolate='smoothstep',
        factor=3
    )
    
    print("\n" + "="*60)
    print("? 所有版本已生成!")
    print("="*60)
    print("\n推荐:")
    print("? v2 (Smoothstep 2x) - 速度快、效果好，推荐首选")
    print("? v3 (Cubic 2x) - 效果最好，但文件较大")
    print("? v5 (Smoothstep 3x) - 超平滑，但可能过度插值")
