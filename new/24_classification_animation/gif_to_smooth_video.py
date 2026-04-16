#!/usr/bin/env python3
"""
Convert GIF to smooth video with frame interpolation.
Handles both direct conversion and optional frame interpolation.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageSequence
import subprocess
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
            duration = img.info.get('duration', 100)  # 默认 100ms
            durations.append(duration)
            frames.append(img.copy().convert('RGB'))
    except EOFError:
        pass
    
    return frames, durations

def interpolate_frames(frames, factor=2):
    """
    使用线性插值增加帧数
    factor: 插值倍数 (2 = 帧数翻倍)
    """
    interpolated = []
    
    for i in range(len(frames) - 1):
        frame1 = np.array(frames[i], dtype=np.float32)
        frame2 = np.array(frames[i + 1], dtype=np.float32)
        
        interpolated.append(frames[i])
        
        # 生成中间帧
        for j in range(1, factor):
            alpha = j / factor
            blended = (1 - alpha) * frame1 + alpha * frame2
            interpolated.append(Image.fromarray(blended.astype(np.uint8)))
    
    interpolated.append(frames[-1])
    return interpolated

def gif_to_video(gif_path, output_video_path, fps=30, interpolate=True, interpolate_factor=2):
    """
    转换 GIF 为 MP4 视频
    
    Args:
        gif_path: GIF 文件路径
        output_video_path: 输出 MP4 文件路径
        fps: 输出视频帧率 (默认 30 FPS)
        interpolate: 是否进行帧插值
        interpolate_factor: 插值倍数
    """
    print(f"?" * 50)
    print(f"? 读取 GIF: {gif_path}")
    
    # 读取 GIF 帧
    frames, durations = get_gif_info(gif_path)
    original_frame_count = len(frames)
    avg_duration_ms = np.mean(durations)
    
    print(f"? 原始帧数: {original_frame_count}, 平均间隔: {avg_duration_ms:.0f}ms")
    
    # 帧插值
    if interpolate and interpolate_factor > 1:
        print(f"? 进行帧插值 (倍数: {interpolate_factor}x)...")
        frames = interpolate_frames(frames, factor=interpolate_factor)
        print(f"? 插值后帧数: {len(frames)}")
    
    # 获取帧尺寸
    frame_array = np.array(frames[0])
    height, width = frame_array.shape[:2]
    
    print(f"? 输出参数: {width}x{height} @ {fps}FPS")
    print(f"? 写入视频: {output_video_path}")
    
    # 使用 OpenCV 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for i, frame in enumerate(frames):
        if i % 10 == 0:
            print(f"  -> {i+1}/{len(frames)}")
        
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)
    
    out.release()
    print(f"? 完成! 输出文件大小: {os.path.getsize(output_video_path) / 1024 / 1024:.2f} MB")

def gif_to_video_ffmpeg(gif_path, output_video_path, fps=30, crf=23):
    """
    使用 FFmpeg 转换 GIF 为 MP4 (备选方案)
    CRF: 质量参数 (0-51, 低=高质量)
    """
    print(f"?" * 50)
    print(f"? 使用 FFmpeg 转换 GIF -> MP4")
    print(f"? 输入: {gif_path}")
    print(f"? 输出: {output_video_path}")
    print(f"? FPS: {fps}, CRF: {crf}")
    
    cmd = [
        'ffmpeg',
        '-i', gif_path,
        '-vf', f'fps={fps}',
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-y',
        output_video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            size_mb = os.path.getsize(output_video_path) / 1024 / 1024
            print(f"? 完成! 文件大小: {size_mb:.2f} MB")
            return True
        else:
            print(f"? FFmpeg 错误:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("? FFmpeg 未找到，请确保已安装")
        return False
    except subprocess.TimeoutExpired:
        print("? FFmpeg 超时")
        return False

if __name__ == '__main__':
    gif_path = r"C:\Users\PC\Desktop\baselines\BrLP-main\new\24_classification_animation\results_v3\005_S_0572_progression.gif"
    output_dir = Path(gif_path).parent
    
    # 输出文件路径
    output_mp4_opencv = output_dir / "005_S_0572_progression_smooth_2x.mp4"
    output_mp4_ffmpeg = output_dir / "005_S_0572_progression_smooth_ffmpeg.mp4"
    
    print("=" * 60)
    print("GIF 转视频 + 平滑处理工具")
    print("=" * 60)
    
    # 方案1: OpenCV + 帧插值 (2倍平滑)
    print("\n[方案 1] OpenCV 方案 - 帧插值 2x + 30FPS")
    try:
        gif_to_video(
            gif_path, 
            str(output_mp4_opencv),
            fps=30,
            interpolate=True,
            interpolate_factor=2
        )
    except Exception as e:
        print(f"? OpenCV 方案失败: {e}")
    
    # 方案2: FFmpeg 转换 (高质量，需要安装 FFmpeg)
    print("\n[方案 2] FFmpeg 方案 - H.264 高质量编码")
    ffmpeg_success = gif_to_video_ffmpeg(
        gif_path,
        str(output_mp4_ffmpeg),
        fps=30,
        crf=20  # 质量范围: 0-51 (低=高质量)
    )
    
    if not ffmpeg_success:
        print("\n? 尝试 FFmpeg 保底方案...")
        gif_to_video_ffmpeg(gif_path, str(output_mp4_ffmpeg), fps=30, crf=23)
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
