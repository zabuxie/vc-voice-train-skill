# -*- coding: utf-8 -*-
"""
40k 数据预处理（不重新切片，保留原有片段边界）

preprocess.py 内部会再跑一次 Slicer，把已经切好的片段再次切碎
（实测 80 段 → 40 段，总时长 441s → 105s，丢 76% 数据）。

本脚本只做 preprocess.py 里除切片外的必需工作：
  1. 48Hz 高通滤波（preprocess.py 同款 butter(5, 48, 'high')）
  2. 峰值归一化到 0.9
  3. 输出 0_gt_wavs（目标采样率）和 1_16k_wavs（16k）

用法：
    <RVC>\\runtime\\python.exe make_40k.py <输入目录> <输出目录> <目标采样率>
    例: runtime\\python.exe make_40k.py E:/rvc_data/wavs40 E:/rvc_data/exp40 40000
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, lfilter

if len(sys.argv) < 3:
    print('用法: python make_40k.py <输入目录> <输出目录> [目标采样率]')
    sys.exit(1)

inp_dir = sys.argv[1]
exp_dir = sys.argv[2]
target_sr = int(sys.argv[3]) if len(sys.argv) > 3 else 40000

gt_dir = os.path.join(exp_dir, '0_gt_wavs')
w16_dir = os.path.join(exp_dir, '1_16k_wavs')
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(w16_dir, exist_ok=True)

# preprocess.py 同样的高通滤波（在目标采样率上做）
b, a = butter(N=5, Wn=48, btype='high', fs=target_sr)

files = sorted(glob.glob(os.path.join(inp_dir, '*.wav')))
print('输入片段: %d' % len(files))
if not files:
    print('错误: 输入目录没有 wav')
    sys.exit(1)

ok = 0
total = 0.0
for f in files:
    try:
        d, sr = sf.read(f, always_2d=False)
        if d.ndim > 1:
            d = d[:, 0]
        d = d.astype(np.float32)
        # 若读进来是 int 范围（不该发生，保险）
        if np.abs(d).max() > 1.5:
            d = d / 32768.0

        # 转目标采样率
        if sr != target_sr:
            g = np.gcd(int(sr), int(target_sr))
            d = resample_poly(d, target_sr // g, sr // g).astype(np.float32)

        # 高通滤波
        d = lfilter(b, a, d).astype(np.float32)

        # 峰值归一化到 0.9（preprocess.py 同款）
        mx = float(np.abs(d).max())
        if mx > 0:
            d = d / mx * 0.9

        stem = os.path.splitext(os.path.basename(f))[0]
        sf.write(os.path.join(gt_dir, stem + '.wav'), d, target_sr, subtype='PCM_16')

        # 16k 版本（RVC 判别器参考）
        g2 = np.gcd(16000, target_sr)
        d16 = resample_poly(d, 16000 // g2, target_sr // g2).astype(np.float32)
        sf.write(os.path.join(w16_dir, stem + '.wav'), d16, 16000, subtype='PCM_16')

        total += len(d) / target_sr
        ok += 1
    except Exception as e:
        print('跳过 %s: %s' % (os.path.basename(f), e))

print('完成: %d 段, 总时长 %.1f 秒 (%.2f 分钟)' % (ok, total, total / 60))
print('输出: %s' % gt_dir)
print('       %s' % w16_dir)
