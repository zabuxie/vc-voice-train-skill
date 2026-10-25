# -*- coding: utf-8 -*-
"""
数据集校验 —— 检查 RVC 训练数据是否完整、帧是否对齐

用法：
    <RVC>/runtime/python.exe check_dataset.py <数据集目录> [采样率]

参数：
    数据集目录  含 0_gt_wavs/1_16k_wavs/2a_f0/2b-f0nsf/3_feature256
    采样率      默认取 0_gt_wavs 的采样率（自动探测）

检查项：
    1. 五个目录的文件数是否一致
    2. 每段的 spec/f0/feat 帧数比例（应为 2.0）
    3. 特征是否含 nan/inf
    4. 音频电平与削波
    5. 时长分布
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf

SUBS = ['0_gt_wavs', '1_16k_wavs', '2a_f0', '2b-f0nsf', '3_feature256']


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    D = os.path.abspath(sys.argv[1])
    if not os.path.isdir(D):
        print('[错误] 目录不存在:', D)
        return 1

    print('=' * 70)
    print('数据集校验:', D)
    print('=' * 70)

    # ---------- 1. 文件数 ----------
    counts = {}
    for sub in SUBS:
        p = os.path.join(D, sub)
        if not os.path.isdir(p):
            print('[错误] 缺少目录:', p)
            return 1
        counts[sub] = len([f for f in os.listdir(p) if not f.endswith('.spec.pt')])
    print()
    print('文件数:')
    for k, v in counts.items():
        print('  %-14s %d' % (k, v))
    ok_count = len(set(counts.values())) == 1
    print('  一致性: %s' % ('OK' if ok_count else '不一致!'))
    if not ok_count:
        print('[错误] 文件数不一致')
        return 1

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(D, '0_gt_wavs', '*.wav')))
    print('  片段数: %d' % len(stems))
    if not stems:
        print('[错误] 0_gt_wavs 为空')
        return 1

    # ---------- 2. 采样率 / hop ----------
    wavs = glob.glob(os.path.join(D, '0_gt_wavs', '*.wav'))
    sr = sf.info(wavs[0]).samplerate
    hop = 400 if sr >= 40000 else 320
    print()
    print('采样率: %d Hz   hop_length: %d（按 RVC 官方 config 推定）' % (sr, hop))

    # ---------- 3. 逐段检查 ----------
    print()
    print('逐段检查（前 10 段）:')
    print('  %-16s %6s %6s %6s  %6s %6s  %s' %
          ('文件', 'dur', 'spec', 'feat', 'f0/ft', 'sp/ft', '状态'))
    bad = []
    durs = []
    for s in stems[:10]:
        gf = os.path.join(D, '0_gt_wavs', s + '.wav')
        ff = os.path.join(D, '3_feature256', s + '.npy')
        p0 = os.path.join(D, '2a_f0', s + '.wav.npy')
        info = sf.info(gf)
        dur = info.frames / info.samplerate
        durs.append(dur)
        try:
            feat = np.load(ff)
            f0 = np.load(p0)
        except Exception as e:
            print('  %-16s 读取失败: %s' % (s, e))
            bad.append(s)
            continue
        spec_frames = info.frames // hop
        r_f0 = f0.shape[0] / feat.shape[0] if feat.shape[0] else 0
        r_sp = spec_frames / feat.shape[0] if feat.shape[0] else 0
        flag = 'OK'
        if abs(r_f0 - 2.0) > 0.1 or abs(r_sp - 2.0) > 0.1:
            flag = 'MISMATCH'
            bad.append(s)
        print('  %-16s %6.2f %6d %6d  %6.2f %6.2f  %s' %
              (s, dur, spec_frames, feat.shape[0], r_f0, r_sp, flag))

    # ---------- 4. 特征 nan 检查 ----------
    print()
    print('特征值检查:')
    nan_files = []
    val_min, val_max = 1e9, -1e9
    for s in stems:
        try:
            a = np.load(os.path.join(D, '3_feature256', s + '.npy')).astype(np.float32)
            if not np.isfinite(a).all():
                nan_files.append(s)
                continue
            val_min = min(val_min, float(a.min()))
            val_max = max(val_max, float(a.max()))
        except Exception:
            nan_files.append(s)
    print('  nan/inf 文件: %d' % len(nan_files))
    if val_min < 1e8:
        print('  值域: %.2f ~ %.2f' % (val_min, val_max))
        if abs(val_min) > 8 or abs(val_max) > 8:
            print('  [警告] 值域异常偏大，特征提取可能有问题')

    # ---------- 5. 音频电平 ----------
    print()
    print('音频电平（抽样 20 段）:')
    peaks, rms_vals, clips = [], [], []
    for f in wavs[:20]:
        d, _ = sf.read(f, always_2d=False)
        if d.ndim > 1:
            d = d[:, 0]
        d = d.astype(np.float32)
        if np.issubdtype(sf.read(f, frames=10)[0].dtype, np.integer):
            d = d / 32768.0
        peaks.append(float(np.abs(d).max()))
        rms_vals.append(float(np.sqrt(np.mean(d ** 2))))
        clips.append(float(np.mean(np.abs(d) > 0.999)))
    print('  peak: %.3f ~ %.3f' % (min(peaks), max(peaks)))
    print('  rms : %.4f ~ %.4f' % (min(rms_vals), max(rms_vals)))
    n_clip = sum(1 for c in clips if c > 0.001)
    print('  有削波的片段: %d / %d' % (n_clip, len(clips)))

    # ---------- 6. 总时长 ----------
    all_durs = [sf.info(f).frames / sf.info(f).samplerate for f in wavs]
    total = sum(all_durs)
    print()
    print('时长分布:')
    print('  总时长: %.1f 秒 (%.1f 分钟)' % (total, total / 60))
    print('  最短: %.2f  最长: %.2f  平均: %.2f' %
          (min(all_durs), max(all_durs), total / len(all_durs)))
    if total < 300:
        print('  [提示] 素材偏少（<5 分钟），音色稳定性可能不足')
    long_segs = sum(1 for x in all_durs if x > 12)
    if long_segs:
        print('  [提示] %d 段超过 12 秒，训练时会被截断' % long_segs)

    # ---------- 结论 ----------
    print()
    print('=' * 70)
    problems = []
    if bad:
        problems.append('帧对齐异常 %d 段: %s' % (len(bad), ', '.join(bad[:3])))
    if nan_files:
        problems.append('特征含 nan %d 段' % len(nan_files))
    if n_clip:
        problems.append('%d 段有削波' % n_clip)

    if problems:
        print('发现问题:')
        for p in problems:
            print('  - ' + p)
        return 1
    else:
        print('数据集校验通过，可以开始训练')
        return 0


if __name__ == '__main__':
    sys.exit(main())
