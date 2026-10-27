# -*- coding: utf-8 -*-
"""
挑参考音频：按音高分布筛选适合 IndexTTS 的参考片段

为什么需要这个：
    IndexTTS 的输出音高 ≈ 参考音频的音高。如果参考片段偏高或波动大，
    合成出来会"音调太高"。实测同一句中文：
        参考中位 396Hz / P90 556Hz  -> 输出中位 441Hz（偏尖）
        参考中位 353Hz / P90 496Hz  -> 输出中位 350Hz
        参考中位 313Hz / P90 441Hz  -> 输出中位 320Hz（更自然）
    所以挑参考时看两个值：中位（整体高低）+ P90（高音峰值，决定刺耳感）。

用法：
    python pick_pitch_ref.py <素材目录> [--out <voices目录>]
                             [--f0-min 300] [--f0-max 380] [--min-dur 5]

    <素材目录>   放 wav/mp3/flac 的目录（或单个文件）
    --out        把最佳片段复制过去并命名（可选）
    --f0-min/max 期望的输出中位音高范围，默认 300~380
    --min-dur    最短时长（秒），默认 5
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.stderr.write('需要 soundfile：pip install soundfile\n')
    sys.exit(1)

EXTS = ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg')


def f0_stats(x, sr):
    """逐帧自相关估基频，返回中位/P10/P90/标准差"""
    vals = []
    for i in range(0, len(x) - 1024, 512):
        fr = x[i:i + 1024]
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, 'full')[len(fr) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lo, hi = int(sr / 900), int(sr / 60)
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        lag = lo + int(np.argmax(seg))
        if ac[lag] > 0.3:
            vals.append(sr / lag)
    if len(vals) < 10:
        return None
    a = np.array(vals)
    return {
        'f0': float(np.median(a)),
        'p10': float(np.percentile(a, 10)),
        'p90': float(np.percentile(a, 90)),
        'std': float(np.std(a)),
    }


def load(path):
    d, sr = sf.read(path, always_2d=False)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d.astype(np.float32)
    try:
        raw, _ = sf.read(path, frames=10)
        if np.issubdtype(raw.dtype, np.integer):
            d = d / 32768.0
    except Exception:
        pass
    return d, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='素材目录或单个音频文件')
    ap.add_argument('--out', default='', help='把最佳片段复制到这个目录')
    ap.add_argument('--f0-min', type=float, default=300.0)
    ap.add_argument('--f0-max', type=float, default=380.0)
    ap.add_argument('--min-dur', type=float, default=5.0)
    ap.add_argument('--name', default='', help='复制时的文件名（不含扩展名）')
    args = ap.parse_args()

    if os.path.isfile(args.src):
        files = [args.src]
    elif os.path.isdir(args.src):
        files = []
        for e in EXTS:
            files.extend(glob.glob(os.path.join(args.src, e)))
        files = sorted(set(files))
    else:
        sys.stderr.write('路径不存在: %s\n' % args.src)
        return 1

    if not files:
        sys.stderr.write('目录里没有音频文件: %s\n' % args.src)
        return 1

    print('分析 %d 个文件...' % len(files))
    rows = []
    for f in files:
        try:
            d, sr = load(f)
        except Exception as e:
            print('  跳过 %s: %s' % (os.path.basename(f), str(e)[:50]))
            continue
        dur = len(d) / sr
        if dur < args.min_dur:
            continue
        s = f0_stats(d, sr)
        if s:
            s.update(path=f, dur=dur, name=os.path.basename(f))
            rows.append(s)

    if not rows:
        print('没有符合条件的片段（时长 >= %gs）' % args.min_dur)
        return 1

    rows.sort(key=lambda r: r['p90'])
    print()
    print('=== 按 P90 从低到高（P90 越低 = 高音峰值越低 = 越不刺耳）===')
    print('%-28s %7s %7s %7s %8s %7s' % ('文件', '中位f0', 'P10', 'P90', '波动', '时长'))
    print('-' * 72)
    for r in rows:
        mark = ''
        if args.f0_min <= r['f0'] <= args.f0_max and r['p90'] < 460:
            mark = '  <-- 候选'
        print('%-28s %6.0fHz %6.0f %6.0f %7.1f %6.1fs%s' % (
            r['name'][:28], r['f0'], r['p10'], r['p90'], r['std'], r['dur'], mark))

    cand = [r for r in rows if args.f0_min <= r['f0'] <= args.f0_max]
    cand.sort(key=lambda r: r['p90'])

    print()
    if cand:
        print('=== 推荐（中位 %.0f~%.0f Hz，P90 最低）===' % (args.f0_min, args.f0_max))
        for r in cand[:3]:
            print('  %-24s f0=%.0f P90=%.0f 波动=%.0f %.1fs' % (
                r['name'][:24], r['f0'], r['p90'], r['std'], r['dur']))
    else:
        print('没有中位落在 %.0f~%.0f Hz 的片段，最接近的是：' % (args.f0_min, args.f0_max))
        near = sorted(rows, key=lambda r: abs(r['f0'] - (args.f0_min + args.f0_max) / 2))[:3]
        for r in near:
            print('  %-24s f0=%.0f P90=%.0f %.1fs' % (
                r['name'][:24], r['f0'], r['p90'], r['dur']))
        cand = near

    if args.out and cand:
        os.makedirs(args.out, exist_ok=True)
        best = cand[0]
        nm = args.name or 'ref'
        dst = os.path.join(args.out, nm + '.wav')
        shutil.copy2(best['path'], dst)
        print()
        print('已复制 -> %s' % dst)
        print('  (源: %s, f0=%.0f P90=%.0f)' % (best['name'], best['f0'], best['p90']))

    return 0


if __name__ == '__main__':
    sys.exit(main())
