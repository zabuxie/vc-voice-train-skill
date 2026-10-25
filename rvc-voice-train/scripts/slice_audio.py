# -*- coding: utf-8 -*-
"""
按静音切片（保持源采样率，不做二次切分）

用法：
    <RVC>/runtime/python.exe slice_audio.py <输入音频> <输出目录> [目标采样率]

参数：
    目标采样率  留空 = 保持源采样率（推荐，避免损失高频）
                传 40000/32000 = 重采样到该值

特点：
    - 一次切分，不做二次切片
    - 片段 3~12 秒（RVC 训练理想区间）
    - 峰值归一化到 0.9
    - 单声道输出

输出：slice_0000.wav, slice_0001.wav, ...
"""
import os
import sys
import numpy as np
import soundfile as sf


def find_rvc():
    """找 RVC 根目录（用于 import slicer）"""
    env = os.environ.get('DSH_RVC_DIR', '')
    if env and os.path.isdir(env):
        return env
    here = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isfile(os.path.join(here, 'infer', 'lib', 'slicer2.py')):
            return here
        p = os.path.dirname(here)
        if p == here:
            break
        here = p
    return ''


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    src = sys.argv[1]
    outdir = sys.argv[2]
    target_sr = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if not os.path.isfile(src):
        print('[错误] 输入不存在:', src)
        return 1
    os.makedirs(outdir, exist_ok=True)

    info = sf.info(src)
    src_sr = info.samplerate
    use_sr = target_sr if target_sr else src_sr

    print('=' * 64)
    print('切片')
    print('输入  : %s' % src)
    print('        %d Hz / %d ch / %.1f 分钟' % (src_sr, info.channels,
                                                 info.frames / src_sr / 60))
    print('输出  : %s' % outdir)
    print('采样率: %d %s' % (use_sr, '(保持源)' if not target_sr else '(重采样)'))
    print('=' * 64)

    # 读取
    d, _ = sf.read(src, always_2d=False)
    if d.ndim > 1:
        d = d.mean(axis=1)  # 立体声转单声道
    d = d.astype(np.float32)
    if np.issubdtype(sf.read(src, frames=10)[0].dtype, np.integer):
        d = d / 32768.0

    # 重采样（如需要）
    if use_sr != src_sr:
        import scipy.signal as ss
        g = np.gcd(int(src_sr), int(use_sr))
        d = ss.resample_poly(d, use_sr // g, src_sr // g).astype(np.float32)

    # 切片：优先用 RVC 自带 Slicer，否则退化为能量切分
    segs = []
    rvc = find_rvc()
    used_rvc_slicer = False
    if rvc:
        try:
            sys.path.insert(0, rvc)
            from infer.lib.slicer2 import Slicer
            slicer = Slicer(sr=use_sr, threshold=-42, min_length=4000,
                            min_interval=300, hop_size=15, max_sil_kept=500)
            segs = list(slicer.slice(d))
            used_rvc_slicer = True
        except Exception as e:
            print('[提示] RVC Slicer 不可用（%s），改用内置切分' % e)

    if not used_rvc_slicer:
        # 内置：基于 RMS 阈值切分
        frame = max(1, int(0.01 * use_sr))
        n = len(d) // frame
        rms = np.sqrt(np.mean(d[:n * frame].reshape(n, frame) ** 2, axis=1))
        thr = 10 ** (-42 / 20.0)
        voiced = rms > thr
        min_len = int(3.0 * use_sr)
        max_len = int(12.0 * use_sr)
        i = 0
        while i < len(voiced):
            if voiced[i]:
                j = i
                while j < len(voiced) and voiced[j]:
                    j += 1
                s, e = i * frame, min(len(d), j * frame)
                if e - s > max_len:
                    k = s
                    while k < e:
                        e2 = min(e, k + max_len)
                        segs.append(d[k:e2])
                        k = e2
                elif e - s >= min_len:
                    segs.append(d[s:e])
                i = j
            else:
                i += 1

    # 写出
    n = 0
    tot = 0.0
    lens = []
    for seg in segs:
        seg = np.asarray(seg, dtype=np.float32)
        dur = len(seg) / use_sr
        if dur < 3.0 or dur > 12.0:
            continue
        pk = float(np.abs(seg).max())
        if pk < 0.005:
            continue
        seg = seg / pk * 0.9
        sf.write(os.path.join(outdir, 'slice_%04d.wav' % n), seg, use_sr, subtype='PCM_16')
        tot += dur
        lens.append(dur)
        n += 1

    print()
    if n == 0:
        print('[错误] 没有切出任何片段。检查阈值/音频内容')
        return 1
    print('完成: %d 段, 总 %.1f 秒 (%.1f 分钟)' % (n, tot, tot / 60))
    print('长度: 最短=%.2f 最长=%.2f 平均=%.2f' % (min(lens), max(lens), sum(lens) / len(lens)))
    print('输出:', outdir)
    if tot < 300:
        print()
        print('[提示] 素材只有 %.1f 分钟，训练出的音色可能不够稳定（建议 10~30 分钟）' % (tot / 60))
    return 0


if __name__ == '__main__':
    sys.exit(main())
