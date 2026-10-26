# -*- coding: utf-8 -*-
"""
音质验收（谐波纯度 + 频段贴合度）—— 比 flat 更可靠的判断方法

用法：
    <RVC>/runtime/python.exe probe_accept.py <模型输出.wav> <真值.wav>

    <模型输出.wav>  模型生成的音频（如 logs/<exp>/probe_output.wav）
    <真值.wav>      同内容的真人原声（对照基准）

输出：
    基频中位、各频段能量占比、谐波纯度，并与真值逐项对比

为什么不用 flat：
    flat（谱平坦度）只反映"能量是否集中"，低不等于自然。
    实测反例：
      - 多萝西降调版 flat=0.040（最低）→ 谐波纯度 0.964（真值 0.928）→ 过度平滑，用户否决
      - 卡提原声   flat=0.21（超标）  → 谐波纯度 0.907（真值 0.882）→ 用户确认"没问题"

达标标准（输出 vs 真值）：
    基频              差 < 30 Hz
    1-3 kHz 占比      差 < 3 个百分点
    3-6 kHz 占比      差 < 2 个百分点
    谐波纯度          差 < 0.05（关键指标）
    flat             仅作参考，阈值随音色变化
"""
import os
import sys
import numpy as np
import soundfile as sf


def load(f):
    d, sr = sf.read(f, always_2d=False)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d.astype(np.float32)
    if np.issubdtype(sf.read(f, frames=10)[0].dtype, np.integer):
        d = d / 32768.0
    return d, sr


def f0_track(x, sr, hop=512, fmin=60, fmax=800):
    """自相关粗估基频（逐帧）"""
    out = []
    for i in range(0, len(x) - 1024, hop):
        fr = x[i:i + 1024]
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            out.append(0.0)
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, 'full')[len(fr) - 1:]
        if ac[0] <= 0:
            out.append(0.0)
            continue
        ac = ac / ac[0]
        lo, hi = int(sr / fmax), int(sr / fmin)
        seg = ac[lo:hi]
        if len(seg) == 0:
            out.append(0.0)
            continue
        lag = lo + int(np.argmax(seg))
        out.append(float(sr / lag) if ac[lag] > 0.3 else 0.0)
    return np.array(out)


def harmonicity(x, sr, f0, frame=2048):
    """谐波纯度：有声帧中 f0 及其谐波附近能量占该帧总能量的比例"""
    vals = []
    n_frames = len(x) // frame
    for k in range(n_frames):
        fr = x[k * frame:(k + 1) * frame] * np.hanning(frame)
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        # 对齐 f0 轨（f0 用 1024 窗 512 跳）
        idx = min(len(f0) - 1, (k * frame) // 512)
        p = f0[idx] if idx >= 0 else 0.0
        if p <= 0:
            continue
        X = np.abs(np.fft.rfft(fr)) ** 2
        freqs = np.fft.rfftfreq(frame, 1 / sr)
        tol = 90.0
        mask = np.zeros_like(freqs, dtype=bool)
        h = 1
        while h * p < freqs[-1]:
            mask |= (freqs >= h * p - tol) & (freqs <= h * p + tol)
            h += 1
        tot = X.sum()
        if tot > 0:
            vals.append(float(X[mask].sum() / tot))
    return float(np.mean(vals)) if vals else float('nan')


def analyze(f, label):
    x, sr = load(f)
    gate = np.sqrt(np.mean(x ** 2)) > 0.01

    frame = 2048
    m = (len(x) // frame) * frame
    fr = x[:m].reshape(-1, frame) * np.hanning(frame)
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-12
    fh = np.fft.rfftfreq(frame, 1 / sr)
    q = np.sqrt(np.mean(fr ** 2, axis=1))
    loud = q > 0.02
    if loud.sum() < 3:
        print('%-10s 无有效语音' % label)
        return None
    A = S[loud].mean(axis=0)
    tot = A.sum()

    def band(lo, hi):
        return 100.0 * A[(fh >= lo) & (fh < hi)].sum() / tot

    g = np.exp(np.mean(np.log(S[loud]), axis=1))
    a = np.mean(S[loud], axis=1)
    flat = float(np.mean(g / a))

    f0 = f0_track(x, sr)
    voiced = f0[f0 > 0]
    f0med = float(np.median(voiced)) if len(voiced) else 0.0
    harm = harmonicity(x, sr, f0)

    print('%-10s sr=%-6d dur=%5.2fs  f0=%5.1fHz  flat=%.3f  harm=%.3f' % (
        label, sr, len(x) / sr, f0med, flat, harm))
    print('           1-3k=%5.2f%%  3-6k=%5.2f%%  6-8k=%5.2f%%  8-12k=%5.2f%%' % (
        band(1000, 3000), band(3000, 6000), band(6000, 8000), band(8000, 12000)))
    return {'sr': sr, 'f0': f0med, 'flat': flat, 'harm': harm,
            'b13': band(1000, 3000), 'b36': band(3000, 6000),
            'b68': band(6000, 8000), 'b812': band(8000, 12000),
            'dur': len(x) / sr}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    out_f, ref_f = sys.argv[1], sys.argv[2]
    for f in (out_f, ref_f):
        if not os.path.isfile(f):
            print('[错误] 文件不存在:', f)
            return 1

    print('=' * 72)
    print('音质验收（输出 vs 真值）')
    print('=' * 72)
    r = analyze(ref_f, '真值')
    o = analyze(out_f, '模型')
    if not r or not o:
        return 1

    print()
    print('=' * 72)
    print('差异对比')
    print('=' * 72)
    rows = [
        ('基频 (Hz)', r['f0'], o['f0'], 30.0, 'abs'),
        ('1-3k (%)', r['b13'], o['b13'], 3.0, 'abs'),
        ('3-6k (%)', r['b36'], o['b36'], 2.0, 'abs'),
        ('谐波纯度', r['harm'], o['harm'], 0.05, 'abs'),
        ('flat', r['flat'], o['flat'], None, 'ref'),
    ]
    all_ok = True
    for name, rv, ov, tol, kind in rows:
        if kind == 'ref':
            print('  %-12s 真值=%.3f  模型=%.3f   (参考，不作硬性判定)' % (name, rv, ov))
            continue
        diff = abs(ov - rv)
        ok = diff <= tol
        if not ok:
            all_ok = False
        print('  %-12s 真值=%7.2f  模型=%7.2f  差=%6.2f  (标准<%g)  %s' % (
            name, rv, ov, diff, tol, 'OK' if ok else 'FAIL'))

    print()
    if all_ok:
        print('结论：达标（谐波纯度与频段均已贴合真值）')
    else:
        print('结论：有偏差，见上面 FAIL 项')
        if abs(o['harm'] - r['harm']) > 0.05:
            if o['harm'] > r['harm']:
                print('  谐波纯度偏高 -> 频谱被过度平滑（听感可能发闷/发僵）')
            else:
                print('  谐波纯度偏低 -> 谐波结构被破坏（听感可能带噪/电音）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
