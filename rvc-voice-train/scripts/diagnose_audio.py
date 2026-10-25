# -*- coding: utf-8 -*-
"""
素材体检 —— 判断一段音频能不能用来训 RVC 音色

用法：
    <RVC>/runtime/python.exe diagnose_audio.py <音频文件 | 目录> [更多路径...]

输出：
    采样率 / 时长 / 电平 / 削波 / 直流
    频段能量分布（重点看 8-12k 高频延伸）
    基频统计
    静音段电平
    结论与建议

判定标准（8-12 kHz 能量占比）：
    > 8%      优秀，直接训练
    4% ~ 8%   可接受，高频有限
    2% ~ 4%   偏差，音色会闷
    < 2%      不可用，换源

背景：低码率编码器（如 Vorbis 80 kb/s）会把 14 kHz 以上硬切掉，
     40k 模型要学到 20 kHz 但素材里没有，模型只能学到压缩伪影，
     输出"过度平滑"（听感像电音）。
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf

# 参考基准（同机实测）
REF = {
    '真人原声 / 高质量 PCM': 0.11,
    '中码率视频': 0.05,
    '低码率 OGG(80k)': 0.015,
}


def analyze(path, max_sec=180):
    info = sf.info(path)
    name = os.path.basename(path)
    print()
    print('=' * 70)
    print(' %s' % name)
    print('=' * 70)
    print('  路径  : %s' % path)
    print('  采样率: %d Hz' % info.samplerate)
    print('  声道  : %d' % info.channels)
    print('  时长  : %.1f 秒 (%.1f 分钟)' % (info.frames / info.samplerate,
                                            info.frames / info.samplerate / 60))
    print('  格式  : %s' % info.subtype)

    sr = info.samplerate
    frames = min(int(max_sec * sr), info.frames)
    if frames < sr:
        print('  [跳过] 音频太短（<1秒）')
        return None

    d, _ = sf.read(path, frames=frames)
    if d.ndim > 1:
        d = d[:, 0]
    d = d.astype(np.float32)
    if np.issubdtype(sf.read(path, frames=10)[0].dtype, np.integer):
        d = d / 32768.0

    peak = float(np.abs(d).max())
    rms = float(np.sqrt(np.mean(d ** 2)))
    clip = float(np.mean(np.abs(d) > 0.999))
    dc = float(np.mean(d))
    print()
    print('  电平: peak=%.3f  rms=%.4f  削波=%.5f  直流=%+.5f' % (peak, rms, clip, dc))
    if clip > 0.001:
        print('    有削波，建议降增益后重新导出')

    # ---------- 频段能量 ----------
    frame = 2048
    m = (len(d) // frame) * frame
    fr = d[:m].reshape(-1, frame) * np.hanning(frame)
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-12
    fh = np.fft.rfftfreq(frame, 1 / sr)
    q = np.sqrt(np.mean(fr ** 2, axis=1))
    loud = q > 0.02
    if loud.sum() < 3:
        print('  [警告] 有效语音帧太少')
        return None
    A = S[loud].mean(axis=0)
    tot = A.sum()

    def band(lo, hi):
        msk = (fh >= lo) & (fh < hi)
        return 100.0 * A[msk].sum() / tot if tot > 0 else 0.0

    print()
    print('  频段能量分布:')
    bands = [(0, 300), (300, 1000), (1000, 3000), (3000, 6000),
             (6000, 8000), (8000, 12000), (12000, 16000), (16000, 20000)]
    for lo, hi in bands:
        v = band(lo, hi)
        bar = '#' * int(v / 2)
        note = '  <-超过奈奎斯特' if lo >= sr / 2 else ''
        print('    %6d-%6d Hz: %5.2f%%  %s%s' % (lo, hi, v, bar, note))

    hf = band(8000, 12000)
    print()
    print('  ★ 高频延伸（8-12k）: %.2f%%' % hf)

    # ---------- 基频 ----------
    try:
        import librosa
        step = int(sr * 0.5)
        f0s = []
        for i in range(0, len(d) - 2048, step):
            fr1 = d[i:i + 2048]
            if np.sqrt(np.mean(fr1 ** 2)) < 0.01:
                continue
            try:
                f0 = librosa.yin(fr1, fmin=60, fmax=800, sr=sr, frame_length=2048)
                f0 = f0[np.isfinite(f0)]
                if len(f0):
                    f0s.append(float(np.median(f0)))
            except Exception:
                pass
            if len(f0s) >= 200:
                break
        if f0s:
            arr = np.array(f0s)
            print('  基频  : 中位=%.0f Hz  P5=%.0f  P95=%.0f  (采样 %d 帧)' % (
                np.median(arr), np.percentile(arr, 5), np.percentile(arr, 95), len(arr)))
            if np.median(arr) > 450:
                print('    基频偏高，训练时可能需要降调处理')
        else:
            print('  基频  : 无法检测')
    except ImportError:
        print('  基频  : (未安装 librosa，跳过)')

    # ---------- 静音段电平 ----------
    quiet = q[q < np.percentile(q, 20)]
    if len(quiet):
        db = 20 * np.log10(max(float(np.mean(quiet)), 1e-10))
        flag = '' if db <= -60 else ('  底噪偏高' if db > -45 else '')
        print('  静音段: %.1f dBFS%s' % (db, flag))

    return {'sr': sr, 'hf': hf, 'dur': info.frames / sr, 'peak': peak, 'clip': clip}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    results = []
    for a in args:
        if os.path.isdir(a):
            files = sorted(
                glob.glob(os.path.join(a, '*.wav')) +
                glob.glob(os.path.join(a, '*.flac')) +
                glob.glob(os.path.join(a, '*.mp3')) +
                glob.glob(os.path.join(a, '*.ogg'))
            )
            if not files:
                print('[跳过] 目录里没有音频:', a)
                continue
            # 目录最多抽查 3 个
            for f in files[:3]:
                results.append((f, analyze(f)))
        elif os.path.isfile(a):
            results.append((a, analyze(a)))
        else:
            print('[错误] 路径不存在:', a)

    # ---------- 总结 ----------
    print()
    print('=' * 70)
    print(' 总结')
    print('=' * 70)
    for path, r in results:
        if r:
            print('  %-40s sr=%-6d 8-12k=%5.2f%%' % (
                os.path.basename(path), r['sr'], r['hf']))

    print()
    valid = [r for _, r in results if r]
    if not valid:
        print('  无可分析结果')
        return 1

    r = valid[0]
    if r['sr'] < 40000:
        print('  [提示] 采样率 %d，只能训 32k 模型（40k 需要 ≥40k 源）' % r['sr'])

    hf = r['hf']
    print()
    if hf > 8:
        print('  [结论] 高频充足（%.1f%%）→ 可以直接训练' % hf)
    elif hf > 4:
        print('  [结论] 高频一般（%.1f%%）→ 可训，但音色可能偏闷' % hf)
    elif hf > 2:
        print('  [结论] 高频偏少（%.1f%%）→ 建议换更高质量的源' % hf)
    else:
        print('  [结论] 高频严重不足（%.2f%%）→ 不要训，换源' % hf)
        print('         低码率编码会把高频硬切，模型只能学到压缩伪影。')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
