# -*- coding: utf-8 -*-
"""
校验 UVR5 人声分离的"提纯"效果

分离后的人声里常残留背景音乐/混响，模型会把它们学成杂音。
这个脚本量化"静音段有多干净"：

    quiet  静音段电平（dBFS）—— 越低越干净，目标 <= -60
    gap    有声段与静音段的差距（dB）—— 越大分离越彻底
    flat   300-3400 Hz 谱平坦度 —— 高表示残留噪声多

用法：
    <RVC>/runtime/python.exe hp5_verify.py <音频1> [音频2 ...]
    <RVC>/runtime/python.exe hp5_verify.py --rvc <RVC根目录> <音频...>

不传 --rvc 时依次尝试环境变量 DSH_RVC_DIR、当前目录、脚本上两级。
"""
import argparse
import os
import sys

import numpy as np


def find_rvc(start):
    cand = [os.environ.get('DSH_RVC_DIR', ''), os.getcwd(), start]
    p = os.path.abspath(start)
    for _ in range(4):
        p = os.path.dirname(p)
        cand.append(p)
    for c in cand:
        if c and os.path.isfile(os.path.join(c, 'infer', 'lib', 'audio.py')):
            return c
    return ''


def stat(path, label, load_audio):
    try:
        d = load_audio(path, 16000)
    except Exception as e:
        print('%-34s LOAD FAIL: %s' % (label, str(e)[:40]))
        return
    if len(d) == 0:
        print('%-34s EMPTY' % label)
        return

    win = 800
    rms = np.array([np.sqrt((d[i:i + win] ** 2).mean())
                    for i in range(0, len(d) - win, win)])
    if len(rms) == 0:
        print('%-34s TOO SHORT' % label)
        return

    thr = rms.max() * 0.02
    q = rms[rms < thr]
    l = rms[rms >= thr]
    if len(q) == 0:
        print('%-34s no quiet segment' % label)
        return

    qdb = 20 * np.log10(q.mean() + 1e-9)
    ldb = 20 * np.log10(l.mean() + 1e-9) if len(l) else float('nan')

    sp = np.abs(np.fft.rfft(d * np.hanning(len(d)))) ** 2
    fr = np.fft.rfftfreq(len(d), 1 / 16000)
    band = sp[(fr > 300) & (fr < 3400)]
    flat = float(np.exp(np.mean(np.log(band + 1e-12))) / (np.mean(band) + 1e-12))

    verdict = ''
    if qdb <= -60:
        verdict = '  干净'
    elif qdb <= -45:
        verdict = '  可接受'
    else:
        verdict = '  残留较多，建议再过一遍 HP5'

    print('%-34s quiet=%7.1f dBFS  gap=%5.1f dB  flat=%.4f%s' % (
        label, qdb, ldb - qdb, flat, verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--rvc', default='')
    args = ap.parse_args()

    rvc = args.rvc or find_rvc(os.path.abspath(__file__))
    if not rvc or not os.path.isfile(os.path.join(rvc, 'infer', 'lib', 'audio.py')):
        sys.stderr.write('找不到 RVC 根目录。用 --rvc 指定，或设 DSH_RVC_DIR。\n')
        return 1

    sys.path.insert(0, rvc)
    try:
        from infer.lib.audio import load_audio
    except Exception as e:
        sys.stderr.write('导入 load_audio 失败: %s\n' % e)
        return 1

    print('RVC 根:', rvc)
    print('%-34s %s' % ('文件', '指标'))
    print('-' * 78)
    for f in args.files:
        if not os.path.isfile(f):
            print('%-34s NOT FOUND' % os.path.basename(f))
            continue
        stat(f, os.path.basename(f)[:34], load_audio)
    return 0


if __name__ == '__main__':
    sys.exit(main())
