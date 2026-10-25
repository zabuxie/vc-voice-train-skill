# -*- coding: utf-8 -*-
"""
模型音质验证（隔离测试，绕过 rvc-server 和 Edge TTS）

用法：
    <RVC>\\runtime\\python.exe probe_quality.py <实验日志目录> [实验名] [驱动音频]

    例:
      runtime\python.exe probe_quality.py <RVC>/logs/<实验名> <实验名>
      runtime\\python.exe probe_quality.py <日志目录> <实验名> <驱动音频.wav>

不传驱动音频时，自动用训练集第一个片段（自转测试）。

输出判定：
    flat < 0.15  可用
    flat > 0.5   噪声（失败）

注意：需先关闭 rvc-server（否则 faiss/GPU context 冲突）
"""
import os
import sys
import glob
import json
import traceback
import numpy as np

if len(sys.argv) < 2:
    print('用法: python probe_quality.py <实验日志目录> [实验名] [驱动音频]')
    sys.exit(1)

LOG = sys.argv[1].replace('\\', '/')
EXP = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(LOG.rstrip('/'))

# RVC 根目录：从日志目录往上找
RVC = None
p = os.path.abspath(LOG)
for _ in range(5):
    p = os.path.dirname(p)
    if os.path.isfile(os.path.join(p, 'infer', 'modules', 'vc', 'pipeline.py')):
        RVC = p.replace('\\', '/')
        break
if RVC is None:
    RVC = r'E:/rvc/RVC20240604Nvidia'

os.chdir(RVC)
sys.path.insert(0, RVC)
sys.argv = ['probe']   # 防止 config.py 解析我们的参数

os.environ['weight_root'] = RVC + '/assets/weights'
os.environ['index_root'] = RVC + '/assets/indices'
os.environ['rmvpe_root'] = RVC + '/assets/rmvpe'
os.environ['DSH_RVC_PEAK'] = '1.0'    # 关闭归一化，看真实电平
os.environ['DSH_RVC_CEIL'] = '1.0'

OUT = os.path.join(LOG, 'probe_result.log')
log = open(OUT, 'w', encoding='utf-8')


def p(s):
    print(s)
    log.write(str(s) + '\n')
    log.flush()


def metrics(x, sr, tag):
    x = np.asarray(x, dtype=np.float32)
    frame = 1024
    m = (len(x) // frame) * frame
    if m < frame:
        p('%-26s 太短' % tag)
        return None
    fr = x[:m].reshape(-1, frame) * np.hanning(frame)
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-12
    f = np.fft.rfftfreq(frame, 1 / sr)
    q = np.sqrt(np.mean(fr ** 2, axis=1))
    loud = q > 0.01
    if loud.sum() < 3:
        p('%-26s 无有效语音段' % tag)
        return None
    E = S[loud].sum(axis=0)
    c = np.cumsum(E) / E.sum()
    g = np.exp(np.mean(np.log(S[loud]), axis=1))
    a = np.mean(S[loud], axis=1)
    flat = float(np.mean(g / a))
    p('%-26s rms=%.4f  peak=%.3f  roll95=%5.0f  flat=%.4f' % (
        tag, float(np.sqrt(np.mean(x ** 2))), float(np.abs(x).max()),
        f[np.searchsorted(c, 0.95)], flat))
    return flat


p('=' * 64)
p('模型音质验证')
p('=' * 64)
p('RVC : %s' % RVC)
p('LOG : %s' % LOG)
p('EXP : %s' % EXP)

# 找权重
wpath = os.path.join(RVC, 'assets', 'weights', EXP + '.pth')
if not os.path.isfile(wpath):
    # 退而找最近的 e*_s*.pth
    cands = sorted(glob.glob(os.path.join(RVC, 'assets', 'weights', EXP + '_e*.pth')))
    if cands:
        wpath = cands[-1]
        p('未找到 %s.pth，使用 %s' % (EXP, os.path.basename(wpath)))
    else:
        p('[错误] 找不到权重: %s' % wpath)
        sys.exit(1)
p('权重: %s (%.1f MB)' % (os.path.basename(wpath), os.path.getsize(wpath) / 1048576))

# 找索引
idx = ''
icands = glob.glob(os.path.join(LOG, '*.index')) + \
         glob.glob(os.path.join(RVC, 'assets', 'indices', '*%s*' % EXP))
if icands:
    idx = icands[0]
    p('索引: %s' % os.path.basename(idx))
else:
    p('索引: (无，免索引模式，音质会略差)')

# 驱动音频
if len(sys.argv) > 3:
    SRC = sys.argv[3]
elif len(sys.argv) > 2 and not sys.argv[2].startswith(LOG):
    SRC = sys.argv[2]
else:
    SRC = None
if SRC is None or not os.path.isfile(SRC):
    # 用训练集第一个片段
    cfg_path = os.path.join(LOG, 'config.json')
    try:
        c = json.load(open(cfg_path, encoding='utf-8'))
        d = os.path.dirname(c['data']['training_files'])
        fl = open(c['data']['training_files'], encoding='utf-8').read().strip().split('\n')
        SRC = fl[0].split('|')[0]
    except Exception:
        pass  # 可在此添加你的默认驱动音频目录（glob 模式列表）
if not SRC or not os.path.isfile(SRC):
    p('[错误] 找不到驱动音频，请通过第 3 个参数指定')
    sys.exit(1)
p('驱动: %s' % os.path.basename(SRC))
p('')

try:
    from configs.config import Config
    from infer.modules.vc.modules import VC
    import torch
    cfg = Config()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.is_half = False
    p('设备: %s' % cfg.device)

    vc = VC(cfg)
    vc.get_vc(os.path.basename(wpath))
    info, opt = vc.vc_single(0, SRC, 0, None, 'rmvpe', '', idx, 0.75, 3, 0, 0.25, 0.5)
    if opt is None or opt[1] is None:
        p('[失败] %s' % str(info)[:200])
        sys.exit(1)
    sr, a = opt
    af = a.astype(np.float32) / 32768.0 if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32)
    p('')
    flat = metrics(af, sr, EXP)

    # 保存试听
    import soundfile as sf
    outwav = os.path.join(LOG, 'probe_output.wav')
    sf.write(outwav, af, sr)
    p('')
    p('试听文件: %s' % outwav)

    p('')
    p('=' * 64)
    if flat is None:
        p('无法判定')
    elif flat < 0.15:
        p('>>> 结论: 音质可用 (flat=%.4f < 0.15)' % flat)
    elif flat < 0.5:
        p('>>> 结论: 音质勉强 (flat=%.4f)，可能有伪影' % flat)
    else:
        p('>>> 结论: 噪声 (flat=%.4f > 0.5)，训练失败' % flat)
    p('=' * 64)

except Exception:
    p('[异常]')
    p(traceback.format_exc())
    sys.exit(1)
finally:
    log.close()
