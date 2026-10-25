# -*- coding: utf-8 -*-
"""
生成 RVC 训练的 filelist + config（幂等，每次运行重新生成）

用法：
    <RVC>/runtime/python.exe make_config.py <数据目录> [日志目录] [RVC根目录] [采样率]

参数：
    数据目录    含 0_gt_wavs/1_16k_wavs/2a_f0/2b-f0nsf/3_feature256 的目录
    日志目录    默认 <RVC>/logs/<数据目录名>
    RVC根目录   默认从数据目录往上找 infer/ 目录；找不到时看 DSH_RVC_DIR 环境变量
    采样率      32000 或 40000，默认 40000

环境变量（可选）：
    EPOCHS     轮数，默认 100（注意：train.py 的 -te 不生效，必须写进 config）
    BATCH      batch_size，默认 2
    SEGMENT    segment_size，默认 12800
    FP16       1 = 开启半精度，默认 0（关闭更稳）

示例：
    runtime/python.exe make_config.py E:/data/exp40
    runtime/python.exe make_config.py E:/data/exp32 E:/rvc/logs/voice32 E:/rvc 32000
"""
import os
import sys
import glob
import json


def find_rvc(start):
    """找 RVC 根目录（有 infer/modules/vc/pipeline.py）

    优先级：
      1. 环境变量 DSH_RVC_DIR
      2. 从 start 往上找
      3. 从当前工作目录往上找
    """
    env = os.environ.get('DSH_RVC_DIR', '')
    if env and os.path.isfile(os.path.join(env, 'infer', 'modules', 'vc', 'pipeline.py')):
        return env

    def walk(p):
        p = os.path.abspath(p)
        for _ in range(8):
            if os.path.isfile(os.path.join(p, 'infer', 'modules', 'vc', 'pipeline.py')):
                return p
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return ''

    return walk(start) or walk(os.getcwd())


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

D = os.path.abspath(sys.argv[1])
RVC = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else find_rvc(D)
if not RVC or not os.path.isdir(RVC):
    print('[错误] 找不到 RVC 根目录。请在第 3 个参数传，或设 DSH_RVC_DIR 环境变量')
    sys.exit(1)

EXP = os.path.basename(D.rstrip(os.sep))
LOG = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(RVC, 'logs', EXP)
SR = int(sys.argv[4]) if len(sys.argv) > 4 else 40000

EPOCHS = int(os.environ.get('EPOCHS', '100'))
BATCH = int(os.environ.get('BATCH', '2'))
SEGMENT = int(os.environ.get('SEGMENT', '12800'))
FP16 = os.environ.get('FP16', '0') == '1'

print('=' * 64)
print('RVC 配置生成')
print('数据目录:', D)
print('日志目录:', LOG)
print('RVC 根  :', RVC)
print('采样率  :', SR)
print('=' * 64)

# ---------- 校验数据完整性 ----------
need = ['0_gt_wavs', '1_16k_wavs', '2a_f0', '2b-f0nsf', '3_feature256']
counts = {}
for sub in need:
    p = os.path.join(D, sub)
    if not os.path.isdir(p):
        print('[错误] 缺少目录:', p)
        sys.exit(1)
    # 排除 RVC 的频谱缓存
    counts[sub] = len([f for f in os.listdir(p) if not f.endswith('.spec.pt')])
print('文件数:', counts)
if len(set(counts.values())) != 1:
    print('[错误] 各目录文件数不一致，数据不完整')
    sys.exit(1)

stems = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(D, '0_gt_wavs', '*.wav')))
print('片段数:', len(stems))

Dposix = D.replace('\\', '/')

# ---------- 生成 filelist（5 列，顺序固定）----------
lines, missing = [], []
for s in stems:
    parts = [
        '%s/0_gt_wavs/%s.wav' % (Dposix, s),
        '%s/3_feature256/%s.npy' % (Dposix, s),
        '%s/2a_f0/%s.wav.npy' % (Dposix, s),
        '%s/2b-f0nsf/%s.wav.npy' % (Dposix, s),
        '0',
    ]
    for x in parts[:4]:
        if not os.path.isfile(x):
            missing.append(x)
    lines.append('|'.join(parts))

if missing:
    print('[错误] 文件缺失:', len(missing))
    for m in missing[:5]:
        print('  ', m)
    sys.exit(1)

os.makedirs(LOG, exist_ok=True)
fl_path = os.path.join(LOG, 'filelist.txt')
# ascii + LF：RVC 用 GBK 读，中文路径会崩
with open(fl_path, 'w', encoding='ascii', newline='\n') as f:
    f.write('\n'.join(lines))
print('filelist:', fl_path, '(%d 行)' % len(lines))

# ---------- 生成 config ----------
tpl_name = '40k.json' if SR >= 40000 else '32k.json'
tpl = os.path.join(RVC, 'configs', 'v1', tpl_name)
if not os.path.isfile(tpl):
    print('[错误] 找不到 config 模板:', tpl)
    sys.exit(1)

cfg = json.load(open(tpl, encoding='utf-8'))
cfg['train']['epochs'] = EPOCHS
cfg['train']['batch_size'] = BATCH
cfg['train']['log_interval'] = 50
cfg['train']['fp16_run'] = FP16
cfg['train']['segment_size'] = SEGMENT
cfg['data']['sampling_rate'] = SR
cfg['data']['max_wav_value'] = 32768.0
cfg['data']['training_files'] = fl_path.replace('\\', '/')
cfg['data']['validation_files'] = os.path.join(D, '1_16k_wavs').replace('\\', '/')
cfg['gpus'] = '0'

cfg_path = os.path.join(LOG, 'config.json')
json.dump(cfg, open(cfg_path, 'w', encoding='utf-8'), ensure_ascii=True, indent=2)
print('config:', cfg_path)

# ---------- 校验 ----------
d = cfg['data']
print()
print('关键参数:')
print('  sr=%d  mel=%d  hop=%d  filter=%d' % (
    d['sampling_rate'], d['n_mel_channels'], d['hop_length'], d['filter_length']))
print('  epochs=%d  bs=%d  segment=%d  fp16=%s' % (
    cfg['train']['epochs'], cfg['train']['batch_size'],
    cfg['train']['segment_size'], cfg['train']['fp16_run']))
print('  max_wav_value=%s' % d['max_wav_value'])

try:
    open(fl_path, 'rb').read().decode('gbk')
    print('  filelist GBK 解码: OK')
except Exception as e:
    print('  [警告] filelist GBK 解码失败:', e)

expect_mel = 125 if SR >= 40000 else 80
if d['n_mel_channels'] != expect_mel:
    print('  [错误] %d 采样的 mel 应为 %d，当前 %d' % (SR, expect_mel, d['n_mel_channels']))
    sys.exit(1)

print()
print('完成。训练命令：')
print('  cd "%s"' % RVC)
print('  export DSH_NUM_WORKERS=0 DSH_PREFETCH=2 DSH_PERSISTENT=0 PATH="%s:$PATH"' % RVC)
sr_flag = '40k' if SR >= 40000 else '32k'
print('  runtime/python.exe infer/modules/train/train.py -e %s -sr %s -f0 1 -bs %d -g 0 \\'
      % (os.path.basename(LOG), sr_flag, BATCH))
print('    -te %d -se 20 -pg assets/pretrained/f0G%s.pth -pd assets/pretrained/f0D%s.pth \\'
      % (EPOCHS, sr_flag, sr_flag))
print('    -l 0 -c 0 -sw 1 -v v1')
print()
print('注意：实际轮数以 config.json 的 train.epochs 为准，-te 不生效。')
