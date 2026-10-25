# -*- coding: utf-8 -*-
"""
RVC 训练环境自检 —— 开工前跑一次，5 分钟内确认能不能用

用法：
    <RVC>/runtime/python.exe verify_env.py [RVC根目录] [数据目录] [日志目录]

    都可省略：
      RVC根目录  默认从当前目录往上找，或看 DSH_RVC_DIR
      数据目录   默认不需要（只检查环境）
      日志目录   默认 <RVC>/logs/<数据目录名>

环境变量：
    DATA_DIR / LOG_DIR  同上

检查项：
    1. Python / torch / CUDA 版本
    2. GPU 型号、显存、是否满足 40k
    3. 显存实际分配测试
    4. 必需模型文件
    5. train.py workers 补丁
    6. 归一化补丁（最关键）
    7. 数据完整性（如果给了数据目录）
    8. config 参数（如果给了日志目录）
    9. faiss 可用性
"""
import os
import sys
import glob
import json

errors = []
warnings = []
ok_count = 0


def ok(msg):
    global ok_count
    ok_count += 1
    print('  [OK]   ' + msg)


def err(msg):
    errors.append(msg)
    print('  [FAIL] ' + msg)


def warn(msg):
    warnings.append(msg)
    print('  [WARN] ' + msg)


def section(t):
    print()
    print('=' * 64)
    print(' ' + t)
    print('=' * 64)


def find_rvc(start):
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


# ---------- 解析参数 ----------
RVC = sys.argv[1] if len(sys.argv) > 1 else find_rvc(os.getcwd())
if not RVC or not os.path.isdir(RVC):
    print('[错误] 找不到 RVC 根目录。请传第 1 个参数，或设 DSH_RVC_DIR')
    sys.exit(1)
RVC = os.path.abspath(RVC)

DATA = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('DATA_DIR', '')
LOG = sys.argv[3] if len(sys.argv) > 3 else os.environ.get('LOG_DIR', '')
if DATA and not LOG:
    LOG = os.path.join(RVC, 'logs', os.path.basename(DATA.rstrip(os.sep)))

print('=' * 64)
print(' RVC 环境自检')
print('=' * 64)
print('  RVC :', RVC)
print('  数据:', DATA or '(未指定，跳过)')
print('  日志:', LOG or '(未指定，跳过)')

if any(ord(c) > 127 for c in RVC):
    err('RVC 路径含非 ASCII 字符（中文路径会导致 GBK 崩溃）')
    print('         建议移到纯英文路径，如 E:\\rvc\\RVC20240604Nvidia')

# ---------- 1. Python / torch ----------
section('1. Python 与 PyTorch')
torch = None
try:
    print('  Python: %s' % sys.version.split()[0])
    print('  架构  : %d bit' % (64 if sys.maxsize > 2 ** 32 else 32))
    ok('Python 可用')
except Exception as e:
    err('Python 检查失败: %s' % e)

try:
    import torch
    print('  torch : %s' % torch.__version__)
    print('  CUDA  : %s' % torch.version.cuda)
    ok('torch 可导入')
except Exception as e:
    err('torch 导入失败: %s' % e)

# ---------- 2. GPU ----------
section('2. GPU 与显存')
gpu_ok = False
if torch is not None:
    try:
        if not torch.cuda.is_available():
            err('CUDA 不可用（驱动问题 / RDP 禁用 GPU / 非直通显卡）')
        else:
            print('  设备数: %d' % torch.cuda.device_count())
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)
            print('  GPU   : %s' % name)
            print('  显存  : %.2f GB' % total_gb)
            print('  算力  : %d.%d' % (props.major, props.minor))
            try:
                free_b, _ = torch.cuda.mem_get_info(0)
                print('  当前空闲: %.2f GB' % (free_b / (1024 ** 3)))
            except Exception:
                pass
            gpu_ok = True
            if total_gb >= 11.0:
                ok('显存 %.1f GB，满足 40k' % total_gb)
            elif total_gb >= 7.5:
                warn('显存 %.1f GB，只能训 32k（40k 需 >=12GB）' % total_gb)
            else:
                err('显存只有 %.1f GB，训练会很吃力' % total_gb)
    except Exception as e:
        err('GPU 查询失败: %s' % e)

# ---------- 3. 显存分配测试 ----------
section('3. 显存分配测试')
if gpu_ok:
    try:
        x = torch.randn(5000, 500, device='cuda')
        y = torch.mm(x, x.t())
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated(0) / (1024 ** 2)
        del x, y
        torch.cuda.empty_cache()
        ok('CUDA 内核可用（分配 %.1f MB 成功）' % used)
    except Exception as e:
        err('显存分配失败: %s' % e)
else:
    warn('跳过（GPU 不可用）')

# ---------- 4. 必需文件 ----------
section('4. 必需模型与配置')
files = [
    ('assets/pretrained/f0G40k.pth', '40k 生成器底模', True),
    ('assets/pretrained/f0D40k.pth', '40k 判别器底模', True),
    ('assets/pretrained/f0G32k.pth', '32k 生成器底模', False),
    ('assets/pretrained/f0D32k.pth', '32k 判别器底模', False),
    ('assets/rmvpe/rmvpe.pt', 'rmvpe 音高模型', True),
    ('assets/hubert/hubert_base.pt', 'HuBERT 特征模型', True),
    ('configs/v1/40k.json', '40k config 模板', True),
    ('configs/v1/32k.json', '32k config 模板', False),
    ('infer/modules/train/train.py', '训练脚本', True),
    ('infer/modules/train/extract/extract_f0_rmvpe.py', 'f0 提取', True),
    ('infer/modules/train/extract_feature_print.py', '特征提取', True),
]
for rel, desc, required in files:
    p = os.path.join(RVC, rel.replace('/', os.sep))
    if os.path.isfile(p):
        sz = os.path.getsize(p) / (1024 ** 2)
        ok('%-40s %.1f MB' % (desc, sz))
    else:
        (err if required else warn)('缺失: %s (%s)' % (rel, desc))

ff = os.path.join(RVC, 'ffmpeg.exe')
if os.path.isfile(ff):
    ok('ffmpeg.exe')
else:
    warn('ffmpeg.exe 不存在（推理/音频转换会失败）')

# ---------- 5. train.py 补丁 ----------
section('5. train.py 补丁')
try:
    tp = os.path.join(RVC, 'infer', 'modules', 'train', 'train.py')
    src = open(tp, encoding='utf-8').read()
    if '_dsh_workers' in src:
        ok('workers=0 补丁已应用（%d 处）' % src.count('_dsh_workers'))
    else:
        warn('未打 workers=0 补丁（遇到 WinError 1455 时用 patch_train_workers.py）')
except Exception as e:
    err('读取 train.py 失败: %s' % e)

# ---------- 6. 归一化补丁（最关键） ----------
section('6. 归一化补丁（data_utils.py）')
try:
    du = os.path.join(RVC, 'infer', 'lib', 'train', 'data_utils.py')
    src = open(du, encoding='utf-8').read()
    fixed = src.count('audio_norm = audio / self.max_wav_value  # DSH fix')
    if fixed >= 2:
        ok('归一化已修复（%d 处）' % fixed)
    elif 'audio_norm = audio\n' in src or 'audio_norm = audio\r\n' in src:
        err('归一化缺陷存在！会训出噪声。运行 fix_audio_norm.py 修复')
        print('         修复后还要删 .spec.pt 缓存，否则无效')
    else:
        warn('未找到 audio_norm 相关代码，可能版本不同')
except Exception as e:
    err('读取 data_utils.py 失败: %s' % e)

# ---------- 7. 数据 ----------
if DATA:
    section('7. 训练数据')
    if not os.path.isdir(DATA):
        err('数据目录不存在: %s' % DATA)
    else:
        if any(ord(c) > 127 for c in DATA):
            err('数据路径含非 ASCII 字符: %s' % DATA)
        counts = {}
        for sub in ['0_gt_wavs', '1_16k_wavs', '2a_f0', '2b-f0nsf', '3_feature256']:
            p = os.path.join(DATA, sub)
            counts[sub] = len([f for f in os.listdir(p) if not f.endswith('.spec.pt')]) \
                if os.path.isdir(p) else 0
        print('  文件数:', counts)
        vals = list(counts.values())
        if min(vals) == 0:
            err('有目录为空，数据不完整')
        elif len(set(vals)) != 1:
            err('各目录文件数不一致')
        else:
            ok('数据完整（%d 段）' % vals[0])
        try:
            import soundfile as sf
            w = sorted(glob.glob(os.path.join(DATA, '0_gt_wavs', '*.wav')))
            if w:
                sr = sf.info(w[0]).samplerate
                print('  采样率: %d' % sr)
                if sr in (32000, 40000):
                    ok('采样率合法')
                else:
                    warn('采样率 %d 不是 32000/40000，可能有问题' % sr)
        except Exception:
            pass

# ---------- 8. config ----------
if LOG:
    section('8. 训练 config')
    cfg_path = os.path.join(LOG, 'config.json')
    if os.path.isfile(cfg_path):
        try:
            c = json.load(open(cfg_path, encoding='utf-8'))
            d = c['data']
            sr = d['sampling_rate']
            exp_mel = 125 if sr >= 40000 else 80
            print('  sr=%s mel=%s hop=%s filter=%s' % (
                sr, d.get('n_mel_channels'), d.get('hop_length'), d.get('filter_length')))
            if d.get('n_mel_channels') == exp_mel:
                ok('mel bins 匹配 (%d)' % exp_mel)
            else:
                err('mel bins 应为 %d，当前 %s（config 与采样率不匹配）'
                    % (exp_mel, d.get('n_mel_channels')))
            print('  epochs=%s bs=%s fp16=%s' % (
                c['train']['epochs'], c['train']['batch_size'], c['train']['fp16_run']))
            fl = d['training_files']
            if os.path.isfile(fl):
                n = len(open(fl, encoding='utf-8').read().strip().split('\n'))
                ok('filelist 存在（%d 行）' % n)
                try:
                    open(fl, 'rb').read().decode('gbk')
                    ok('filelist GBK 解码通过')
                except Exception:
                    err('filelist GBK 解码失败（含中文路径）')
            else:
                err('filelist 不存在: %s' % fl)
        except Exception as e:
            err('config 解析失败: %s' % e)
    else:
        warn('config.json 不存在（还没生成？用 make_config.py）')

# ---------- 9. faiss ----------
section('9. faiss 可用性')
try:
    import faiss  # noqa
    ok('faiss 可导入')
except Exception as e:
    err('faiss 导入失败: %s（若为 Segfault，先关掉 rvc-server）' % e)

# ---------- 汇总 ----------
section('验证结果')
print('  通过: %d' % ok_count)
print('  警告: %d' % len(warnings))
print('  失败: %d' % len(errors))

if errors:
    print()
    print('  必须修复:')
    for e in errors:
        print('    - ' + e)
    print()
    sys.exit(1)
elif warnings:
    print()
    print('  可忽略的警告:')
    for w in warnings:
        print('    - ' + w)
    print()
    print('  >>> 可以开始训练（注意上面警告）')
    sys.exit(0)
else:
    print()
    print('  >>> 环境完全就绪')
    sys.exit(0)
