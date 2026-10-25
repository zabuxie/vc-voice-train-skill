# -*- coding: utf-8 -*-
"""
修复 RVC 20240604 的 data_utils.py 归一化缺陷（幂等）

问题：load_wav_to_torch() 用 scipy.io.wavfile.read 读 16-bit PCM，得到 int16 原始
     幅度（峰值 ~29500），但 audio_norm 的归一化被注释掉了，直接送进
     spectrogram_torch()，导致训练真值频谱比推理期大 32768 倍。
     生成器隐空间被污染 → 输出宽带噪声（听起来像"电音"）。
     该缺陷被 train.py 的 `if loss_mel > 75: loss_mel = 75` 掩盖。

修复：把两处 `audio_norm = audio` 改成 `audio_norm = audio / self.max_wav_value`

用法：
    <RVC>\\runtime\\python.exe fix_audio_norm.py [RVC根目录]
"""
import os
import sys
import shutil
import datetime

RVC = sys.argv[1] if len(sys.argv) > 1 else r'E:\rvc\RVC20240604Nvidia'
TARGET = os.path.join(RVC, 'infer', 'lib', 'train', 'data_utils.py')

print('=' * 64)
print('RVC data_utils.py 归一化缺陷修复')
print('目标:', TARGET)
print('=' * 64)

if not os.path.isfile(TARGET):
    print('[错误] 文件不存在')
    sys.exit(1)

with open(TARGET, 'r', encoding='utf-8', newline='') as f:
    src = f.read()

# 已经是修复状态？（必须匹配带 DSH fix 标记的生效行，
# 不能用 'audio_norm = audio / self.max_wav_value' 判断——原文件里这行是注释，会误判）
if 'audio_norm = audio / self.max_wav_value  # DSH fix' in src:
    n = src.count('audio_norm = audio / self.max_wav_value  # DSH fix')
    print(f'[跳过] 已修复（{n} 处），无需重复')
    sys.exit(0)

# 计算需要替换的位置：只替换 `audio_norm = audio` 这一行（后面紧跟被注释的归一化）
OLD = '        audio_norm = audio\n        #        audio_norm = audio / self.max_wav_value'
NEW = '        audio_norm = audio / self.max_wav_value  # DSH fix\n        #        audio_norm = audio / self.max_wav_value'

if '\r\n' in src:
    OLD = OLD.replace('\n', '\r\n')
    NEW = NEW.replace('\n', '\r\n')

count = src.count(OLD)
print(f'找到 {count} 处待修复位置（预期 2）')
if count == 0:
    print('[错误] 未找到预期代码。请手动检查 data_utils.py 第 106/298 行附近')
    sys.exit(2)

# 备份
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
bak = TARGET + '.bak_' + stamp
shutil.copy2(TARGET, bak)
print('[备份]', os.path.basename(bak))

new_src = src.replace(OLD, NEW)
with open(TARGET, 'w', encoding='utf-8', newline='') as f:
    f.write(new_src)

# 校验
with open(TARGET, 'r', encoding='utf-8', newline='') as f:
    check = f.read()
if check.count('audio_norm = audio / self.max_wav_value  # DSH fix') == count:
    print(f'[成功] 已修复 {count} 处')
    print()
    print('重要：修复后必须删除旧的 .spec.pt 缓存，否则修复无效！')
    print('  find E:/rvc_data -name "*.spec.pt" -delete')
else:
    print('[失败] 校验不通过，正在恢复备份')
    shutil.copy2(bak, TARGET)
    sys.exit(3)
