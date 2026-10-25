# -*- coding: utf-8 -*-
"""
构建 RVC faiss 索引（CLI 训练流程不含此步，必须手动跑）

用法：
    <RVC>\\runtime\\python.exe make_index.py <实验日志目录> [版本] [特征目录]

    [版本]       v1（256维，默认）或 v2（768维）
    [特征目录]  3_feature256 所在目录；默认自动查找：
                  1) <实验日志目录>/3_feature256
                  2) config.json 里 0_gt_wavs 的同级 ../3_feature256
                  3) E:/rvc_data/exp40/3_feature256

示例：
    runtime\python.exe make_index.py <RVC>/logs/<实验名> v1

注意：
  - faiss 与运行中的 rvc-server 冲突，**必须先停服务**，否则段错误 exit 139
  - 索引路径必须纯 ASCII，否则 faiss 静默失败
"""
import os
import sys
import glob
import json
import numpy as np
import faiss

if len(sys.argv) < 2:
    print('用法: python make_index.py <实验日志目录> [v1|v2] [特征目录]')
    sys.exit(1)

LOG = os.path.abspath(sys.argv[1])
version = sys.argv[2] if len(sys.argv) > 2 else 'v1'
exp_name = os.path.basename(LOG.rstrip(os.sep))

# 找 RVC 根目录
RVC = None
p = LOG
for _ in range(6):
    p = os.path.dirname(p)
    if os.path.isfile(os.path.join(p, 'infer', 'modules', 'vc', 'pipeline.py')):
        RVC = p
        break
if RVC is None:
    RVC = r'E:\rvc\RVC20240604Nvidia'
print('RVC :', RVC)
print('LOG :', LOG)
print('EXP :', exp_name)
os.chdir(RVC)
sys.path.insert(0, RVC)

# 找特征目录
cands = []
if len(sys.argv) > 3:
    cands.append(sys.argv[3])
cands.append(os.path.join(LOG, '3_feature' + ('256' if version == 'v1' else '768')))
# 从 config 推
try:
    c = json.load(open(os.path.join(LOG, 'config.json'), encoding='utf-8'))
    tf = c['data']['training_files']
    if os.path.isfile(tf):
        first = open(tf, encoding='utf-8').read().strip().split('\n')[0]
        feats = first.split('|')[1]
        cands.append(os.path.dirname(feats))
except Exception:
    pass
cands.append(r'E:/rvc_data/exp40/3_feature256')

fea_dir = None
for c in cands:
    if c and os.path.isdir(c) and glob.glob(os.path.join(c, '*.npy')):
        fea_dir = c
        break

if fea_dir is None:
    print('[错误] 找不到特征目录，尝试过:')
    for c in cands:
        print('   ', c)
    sys.exit(1)

files = sorted(glob.glob(os.path.join(fea_dir, '*.npy')))
print('特征目录:', fea_dir)
print('特征文件:', len(files))
if not files:
    print('[错误] 特征目录为空')
    sys.exit(1)

feats = []
for f in files:
    a = np.load(f)
    if a.ndim == 3:
        a = a[0]
    feats.append(a.astype(np.float32))

big = np.concatenate(feats, axis=0).astype(np.float32)
print('特征矩阵:', big.shape)
dim = 256 if version == 'v1' else 768
if big.shape[1] != dim:
    print('[错误] 特征维度 %d 与 %s 要求 %d 不符' % (big.shape[1], version, dim))
    sys.exit(1)

# RVC 原逻辑：样本过多时先用 kmeans 降采样
if big.shape[0] > 200000:
    from sklearn.cluster import MiniBatchKMeans
    print('样本过多，kmeans 降到 1 万中心 ...')
    big = MiniBatchKMeans(n_clusters=10000, verbose=False, batch_size=4096,
                          compute_labels=False, init='random').fit(big).cluster_centers_
    print('降采样后:', big.shape)

n_ivf = min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39)
n_ivf = max(n_ivf, 1)
print('n_ivf =', n_ivf)

index = faiss.index_factory(dim, 'IVF%d,Flat' % n_ivf)
ivf = faiss.extract_index_ivf(index)
ivf.nprobe = 1
index.train(big)
for i in range(0, big.shape[0], 8192):
    index.add(big[i:i + 8192])

fname = 'added_IVF%d_Flat_nprobe_1_%s_%s.index' % (n_ivf, exp_name, version)
out = os.path.join(LOG, fname)
faiss.write_index(index, out)
print('SAVED', out, '%.1f MB' % (os.path.getsize(out) / 1048576))

# 同时复制到 assets/indices
try:
    dst_dir = os.path.join(RVC, 'assets', 'indices')
    os.makedirs(dst_dir, exist_ok=True)
    import shutil
    dst = os.path.join(dst_dir, fname)
    shutil.copyfile(out, dst)
    print('COPIED', dst)
except Exception as e:
    print('[警告] 复制到 assets/indices 失败:', e)

print()
print('完成。索引可用于推理：')
print('  ', out)
