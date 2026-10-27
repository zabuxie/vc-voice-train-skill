---
name: rvc-voice-train
description: 让 AI 用指定人物的声音说话：两条路线——RVC 音色训练（素材→训练→变声）与 IndexTTS 零样本克隆（参考音频→直接合成）。含素材质量判定、归一化缺陷修复、音高校准、确定性采样、音质验收、跨语种方案选型。当用户说"训练一个音色/声音"、"让AI用某个人的声音说话"、"训个RVC"、"克隆声音"、"中文不自然/音调太高"时使用。
whenToUse: 用户提供了人声素材（视频/音频/游戏语音）并希望让 AI 用该声音说话；或需要排查已有音色的音质问题（噪声/电音/发闷/跨语种不连贯/音调偏高/声调不稳）时使用。
---

# AI 音色克隆（RVC 训练 + IndexTTS 零样本）

让 AI 用指定人物的声音说话。两条技术路线，**先看选型再动手**。

---

## 路线选型（先看这个表）

| 需求 | 推荐 | 原因 |
|---|---|---|
| **跨语种**（如日语素材说中文） | **IndexTTS** | RVC 音素缺失，跨语种必然不连贯 |
| **音色高度还原**（同语种） | **RVC** | 训练过，音色更准 |
| **音调可精确控制** | **RVC** | 音高由驱动音频决定 |
| **零训练、快速上线** | **IndexTTS** | 一段参考音频即可 |
| **需要情感/语气控制** | IndexTTS | 支持情感参考音频 |

> **实测对照**（同一素材：日语 CV 素材训多萝西）：
> - RVC 说日语：音色准、自然 ✅
> - RVC 说中文：**不连贯**（素材无中文音素）❌
> - IndexTTS 说中文：**连贯自然**，但音调偏高（可校准）✅

---

## 第一部分：RVC 音色训练

把一段人声素材训练成 RVC 音色，供 TTS 使用（如 dsh-plugin-tts 的 RVC provider）。

**总耗时**：数据准备 20~40 分钟 + 训练 40~90 分钟 + 验证 10 分钟。

---

## 0. 开始之前：三件事必须确认

这三条能拦住 80% 的失败。**任何一条不过，不要开始训练。**

### 0.1 RVC 版本与归一化缺陷

**只在 RVC 20240604 Nvidia 整合包上验证过**（Python 3.9.13 / torch 2.0.0+cu118）。

该版本的 `infer/lib/train/data_utils.py` 有一个**关键缺陷**：

```python
audio_norm = audio
#        audio_norm = audio / self.max_wav_value    ← 被注释掉
```

`load_wav_to_torch()` 用 `scipy.io.wavfile.read` 读 16-bit PCM，得到 **int16 原始幅度**
（峰值 ~29500），未归一化就送进 `spectrogram_torch()`，训练真值频谱比推理期大 **32768 倍**。
生成器隐空间被污染 → **输出宽带噪声**（听起来像"电音"）。

这个缺陷被 `train.py` 的 `if loss_mel > 75: loss_mel = 75` 掩盖，日志看起来"正常"。

**检查并修复：**

```bash
# 检查：应命中 2 行带 "# DSH fix" 的生效代码
grep -n "audio_norm = audio / self.max_wav_value  # DSH fix" <RVC>/infer/lib/train/data_utils.py

# 若没有，运行修复脚本（幂等，自动备份）
<RVC>/runtime/python.exe <skill>/scripts/fix_audio_norm.py <RVC>
```

> ⚠️ **修复后必须删除 `.spec.pt` 缓存**，否则旧的错误频谱被复用，修复完全无效：
> ```bash
> find <数据目录> -name '*.spec.pt' -delete
> ```

> 💡 **验证陷阱**：`grep -c "audio_norm = audio"` 会数出 **4** 行，因为注释行也含该字符串。
> 判断标准是**第 106 / 298 行行首没有 `#`**，或用 `# DSH fix` 后缀精确匹配。

### 0.1b i18n 模块缺失（导出权重时炸）

RVC 20240604 整合包**缺少 `i18n/` 目录**，但 `infer/lib/train/process_ckpt.py` 依赖它：

```
ModuleNotFoundError: No module named 'i18n'
```

影响 `-sw 1` 自动导出推理权重、或手动转格式时失败。

```bash
<RVC>/runtime/python.exe <skill>/scripts/fix_i18n.py <RVC>
```

（幂等；已存在则跳过并验证可导入）

### 0.2 素材质量（决定上限）

**素材决定天花板，训练救不回来。** 先用体检脚本量化：

```bash
<RVC>/runtime/python.exe <skill>/scripts/diagnose_audio.py <素材文件或目录>
```

**关键：用"相对 1–3 kHz 的能量倍数"判断高频延伸，别看绝对百分比。**

| 12–16 kHz 相对 1–3k | 结论 | 样本 |
|---|---|---|
| **≥ 0.20×** | ✅ 优秀 | 卡提 0.251× / 多萝西 PRTS 0.219× |
| 0.10× ~ 0.20× | 可用 | 高码率编码音频 |
| 0.05× ~ 0.10× | 偏差 | 中码率 |
| **< 0.05×** | ❌ 换源 | 低码率编码（高频被硬切） |

> ⚠️ **不要用"8–16k 占全频段百分比"判断**——分母包含 20 kHz 以上的空频段，
> 比例会被放大，导致误判。实测踩过：某工具报"8–16k 35.8%"看似优秀，
> 但切片后真实值只有 1.1%，素材其实不能用。
>
> **正确做法**：以 1–3 kHz 为基准算倍数，或看谐波纯度。

**实测素材档案：**

| 素材 | 规格 | 12-16k 相对 1-3k | 结果 |
|---|---|---|---|
| 80 kb/s Vorbis OGG | 有损 | ≈0.03×（14k 被硬切） | ❌ 训练出"过度平滑" |
| PRTS PCM | 16-bit / 44.1k / 705 kb/s | **0.219×** | ✅ 可用 |
| 卡提原始音频 | **24-bit / 48k / 1152 kb/s** | **0.251×** | ✅ 一次成功 |

**素材优先级的经验值：**

| 来源 | 典型质量 |
|---|---|
| 录音棚 / 本地录音 PCM 44.1k+ | ⭐⭐⭐⭐⭐ |
| OBS 本地录制 / 游戏 PCM 原始资源 | ⭐⭐⭐⭐ |
| 高码率（≥192 kb/s）视频/直播回放 | ⭐⭐⭐ |
| **低码率 OGG/MP3（≤96 kb/s）** | ⭐⭐ 高频被硬切，**不推荐** |
| 电话/微信语音 | ⭐ 不要用 |

> 实测案例：80 kb/s Vorbis 素材 14 kHz 以上被编码器切掉（16 kHz 只剩 0.008×能量），
> 训出的模型"过度平滑"；换成 705 kb/s PCM 后彻底解决。

### 0.3 GPU 与显存

| 显存 | 能训 | 建议参数 |
|---|---|---|
| **8 GB** | 32k | `-bs 1`，`segment_size=3200` |
| **12 GB** | 32k / **40k** | `-bs 2`，`segment_size=12800` |
| **16 GB+** | 40k 很轻松 | 同上，可加大 batch |

> ⚠️ **40k 在 8G 上跑不了**（实测连 `bs=1` 都 OOM）。
> 显存不够时用云电脑（12G/16G 独立 GPU），本地 8G 只做推理。

**训练前必须清空 GPU：**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# 期望：只有表头（空）

# 常见占用者
taskkill /F /IM QQ.exe; taskkill /F /IM douyin.exe; taskkill /F /IM WeChat.exe
# rvc-server 也要停（见 5.3）
```

---

## 1. 数据准备

### 1.1 从素材得到人声

**已有干净人声（游戏语音/录音）→ 跳到 1.2。**

需要从视频/直播提取人声时：

```bash
cd <RVC>
UVR_SRC="<原视频>" UVR_MODEL=HP2_all_vocals UVR_OUT="<工作目录>/separated" \
  runtime/python.exe uvr-split.py
# 产出 *_vocals.wav（人声）+ *_instrumental.wav（伴奏）
```

**必须做二次提纯**（HP2 后人声里仍混背景音乐，模型会学成"电音"）：

```bash
UVR_SRC="<工作目录>/separated/xxx_vocals.wav" UVR_MODEL=HP5_only_main_vocal \
UVR_OUT="<工作目录>/hp5" runtime/python.exe uvr-split.py
```

**校验提纯效果**（静音段电平越低越干净，目标 ≤ -60 dBFS）：

```bash
<RVC>/runtime/python.exe <skill>/scripts/hp5_verify.py <hp5目录>
```

> 长视频（>10 分钟）必须分段处理，否则内存爆。
> `uvr-split.py` 内部用相对路径读 `configs/`，**CWD 必须是 RVC 根目录**。

### 1.2 切片

按静音切成 3~12 秒的片段：

```bash
cd <RVC>
PATH="<RVC>:$PATH" runtime/python.exe <skill>/scripts/slice_audio.py \
  "<人声wav>" "<输出目录>/wavs" [目标采样率]
```

- 不传目标采样率时**保持源采样率**（推荐，避免损失高频）
- 切片条件：3~12 秒、peak > 0.005、归一化到 0.9
- **数据量参考**：10~30 分钟素材足够；超过 40 分钟收益递减

### 1.3 预处理（关键：不要用 preprocess.py）

> ⚠️ **`infer/modules/train/preprocess.py` 会二次切片！**
> 它内部用 `Slicer(min_length=1500)` 再切一遍，把已切好的片段切碎
> （实测 80 段 → 40 段，总时长 441s → 105s，**丢 76% 数据**）。

用本项目脚本代替（只做高通 + 峰值归一化 + 转 16k）：

```bash
<RVC>/runtime/python.exe <skill>/scripts/make_dataset.py \
  "<wavs目录>" "<exp目录>" <采样率>
# 例：make_dataset.py E:/data/wavs E:/data/exp 40000
```

产出标准 RVC 数据集结构：

```
exp/
├── 0_gt_wavs/      目标采样率音频（采样率 = 上一步传的值）
└── 1_16k_wavs/     16kHz（判别器参考）
```

### 1.4 音高 + 特征

```bash
cd <RVC>
export PATH="<RVC>:$PATH"

# 音高（rmvpe，比 harvest 噪声低）
runtime/python.exe infer/modules/train/extract/extract_f0_rmvpe.py \
  1 0 0 "<exp目录>" True

# HuBERT 特征（v1 = 256 维）
runtime/python.exe infer/modules/train/extract_feature_print.py \
  cuda:0 1 0 0 "<exp目录>" v1 True
```

**校验**（帧数比例应为 2.0）：

```bash
<RVC>/runtime/python.exe <skill>/scripts/check_dataset.py "<exp目录>" <采样率>
```

### 1.5 生成 filelist（5 列，顺序固定）

```
0_gt_wavs/xxx.wav|3_feature256/xxx.npy|2a_f0/xxx.wav.npy|2b-f0nsf/xxx.wav.npy|0
```

- 第 1 列：**目标采样率音频**
- 第 2 列：**特征 .npy**（不是 stem！）
- 5 列用 `|` 分隔，缺列会 `KeyError`

用脚本生成（同时写 config）：

```bash
<RVC>/runtime/python.exe <skill>/scripts/make_config.py \
  "<exp目录>" "<RVC>/logs/<实验名>" "<RVC>" [采样率]
```

> ⚠️ **路径必须纯 ASCII**。RVC 用 GBK 读 filelist，中文路径会 `UnicodeDecodeError`。

---

## 2. 训练

### 2.1 启动

```bash
cd <RVC>
export DSH_NUM_WORKERS=0 DSH_PREFETCH=2 DSH_PERSISTENT=0
export PATH="<RVC>:$PATH"

# 32k
runtime/python.exe infer/modules/train/train.py \
  -e <实验名> -sr 32k -f0 1 -bs 1 -g 0 -te 100 -se 20 \
  -pg assets/pretrained/f0G32k.pth -pd assets/pretrained/f0D32k.pth \
  -l 0 -c 0 -sw 1 -v v1

# 40k（显存必须 ≥12G）
runtime/python.exe infer/modules/train/train.py \
  -e <实验名> -sr 40k -f0 1 -bs 2 -g 0 -te 100 -se 20 \
  -pg assets/pretrained/f0G40k.pth -pd assets/pretrained/f0D40k.pth \
  -l 0 -c 0 -sw 1 -v v1
```

**参数要点：**

| 参数 | 坑 |
|---|---|
| `-sr` | **必须是 `32k`/`40k`（字符串）**，写 `32000` 会 `KeyError` |
| `-te` | ⚠️ **不控制实际轮数！** 循环读 `config.json` 的 `train.epochs`，改 `-te` 无效 |
| 底模 | 必须匹配采样率：32k→`f0G32k`，40k→`f0G40k` |
| config | 40k 用 `configs/v1/40k.json`（mel=125, filter=2048），32k 用 `32k.json`（mel=80, filter=1024） |

> **改轮数要改 config.json：**
> ```bash
> runtime/python.exe -c "import json;p='<logs>/<exp>/config.json';c=json.load(open(p));c['train']['epochs']=200;json.dump(c,open(p,'w'),indent=2)"
> ```

### 2.2 ★ 前 3 轮就要判断成败（最重要）

日志每次打印形如：

```
loss_disc=3.2, loss_gen=8.4, loss_fm=2274.4, loss_mel=31.2, loss_kl=6.1
```

| 指标 | 健康 ✅ | 异常 ❌ | 异常含义 |
|---|---|---|---|
| **`loss_disc`** | **个位数（2~6）** | **1e8 以上** | **归一化缺陷没修好** |
| `loss_mel` | 10~30 波动下降 | 恒 `75.000` | 同上（被截断） |
| `loss_kl` | 9 → 0.6~2 | 一直 9 不动 | 训练没生效 |

> ⚠️ **`loss_mel=75.000` 是硬截断**（`train.py:533`），不是真实值。
> 恒 75 说明真实误差远超 75。
> ⚠️ `loss_disc` 数值巨大（1e8~1e9）**不是正常现象**——这是对抗训练里唯一的强信号。

**如果前 3 轮 `loss_disc` 是 1e8 级 → 立即停，回去检查 0.1 的归一化修复和 `.spec.pt` 缓存。**

### 2.3 训练完做什么

`-sw 1` **会自动导出推理权重**到 `assets/weights/<exp>.pth`（格式已正确，含 config/f0/version，无 `enc_q`）。

**但索引不会自动建**，必须手动（见 3.1）。

---

## 3. 索引与验收

### 3.1 建 faiss 索引

> ⚠️ **faiss 与运行中的 rvc-server 冲突，会 Segfault（exit 139）。先停服务。**

```bash
# 确认没有 python 进程占用
nvidia-smi --query-compute-apps=pid --format=csv

<RVC>/runtime/python.exe <skill>/scripts/make_index.py "<logs/实验名>" v1
```

**没有索引 → 免索引模式 → 音质明显变差。**

> 索引路径必须纯 ASCII，faiss 读中文路径会**静默失败**（不报错，只是没加载）。

### 3.2 ★ 音质验收（用 `probe_accept.py`，不要只看 flat）

**不要只用 `speak` 接口试听**——那条链路上 TTS 前端可能出问题，干扰判断。

**第一步：生成输出**
```bash
<RVC>/runtime/python.exe <skill>/scripts/probe_quality.py "<logs/实验名>" <实验名> [驱动音频]
# 产物：logs/<实验名>/probe_output.wav
```

**第二步：与真值对照验收（关键）**
```bash
<RVC>/runtime/python.exe <skill>/scripts/probe_accept.py \
  <logs/实验名>/probe_output.wav <同内容的真人原声.wav>
```

**达标标准（输出 vs 真值）：**

| 指标 | 标准 |
|---|---|
| 基频 | 差 < 30 Hz |
| 1–3 kHz 占比 | 差 < 3 个百分点 |
| 3–6 kHz 占比 | 差 < 2 个百分点 |
| **谐波纯度** | **差 < 0.05**（最关键） |
| flat | **仅作参考**，阈值随音色变化 |

> ⚠️ **flat 不是通用标准**（这是最容易踩的坑）。实测反例：
>
> | 模型 | flat | 谐波纯度 | 真值谐波 | 结果 |
> |---|---|---|---|---|
> | 多萝西降调版 | **0.040（最低）** | 0.964 | 0.928 | ❌ 过度平滑，用户否决 |
> | 卡提原声 | **0.21（超标）** | 0.907 | 0.882 | ✅ 用户确认"没问题" |
> | 多萝西 v2 | 0.105 | 0.909 | 0.909 | ✅ 定稿 |
>
> **flat 低 ≠ 好听。** 高频丰富的音色（如卡提）flat 天然偏高。

**谐波纯度怎么读：**

| 情况 | 含义 | 听感 |
|---|---|---|
| 与真值接近（差 < 0.05） | 自然 | 正常 |
| **明显偏高**（如 0.96+ vs 真值 0.91） | **频谱被过度平滑** | 发闷、发僵、"电音感" |
| 明显偏低 | 谐波结构被破坏 | 带噪、沙哑 |

> 该指标由 `probe_accept.py` 计算：对有声音帧，取 f0 及其各次谐波 ±90 Hz 内的能量，
> 除以该帧总能量，再对所有有声帧求均值。

---

## 4. 部署与试听

### 4.1 放置文件

```
<RVC>/assets/weights/<名字>.pth
<RVC>/assets/indices/added_IVF*_<名字>_v1.index
```

### 4.2 热切换（推荐，不用重启）

```bash
curl -X POST http://127.0.0.1:4892/load \
  -H "Content-Type: application/json" \
  -d '{"model":"<正斜杠路径>/<名字>.pth","index":"<正斜杠路径>/xxx.index"}'
```

> ⚠️ 路径**必须用正斜杠** `/`，反斜杠会 JSON 转义失败。

### 4.3 启动脚本（如果要设为默认）

编辑 `rvc-server.cmd`，改 `--model` / `--index`，并确保：

| 要求 | 原因 |
|---|---|
| **CRLF 换行** | LF 会导致 `'ght_root' is not recognized` |
| **纯 ASCII** | 中文注释也会破坏解析 |
| **3 个环境变量** | `weight_root` / `index_root` / `rmvpe_root`（缺 rmvpe_root → `KeyError: 'rmvpe_root'`） |

> ⚠️ **改完 .cmd 务必验证换行符**（编辑工具常插入 LF）：
> ```powershell
> $p="<路径>"; $c=[IO.File]::ReadAllText($p) -replace "`r?`n","`r`n"
> [IO.File]::WriteAllText($p,$c,[Text.UTF8Encoding]::new($false))
> ```

### 4.4 推理参数（已实测最优，别乱调）

```javascript
baseVoice: 'zh-CN-XiaoyiNeural',  // 女性驱动！男声驱动会出电音（跨性别转换伪影）
f0Method: 'rmvpe',                 // 比 harvest 噪声低
indexRate: 0.75,                   // 不是越高越好，0.85/0.90 反而变差
protect: 0.50,                     // 0.7 会严重发闷
rms_mix_rate: 0.25,
```

> 参数扫描实测：`index_rate 0.75→1.00`、`protect 0.25→0.50` 对频谱分布影响 < 0.5%，
> **默认值即最优**。
> ⚠️ `pipeline.py` 里 `if protect < 0.5` 才生效，0.5 是边界值。

### 4.5 试听时注意

**如果用 HTTP 接口试听，中文必须用 UTF-8 正确发送。**

```bash
# ❌ 错误：bash 内联中文可能被 shell 破坏编码
curl -d '{"text":"博士你好"}' ...

# ✅ 正确：用 Node/文件方式发送
node -e "fetch(url,{method:'POST',body:Buffer.from(JSON.stringify({text:'博士你好'}),'utf8')})"
```

> 实测教训：bash 的 `curl -d "中文"` 传过去会变乱码（29 字变成 46 个乱码字符），
> 表现为"音频只有 1.87 秒"——**看起来像 TTS 截短，实则是输入就坏了**。

---

## 5. 常见故障速查

### 5.1 训练阶段

| 现象 | 根因 | 解法 |
|---|---|---|
| `WinError 1455 页面文件太小` | 页面文件固定太小 + DataLoader spawn 子进程 | 页面文件改**自动管理**（≥16G）；或 `DSH_NUM_WORKERS=0` + 打 workers 补丁 |
| `CUDA error: resource already mapped` | 其他进程占 CUDA context | 关 QQ/抖音/浏览器/rvc-server |
| `CUDA out of memory` | 显存不够（显示 free 也不一定可用） | 降 `-bs`；40k 需 ≥12G |
| `loss 全是 nan` | fp16 溢出 | config 设 `fp16_run=False` |
| `KeyError: '40000'` | `-sr 40000` | 改 `-sr 40k` |
| `size mismatch 1025 vs 513` | config 与采样率不匹配 | 32k 用 32k.json，40k 用 40k.json |
| `FileNotFoundError: filelist.txt` | filelist 缺失或列数错 | 必须 5 列，用 make_config.py 生成 |
| `preprocess.py` 静默无输出 | 第 3 参数（进程数）填了 0 | 改成 `4`；或改用 make_dataset.py |
| `loss_mel` 恒 75 / `loss_disc` 1e8 | **归一化缺陷** | 跑 fix_audio_norm.py + 删 .spec.pt |

### 5.2 索引阶段

| 现象 | 根因 | 解法 |
|---|---|---|
| `Segmentation fault` exit 139 | faiss 与 rvc-server 冲突 | **先停所有 python 进程** |
| `feature files: 0` | 找不到特征目录 | 用 make_index.py（会自动找）或指定路径 |

### 5.3 部署与推理

| 现象 | 根因 | 解法 |
|---|---|---|
| `KeyError: 'rmvpe_root'` | 环境变量缺失 | 启动脚本加 `set rmvpe_root=...` |
| `'ght_root' is not recognized` | .cmd 是 LF 换行 | 转 CRLF |
| 加载错模型 | 字母序自动匹配取第一个 | **weights 目录只留一个同名模型** |
| 输出是噪声 | 索引不配套 / 模型本身坏 | 确认索引对应版本；用 probe_quality.py 验收 |
| 声音发闷 | `protect` 太高 / 驱动音频太闷 | `protect=0.50`，换清晰驱动 |
| 有电音 | 男声驱动 / 归一化缺陷 | 换 `zh-CN-XiaoyiNeural`；检查 0.1 |
| 完全不像 | 加载了旧模型 | 核对文件名精确匹配 |

---

## 6. 云电脑训练 SOP（本地显存不足时）

**目标**：在租来的 12G/16G 独立 GPU 机器上跑完整个流程。

### 6.1 租之前问清 3 件事

1. **GPU 是直通独占吗？** `nvidia-smi` 能看到完整显存吗？（共享/vGPU 直接换家）
2. **RDP 会话里 CUDA 能用吗？**（部分配置会禁用 GPU）
3. **大文件上传走什么通道？**（7GB 走剪贴板要几小时）

### 6.2 时间预算

| 阶段 | 12G 3060 | 16G 5060Ti |
|---|---|---|
| 上传 7GB | 20~40 min | 20~40 min |
| 环境验证 | 2 min | 2 min |
| 训练（100 轮，80 段） | ~70 min | ~42 min |
| 建索引 + 验收 | 12 min | 12 min |
| **合计** | **~2h** | **~1.3h** |

> **按日租比按小时划算**，且不用担心超时中断。

### 6.3 云端必做

```bash
# 1. 环境自检
<RVC>/runtime/python.exe <skill>/scripts/verify_env.py <RVC>

# 2. 确认归一化补丁在位
grep -c "audio_norm = audio / self.max_wav_value  # DSH fix" <RVC>/infer/lib/train/data_utils.py
# 期望 2

# 3. 清 GPU
nvidia-smi --query-compute-apps=pid --format=csv
```

### 6.4 路径必须纯 ASCII

云机器上也要遵守：`E:/rvc/rvc_env` 可以，`E:/我的音色任务/` 不行。
中文路径会导致 filelist GBK 解码失败、faiss 静默失败。

---

## 7. 关键参考数据

### 7.1 40k vs 32k 参数对照

| 参数 | 32k | 40k |
|---|---|---|
| config 模板 | `configs/v1/32k.json` | `configs/v1/40k.json` |
| `-sr` | `32k` | `40k` |
| 底模 | `f0G32k` / `f0D32k` | `f0G40k` / `f0D40k` |
| mel bins | 80 | 125 |
| filter_length | 1024 | 2048 |
| hop_length | 320 | 400 |
| upsample_rates | [10,4,2,2,2] | [10,10,2,2] |

### 7.2 f0 是量化值不是 Hz

`2a_f0/*.npy` 存的是 **1~255 的量化值**，不是频率。反解：

```python
mel_min = 1127 * ln(1 + 50/700)
mel_max = 1127 * ln(1 + 1100/700)
mel = (quant - 1) * (mel_max - mel_min) / 255 + mel_min
hz  = 700 * (exp(mel/1127) - 1)
```

实测：quant=94 → 328.6 Hz（与自相关测的 333 Hz 吻合）。

> 看到 f0 文件里是几十，**不要以为是几十 Hz**。

### 7.3 数据体检正常值

| 项 | 正常范围 |
|---|---|
| 削波率 | 0.00000 |
| 直流偏移 | < 0.001 |
| 静音段电平 | ≤ -60 dBFS（越低越干净） |
| 片段长度 | 3~12 秒 |
| f0/feat 帧数比 | 2.00 ± 0.05 |
| spec/feat 帧数比 | 2.00 ± 0.05（注意 hop 值随采样率变） |

---

## 8. 脚本清单

所有脚本在 `<skill>/scripts/`，均支持 `--help` 或第一个参数传路径。

| 脚本 | 用途 | 何时用 |
|---|---|---|
| `fix_audio_norm.py` | 修复归一化缺陷（幂等+备份） | **训练前必跑** |
| `diagnose_audio.py` | 素材体检（频段/基频/电平） | **选素材时** |
| `slice_audio.py` | 切片（保持源采样率，不二次切） | 数据准备 |
| `make_dataset.py` | 预处理（高通+归一化+转16k，不二次切） | 数据准备 |
| `check_dataset.py` | 校验数据集完整性+帧对齐 | 提取完特征后 |
| `make_config.py` | 生成 filelist + config | 训练前 |
| `make_index.py` | 建 faiss 索引 | 训练后 |
| `probe_quality.py` | 音质验收（flat/谐波/频段） | **训练后必跑** |
| `verify_env.py` | 环境自检（GPU/文件/数据/config） | **开工前** |
| `patch_train_workers.py` | 打 workers=0 补丁 | 遇 1455 时 |
| **`fix_i18n.py`** | **补齐 RVC 缺失的 i18n 模块** | **训练前**（导出权重要用） |
| **`probe_accept.py`** | **输出 vs 真值验收（谐波纯度/频段贴合）** | **训练后必跑** |
| `hp5_verify.py` | 校验 HP5 提纯效果 | 分离后 |

---

## 9. 绝对不要做的事

- ❌ 用 `preprocess.py` 处理已切好的数据（会二次切片丢 76%）
- ❌ 只改 `-te` 就以为改了轮数（要改 config.json）
- ❌ 改完 `data_utils.py` 不删 `.spec.pt`（修复完全无效）
- ❌ 在中文路径下训练（GBK 崩溃 / faiss 静默失败）
- ❌ 训练时开着 QQ/抖音/浏览器（CUDA context 冲突）
- ❌ 建索引时服务还在跑（Segfault）
- ❌ 用 bash 内联中文调 HTTP 接口（编码被破坏）
- ❌ 只靠 flat 判断音质（要配合频段分布 + 谐波纯度）
- ❌ 把多个同名模型放在 weights 目录（字母序抢占）
- ❌ 认为 `loss_disc` 1e8 是"正常的对抗训练震荡"（那是缺陷信号）

---

# 第二部分：IndexTTS 零样本克隆（跨语种首选）

**不训练**，给一段参考音频就能合成目标音色。跨语种能力远超 RVC。

## 10. 安装

### 10.1 硬件

| 项 | 要求 |
|---|---|
| 显存 | **≥ 6 GB**（8G 卡可跑，自动低显存模式） |
| 磁盘 | ~20 GB（环境 8G + 模型 5.2G） |
| Python | **3.10~3.11**（3.12+ 不支持） |

### 10.2 步骤

```bash
# 1. 装 uv（比 pip 快很多）
curl -L -o uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
# 解压得到 uv.exe

# 2. 装 Python 3.11（uv 自带下载慢时，用 gh 下构建包）
#    gh release download 20261021 --repo astral-sh/python-build-standalone \
#      --pattern "*cpython-3.11*install_only_stripped.tar.gz" --dir .
#    tar -xzf <包名>

# 3. 克隆源码
git clone --depth 1 https://github.com/index-tts/index-tts.git

# 4. 建环境（走清华镜像，否则 PyTorch 下载极慢）
cd index-tts
export UV_CACHE_DIR=<ENV_ROOT>/cache UV_PYTHON_DOWNLOADS=never
../uv/uv.exe venv --python <python3.11路径>
../uv/uv.exe sync --python .venv/Scripts/python.exe \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 装适配器额外依赖
../uv/uv.exe pip install fastapi uvicorn --python .venv/Scripts/python.exe \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 10.3 下载模型（5.2 GB）

从 [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) 下到 `checkpoints/`：

| 文件 | 大小 |
|---|---|
| `gpt.pth` | **3.1 GB** |
| `s2mel.pth` | 396 MB |
| `codec.pth` | 579 MB |
| `qwen0.6bemo4-merge/model.safetensors` | **1.1 GB** |
| `config.yaml` / `feat1.pt` / `feat2.pt` / `bpe.model` / `wav2vec2bert_stats.pt` | 小文件 |

> ⚠️ **仓库里没有 `bpe.model` 和 `campplus_cn_common.bin`**（我按常见文件名猜的，实测 404）。
> 只有上面列出的 22 个文件。`campplus`/`bigvgan` 会在首次加载时自动下到 `checkpoints/hf_cache/`。
>
> ⚠️ **hf-mirror 会限流**：并发下载小文件会 403。大文件串行、小文件带重试。

### 10.4 部署适配器

把 `indextts/adapter.py` 放到 `<ENV_ROOT>/indextts/`，参考音色放 `<ENV_ROOT>/voices/`：

```
<ENV_ROOT>/
├── index-tts/          源码 + .venv
├── checkpoints/        模型权重（或 index-tts/checkpoints）
├── voices/             参考音色 *.wav
└── indextts/
    ├── adapter.py
    ├── 启动IndexTTS.cmd
    └── launch_hidden.vbs
```

启动：**双击 `启动IndexTTS.cmd`**（或 `wscript launch_hidden.vbs`）。

## 11. 音高校准（最容易被忽略的一步）

**IndexTTS 的输出音高 ≈ 参考音频的音高。** 参考选错，合成出来就"音调太高"。

### 11.1 用脚本挑参考

```bash
<RVC>/runtime/python.exe scripts/pick_pitch_ref.py <素材目录> \
  --out <ENV_ROOT>/voices --name myvoice --f0-min 300 --f0-max 380
```

输出每个片段的中位 f0 / P10 / P90 / 波动，按 P90 升序排列。

### 11.2 判定标准

| 指标 | 含义 | 参考值 |
|---|---|---|
| **中位 f0** | 整体音高 | 成年女声日常说话 **200~260 Hz**；偏高音色 300~380 |
| **P90** | 高音峰值，决定刺耳感 | **越低越稳**；> 500 Hz 会明显发尖 |
| 时长 | 太短学不稳 | **≥ 5 秒**，10~30 秒最佳 |

**实测对照（同一句中文，输出音高）**：

| 参考 | 参考中位/P90 | 输出中位/P90 | 听感 |
|---|---|---|---|
| 高音台词 | 396 / 556 Hz | **441 / 525 Hz** | ❌ 偏尖 |
| 中等台词 | 353 / 496 Hz | 350 / 507 Hz | ⚠️ 可接受但偶有尖 |
| 平稳片段 | 313 / 441 Hz | 320 / 384 Hz | ✅ 自然 |

> **规律：输出中位 ≈ 参考中位（接近 1:1）。** 想要什么音高，就挑什么音高的参考。

## 12. 标点情感（按句尾标点自动调语气）

**RVC 做不到**——它只是变声器，语气完全由驱动音频决定。
**IndexTTS 可以**，用 8 维情感向量（`emo_vector`）。

适配器已内置按句尾标点自动推断（`INDEX_TTS_EMO=0` 可关）：

| 句尾 | 情感向量（喜/怒/哀/惧/厌/低落/惊喜/平静） | 实测 f0 |
|---|---|---|
| `。` 或无标点 | 平静 0.60 | 324 Hz |
| `？` | 平静 0.55 + 惊喜 0.15 | 322 Hz |
| **`！`** | 喜 0.35 + 惊喜 0.45 | **469 Hz（+145）** |
| `！！` | 喜 0.50 + 惊喜 0.50 | 394 Hz |
| `…` | 低落 0.35 + 平静 0.35 | 犹豫感 |
| `~` | 喜 0.40 + 平静 0.30 | 轻松 |

请求里显式传 `emo_vector: [8个数]` 可覆盖自动推断。

> ⚠️ **只按句尾标点判断整段情绪**，不逐句分析——逐句切分会破坏停顿连贯性。
>
> ⚠️ `use_emo_text`（情感文本描述）需加载 QwenEmotion（+1.1 GB 显存），
> 8G 卡不建议；`emo_vector` 不额外占显存。

## 13. 确定性采样（解决声调忽好忽坏）

**问题**：同一句话每次合成，末尾声调不同。实测"试"字的去声斜率在
**-7.7 / -1.8 / -0.2** 之间跳动（-0.2 就是听感"读平了"）。

**根因**：`do_sample=true` + 支持扩散采样，每次随机种子不同。

**修复**（适配器已内置）：

```python
# 1) 强制 do_sample=False
# 2) 每次合成前固定种子
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

> ⚠️ **只设 `do_sample=False` 不够**——实测三次输出 MD5 仍不同（内部还有其他随机源）。
> **必须同时固定种子**才能完全可复现。
>
> 验证方法：连续合成 3 次同一句话，输出 MD5 应完全一致。

## 14. 适配器要点（踩过的坑）

| 坑 | 现象 | 解决 |
|---|---|---|
| **pydantic 可选字段** | 插件传 `null` 报 **422** | 用 `Optional[str] = None`，不能写 `str = None` |
| **emoji 编码** | 文本含 🌙 报 **500 GBK 错误** | ① `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` ② 清洗 emoji |
| **低显存分段** | 8G 卡 >40 字强制切段，停顿怪 | 把 `split_text_by_punctuation` 阈值从 40 调到 **90** |
| **完全关闭分段** | 63 字合成出现 **12.7 秒静音** | 不要关，只调阈值；90 字是实测可用值 |
| **进程随父退出** | 脚本一关服务就死 | 用 **VBScript + `WScript.Shell.Run`**（`Start-Process` 仍会跟着死） |
| **WinError 1455** | 加载模型时报"页面文件太小" | 关 QQ/浏览器/音乐播放器；适配器自己占 **5.2 GB 内存** |
| **首次加载慢** | 30~60 秒无响应 | 正常，模型加载 + 预热 |

## 15. 音高/声调自查脚本

| 脚本 | 用途 |
|---|---|
| `scripts/pick_pitch_ref.py` | 挑参考音频（按中位 f0 + P90 排序） |
| `scripts/probe_accept.py` | 输出 vs 真值验收（谐波纯度/频段） |

## 16. QQ 语音桥接（可选）

把合成结果发到 QQ 语音消息（需要 [NapCat](https://github.com/NapNeko/NapCatQQ)）：

```bash
node qq/qq_voice.mjs send <QQ号> "要说的话"
node qq/qq_voice.mjs send-group <群号> "要说的话"
```

前置：NapCat 开着 WebSocket（默认 `ws://127.0.0.1:3001`）+ ffmpeg。

> - NapCat 的 `can_send_record` 应为 `{yes: true}`
> - `record` 消息段同时接受 **mp3 和 wav**（mp3 体积小 7 倍）
> - 大模型推理 + QQ 发送，5 秒内的短句约 5 秒完成

## 17. 两条路线怎么选（再强调一次）

| 你的情况 | 选 |
|---|---|
| 素材是**外语**，要它说中文 | **IndexTTS**（RVC 必然不连贯） |
| 素材和目标语言一致 | **RVC**（音色更准） |
| 想要**低音**的跨语种效果 | RVC（驱动音频决定音高）；或 IndexTTS 挑低音参考 |
| 不想训练 | IndexTTS |
| 8G 显存 | 都行（IndexTTS 5.2G 内存 + 5G 显存；RVC 更省） |

> **两者不能同时占显卡**：IndexTTS ~5 GB，RVC ~2-3 GB，8G 卡装不下两个。
> 切换时要停一个起另一个（见 `patches-index.mjs.diff` 里的 `backend-control` 路由 +
> `patches-client.js.diff` 里的「快速切换后端」按钮；启动脚本路径由 `DSH_RVC_START` /
> `DSH_INDEX_START` 环境变量提供）。
