# RVC 音色训练 · AI 执行手册

> 给接手的 AI 看。读完这一份就能独立完成训练，**不需要向用户提问**。
> 配套：`SKILL.md`（简版流程）、`scripts/`（可执行脚本）

---

## 0. 你的任务

用户提供了人声素材（视频/音频/游戏语音），要训练成可复用的 RVC 音色，供 TTS 使用。

**如果用户没说清楚，按这个默认策略做：**
- 显存 ≥12G → 训 40k
- 显存 <12G → 训 32k，或建议用户用云电脑训 40k
- 素材 <5 分钟 → 提醒用户音色可能不稳定，但仍可训
- 素材 >40 分钟 → 只取前 20~30 分钟（收益递减）

**不要做的事：**
- 不要用 `preprocess.py`（会二次切片丢 76% 数据）
- 不要只改 `-te` 就想改轮数（要改 config.json）
- 不要改完 `data_utils.py` 不删 `.spec.pt`
- 不要在中文路径下训练
- 不要用 bash 内联中文调 HTTP 接口

---

## 1. 开工前：三道必检（5 分钟）

**任何一道不过，停下来报告用户，不要硬训。**

### 1.1 环境自检

```bash
<RVC>/runtime/python.exe <skill>/scripts/verify_env.py <RVC>
```

**判定**：看输出的 `失败: N`。
- N=0 → 继续
- N>0 且是"显存不足" → 告知用户换 40k 需 ≥12G，或改用 32k
- 其他失败 → 按提示修

### 1.2 归一化缺陷补丁

```bash
<RVC>/runtime/python.exe <skill>/scripts/fix_audio_norm.py <RVC>
```

**判定**：
- `[跳过] 已修复（2 处）` → 已有补丁，继续
- `[成功] 已修复 2 处` → 刚打的补丁，**必须接着删 .spec.pt**
- `[错误] 未找到预期代码` → 版本可能不同，**告诉用户**并附带脚本里的预期代码

**删除旧缓存（打补丁后必做）：**

```bash
find <数据目录> -name '*.spec.pt' -delete
```

### 1.3 素材体检

```bash
<RVC>/runtime/python.exe <skill>/scripts/diagnose_audio.py <素材路径>
```

**判定标准（看"8-12k"这一行）：**

| 8-12k 占比 | 结论 | 行动 |
|---|---|---|
| > 8% | 优秀 | 直接训 |
| 4~8% | 可接受 | 训，告知用户高频有限 |
| 2~4% | 偏差 | **先问用户有没有更好质量的源** |
| < 2% | 不可用 | **停止**，明确告诉用户换源 |

**同时检查：**
- 削波 > 0.001 → 让用户降增益重导出
- 采样率 < 40000 且想训 40k → 只能训 32k
- 基频中位 > 450 Hz → 可能要降调，先问用户

---

## 2. 数据准备（20~40 分钟）

### 2.1 得到干净人声

| 用户给的是什么 | 怎么做 |
|---|---|
| 已经是干净人声（游戏语音/录音） | 跳到 2.2 |
| 视频/直播（有背景音） | 跑 UVR5 分离 + HP5 提纯 |
| 多个人说话 | **问用户**要哪个人的时间段 |

**UVR5 分离**（CWD 必须是 RVC 根目录）：

```bash
cd <RVC>
UVR_SRC="<视频>" UVR_MODEL=HP2_all_vocals UVR_OUT="<工作>/sep" \
  runtime/python.exe uvr-split.py
UVR_SRC="<工作>/sep/xxx_vocals.wav" UVR_MODEL=HP5_only_main_vocal \
UVR_OUT="<工作>/hp5" runtime/python.exe uvr-split.py
```

**校验提纯效果：**

```bash
<RVC>/runtime/python.exe <skill>/scripts/hp5_verify.py <hp5目录>
```

判定：静音段应 ≤ -60 dBFS。高于 -50 说明伴奏残留多，建议重提纯或换素材。

### 2.2 切片

```bash
<RVC>/runtime/python.exe <skill>/scripts/slice_audio.py <人声wav> <输出>/wavs
```

- 不传第 3 个参数 = **保持源采样率**（推荐）
- 输出 3~12 秒片段
- 如果提示"素材只有 X 分钟"，转告用户

### 2.3 预处理（不要用 preprocess.py！）

```bash
<RVC>/runtime/python.exe <skill>/scripts/make_dataset.py <wavs> <exp> <采样率>
```

采样率：40k 训练用 `40000`，32k 训练用 `32000`。

产出 `exp/0_gt_wavs` + `exp/1_16k_wavs`。

### 2.4 音高 + 特征

```bash
cd <RVC>
export PATH="<RVC>:$PATH"

runtime/python.exe infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 "<exp>" True
runtime/python.exe infer/modules/train/extract_feature_print.py cuda:0 1 0 0 "<exp>" v1 True
```

### 2.5 校验数据集

```bash
<RVC>/runtime/python.exe <skill>/scripts/check_dataset.py <exp>
```

**必须看到** `数据集校验通过`。如果报帧对齐异常或 nan，**停下来排查**，不要继续训。

### 2.6 生成 config

```bash
<RVC>/runtime/python.exe <skill>/scripts/make_config.py <exp>
# 会自动找 RVC 根目录、推断采样率、生成 filelist + config
```

**校验**：输出的 `sr` / `mel` 要对（40000→125，32000→80），`filelist GBK 解码: OK`。

---

## 3. 训练（40~90 分钟）

### 3.1 清空 GPU

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# 必须是空的

# 常见占用者
taskkill /F /IM QQ.exe 2>nul; taskkill /F /IM douyin.exe 2>nul
taskkill /F /IM WeChat.exe 2>nul
```

**如果 rvc-server 在跑，也要停**（否则冲突）。

### 3.2 启动

```bash
cd <RVC>
export DSH_NUM_WORKERS=0 DSH_PREFETCH=2 DSH_PERSISTENT=0
export PATH="<RVC>:$PATH"

# 40k（显存 >=12G）
runtime/python.exe infer/modules/train/train.py \
  -e <实验名> -sr 40k -f0 1 -bs 2 -g 0 -te 100 -se 20 \
  -pg assets/pretrained/f0G40k.pth -pd assets/pretrained/f0D40k.pth \
  -l 0 -c 0 -sw 1 -v v1

# 32k（8G 也能跑）
runtime/python.exe infer/modules/train/train.py \
  -e <实验名> -sr 32k -f0 1 -bs 1 -g 0 -te 100 -se 20 \
  -pg assets/pretrained/f0G32k.pth -pd assets/pretrained/f0D32k.pth \
  -l 0 -c 0 -sw 1 -v v1
```

### 3.3 ★★ 前 3 轮必须判断（最重要的一步）

找到日志里的：

```
loss_disc=..., loss_gen=..., loss_fm=..., loss_mel=..., loss_kl=...
```

| 指标 | 健康 | 异常 | 异常时的行动 |
|---|---|---|---|
| **`loss_disc`** | **2 ~ 6** | **1e8 以上** | **立即停止**，回 1.2 检查归一化 + 删 .spec.pt |
| `loss_mel` | 10~30 波动 | 恒 `75.000` | 同上 |
| `loss_kl` | 9 → 0.6~2 | 一直 9 | 训练没生效，检查 config |

**只有三项都健康才继续等训练完成。**

> 实测参考：坏的训练 `loss_disc=1.8e9`，修好后 `loss_disc=3.2`。
> 这个差异是 9 个数量级，非常明确。

### 3.4 训练耗时参考

| 数据量 | 12G 3060 | 16G 5060Ti |
|---|---|---|
| 80 段（8 分钟）× 100 轮 | ~70 min | ~42 min |
| 200 段（20 分钟）× 100 轮 | ~150 min | ~90 min |

---

## 4. 索引 + 验收（10 分钟）

### 4.1 停服务再建索引

```bash
# faiss 与 rvc-server 冲突会 Segfault
taskkill /F /IM python.exe 2>nul
nvidia-smi --query-compute-apps=pid --format=csv   # 确认空

<RVC>/runtime/python.exe <skill>/scripts/make_index.py "<RVC>/logs/<实验名>" v1
```

**校验**：输出 `SAVED ... KB` 且文件 > 1 MB。

### 4.2 音质验收

```bash
<RVC>/runtime/python.exe <skill>/scripts/probe_quality.py "<RVC>/logs/<实验名>" <实验名>
```

**判定（看 flat）：**

| flat | 结论 | 行动 |
|---|---|---|
| < 0.15 | 可用 | 继续部署 |
| 0.15~0.5 | 有伪影 | 告知用户，问是否接受 |
| > 0.5 | 噪声 | **训练失败**，回查归一化/素材 |

**同时看谐波纯度**（越接近真值越好，真值参考 0.909）：
- 差 < 0.02 → 自然
- 明显偏高（如 0.97+）→ **过度平滑**，听感像电音

**还要看频段分布**（1-3k / 3-6k / 6-8k / 8-12k）与真值差应 < 3 个百分点。

### 4.3 给用户试听

把 `logs/<实验名>/probe_output.wav` 路径告诉用户，让他先听再决定部署。

**不要跳过这步直接部署。**

---

## 5. 部署

### 5.1 放文件

```bash
cp <RVC>/assets/weights/<exp>.pth <RVC>/assets/weights/<用户要的名字>.pth
cp <RVC>/logs/<exp>/added_IVF*.index <RVC>/assets/indices/
```

**清理同名旧模型**（RVC 按字母序自动匹配，会选错）：

```bash
mkdir -p <RVC>/assets/weights/_archive
mv <RVC>/assets/weights/<旧的>.pth <RVC>/assets/weights/_archive/
```

### 5.2 启动服务 + 热切换

```bash
# 起服务（如果 rvc-server.cmd 存在）
Start-Process cmd -ArgumentList "/c","<rvc-server.cmd>" -WindowStyle Hidden

# 等 30 秒后热切换
curl -X POST http://127.0.0.1:4892/load -H "Content-Type: application/json" \
  -d '{"model":"<正斜杠路径>/<名字>.pth","index":"<正斜杠路径>/xxx.index"}'
```

**路径用正斜杠**，反斜杠会 JSON 转义失败。

### 5.3 验证

```bash
curl http://127.0.0.1:4892/health
# 期望: "model":"<名字>.pth", "index":"<对应索引>"
```

### 5.4 设为默认（可选）

改 `rvc-server.cmd` 的 `--model` / `--index`。**改完必须验证**：

```powershell
# .cmd 必须是 CRLF + 纯 ASCII
$p="<路径>"; $b=[IO.File]::ReadAllBytes($p)
$cr=($b|?{$_ -eq 13}).Count; $lf=($b|?{$_ -eq 10}).Count
$nonAscii=($b|?{$_ -gt 127}).Count
"CR=$cr LF=$lf 非ASCII=$nonAscii"
# 期望 CR=LF 且 非ASCII=0
```

---

## 6. 结果汇报模板

### 6.1 成功

```
音色训练完成

模型：<名字>.pth + <索引名>.index
音质：flat=0.0xx（真值 0.083），谐波纯度 0.xxx
试听：<绝对路径>/probe_output.wav

已部署并加载，可以用了。
```

### 6.2 失败（音质不达标）

```
训练完成但音质不达标

flat=0.xx（应 <0.15），谐波纯度 0.xxx

排查结果：
- 归一化补丁：[已确认/已修复]
- 素材高频（8-12k）：x.x%（<2% 为不可用）
- 其他：<列出发现的问题>

建议：<换源 / 换40k / 降调 / ...>
```

**汇报时必须带上具体数字，不要只说"效果不好"。**

---

## 7. 故障处理决策树

```
训练报错
├─ WinError 1455
│  ├─ 页面文件 < 16G → 让用户改自动管理（需重启）
│  └─ 已够大 → 跑 patch_train_workers.py + DSH_NUM_WORKERS=0
├─ CUDA out of memory
│  ├─ 40k 且显存 < 12G → 改用 32k，或建议云电脑
│  └─ 其他 → 降 -bs，清 GPU 占用
├─ resource already mapped
│  └─ 关 QQ/抖音/rvc-server
├─ loss 全 nan
│  └─ config 设 fp16_run=False
├─ KeyError '40000'
│  └─ -sr 改成 40k
├─ size mismatch
│  └─ config 与采样率不匹配
├─ loss_mel 恒 75 / loss_disc 1e8
│  └─ 归一化缺陷：fix_audio_norm.py + 删 .spec.pt
└─ 帧对齐异常
   └─ 数据提取有问题：重跑 f0/特征
```

**音质不好但没报错**
```
flat > 0.5（噪声）
├─ 索引没建 → make_index.py
├─ 归一化没修 → fix_audio_norm.py + 删 .spec.pt
└─ 素材太差 → diagnose_audio.py

flat < 0.15 但用户说"闷"
├─ protect > 0.5 → 改成 0.50
├─ 驱动音色不对 → 用 zh-CN-XiaoyiNeural
└─ 素材高频不足 → diagnose_audio.py

用户说"有电音"
├─ 男声驱动 → 换女声驱动
├─ 谐波纯度偏高（过度平滑）→ 素材问题
└─ 索引不配套 → 核对 IVF 号
```

---

## 8. 关键数字速查

| 项 | 值 |
|---|---|
| **健康的 `loss_disc`** | 2 ~ 6 |
| **异常的 `loss_disc`** | 1e8 以上 |
| `loss_mel` 截断值 | 75.000（恒等即异常） |
| 可用 flat | < 0.15 |
| 真值 flat 参考 | 0.083 |
| 好模型 flat 参考 | 0.10 |
| 坏模型 flat 参考 | 0.64 |
| 谐波纯度真值参考 | 0.909（差 >0.02 异常） |
| 素材 8-12k 可用下限 | 2% |
| 素材 8-12k 优秀 | > 8% |
| 40k 显存下限 | 12 GB |
| 静音段电平目标 | ≤ -60 dBFS |

---

## 9. 脚本速查

| 脚本 | 命令 | 何时 |
|---|---|---|
| `verify_env.py` | `verify_env.py <RVC>` | **开工前** |
| `fix_audio_norm.py` | `fix_audio_norm.py <RVC>` | **开工前** |
| `diagnose_audio.py` | `diagnose_audio.py <素材>` | **选素材** |
| `slice_audio.py` | `slice_audio.py <wav> <out>` | 准备数据 |
| `make_dataset.py` | `make_dataset.py <wavs> <exp> <sr>` | 准备数据 |
| `check_dataset.py` | `check_dataset.py <exp>` | 提特征后 |
| `make_config.py` | `make_config.py <exp>` | 训练前 |
| `make_index.py` | `make_index.py <logs/exp> v1` | 训练后 |
| `probe_quality.py` | `probe_quality.py <logs/exp> <exp>` | **验收** |
| `patch_train_workers.py` | `patch_train_workers.py <RVC>` | 遇 1455 |
| `hp5_verify.py` | `hp5_verify.py <hp5目录>` | 分离后 |

所有脚本：无参数运行时打印用法。

---

## 10. 环境要求

| 项 | 要求 |
|---|---|
| RVC 版本 | **20240604 Nvidia 整合包**（Python 3.9.13 / torch 2.0.0+cu118） |
| 路径 | **纯 ASCII**（中文会导致 GBK 崩溃和 faiss 静默失败） |
| GPU | 32k: ≥8G ｜ 40k: ≥12G |
| 页面文件 | ≥16G 或设"自动管理" |
| 必需文件 | `assets/pretrained/f0G40k.pth` `f0D40k.pth` `configs/v1/40k.json`（40k 用） |

**其他 RVC 版本注意**：`audio_norm` 缺陷未必存在。`fix_audio_norm.py` 会先检查，
不存在时提示"未找到预期代码"，此时按提示判断——**如果原版就是对的，不用修**。
