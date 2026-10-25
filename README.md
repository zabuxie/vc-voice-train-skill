# rvc-voice-train-skill

DeepSeek Harness (DSH) skill: train RVC voice models from voice clips — with dataset QA, a fix for the RVC 20240604 audio-norm defect, objective quality acceptance, and a full troubleshooting table.

DSH 技能：从人声素材训练 RVC 音色模型。包含素材体检、RVC 20240604 归一化缺陷修复、客观数据集与音质验收、完整踩坑排查表。

## What's inside / 包含什么

```
rvc-voice-train/
├── SKILL.md            # skill definition (frontmatter + full workflow)
└── scripts/            # 12 executable helper scripts
    ├── fix_audio_norm.py       # fix RVC 20240604 audio_norm defect (idempotent)
    ├── diagnose_audio.py       # source material quality check (8-12k energy, pitch, clipping)
    ├── verify_env.py           # pre-flight environment check (GPU/VRAM/patches/config)
    ├── slice_audio.py          # silence-based slicing, keeps source sample rate
    ├── make_dataset.py         # preprocess WITHOUT re-slicing (preprocess.py loses 76% of data)
    ├── check_dataset.py        # dataset integrity + frame alignment validation
    ├── make_config.py          # generate filelist + config.json (40k/32k)
    ├── make_index.py           # build faiss index (the step CLI training skips!)
    ├── probe_quality.py        # objective quality acceptance (flat / harmonics / bands)
    ├── patch_train_workers.py  # DataLoader workers=0 patch (fixes WinError 1455)
    ├── hp5_verify.py           # vocal separation purity check
    └── hp5_batch.sh            # batched UVR5 separation for long audio
```

## Why / 为什么需要这个 skill

Training an RVC voice on RVC 20240604 (Nvidia bundle) fails silently in a way that
looks like "the model is just bad":

- **`infer/lib/train/data_utils.py` never normalizes audio** (`audio_norm = audio`,
  normalization commented out). 16-bit PCM goes into `spectrogram_torch()` with
  amplitude ~29500 → training target spectrum is **32768x larger** than inference →
  generator latent space is polluted → output is broadband noise ("电音").
- The defect is masked by `train.py`: `if loss_mel > 75: loss_mel = 75`, so logs
  look "normal" while training is broken.
- **The fix is one line, plus deleting stale `.spec.pt` caches** — which nobody tells you.

This skill detects and fixes that first, then checks your source material quality
(low-bitrate encoders hard-cut 14 kHz+ which no amount of training restores), then
walks the full pipeline with objective acceptance metrics instead of "listen and hope".

## Quick start / 快速开始

1. Install RVC 20240604 Nvidia bundle (includes Python 3.9.13 + torch 2.0.0+cu118)
2. Copy `rvc-voice-train/` into your DSH skills directory:

```
<DSH_HOME>/skills/rvc-voice-train/     # default ~/.dsh/skills/
```

3. Restart DSH. Trigger it by saying things like「训个RVC」「训练一个音色」.

Or run the scripts directly without the skill loader:

```bash
<RVC>/runtime/python.exe <skill>/scripts/verify_env.py <RVC>
<RVC>/runtime/python.exe <skill>/scripts/fix_audio_norm.py <RVC>
<RVC>/runtime/python.exe <skill>/scripts/diagnose_audio.py <your_audio.wav>
```

## Requirements / 环境要求

| Item | Requirement |
|---|---|
| RVC bundle | **20240604 Nvidia** (verified; other versions untested — the norm defect may not exist there) |
| GPU VRAM | 32k training: ≥8 GB · 40k training: **≥12 GB** |
| Paths | **ASCII only** (Chinese paths break GBK filelist reads and faiss silently) |
| Pagefile | ≥16 GB or "automatically managed" |

## Key signals / 关键信号

| Signal | Healthy | Broken |
|---|---|---|
| `loss_disc` (first 3 epochs) | **2 ~ 6** | **1e8+** → norm defect |
| `loss_mel` | 10–30, fluctuating down | pinned at `75.000` (hard clamp) |
| flat (output) | **< 0.15** | > 0.5 → noise |
| source 8–12 kHz energy | > 8% great · 2–8% ok · **< 2% unusable** | |

## Notes / 注意

- Voice models you train from copyrighted material (game voices, etc.) stay under
  the original copyright holder. This repo ships **no audio, no models** — methods only.
- Verified on Windows + RTX 4060 Laptop 8G (32k) and RTX 3060 12G cloud (40k).
- The `audio_norm` defect may not exist in other RVC versions; `fix_audio_norm.py`
  detects this and tells you.

## License

MIT for the scripts and docs. RVC itself follows its own license (MIT, RVC-Project).
