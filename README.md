# vc-voice-train-skill

Make your AI speak in someone's voice — two routes, pick by need:

- **RVC training** — train a model from voice clips; best timbre accuracy, same-language.
- **IndexTTS zero-shot** — synthesize straight from a reference clip; **the only sane choice for cross-language**.

DSH 技能：让 AI 用指定人物的声音说话。两条路线：

- **RVC 训练** —— 用素材训模型，音色准，适合同语种；
- **IndexTTS 零样本** —— 给一段参考音频直接合成，**跨语种首选**。

## Which one? / 怎么选

| Need | Use | Why |
|---|---|---|
| Foreign-language source, want Chinese output | **IndexTTS** | RVC has no Chinese phonemes → choppy |
| Source language == target language | **RVC** | trained, more accurate timbre |
| Precise pitch control | **RVC** | pitch comes from the driver audio |
| No training budget | **IndexTTS** | one reference clip is enough |

> Measured on the same source (Japanese CV clips, Chinese output):
> RVC = choppy · IndexTTS = natural but ~90 Hz too high (fixable via reference selection).

## What's inside / 包含什么

```
rvc-voice-train/
├── SKILL.md                  # full workflow, selection guide, 16 sections
├── scripts/                  # RVC training + acceptance helpers
│   ├── fix_audio_norm.py     # fix RVC 20240604 audio_norm defect (silent noise output)
│   ├── fix_i18n.py           # add missing i18n module (breaks weight export)
│   ├── probe_accept.py       # output-vs-truth acceptance (harmonic purity, bands)
│   ├── pick_pitch_ref.py     # choose reference clips by pitch (median f0 + P90)
│   ├── diagnose_audio.py     # source material quality check
│   ├── verify_env.py         # pre-flight environment check
│   ├── slice_audio.py        # silence-based slicing, keeps source sample rate
│   ├── make_dataset.py       # preprocess WITHOUT re-slicing (preprocess.py loses 76%)
│   ├── check_dataset.py      # dataset integrity + frame alignment
│   ├── make_config.py        # filelist + config.json (40k/32k)
│   ├── make_index.py         # build faiss index (the step CLI training skips!)
│   ├── probe_quality.py      # flat / harmonics / band report
│   ├── patch_train_workers.py# DataLoader workers=0 patch (WinError 1455)
│   ├── hp5_verify.py         # vocal separation purity check
│   └── hp5_batch.sh          # batched UVR5 separation
├── indextts/                 # IndexTTS deployment
│   ├── adapter.py            # FastAPI adapter: dsh-plugin-tts <-> IndexTTS-2.5
│   ├── 启动IndexTTS.cmd      # one-click launcher (frees GPU, checks memory)
│   └── launch_hidden.vbs     # detached launcher (survives .cmd exit)
├── qq/
│   └── qq_voice.mjs          # send synthesized voice to QQ via NapCat
├── patches-index.mjs.diff    # plugin host patch: lang auto-switch + backend control
└── patches-client.js.diff    # plugin UI patch: one-click RVC <-> IndexTTS switch
```

## Why / 为什么需要这个 skill

Both RVC and IndexTTS fail in ways that look like "the model is just bad", with no error:

**RVC 20240604** never normalizes audio (`audio_norm = audio`, normalization commented out).
16-bit PCM hits `spectrogram_torch()` ~32768× too large → generator latent space is polluted →
**output is broadband noise**. The failure is masked by `if loss_mel > 75: loss_mel = 75`,
so training logs look normal. Detection: `loss_disc` is **1e8+** instead of 2–6.

**IndexTTS** has no such defect, but three things bite:

- Output pitch ≈ **reference pitch**. Pick a shrill reference and every sentence sounds shrill.
  Measured: reference 396 Hz → output 441 Hz; reference 313 Hz → output 320 Hz.
- Random sampling makes **tone contours wobble**. The 4th tone (去声) slope measured
  −7.7 / −1.8 / **−0.2** across runs — the −0.2 case is audibly "flat".
  Setting `do_sample=False` alone does **not** fix it; you must also pin the RNG seed.
- On 8 GB cards the built-in low-VRAM chunking splits every 40 characters, which mangles
  pauses. Raising the threshold to 90 helps; disabling chunking entirely produces a
  **12.7-second dead tail** on long inputs.

## Quick start / 快速开始

### RVC

```bash
<RVC>/runtime/python.exe <skill>/scripts/verify_env.py <RVC>
<RVC>/runtime/python.exe <skill>/scripts/fix_audio_norm.py <RVC>
<RVC>/runtime/python.exe <skill>/scripts/fix_i18n.py <RVC>
# then follow SKILL.md sections 1-4
```

### IndexTTS

```bash
# install (see SKILL.md section 10), then:
cp indextts/* <ENV_ROOT>/indextts/
cp -r <voices>/* <ENV_ROOT>/voices/
# double-click 启动IndexTTS.cmd
```

Point dsh-plugin-tts at provider "Index-TTS2", address `http://127.0.0.1:7880`.

### Plugin patches（可选）

Two unified diffs for `@memef1f1y/dsh-plugin-tts@0.6.7`:

- `patches-index.mjs.diff` — host side:
  - **language auto-switch**: a `zh-CN` voice cannot read kana (measured: 0.24 s of
    near-silence vs 2.6 s correct). Detects kana / hangul / cyrillic / arabic / thai in
    the text and picks the matching Edge driver voice per request.
  - **`/dsh-tts-api/backend-control`**: start / stop either backend, resolve its PID
    from the listening port, and report `lang-patch` status.
  - **`lang-patch-status` / `lang-patch-restore`**: DSH reinstalls `node_modules` on
    restart, which silently reverts manual patches. These actions detect that and
    restore from `index.mjs.langpatch.bak`.
- `patches-client.js.diff` — UI: a "快速切换后端" button that stops one backend, starts
  the other (they cannot share an 8 GB card), switches the provider, waits for
  readiness, and shows live status + patch health.

Apply from the package root (`patch -p1`, Git Bash / WSL / `patch.exe`):

```bash
cd <node_modules>/@memef1f1y/dsh-plugin-tts
patch -p1 < <skill>/rvc-voice-train/patches-index.mjs.diff
patch -p1 < <skill>/rvc-voice-train/patches-client.js.diff

# keep a known-good copy so the self-heal action can restore it after a reinstall:
cp lib/index.mjs lib/index.mjs.langpatch.bak
```

## Requirements / 环境要求

| Item | RVC route | IndexTTS route |
|---|---|---|
| Bundle | **RVC 20240604 Nvidia** (verified; defect may not exist elsewhere) | IndexTTS-2.5 |
| GPU VRAM | 32k: ≥8 GB · 40k: **≥12 GB** | **≥6 GB** (8 GB works) |
| RAM | — | **≥16 GB** (model load uses ~5 GB) |
| Python | 3.9.13 (bundled) | **3.10–3.11 only** |
| Disk | ~10 GB | ~20 GB |
| Paths | **ASCII only** | ASCII only |

## Key signals / 关键信号

| Signal | Healthy | Broken |
|---|---|---|
| RVC `loss_disc` | **2–6** | **1e8+** → audio_norm defect |
| RVC `loss_mel` | 10–30, decreasing | pinned at `75.000` (hard clamp) |
| RVC harmonic purity | within 0.05 of truth | higher → over-smoothed ("电音") |
| IndexTTS output pitch / P90 | close to reference | reference too shrill |
| IndexTTS repeatability | 3 runs → identical MD5 | seed not pinned |

> `flat` (spectral flatness) is **not** a universal metric. Measured counterexamples:
> a rejected model had **flat 0.040** (lowest) while a user-approved one had **0.21** (over
> the usual 0.15 threshold). Always compare against ground truth.

## Notes / 注意

- Voice models trained from copyrighted material stay under the original rights holder.
  This repo ships **no audio and no models** — methods only.
- Verified on Windows + RTX 4060 Laptop 8 GB (32k RVC, IndexTTS) and RTX 3060 12 GB (40k RVC).
- The two backends **cannot run simultaneously on an 8 GB card** (IndexTTS ~5 GB + RVC ~2–3 GB).

## License

MIT for scripts and docs. RVC and IndexTTS follow their own licenses.
