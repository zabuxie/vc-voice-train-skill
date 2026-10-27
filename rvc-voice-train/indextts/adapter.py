# -*- coding: utf-8 -*-
"""
dsh-plugin-tts <- IndexTTS-2.5 适配器

在本地 7880 端口提供 dsh-plugin-tts 期望的三个接口：
    GET  /health              -> {"ok": true, ...}
    GET  /api/v1/voices       -> {"voices": [名称, ...]}
    POST /api/v1/tts/tasks    -> 合成，返回 wav 字节

它把请求翻译成 IndexTTS-2.5 的 Python 调用（进程内，模型常驻）。

部署位置（默认）：
    <ENV_ROOT>/index-tts/       IndexTTS 源码 + .venv
    <ENV_ROOT>/checkpoints/     模型权重（或用 index-tts/checkpoints）
    <ENV_ROOT>/voices/          参考音色（wav/mp3/flac）

环境变量（都可选）：
    INDEX_TTS_ENV_ROOT  部署根目录，默认脚本所在目录的上一级
    INDEX_TTS_PORT      监听端口，默认 7880
    INDEX_TTS_VOICE_DIR 参考音色目录，默认 <ENV_ROOT>/voices
    INDEX_TTS_CHUNK_CHARS 低显存分段阈值，默认 90（0=不分段）
    INDEX_TTS_INTERVAL_MS 段间静音毫秒，默认 120
    INDEX_TTS_DO_SAMPLE 置 1 恢复随机采样（默认确定性）
    INDEX_TTS_SEED      固定种子，默认 1234

请求体里的 prompt_audio 可以是：
    - 音色名（不带扩展名）  例如 "myvoice"
    - 绝对路径
    - 留空 -> 用默认音色（voices 目录里第一个）
"""
import io
import os
import sys
import time
import traceback
import glob
from typing import Optional

# ENV_ROOT 默认取「本脚本所在目录的上一级」；放在 indextts/ 子目录时需指到部署根。
_HERE = os.path.dirname(os.path.abspath(__file__))
ENV_ROOT = os.environ.get('INDEX_TTS_ENV_ROOT') or (
    os.path.dirname(_HERE) if os.path.basename(_HERE) == 'indextts' else _HERE
)
ROOT = os.path.join(ENV_ROOT, 'index-tts')
# 模型权重：优先 <ENV_ROOT>/checkpoints，其次 index-tts/checkpoints
_CKPT_A = os.path.join(ENV_ROOT, 'checkpoints')
_CKPT_B = os.path.join(ROOT, 'checkpoints')
CKPT = _CKPT_A if os.path.isdir(_CKPT_A) else _CKPT_B
VOICE_DIR = os.environ.get('INDEX_TTS_VOICE_DIR') or os.path.join(ENV_ROOT, 'voices')
PORT = int(os.environ.get('INDEX_TTS_PORT', '7880'))

if not os.path.isdir(ROOT):
    sys.stderr.write('[FATAL] 找不到 IndexTTS 源码目录：%s\n'
                     '        请设 INDEX_TTS_ENV_ROOT 指向部署根目录。\n' % ROOT)
    sys.exit(1)

os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 关键：Windows 控制台默认 GBK，IndexTTS 内部会 print 原文（可能含 emoji），
# 触发 UnicodeEncodeError 并让整次合成失败。
# 强制 stdout/stderr 走 UTF-8，且永不因编码失败中断。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
os.environ.setdefault('PYTHONIOENCODING', 'utf-8:replace')

os.makedirs(VOICE_DIR, exist_ok=True)

# ---------------- 依赖 ----------------
try:
    import numpy as np
    import soundfile as sf
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except Exception as e:
    print('[FATAL] 依赖缺失:', e)
    sys.exit(1)

app = FastAPI(title='IndexTTS-2.5 adapter for dsh-plugin-tts')

_tts = None
_load_error = None
_loaded_at = None


def get_tts():
    """懒加载模型（首次调用时加载，约 9 分钟）"""
    global _tts, _load_error, _loaded_at
    if _tts is not None:
        return _tts
    if _load_error:
        raise HTTPException(500, 'model load failed earlier: %s' % _load_error)
    try:
        from indextts.infer_v2_5 import IndexTTS2
        t0 = time.time()
        print('>> loading IndexTTS-2.5 ...', flush=True)
        _tts = IndexTTS2(
            cfg_path=os.path.join(CKPT, 'config.yaml'),
            model_dir=CKPT,
            use_bf16=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_qwen_emo=False,
        )
        # 8G 显存会被自动判定为 low_vram，导致 >40 字强制分段合成，
        # 造成断句/停顿不自然。
        #
        # 实测发现：
        #   - 完全关闭分段 -> 63 字文本出现 12.7 秒异常静音（尾部生成失败），不可接受
        #   - 保持分段但调大阈值 -> 更稳妥
        #
        # 策略：保留 low_vram 的分段逻辑，但把切分阈值从 40 字调到可配置值。
        # 环境变量 INDEX_TTS_CHUNK_CHARS 控制（默认 90），设为 0 则完全关闭分段。
        chunk_chars = int(os.environ.get('INDEX_TTS_CHUNK_CHARS', '90'))
        if chunk_chars <= 0:
            try:
                if getattr(_tts, 'low_vram', False):
                    _tts.low_vram = False
                    print('>> low_vram chunking DISABLED', flush=True)
            except Exception:
                pass
        else:
            # 打补丁：把 40 字阈值换成 chunk_chars
            try:
                import re as _re
                from indextts import infer_v2_5 as _mod
                _orig = _mod.IndexTTS2.split_text_by_punctuation
                if not getattr(_mod.IndexTTS2, '_dsh_patched', False):
                    def _patched(text, max_chars=40, _cc=chunk_chars):
                        return _orig(text, max_chars=_cc)
                    _mod.IndexTTS2.split_text_by_punctuation = staticmethod(_patched)
                    _mod.IndexTTS2._dsh_patched = True
                print('>> low_vram chunk size set to %d chars (was 40)' % chunk_chars, flush=True)
            except Exception as e:
                print('>> chunk patch skipped:', e, flush=True)
        _loaded_at = time.time()
        print('>> model loaded in %.1fs' % (_loaded_at - t0), flush=True)
        if torch.cuda.is_available():
            print('>> VRAM: %.2f GB' % (torch.cuda.memory_allocated(0) / 1024 ** 3), flush=True)
        return _tts
    except Exception as e:
        _load_error = str(e)[:300]
        print('>> load failed:', _load_error, flush=True)
        traceback.print_exc()
        raise HTTPException(500, 'model load failed: %s' % _load_error)


def list_voices():
    """voices 目录支持的音频 -> 名称列表"""
    out = []
    for ext in ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg'):
        for p in glob.glob(os.path.join(VOICE_DIR, ext)):
            out.append(os.path.splitext(os.path.basename(p))[0])
    return sorted(set(out))


def resolve_voice(name_or_path):
    """把 prompt_audio 解析成真实文件路径"""
    s = str(name_or_path or '').strip().strip('"')
    if s and os.path.isfile(s):
        return s
    if s:
        # 当成音色名
        for ext in ('.wav', '.mp3', '.flac', '.m4a', '.ogg'):
            p = os.path.join(VOICE_DIR, s + ext)
            if os.path.isfile(p):
                return p
    # 默认：目录里第一个
    for ext in ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg'):
        g = sorted(glob.glob(os.path.join(VOICE_DIR, ext)))
        if g:
            return g[0]
    return None


# ---------------- 接口 ----------------

@app.get('/health')
def health():
    ok = _tts is not None
    return {
        'ok': True,
        'model_loaded': ok,
        'loaded_at': _loaded_at,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'voices': list_voices(),
        'load_error': _load_error,
    }


@app.get('/api/v1/voices')
def voices():
    names = list_voices()
    return {'voices': names}


class TTSRequest(BaseModel):
    text: str = ''
    prompt_audio: str = ''
    clean_text: bool = True
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 0.65
    emo_text: Optional[str] = None
    emo_random: bool = False
    max_text_tokens_per_segment: int = 120
    # 确定性采样：同一句话每次输出完全一致，避免末尾声调忽正忽平（实测随机
    # 采样时"试"字的去声斜率在 -7.7 ~ -0.2 之间跳动）。可用 INDEX_TTS_DO_SAMPLE=1 恢复。
    do_sample: bool = False
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500


def post_process(wav_path):
    """读 wav -> 峰值归一到 0.89，削波保护"""
    d, sr = sf.read(wav_path, always_2d=False)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d.astype(np.float32)
    if np.issubdtype(sf.read(wav_path, frames=10)[0].dtype, np.integer):
        # soundfile 已转 float，这里兜底
        pass
    pk = float(np.abs(d).max())
    if pk > 0:
        d = d / pk * 0.89
    d = np.clip(d, -0.97, 0.97)
    return d, sr


@app.post('/api/v1/tts/tasks')
def tts_tasks(req: TTSRequest):
    text = (req.text or '').strip()
    if not text:
        raise HTTPException(400, 'empty text')

    # 清洗无法朗读的字符：emoji / 私有区 / 控制符
    # （不删标点，保留断句信息）
    if req.clean_text:
        cleaned_chars = []
        for ch in text:
            cp = ord(ch)
            # emoji 与符号区段
            if (0x1F000 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF) or \
               (0x2B00 <= cp <= 0x2BFF) or (0xFE00 <= cp <= 0xFE0F) or \
               (0xE000 <= cp <= 0xF8FF) or (0x1F1E6 <= cp <= 0x1F1FF) or \
               cp == 0x200D:
                cleaned_chars.append(' ')
            elif cp < 0x20 and ch not in '\n\t':
                cleaned_chars.append(' ')
            else:
                cleaned_chars.append(ch)
        text = ''.join(cleaned_chars)
        # 折叠空白
        import re as _re2
        text = _re2.sub(r'[ \t]{2,}', ' ', text).strip()
        text = _re2.sub(r'\s*\n\s*', '\n', text)
        if not text:
            raise HTTPException(400, 'text has no readable content after cleanup')

    ref = resolve_voice(req.prompt_audio)
    if not ref:
        raise HTTPException(400,
                            'no reference audio; put a wav/mp3 into %s' % VOICE_DIR)

    model = get_tts()

    # 强制确定性采样：插件默认传 do_sample=true，且 IndexTTS 内部与扩散采样
    # 都有随机源，单靠 do_sample=false 不够（实测三次输出 MD5 仍不同）。
    # 因此这里在每次合成前固定 torch/numpy/random 的种子。
    # INDEX_TTS_DO_SAMPLE=1 可完全恢复随机。
    do_sample = bool(req.do_sample)
    force_det = os.environ.get('INDEX_TTS_DO_SAMPLE', '0') != '1'
    if force_det:
        do_sample = False
        seed = int(os.environ.get('INDEX_TTS_SEED', '1234'))
        try:
            import random as _py_random
            import torch as _torch
            _py_random.seed(seed)
            np.random.seed(seed)
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
            # 注意：不强制 deterministic=True（会拖慢且可能不兼容），
            # 只固定种子即可让结果可复现。
        except Exception:
            pass

    # 语言：按文本自动判断（中文/日文/英文）
    lang = 'ZH'
    if any('\u3040' <= ch <= '\u30ff' for ch in text):
        lang = 'JA'
    elif all(ord(ch) < 128 for ch in text):
        lang = 'EN'

    out_dir = os.path.join(ENV_ROOT, 'output')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'tts_%d.wav' % int(time.time() * 1000))

    t0 = time.time()
    try:
        model.infer(
            spk_audio_prompt=ref,
            text=text,
            output_path=out,
            lang=lang,
            emo_audio_prompt=(req.emo_ref_path or None),
            emo_alpha=float(req.emo_weight or 1.0),
            emo_vector=None,
            use_emo_text=bool(req.emo_text) and req.emo_control_method == 3,
            emo_text=(req.emo_text or None),
            use_random=bool(req.emo_random),
            interval_silence=int(os.environ.get('INDEX_TTS_INTERVAL_MS', '120')),
            max_text_tokens_per_segment=int(req.max_text_tokens_per_segment or 120),
            verbose=False,
            # 生成参数（透传给 GPT 采样）
            do_sample=do_sample,
            top_p=float(req.top_p),
            top_k=int(req.top_k),
            temperature=float(req.temperature),
            length_penalty=float(req.length_penalty),
            num_beams=int(req.num_beams),
            repetition_penalty=float(req.repetition_penalty),
            max_mel_tokens=int(req.max_mel_tokens),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, 'synthesis failed: %s' % str(e)[:250])

    dt = time.time() - t0
    if not os.path.isfile(out):
        raise HTTPException(500, 'synthesis produced no file')

    d, sr = post_process(out)
    buf = io.BytesIO()
    sf.write(buf, d, sr, format='WAV', subtype='PCM_16')
    data = buf.getvalue()

    # 清理中间文件
    try:
        os.remove(out)
    except OSError:
        pass

    # 日志本身也要防编码异常（绝不能因 print 失败丢掉已合成好的音频）
    try:
        print('>> synthesized %.1fs audio in %.1fs (ref=%s, lang=%s)' % (
            len(d) / sr, dt, os.path.basename(ref), lang), flush=True)
    except Exception:
        pass

    return Response(content=data, media_type='audio/wav')


if __name__ == '__main__':
    # 自己写日志文件：VBS/wscript 启动时无法用 shell 重定向，靠这里落盘。
    class _Tee(object):
        def __init__(self, stream, path):
            self._s = stream
            try:
                self._f = open(path, 'a', encoding='utf-8', errors='replace')
            except Exception:
                self._f = None

        def write(self, data):
            try:
                self._s.write(data)
            except Exception:
                pass
            if self._f:
                try:
                    self._f.write(data)
                    self._f.flush()
                except Exception:
                    pass
            return len(data) if isinstance(data, str) else 0

        def flush(self):
            for s in (self._s, self._f):
                if s:
                    try:
                        s.flush()
                    except Exception:
                        pass

        def isatty(self):
            return False

    _log = os.path.join(ENV_ROOT, 'adapter.log')
    try:
        open(_log, 'w', encoding='utf-8').close()  # 每次启动清空
    except Exception:
        pass
    sys.stdout = _Tee(sys.stdout, _log)
    sys.stderr = _Tee(sys.stderr, _log)

    print('=' * 60)
    print('IndexTTS-2.5 adapter for dsh-plugin-tts')
    print('=' * 60)
    print('端口  :', PORT)
    print('音色  :', VOICE_DIR)
    print('现有  :', list_voices() or '(空，请放入参考音频)')
    print('模型  :', CKPT)
    print('=' * 60)
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='info')
