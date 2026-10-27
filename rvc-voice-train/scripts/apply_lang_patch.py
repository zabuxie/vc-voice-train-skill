# -*- coding: utf-8 -*-
"""
给 @memef1f1y/dsh-plugin-tts 重新打上语言自动切换补丁

背景：
    DSH 重启/重装插件时 node_modules 会被还原，手工改动丢失。
    这个脚本幂等——已打过就跳过，可以每次重启后跑一遍。

用法：
    <RVC>/runtime/python.exe apply_lang_patch.py
"""
import io
import os
import re
import sys

P = r'C:\Users\Administrator\.dsh\profiles\desktop\node_modules\@memef1f1y\dsh-plugin-tts\lib\index.mjs'

MARKER = 'DSH fix 2026-10-26: 按文本语言自动切换'

BLOCK = '''
// DSH fix 2026-10-26: 按文本语言自动切换 RVC 驱动音色。
// 背景：TTS 语音只能读自己的语言（zh-CN 晓伊读不了假名，实测日文文本只出 0.24s 近空音频）。
// RVC 模型本身只换音色不挑语言，把驱动源换成对应语言的 Edge 语音即可。
const LANG_VOICE_RE = [
  ['ja', /[\\u3040-\\u30ff\\u31f0-\\u31ff]/],   // 平假名/片假名
  ['ko', /[\\uac00-\\ud7af\\u1100-\\u11ff]/],   // 谚文
  ['ru', /[\\u0400-\\u04ff]/],                  // 西里尔
  ['ar', /[\\u0600-\\u06ff]/],                  // 阿拉伯
  ['th', /[\\u0e00-\\u0e7f]/],                  // 泰文
];
const LANG_VOICES = {
  ja: 'ja-JP-NanamiNeural',
  ko: 'ko-KR-SunHiNeural',
  ru: 'ru-RU-SvetlanaNeural',
  ar: 'ar-SA-ZariyahNeural',
  th: 'th-TH-PremwadeeNeural',
};

function pickBaseVoice(text, fallbackVoice) {
  const t = String(text || '');
  for (const pair of LANG_VOICE_RE) {
    if (pair[1].test(t)) return LANG_VOICES[pair[0]] || fallbackVoice;
  }
  return fallbackVoice;
}
'''


def main():
    if not os.path.isfile(P):
        print('[ERROR] 插件文件不存在:', P)
        return 1

    src = io.open(P, encoding='utf-8').read()

    if MARKER in src:
        print('[SKIP] 已打过补丁')
        return 0

    # 1. 在 rvcLoadedKey 声明后插入语言表
    anchor = "let rvcLoadedKey = '';"
    if anchor not in src:
        print('[ERROR] 找不到插入锚点 rvcLoadedKey')
        return 1
    idx = src.find(anchor)
    end = src.find('\n', idx) + 1
    src = src[:end] + BLOCK + src[end:]

    # 2. 在 synthesizeRvc 里应用
    target = "    const baseVoice = cfg.baseVoice || voice;"
    if target not in src:
        print('[ERROR] 找不到 synthesizeRvc 里的 baseVoice 行')
        return 1
    repl = """    // DSH fix 2026-10-26: 按文本语言自动切换驱动音色。
    let baseVoice = cfg.baseVoice || voice;
    const autoVoice = pickBaseVoice(text, baseVoice);
    if (autoVoice !== baseVoice) {
      baseVoice = autoVoice;
    }"""
    src = src.replace(target, repl, 1)

    # 3. 纯 Edge 链路也应用
    edge_target = "registerProvider('edge-tts', {\n  synthesizeShort(text, voice, prosody, custom) {\n    return synthesize(text, voice, prosody || undefined);"
    if edge_target in src:
        src = src.replace(
            edge_target,
            "registerProvider('edge-tts', {\n  synthesizeShort(text, voice, prosody, custom) {\n    const v = pickBaseVoice(text, voice);\n    return synthesize(text, v, prosody || undefined);",
            1)

    io.open(P, 'w', encoding='utf-8', newline='').write(src)
    print('[OK] 补丁已应用')
    print('     pickBaseVoice:', 'pickBaseVoice' in src)
    print('     中文注释标记:', MARKER in src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
