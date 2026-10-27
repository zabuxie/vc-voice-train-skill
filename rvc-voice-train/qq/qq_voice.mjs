#!/usr/bin/env node
/**
 * QQ 语音桥接：文本 -> IndexTTS 合成 -> mp3 -> NapCat -> QQ 语音消息
 *
 * 用法：
 *   node qq_voice.mjs send <QQ号> <文本>
 *   node qq_voice.mjs send-group <群号> <文本>
 *
 * 环境变量（都可选）：
 *   QQ_INDEX_URL   IndexTTS 适配器地址，默认 http://127.0.0.1:7880
 *   QQ_NAPCAT_WS   NapCat WebSocket，默认 ws://127.0.0.1:3001
 *   QQ_FFMPEG      ffmpeg 路径；默认依次找 PATH / RVC 目录 / 环境变量
 *   QQ_VOICE_NAME  参考音色名，默认 ref
 *
 * 前置条件：
 *   1. IndexTTS 适配器已在跑（indextts/启动IndexTTS.cmd）
 *   2. NapCat 已登录并开着 WebSocket 服务
 *   3. ffmpeg 可用
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';

const INDEX_URL = (process.env.QQ_INDEX_URL || 'http://127.0.0.1:7880').replace(/\/+$/, '') + '/api/v1/tts/tasks';
const NAPCAT_WS = process.env.QQ_NAPCAT_WS || 'ws://127.0.0.1:3001';
const VOICE = process.env.QQ_VOICE_NAME || 'ref';
const TMP = path.join(os.tmpdir(), 'qq-voice');

function findFfmpeg() {
  if (process.env.QQ_FFMPEG && fs.existsSync(process.env.QQ_FFMPEG)) return process.env.QQ_FFMPEG;
  const cands = [
    'ffmpeg',
    'E:/rvc/RVC20240604Nvidia/ffmpeg.exe',
    'C:/ffmpeg/bin/ffmpeg.exe',
  ];
  for (const c of cands) {
    try {
      execFileSync(c, ['-version'], { stdio: 'ignore', timeout: 5000 });
      return c;
    } catch (e) { }
  }
  return null;
}

const FFMPEG = findFfmpeg();

fs.mkdirSync(TMP, { recursive: true });

function log(...a) { console.log('[qq-voice]', ...a); }

// ---------- 1. 文本 -> wav ----------
async function synth(text) {
  const payload = {
    text,
    prompt_audio: VOICE,
    clean_text: true,
    emo_control_method: 0,
    emo_ref_path: null,
    emo_weight: 0.65,
    emo_text: null,
    emo_random: false,
    max_text_tokens_per_segment: 120,
    do_sample: true,
    top_p: 0.8,
    top_k: 30,
    temperature: 0.8,
    length_penalty: 0.0,
    num_beams: 3,
    repetition_penalty: 10.0,
    max_mel_tokens: 1500,
  };
  const r = await fetch(INDEX_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(600000),
  });
  if (!r.ok) throw new Error(`IndexTTS HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const buf = Buffer.from(await r.arrayBuffer());
  if (buf.length < 1000) throw new Error('IndexTTS 返回音频过小');
  const wav = path.join(TMP, `t_${Date.now()}.wav`);
  fs.writeFileSync(wav, buf);
  log('wav:', wav, (buf.length / 1024).toFixed(0) + 'KB');
  return wav;
}

// ---------- 2. wav -> mp3 ----------
// QQ 语音用 mp3 体积小（203KB wav -> 28KB mp3），且 NapCat 两种都接受。
// ffmpeg 不可用时退回原始 wav（也能发送，只是体积大）。
function toMp3(wav) {
  if (!FFMPEG) {
    log('未找到 ffmpeg，直接发送 wav（体积较大）');
    return wav;
  }
  const mp3 = wav.replace(/\.wav$/, '.mp3');
  try {
    execFileSync(FFMPEG, ['-y', '-i', wav, '-ar', '24000', '-ac', '1', '-b:a', '48k', mp3], { stdio: 'ignore' });
    log('mp3:', (fs.statSync(mp3).size / 1024).toFixed(0) + 'KB');
    return mp3;
  } catch (e) {
    log('转码失败，发送原始 wav:', e.message.slice(0, 80));
    return wav;
  }
}

// ---------- 3. 发到 QQ ----------
function sendQQ(target, file, isGroup) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(NAPCAT_WS);
    let settled = false;
    const finish = (fn, v) => { if (!settled) { settled = true; try { ws.close(); } catch (e) { } fn(v); } };

    ws.onerror = (e) => finish(reject, new Error('NapCat 连接失败'));
    ws.onopen = () => {
      const msg = [{ type: 'record', data: { file: 'file:///' + file } }];
      const req = isGroup
        ? { action: 'send_group_msg', params: { group_id: target, message: msg }, echo: 'x' }
        : { action: 'send_private_msg', params: { user_id: target, message: msg }, echo: 'x' };
      ws.send(JSON.stringify(req));
    };
    ws.onmessage = (ev) => {
      try {
        const j = JSON.parse(ev.data);
        if (j.post_type === 'meta_event') return;
        if (j.echo === 'x') {
          if (j.status === 'ok' && j.retcode === 0) finish(resolve, j.data);
          else finish(reject, new Error(`发送失败: ${j.wording || j.retcode}`));
        }
      } catch (e) { }
    };
    setTimeout(() => finish(reject, new Error('发送超时')), 20000);
  });
}

// ---------- main ----------
async function main() {
  const [cmd, target, ...rest] = process.argv.slice(2);
  const text = rest.join(' ').trim();
  if (!cmd || !target || !text) {
    console.log('用法: node qq_voice.mjs send <QQ号> <文本>');
    console.log('      node qq_voice.mjs send-group <群号> <文本>');
    process.exit(1);
  }

  const isGroup = cmd === 'send-group';
  log(`目标 ${isGroup ? '群' : '私聊'} ${target}  音色 ${VOICE}`);
  log('文本:', text.slice(0, 50) + (text.length > 50 ? '...' : ''));

  const t0 = Date.now();
  const wav = await synth(text);
  const mp3 = toMp3(wav);
  const res = await sendQQ(target, mp3, isGroup);
  log(`✅ 已发送 (${((Date.now() - t0) / 1000).toFixed(1)}s)`, JSON.stringify(res));
}

main().catch(e => { console.error('[qq-voice] 失败:', e.message); process.exit(1); });
