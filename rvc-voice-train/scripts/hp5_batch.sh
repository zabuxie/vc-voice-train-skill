#!/bin/bash
# HP5 separation in chunks. uvr-split.py writes <base>_vocals.wav / <base>_instrumental.wav
# directly into UVR_OUT, so collect from there (NOT from _v/_i subdirs).
SRC="$1"; OUT="$2"; SEG="${3:-240}"
WORK=/e/rvc/_hp5work
FF="D:/FormatFactory/ffmpeg.exe"
mkdir -p "$OUT" "$WORK"
T=0; N=0
while : ; do
  rm -rf "$WORK/out"; mkdir -p "$WORK/out"
  "$FF" -hide_banner -loglevel error -y -ss $T -i "$SRC" -t $SEG -vn -ac 2 -ar 44100 -c:a pcm_s16le "$WORK/seg.wav" 2>/dev/null
  SZ=$(stat -c%s "$WORK/seg.wav" 2>/dev/null || echo 0)
  if [ "$SZ" -lt 10000 ]; then echo "END t=$T"; break; fi
  UVR_SRC="$WORK/seg.wav" UVR_MODEL=HP5_only_main_vocal UVR_OUT="$WORK/out" \
    /e/rvc/RVC20240604Nvidia/runtime/python.exe /e/rvc/RVC20240604Nvidia/uvr-split.py > "$WORK/uvr.log" 2>&1
  V="$WORK/out/seg_vocals.wav"; I="$WORK/out/seg_instrumental.wav"
  if [ -f "$V" ]; then mv "$V" "$OUT/seg_$(printf %03d $N)_vocal.wav"; else echo "seg $N VOCAL MISSING"; fi
  if [ -f "$I" ]; then mv "$I" "$OUT/seg_$(printf %03d $N)_instr.wav"; else echo "seg $N INSTR MISSING"; fi
  N=$((N+1)); T=$((T+SEG))
  echo "seg $N t=$T"
done
echo "DONE segments=$N"
