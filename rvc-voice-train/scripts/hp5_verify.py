# -*- coding: utf-8 -*-
import sys, os, numpy as np
sys.path.insert(0, r'E:\rvc\RVC20240604Nvidia')
from infer.lib.audio import load_audio
def stat(path, label):
    d = load_audio(path, 16000)
    if len(d)==0: print('%-26s EMPTY' % label); return
    win=800; rms=np.array([np.sqrt((d[i:i+win]**2).mean()) for i in range(0,len(d)-win,win)])
    thr = rms.max()*0.02
    q=rms[rms<thr]; l=rms[rms>=thr]
    if len(q)==0: print('%-26s no quiet' % label); return
    qdb=20*np.log10(q.mean()+1e-9); ldb=20*np.log10(l.mean()+1e-9)
    sp=np.abs(np.fft.rfft(d*np.hanning(len(d))))**2
    fr=np.fft.rfftfreq(len(d),1/16000)
    band=sp[(fr>300)&(fr<3400)]
    flat=float(np.exp(np.mean(np.log(band+1e-12)))/(np.mean(band)+1e-12))
    print('%-26s quiet=%6.1f dB gap=%5.1f dB flat=%.4f rms=%.4f' % (label,qdb,ldb-qdb,flat,float(np.sqrt((d**2).mean()))))
# 同源对比: 同一段 300-540s
stat(r'E:\rvc_train\seg1.wav', 'HP2 人声轨(原始)')
stat(r'E:\rvc_train\hp5_1seg\seg1_vocals.wav', 'HP5 人声轨')
stat(r'E:\rvc_train\hp5_1seg\seg1_instrumental.wav', 'HP5 伴奏轨')
stat(r'E:\rvc_train\test_src.mp3', 'Edge 干净原声(参照)')
