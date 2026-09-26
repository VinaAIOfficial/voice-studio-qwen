import base64
import io
import logging
import os
import threading
import time
from voice_studio.common import PROTOCOL, digest, validate_payload, wav_info
from worker.backends import ModelPool

LOG=logging.getLogger('voice_worker')
POOL=None
LOCK=threading.Lock()

def encode_wav(audio,sr):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    if sr!=24000:
        factor=gcd(sr,24000)
        audio=resample_poly(audio,24000//factor,sr//factor)
    peak=float(np.max(np.abs(audio)))
    if peak<0.00001: raise RuntimeError('Model trả về audio gần như im lặng.')
    # Only protect against clipping. No aggressive denoising / pitch shift.
    if peak>0.99: audio=audio*(0.99/peak)
    buf=io.BytesIO(); sf.write(buf,audio,24000,format='WAV',subtype='PCM_16')
    return buf.getvalue()

def handler(job):
    global POOL
    metrics={}; started=time.perf_counter()
    p=job.get('input') if isinstance(job,dict) else None
    try:
        validate_payload(p)
        engine=os.getenv('TTS_ENGINE','qwen')
        if engine not in ('qwen','chatterbox'): raise ValueError('TTS_ENGINE cần qwen hoặc chatterbox.')
        with LOCK:
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('GPU CUDA không có trong worker.')
            torch.cuda.reset_peak_memory_stats()
            if POOL is None: POOL=ModelPool(engine,os.getenv('MODELS_DIR','/models'))
            audio,sr,metrics=POOL.generate(p)
            raw=encode_wav(audio,sr); info,_=wav_info(raw)
            metrics.update(audio_seconds=round(info['seconds'],3),
                rtf=round(metrics['inference_seconds']/info['seconds'],4),
                handler_seconds=round(time.perf_counter()-started,3),
                peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
                peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,3),
                gpu=torch.cuda.get_device_name(0),worker_id=os.getenv('RUNPOD_POD_ID',''))
            result={k:p[k] for k in ('chunk_id','seed','model_key','model_revision')}
            result.update(protocol=PROTOCOL,demo=False,audio_b64=base64.b64encode(raw).decode(),
                          sha256=digest(raw),metrics=metrics,sample_rate=24000)
            LOG.info('%s | %s | %.2fs audio | %.2fs inference',p['chunk_id'],p['model_key'],info['seconds'],metrics['inference_seconds'])
            return result
    except Exception as e:
        LOG.exception('TTS job failed')
        # Do not return full requests, reference audio, credentials or environment.
        return {'protocol':PROTOCOL,'error':f'{type(e).__name__}: {str(e)[:1800]}','metrics':metrics}

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    import runpod
    runpod.serverless.start({'handler':handler,'concurrency_modifier':lambda current:1})
