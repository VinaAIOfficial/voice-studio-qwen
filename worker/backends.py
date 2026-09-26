"""Official model adapters. No model download during requests."""
import base64
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from voice_studio.common import MODELS

class ModelPool:
    def __init__(self, engine, models_dir='/models'):
        self.engine=engine; self.root=Path(models_dir)
        self.model=None; self.key=None; self.prompt=None; self.prompt_key=None
    def get(self,key):
        allowed={'qwen_design','qwen_clone'} if self.engine=='qwen' else {'chatterbox'}
        if key not in allowed: raise ValueError('Endpoint sai loại. Qwen và Chatterbox dùng hai endpoint riêng.')
        if key==self.key and self.model is not None: return self.model,0.0
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('Worker cần GPU CUDA. Không tự chuyển CPU trên RunPod.')
        self.prompt=None; self.prompt_key=None; self.model=None; self.key=None
        gc.collect(); torch.cuda.empty_cache()
        folder=self.root/key
        manifest=json.loads((self.root/'models.json').read_text())
        if manifest.get(key)!=MODELS[key]: raise RuntimeError('Model manifest không khớp bản client. Build lại đúng phiên bản.')
        t=time.perf_counter()
        if key.startswith('qwen'):
            from qwen_tts import Qwen3TTSModel
            self.model=Qwen3TTSModel.from_pretrained(str(folder),device_map='cuda:0',
                          dtype=torch.bfloat16,attn_implementation='sdpa')
        else:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.model=ChatterboxTurboTTS.from_local(str(folder),device='cuda')
        torch.cuda.synchronize(); self.key=key
        return self.model,time.perf_counter()-t

    def generate(self,p):
        import torch
        import numpy as np
        model,load_s=self.get(p['model_key'])
        torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
        np.random.seed(p['seed'])
        import random
        random.seed(p['seed'])
        prompt_s=0.0
        if p['model_key']!='qwen_design':
            raw=base64.b64decode(p['reference_b64'],validate=True)
            key=hashlib.sha256(raw+p['reference_text'].encode()).hexdigest()
            if key!=self.prompt_key:
                self.prompt=None; self.prompt_key=None
                t=time.perf_counter()
                with tempfile.TemporaryDirectory(prefix='voice_ref_') as temp:
                    path=Path(temp)/'reference.wav'; path.write_bytes(raw)
                    if p['model_key']=='qwen_clone':
                        self.prompt=model.create_voice_clone_prompt(ref_audio=str(path),
                            ref_text=p['reference_text'],x_vector_only_mode=False)
                    else:
                        model.prepare_conditionals(str(path),exaggeration=0.0)
                torch.cuda.synchronize(); prompt_s=time.perf_counter()-t; self.prompt_key=key
        t=time.perf_counter()
        with torch.inference_mode():
            if p['model_key']=='qwen_design':
                wavs,sr=model.generate_voice_design(text=p['text'],language='English',
                             instruct=p['description'],max_new_tokens=1024)
                audio=wavs[0]
            elif p['model_key']=='qwen_clone':
                wavs,sr=model.generate_voice_clone(text=p['text'],language='English',
                             voice_clone_prompt=self.prompt,max_new_tokens=1024)
                audio=wavs[0]
            else:
                # Upstream silently returns a truncated token sequence at its cap.
                # Detect saturation before decoding, without editing upstream weights/code.
                original=model.t3.inference_turbo
                def checked_tokens(*args,**kwargs):
                    tokens=original(*args,**kwargs)
                    if tokens.shape[-1]>=kwargs.get('max_gen_len',1000):
                        raise RuntimeError('Chatterbox chạm giới hạn sinh. Chia đoạn ngắn hơn rồi thử lại.')
                    return tokens
                model.t3.inference_turbo=checked_tokens
                try: audio=model.generate(text=p['text'],audio_prompt_path=None)
                finally: model.t3.inference_turbo=original
                sr=model.sr
                if hasattr(audio,'detach'): audio=audio.detach().cpu().numpy()
        torch.cuda.synchronize(); inference_s=time.perf_counter()-t
        audio=np.asarray(audio,dtype=np.float32).reshape(-1)
        if not len(audio) or not np.isfinite(audio).all(): raise RuntimeError('Model trả về audio rỗng hoặc NaN.')
        if p['model_key'].startswith('qwen') and len(audio)/sr>=78:
            raise RuntimeError('Audio chạm giới hạn sinh. Chia văn bản thành đoạn ngắn hơn, tránh dùng bản có thể bị cắt.')
        if len(audio)/sr>90: raise RuntimeError('Audio vượt 90 giây cho một đoạn; cần kiểm tra lặp lời.')
        return audio,int(sr),{'load_seconds':round(load_s,3),'reference_seconds':round(prompt_s,3),
                             'inference_seconds':round(inference_s,3)}
