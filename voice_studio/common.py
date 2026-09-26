"""Shared wire format. Standard library only; safe to import without a GPU."""
import array
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

PROTOCOL = 'voice-studio/0.1'
MAX_CHARS = 320
MODELS = {
    'qwen_design': {'repo': 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
                    'revision': '5ecdb67327fd37bb2e042aab12ff7391903235d3'},
    'qwen_clone': {'repo': 'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                   'revision': '5d83992436eae1d760afd27aff78a71d676296fc'},
    'chatterbox': {'repo': 'ResembleAI/chatterbox-turbo',
                  'revision': '749d1c1a46eb10492095d68fbcf55691ccf137cd'},
}
DESIGN_TEXT = ('On Sunday mornings, I still make enough coffee for two. Some habits stay with us. '
               'But this year, I decided to open my door a little wider, call an old friend, '
               'and see what might happen next.')
DESIGN_PROMPTS = {
    'Nam Mỹ lớn tuổi': ('A fictional American man in his late sixties. Warm, mature, grounded voice; '
                        'natural General American accent, clear words, relaxed conversational pacing. '
                        'He is quietly reflecting on life with gentle warmth and occasional light humor. '
                        'Subtle age in the voice, steady delivery, not a dramatic announcer.'),
    'Nữ Mỹ lớn tuổi': ('A fictional American woman in her late sixties. Warm, mature, reassuring voice; '
                        'natural General American accent, clear words, relaxed conversational pacing. '
                        'She is sharing a personal memory with gentle warmth and quiet confidence. '
                        'Subtle age in the voice, steady delivery, not a dramatic announcer.'),
}

def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(value).hexdigest()

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_json(path, data):
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

class FolderLock:
    """Kernel-owned lock: crash releases it, stale lock files do not block restart."""
    def __init__(self, folder):
        self.path = Path(folder) / '.run.lock'
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, 'a+b')
        self.f.seek(0)
        if not self.f.read(1): self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise RuntimeError('Dự án/công cụ đang được một tiến trình khác sử dụng.') from None
        return self
    def __exit__(self, *args):
        if os.name == 'nt':
            import msvcrt
            self.f.seek(0); msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()

def normalize_text(text):
    if not isinstance(text, str): raise ValueError('Văn bản phải là chuỗi.')
    return re.sub(r'\s+', ' ', text).strip()

def split_text(text, limit=280):
    if type(limit) is not int or not 80 <= limit <= MAX_CHARS:
        raise ValueError('Giới hạn đoạn phải từ 80 đến 320 ký tự.')
    if not isinstance(text, str) or not text.strip() or len(text) > 30000:
        raise ValueError('Cần 1–30.000 ký tự văn bản.')
    chunks = []
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        remaining = normalize_text(paragraph)
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining); break
            window = remaining[:limit+1]
            candidates = []
            for m in re.finditer(r'[.!?][\"\u201d\u2019\']?\s+', window):
                left = window[:m.start()+1]
                if re.search(r'\b(?:Mr|Mrs|Ms|Dr|St|Prof|Sr|Jr|vs|e\.g|i\.e)\.$', left, re.I):
                    continue
                candidates.append(m.start()+len(m.group().rstrip()))
            cut = max((c for c in candidates if c >= min(60, limit//2)), default=0)
            if not cut:
                cut = max((m.start()+1 for m in re.finditer(r'[,;:]\s+', window)
                           if m.start() >= limit//2), default=0)
            if not cut: cut = window.rfind(' ', 1, limit+1)
            if cut <= 0:
                raise ValueError('Có một từ/URL dài hơn giới hạn đoạn. Hãy sửa văn bản.')
            chunks.append(remaining[:cut].strip()); remaining = remaining[cut:].strip()
    if len(chunks) > 200: raise ValueError('Tối đa 200 đoạn trong một dự án thử nghiệm.')
    if normalize_text(' '.join(chunks)) != normalize_text(text):
        raise AssertionError('Text split changed content.')
    return chunks

def read_input(path):
    path = Path(path)
    if path.stat().st_size > 2_000_000: raise ValueError('Input tối đa 2 MB.')
    if path.suffix.lower() == '.txt':
        return {'title': path.stem, 'text': path.read_text(encoding='utf-8-sig')}
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            matches = [x for x in z.infolist() if x.filename == 'tts_input.json']
            if len(matches) != 1 or matches[0].file_size > 1_000_000:
                raise ValueError('ZIP cần đúng một tts_input.json ở thư mục gốc, tối đa 1 MB.')
            obj = json.loads(z.read(matches[0]).decode('utf-8-sig'))
    elif path.suffix.lower() == '.json': obj = read_json(path)
    else: raise ValueError('Chọn TXT, JSON hoặc ZIP của Voice Studio.')
    if not isinstance(obj, dict) or obj.get('schema') != 'tts-input/0.1':
        raise ValueError('Schema cần là tts-input/0.1. ZIP Full Pipeline cũ chưa được nhập trực tiếp.')
    if not isinstance(obj.get('title'), str) or not obj['title'].strip(): raise ValueError('Thiếu title.')
    split_text(obj.get('text'))
    if obj.get('language', 'English') != 'English': raise ValueError('V0.1 hỗ trợ tiếng Anh.')
    return obj

def wav_info(raw, reference=False):
    if not isinstance(raw, bytes) or len(raw) > (6_000_000 if reference else 12_000_000):
        raise ValueError('File WAV quá lớn.')
    try:
        with wave.open(io.BytesIO(raw), 'rb') as w:
            sr, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            if w.getcomptype() != 'NONE' or width != 2 or channels not in (1, 2):
                raise ValueError('Cần WAV PCM 16-bit, mono hoặc stereo.')
            if sr not in (16000, 22050, 24000, 44100, 48000): raise ValueError('Sample rate WAV không hỗ trợ.')
            seconds = n/sr
            if not (6 <= seconds <= 30 if reference else 0.05 <= seconds <= 90):
                raise ValueError('Mẫu giọng cần 6–30 giây.' if reference else 'Audio trả về cần 0.05–90 giây.')
            frames = w.readframes(n)
            if len(frames) != n*channels*width: raise ValueError('WAV bị thiếu dữ liệu.')
    except (wave.Error, EOFError) as e:
        raise ValueError('Không đọc được WAV PCM 16-bit. Xuất lại file âm thanh.') from e
    return {'sample_rate': sr, 'channels': channels, 'sample_width': width,
            'frames': n, 'seconds': seconds}, frames

def mono_wav(raw, reference=True):
    info, frames = wav_info(raw, reference)
    if info['channels'] == 2:
        a = array.array('h'); a.frombytes(frames)
        if sys.byteorder != 'little': a.byteswap()
        a = array.array('h', (int((a[i]+a[i+1])/2) for i in range(0, len(a), 2)))
        if sys.byteorder != 'little': a.byteswap()
        frames = a.tobytes()
    out = io.BytesIO()
    with wave.open(out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(info['sample_rate']); w.writeframes(frames)
    return out.getvalue()

def validate_payload(p):
    if not isinstance(p, dict) or p.get('protocol') != PROTOCOL: raise ValueError('Sai protocol client/worker.')
    if p.get('model_key') not in MODELS: raise ValueError('Model không hỗ trợ.')
    if p.get('model_revision') != MODELS[p['model_key']]['revision']: raise ValueError('Sai revision model.')
    if not isinstance(p.get('chunk_id'), str) or not re.fullmatch(r'C\d{3}', p['chunk_id']): raise ValueError('Sai chunk_id.')
    if type(p.get('seed')) is not int or not 0 <= p['seed'] < 2**32: raise ValueError('Seed không hợp lệ.')
    if not isinstance(p.get('text'), str) or not 1 <= len(p['text'].strip()) <= MAX_CHARS:
        raise ValueError('Mỗi đoạn cần 1–320 ký tự.')
    if p.get('language') != 'English': raise ValueError('V0.1 chỉ nhận English.')
    if p['model_key'] == 'qwen_design':
        if not isinstance(p.get('description'), str) or not 10 <= len(p['description']) <= 1500:
            raise ValueError('Mô tả giọng cần 10–1500 ký tự.')
    else:
        if not isinstance(p.get('reference_b64'), str) or len(p['reference_b64']) > 6_000_000:
            raise ValueError('Thiếu mẫu giọng hoặc mẫu quá lớn.')
        raw = base64.b64decode(p['reference_b64'], validate=True)
        wav_info(raw, reference=True)
        if not isinstance(p.get('reference_text'), str) or not 1 <= len(p['reference_text'].strip()) <= 2000:
            raise ValueError('Cần bản chép lời chính xác của mẫu giọng.')
    return p

def make_payload(identity, chunk, seed, reference=None):
    p = {k: identity[k] for k in ('model_key', 'model_revision', 'language')}
    p.update(protocol=PROTOCOL, chunk_id=chunk['id'], text=chunk['text'], seed=seed)
    if p['model_key'] == 'qwen_design': p['description'] = identity['description']
    else:
        p['reference_b64'] = base64.b64encode(reference).decode('ascii')
        p['reference_text'] = identity['reference_text']
    return validate_payload(p)
