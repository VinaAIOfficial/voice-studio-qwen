"""Build-time download; includes immutable revisions and an audit manifest."""
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from voice_studio.common import MODELS

root=Path(os.getenv('MODELS_DIR','/models')); root.mkdir(parents=True,exist_ok=True)
engine=os.getenv('TTS_ENGINE','qwen')
keys=['qwen_design','qwen_clone'] if engine=='qwen' else ['chatterbox']
for key in keys:
    spec=MODELS[key]
    snapshot_download(repo_id=spec['repo'],revision=spec['revision'],local_dir=str(root/key),
                      ignore_patterns=['*.png','*.jpg','*.mp4','*.wav'])
(root/'models.json').write_text(json.dumps({k:MODELS[k] for k in keys},indent=2))
print('Downloaded pinned models:', ', '.join(keys))
