"""Fast import/signature gate; deliberately does not claim GPU inference validation."""
import inspect
import json
import os
from pathlib import Path
from voice_studio.common import MODELS
from worker.handler import handler

engine=os.environ['TTS_ENGINE']; root=Path(os.getenv('MODELS_DIR','/models'))
manifest=json.loads((root/'models.json').read_text())
for key,spec in manifest.items():
    assert spec==MODELS[key]
    assert (root/key/'config.json').is_file() or (root/key/'t3_turbo_v1.safetensors').is_file()
if engine=='qwen':
    from qwen_tts import Qwen3TTSModel
    assert 'voice_clone_prompt' in inspect.signature(Qwen3TTSModel.generate_voice_clone).parameters
    assert 'instruct' in inspect.signature(Qwen3TTSModel.generate_voice_design).parameters
else:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    assert 'audio_prompt_path' in inspect.signature(ChatterboxTurboTTS.generate).parameters
    assert hasattr(ChatterboxTurboTTS,'prepare_conditionals')
print('PASS import/signature/model-manifest checks. This is not a GPU generation test.')
