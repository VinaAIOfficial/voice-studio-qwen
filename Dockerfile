FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TTS_ENGINE=qwen MODELS_DIR=/models HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 sox libsox-fmt-all ffmpeg && rm -rf /var/lib/apt/lists/*
COPY worker/requirements-qwen.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt && python -m pip check
COPY voice_studio /app/voice_studio
COPY worker /app/worker
RUN python -m worker.download_models && python -m pip freeze > /models/installed-packages.txt
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN python -m worker.check_install
CMD ["python", "-u", "-m", "worker.handler"]
