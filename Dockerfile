FROM python:3.12-slim

ARG ASR_MODEL_REPOSITORY=Systran/faster-whisper-medium
ARG ASR_MODEL_REVISION=08e178d48790749d25932bbc082711ddcfdfbc4f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt /tmp/
# CPU torch wheels for the WavLM audio anti-spoof model; the default PyPI
# build would drag the full CUDA stack into the image.
RUN pip install --no-cache-dir "torch>=2.1,<3" "torchaudio>=2.1,<3" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-ml.txt

# Fetch a pinned CTranslate2 model while building. This layer is reused while
# the dependency layer and model revision stay unchanged; requests never need
# external network access after the image has been built.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${ASR_MODEL_REPOSITORY}', revision='${ASR_MODEL_REVISION}', local_dir='/app/models/asr/faster-whisper-medium')" \
    && rm -rf /app/models/asr/faster-whisper-medium/.cache

COPY ml_service /app/ml_service
COPY deepfake_audio /app/deepfake_audio
COPY face_flashing /app/face_flashing
COPY train /app/train

# The training cropper uses InsightFace buffalo_l. Fetch the same detector and
# landmark model at build time so classifier requests remain offline and use
# exactly the alignment that produced the XGBoost training data.
RUN python -c "from train.face_crop import _app; _app()"

# The XGBoost ensemble is a runtime dependency of the CPU image. The pinned
# Faster-Whisper model above is already stored under /app/models/asr/.
COPY models/xgb /app/models/xgb

# Heavy classifier models (the WavLM checkpoint and neiro_model checkpoints)
# remain external to the image. WavLM is mounted by compose; full GPU images
# can additionally COPY neiro_model checkpoints into /app/models before CMD.

EXPOSE 8100
# Sessions/challenges live in process memory (ChallengeStore), so the service
# must run as exactly one process: workers are pinned to 1 (the CLI flag also
# beats a stray WEB_CONCURRENCY env), and it must not be scaled to multiple
# replicas either — a session created in one process 404s in any other, and the
# consume-after-scoring replay protection is per-process too. Externalize the
# store before scaling; note each extra worker would also duplicate the loaded
# model caches in RAM/VRAM on the full GPU image.
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
