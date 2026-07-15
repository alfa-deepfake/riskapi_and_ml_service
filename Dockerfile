FROM python:3.12-slim

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

COPY ml_service /app/ml_service
COPY deepfake_audio /app/deepfake_audio
COPY face_flashing /app/face_flashing

# The XGBoost ensemble and the Whisper ASR snapshot are runtime dependencies of
# the CPU image. Keep them in separate COPY layers: compose mounts only the
# optional WavLM checkpoint below, so it cannot hide the server-side ASR model.
COPY models/xgb /app/models/xgb
COPY models/asr/whisper-tiny.en /app/models/asr/whisper-tiny.en

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
