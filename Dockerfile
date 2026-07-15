FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-ml.txt

COPY ml_service /app/ml_service

# The XGBoost video-classifier ensemble (models/xgb, ~20MB CPU models) ships in
# the image. Heavy classifier models (torch + neiro_model checkpoints) remain
# optional at boot: the audio/CLIP adapters degrade to "unavailable" when their
# checkpoints are absent (see ml_service/services/*_service.py). To build the
# full GPU model image, install torch and COPY the neiro_model/ checkpoints
# into /app/models before the CMD.
COPY models /app/models

EXPOSE 8100
# Sessions/challenges live in process memory (ChallengeStore), so the service
# must run as exactly one process: workers are pinned to 1 (the CLI flag also
# beats a stray WEB_CONCURRENCY env), and it must not be scaled to multiple
# replicas either — a session created in one process 404s in any other, and the
# consume-after-scoring replay protection is per-process too. Externalize the
# store before scaling; note each extra worker would also duplicate the loaded
# model caches in RAM/VRAM on the full GPU image.
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
