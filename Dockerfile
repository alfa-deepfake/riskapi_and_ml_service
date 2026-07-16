FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# InsightFace 0.7.3 is distributed as source and builds one C++ extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt /tmp/
# CPU torch wheels for the WavLM audio anti-spoof model and the v15 Noise-CNN
# (timm ConvNeXt); the default PyPI build would drag the full CUDA stack in.
RUN pip install --no-cache-dir "torch>=2.1,<3" "torchaudio>=2.1,<3" "torchvision>=0.16,<1" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-ml.txt

COPY ml_service /app/ml_service
COPY deepfake_audio /app/deepfake_audio
COPY face_flashing /app/face_flashing
COPY train /app/train

# Model weights are deliberately excluded from the image. Docker Compose mounts
# them read-only from ./models at runtime, including the InsightFace cache.

EXPOSE 8100
# Sessions/challenges live in process memory (ChallengeStore), so the service
# must run as exactly one process: workers are pinned to 1 (the CLI flag also
# beats a stray WEB_CONCURRENCY env), and it must not be scaled to multiple
# replicas either — a session created in one process 404s in any other, and the
# consume-after-scoring replay protection is per-process too. Externalize the
# store before scaling; note each extra worker would also duplicate the loaded
# model caches in RAM/VRAM on the full GPU image.
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
