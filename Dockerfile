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

# Heavy classifier models (torch + neiro_model checkpoints) are optional at boot:
# the audio/video adapters degrade to "unavailable" when the checkpoints are absent
# (see ml_service/services/*_service.py). This workspace ships the service without
# them. To build the full GPU model image, install torch and COPY the neiro_model/
# checkpoints into /app/models before the CMD.

EXPOSE 8100
CMD ["sh", "-c", "uvicorn ml_service.main:app --host 0.0.0.0 --port 8100 --workers ${ML_UVICORN_WORKERS:-1}"]
