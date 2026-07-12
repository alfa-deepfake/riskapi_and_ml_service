FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGE=torch==2.5.1+cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY production/requirements.txt /tmp/requirements.txt
COPY production/requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" "${TORCH_PACKAGE}" \
    && pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-ml.txt

COPY production/ml_service /app/ml_service
COPY face_flashing /app/face_flashing
COPY puls_from_video /app/puls_from_video
COPY neiro_model/audio/infer.py /app/neiro_model/audio/infer.py
COPY neiro_model/audio/cnn_model.pt /app/models/audio/cnn_model.pt
COPY neiro_model/video_infer.py /app/neiro_model/video_infer.py
COPY neiro_model/clip_vit_b16_deepfake_best.pt /app/models/video/clip_vit_b16_deepfake_best.pt

RUN touch /app/neiro_model/__init__.py /app/neiro_model/audio/__init__.py

EXPOSE 8100
CMD ["sh", "-c", "uvicorn ml_service.main:app --host 0.0.0.0 --port 8100 --workers ${ML_UVICORN_WORKERS:-1}"]
