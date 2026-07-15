# Production ML Service

Production contour for anti-deepfake verification. The service owns challenge generation, cascaded ML/liveness scoring, and optional result delivery to `risk_api`.

## What is included

- FastAPI ML service with `/health`, `/v1/sessions`, `/v1/sessions/{id}/challenge`, `/v1/sessions/{id}/evidence`, and `/v1/score`.
- Cascaded checks for frame classifier output, active light response, rPPG signal, gesture challenge, and audio challenge.
- Explainable score payload compatible with the existing `deepfake-riskapi` `/checks/{check_id}/result` JSON contract.
- Dockerfile and docker compose for `ml-service`, `risk-api`, and MongoDB.
- Unit tests for challenge generation and score aggregation.

## Run locally

```bash
cd riskapi_and_ml_service
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. pytest
uvicorn ml_service.main:app --reload --port 8100
```

## Run with Docker Compose

```bash
cd riskapi_and_ml_service
docker compose up --build
```

The compose stack builds `risk-api` from the sibling `../deepfake-riskapi` repo,
so both must sit side by side in the `alfa-deepfake/` workspace.

Services:

- Frontend: `http://localhost:8080`
- ML service: `http://localhost:8100`
- Risk API: `http://localhost:8000`
- MongoDB: `localhost:27017`

## API Flow

1. Create a verification session:

```bash
curl -X POST http://localhost:8100/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"uid":"u-1","check_id":"check-1","scenario":"video_call"}'
```

2. Render the returned challenge on the frontend and collect telemetry.
3. Submit evidence:

```bash
curl -X POST http://localhost:8100/v1/sessions/{session_id}/evidence \
  -H 'Content-Type: application/json' \
  -d @sample_evidence.json
```

4. The ML service returns a score and, if `RISK_API_URL` is set, sends status/result to risk-api.

## Frontend

The frontend in `frontend/` is a browser challenge console. It creates a session, renders active light flashes, samples camera luminance, asks for the gesture challenge, records an audio snippet, and submits evidence to the ML service.

Current MVP limitations:

- Gesture verification is a user-confirmed placeholder until the MediaPipe detector is moved into the frontend or a backend upload flow.
- Audio recording is captured in-browser, while ASR and synthetic speech inference are represented as evidence fields until the audio model endpoint is wired.
- rPPG is sent as a provisional signal in the demo flow; the service-side detector interface is ready for the real rPPG adapter.

## Model adapters

The current production code is adapter-based. Heavy models are optional at service boot:

- Video classifier: primary path is the XGBoost forensic ensemble in `models/xgb/`
  (6 models from the training repo's `infer.py`: one in-distribution + five
  leave-one-generator-out). Per model, frame scores are averaged; the per-model
  scores are then averaged with a lone-dissenter rule — if exactly one model
  votes on the opposite side of the threshold from all the others, it is
  ignored. Fake/not-fake threshold is `ML_VIDEO_XGB_THRESHOLD` (default 0.45);
  models dir is `ML_VIDEO_XGB_MODELS_DIR` (default `models/xgb`). When the
  ensemble is absent, the adapter falls back to `neiro_model/video_infer.py`
  CLIP checkpoints.
- Audio anti-spoof: WavLM classifier (vendored `deepfake_audio/` inference code,
  4-generator checkpoint at `ML_AUDIO_MODEL_PATH`, default
  `models/audio/wavlm_all4_best.pt`). The checkpoint is git-ignored (380MB) and
  reaches the container through the compose `./models/audio` volume; the encoder is
  built offline from the vendored `wavlm_config`, no HuggingFace download.
- Audio phrase ASR: `faster-whisper` runs the locally stored CTranslate2
  `Systran/faster-whisper-medium` model. It uses `int8` CPU inference by
  default; the GPU Compose override uses `float16`. A pinned model revision is
  downloaded during `docker build` and stored in the image, so serving does
  not require Hugging Face access. Compose intentionally mounts only
  `models/audio/`, so the external WavLM checkpoint cannot mask the ASR model.
- rPPG: the `open-rppg` package (FacePhys model) processes the uploaded pulse
  clip; the model is warmed in a background thread at startup because it takes
  ~1 minute to build. Detector name: `open-rppg-facephys`.
- Active light, rPPG, gesture, and audio challenge checks have deterministic scoring logic that works from frontend telemetry.

This makes the service deployable before GPU dependencies and final model packaging are stabilized.

## How to add a new model/check

1. Add a Pydantic evidence schema in `ml_service/api/schemas.py` if the model needs new input fields.
2. Add scoring logic in `ml_service/core/checks.py` that returns `CheckScore`.
3. Implement `SignalDetector` in `ml_service/core/detectors.py` or a new module.
4. Register it in `default_detector_registry()`.

The HTTP API and final risk aggregation do not need to change when a detector only adds a new `CheckScore`.
