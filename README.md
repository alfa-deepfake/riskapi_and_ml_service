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
cd production
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. pytest
uvicorn ml_service.main:app --reload --port 8100
```

## Run with Docker Compose

```bash
cd production
docker compose up --build
```

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

The frontend in `production/frontend` is a browser challenge console. It creates a session, renders active light flashes, samples camera luminance, asks for the gesture challenge, records an audio snippet, and submits evidence to the ML service.

Current MVP limitations:

- Gesture verification is a user-confirmed placeholder until the MediaPipe detector is moved into the frontend or a backend upload flow.
- Audio recording is captured in-browser, while ASR and synthetic speech inference are represented as evidence fields until the audio model endpoint is wired.
- rPPG is sent as a provisional signal in the demo flow; the service-side detector interface is ready for the real rPPG adapter.

## Model adapters

The current production code is adapter-based. Heavy models are optional at service boot:

- Video classifier adapter is designed to use `neiro_model/video_infer.py` checkpoints.
- Audio classifier adapter is designed to use `neiro_model/audio/infer.py`.
- Active light, rPPG, gesture, and audio challenge checks have deterministic scoring logic that works from frontend telemetry.

This makes the service deployable before GPU dependencies and final model packaging are stabilized.

## How to add a new model/check

1. Add a Pydantic evidence schema in `ml_service/api/schemas.py` if the model needs new input fields.
2. Add scoring logic in `ml_service/core/checks.py` that returns `CheckScore`.
3. Implement `SignalDetector` in `ml_service/core/detectors.py` or a new module.
4. Register it in `default_detector_registry()`.

The HTTP API and final risk aggregation do not need to change when a detector only adds a new `CheckScore`.
