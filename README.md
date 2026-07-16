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

- Video classifier: primary path is the v15 two-modality ensemble in
  `models/v15/` (training repo's v15 release). Per sampled frame, a 512×512
  InsightFace-aligned crop is scored by 6 XGBoost v13 trees on 73
  forensic/quality features and by a Noise-CNN (5 ConvNeXt-Tiny folds on a
  256px residual map, temperature-scaled and logistic-calibrated); the scores
  are blended `p = 0.4·trees + 0.6·cnn` and median-smoothed across frames.
  Thresholds (`t_bin` for fake/real, `t_lo`/`t_hi` grey band) ship with the
  bundle in `v15_blend_config.json` and are not env-tunable. A 5-class
  condition gate classifies the input: `restored` (GFPGAN-like AI processing)
  is rejected outright; `vidcall*` is annotated. The asymmetric low-info gate
  withholds FAKE verdicts (status `unknown`) when the source face is <180px
  or the input is wholly upscaled — such input fabricates the generator
  HF-loss signature. Models dir is `ML_VIDEO_V15_DIR` (default `models/v15`).
  When the bundle is absent, the adapter falls back to
  `neiro_model/video_infer.py` CLIP checkpoints.
- Audio anti-spoof: WavLM classifier (vendored `deepfake_audio/` inference code,
  4-generator checkpoint at `ML_AUDIO_MODEL_PATH`, default
  `models/audio/wavlm_all4_best.pt`). The checkpoint is git-ignored (380MB) and
  reaches the container through the compose `./models/audio` volume; the encoder is
  built offline from the vendored `wavlm_config`, no HuggingFace download.
- Audio phrase ASR: `faster-whisper` runs the locally stored CTranslate2
  `Systran/faster-whisper-medium` model. It uses `int8` CPU inference by
  default; the GPU Compose override uses `float16`.

### Runtime model directory

Model weights are not part of the Docker image. Before starting Compose, put
them in these host paths (all are mounted read-only):

- `models/v15/`: committed with the repo (v13 trees, condition gate, blend
  config, CNN calibrator) — except `models/v15/cnn/noise_cnn_holdout_*.pt`
  (5×111MB ConvNeXt fold weights, git-ignored): copy them from the v15
  release bundle (`cnn/artifacts_v15b/noise_cnn_global/`) via scp;
- `models/audio/wavlm_all4_best.pt`: WavLM anti-spoof checkpoint;
- `models/asr/faster-whisper-medium/`: local CTranslate2 Faster-Whisper model
  (`model.bin`, `config.json`, `tokenizer.json`, `vocabulary.txt`);
- `models/insightface/models/buffalo_l/`: the InsightFace `buffalo_l` model
  pack required by `train/face_crop.py` for the aligned 512×512 face crop.

The rPPG weights are packaged by the `open-rppg` dependency. The optional CLIP
fallback, if used instead of the v15 ensemble, also remains external at
`models/video/clip_vit_b16_deepfake_best.pt`.

From the repository root, download the ASR and alignment models into those
directories with:

```bash
bash scripts/download_runtime_models.sh
```

Sources: [Systran Faster-Whisper medium](https://huggingface.co/Systran/faster-whisper-medium)
and [InsightFace buffalo_l v0.7](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip).
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
