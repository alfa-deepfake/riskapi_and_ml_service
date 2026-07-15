from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import ClassifierEvidence, ServiceAnalyzeResponse
from ml_service.config import settings
from ml_service.core.checks import score_classifier
from ml_service.services.common import read_upload, service_response, unavailable_check


class ClassifierService:
    name = "classifier"

    async def analyze_video(self, file: UploadFile, *, face_present: bool | None, face_confidence: float | None) -> ServiceAnalyzeResponse:
        suffix = Path(file.filename or "video.webm").suffix or ".webm"
        error_message = None
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await read_upload(file))
            tmp.flush()
            try:
                result = await run_in_threadpool(_run_video_model, Path(tmp.name))
            except Exception as exc:
                result = None
                error_message = f"video classifier inference failed: {type(exc).__name__}"

        if result is None:
            evidence = ClassifierEvidence(face_present=face_present, face_confidence=face_confidence, frame_count=None)
            reason = error_message or "video classifier model is not configured"
            check = unavailable_check("classifier", 0.25, reason)
            return service_response(self.name, evidence, check, message=check.reason)

        detected_face_present = result.get("face_present")
        detected_face_confidence = result.get("face_confidence")
        evidence = ClassifierEvidence(
            fake_probability=result.get("fake_probability"),
            confidence=result.get("confidence"),
            threshold=result.get("threshold"),
            model_name=result.get("model_name"),
            model_scores=result.get("model_scores"),
            dropped_models=result.get("dropped_models"),
            frame_count=result.get("frame_count"),
            face_present=detected_face_present if detected_face_present is not None else face_present,
            face_confidence=detected_face_confidence if detected_face_confidence is not None else face_confidence,
        )
        check = score_classifier(evidence)
        return service_response(self.name, evidence, check)


def _run_video_model(video_path: Path) -> dict | None:
    xgb_adapter = _get_xgb_adapter()
    if xgb_adapter is not None:
        return xgb_adapter.predict(video_path)
    model_path = Path(settings.video_clip_checkpoint_path)
    if not model_path.exists():
        return None
    adapter = _get_adapter()
    if adapter is None:
        return None
    return adapter.predict(video_path)


@lru_cache(maxsize=1)
def _get_xgb_adapter():
    models_dir = Path(settings.video_xgb_models_dir)
    if not (models_dir / "feature_names.txt").exists():
        return None
    try:
        from ml_service.adapters.xgb_video_adapter import XgbVideoEnsembleAdapter
    except ImportError:
        return None
    return XgbVideoEnsembleAdapter(
        models_dir=models_dir,
        threshold=settings.video_xgb_threshold,
        max_inferences=settings.video_max_inferences,
        infer_every=settings.video_infer_every,
    )


@lru_cache(maxsize=1)
def _get_adapter():
    # One adapter for the process lifetime — the checkpoint load inside the
    # adapter is cached per instance, so a per-request adapter reloads the model
    # on every call.
    try:
        from ml_service.adapters.video_adapter import VideoModelAdapter
    except ImportError:
        return None
    convnext_path = Path(settings.video_convnext_checkpoint_path) if settings.video_convnext_checkpoint_path else None
    return VideoModelAdapter(
        checkpoint_path=Path(settings.video_clip_checkpoint_path),
        convnext_checkpoint_path=convnext_path,
        device=settings.video_device,
        max_inferences=settings.video_max_inferences,
        infer_every=settings.video_infer_every,
    )
