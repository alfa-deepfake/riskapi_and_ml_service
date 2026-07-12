from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import ClassifierEvidence, ServiceAnalyzeResponse
from ml_service.config import settings
from ml_service.core.checks import score_classifier
from ml_service.services.common import service_response, unavailable_check


class ClassifierService:
    name = "classifier"

    async def analyze_video(self, file: UploadFile, *, face_present: bool | None, face_confidence: float | None) -> ServiceAnalyzeResponse:
        suffix = Path(file.filename or "video.webm").suffix or ".webm"
        error_message = None
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await file.read())
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
            model_name=result.get("model_name"),
            frame_count=result.get("frame_count"),
            face_present=detected_face_present if detected_face_present is not None else face_present,
            face_confidence=detected_face_confidence if detected_face_confidence is not None else face_confidence,
        )
        check = score_classifier(evidence)
        return service_response(self.name, evidence, check)


def _run_video_model(video_path: Path) -> dict | None:
    model_path = Path(settings.video_clip_checkpoint_path)
    if not model_path.exists():
        return None
    convnext_path = Path(settings.video_convnext_checkpoint_path) if settings.video_convnext_checkpoint_path else None
    try:
        from ml_service.adapters.video_adapter import VideoModelAdapter
    except ImportError:
        return None
    return VideoModelAdapter(
        checkpoint_path=model_path,
        convnext_checkpoint_path=convnext_path,
        device=settings.video_device,
        max_inferences=settings.video_max_inferences,
        infer_every=settings.video_infer_every,
    ).predict(video_path)
