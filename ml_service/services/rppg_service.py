from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import RppgAnalyzeRequest, RppgEvidence, ServiceAnalyzeResponse
from ml_service.config import Settings
from ml_service.core.checks import score_rppg
from ml_service.services.common import read_upload, service_response, unavailable_check


class RppgService:
    name = "rppg"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze_samples(self, payload: RppgAnalyzeRequest) -> ServiceAnalyzeResponse:
        evidence = RppgEvidence(
            samples=payload.samples,
            sample_rate_hz=payload.sample_rate_hz,
            window_seconds=payload.window_seconds,
            detector="browser-luma-samples",
            face_present=payload.face_present,
            face_confidence=payload.face_confidence,
        )
        check = score_rppg(evidence, self._settings)
        return service_response(self.name, evidence, check)

    async def analyze_video(self, file: UploadFile, *, face_present: bool | None, face_confidence: float | None) -> ServiceAnalyzeResponse:
        suffix = Path(file.filename or "rppg.webm").suffix or ".webm"
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await read_upload(file))
            tmp.flush()
            try:
                result = await run_in_threadpool(_run_rppg_runtime, Path(tmp.name))
            except RuntimeError as exc:
                evidence = RppgEvidence(face_present=face_present, face_confidence=face_confidence)
                check = unavailable_check("rppg", 0.18, str(exc))
                return service_response(self.name, evidence, check, message=str(exc))

        evidence = RppgEvidence(
            bpm=result.get("bpm"),
            signal_quality=result.get("signal_quality"),
            latency=result.get("latency"),
            hrv=result.get("hrv", {}),
            samples=result.get("samples", []),
            sample_rate_hz=result.get("sample_rate_hz"),
            detector=result.get("detector"),
            face_present=result.get("face_present") if result.get("face_present") is not None else face_present,
            face_confidence=result.get("face_confidence") if result.get("face_confidence") is not None else face_confidence,
        )
        check = score_rppg(evidence, self._settings)
        return service_response(self.name, evidence, check)


def _run_rppg_runtime(video_path: Path) -> dict:
    try:
        from puls_from_video.for_integration_puls import process_video_file
    except ImportError as exc:
        raise RuntimeError("rPPG integration module is unavailable") from exc

    try:
        result = process_video_file(video_path)
    except ImportError as exc:
        raise RuntimeError("rPPG runtime dependency is missing: pip install rppg") from exc
    except Exception as exc:
        raise RuntimeError(f"rPPG runtime failed: {type(exc).__name__}") from exc

    bpm = _to_float(result.get("hr_bpm"))
    if bpm is not None and not (20.0 <= bpm <= 220.0):
        bpm = None
    sqi = _to_float(result.get("signal_quality"))
    if sqi is not None:
        sqi = max(0.0, min(1.0, sqi))
    return {
        "bpm": bpm,
        "signal_quality": sqi,
        "latency": _to_float(result.get("latency")),
        "hrv": {key: _to_float(value) for key, value in result.get("hrv", {}).items()},
        "samples": [],
        "sample_rate_hz": None,
        "detector": result.get("method") or "rppg-toolbox-pos",
        "face_present": None,
        "face_confidence": None,
    }


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
