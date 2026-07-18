from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import RppgAnalyzeRequest, RppgEvidence, ServiceAnalyzeResponse
from ml_service.config import Settings
from ml_service.core.checks import score_rppg
from ml_service.services.common import read_upload, safe_suffix, service_response, unavailable_check


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
        suffix = safe_suffix(file.filename, ".webm")
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await read_upload(file))
            tmp.flush()
            try:
                result = await run_in_threadpool(_run_rppg_runtime, Path(tmp.name))
            except Exception as exc:
                # _run_rppg_runtime wraps known failures in RuntimeError; anything
                # else is still a degraded check, never an HTTP 500.
                reason = str(exc) if isinstance(exc, RuntimeError) else f"rPPG inference failed: {type(exc).__name__}"
                evidence = RppgEvidence(face_present=face_present, face_confidence=face_confidence)
                check = unavailable_check("rppg", 0.18, reason)
                return service_response(self.name, evidence, check, message=reason)

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


_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _rppg_model():
    # ~65s to build (ONNX session + weights bundled in the open-rppg wheel):
    # cache for the process lifetime and warm off the request path at startup.
    import rppg

    return rppg.Model()


def warm_rppg_model() -> None:
    """Preload the rPPG model in the background; never raises."""
    try:
        _rppg_model()
    except Exception:
        pass


def _run_rppg_runtime(video_path: Path) -> dict:
    try:
        model = _rppg_model()
    except ImportError as exc:
        raise RuntimeError("rPPG runtime dependency is missing: pip install open-rppg") from exc
    except Exception as exc:
        raise RuntimeError(f"rPPG model load failed: {type(exc).__name__}") from exc

    try:
        # The model is a stateful stream processor — one video at a time.
        with _MODEL_LOCK:
            result = model.process_video(str(video_path)) or {}
    except Exception as exc:
        raise RuntimeError(f"rPPG runtime failed: {type(exc).__name__}") from exc

    bpm = _to_float(result.get("hr"))
    if bpm is not None and not (20.0 <= bpm <= 220.0):
        bpm = None
    sqi = _to_float(result.get("SQI"))
    if sqi is not None:
        sqi = max(0.0, min(1.0, sqi))
    latency = _to_float(result.get("latency"))
    if latency is not None and latency < 0:
        # schema requires latency >= 0; a junk model value must not 500 the check
        latency = None
    # Face presence stays with the caller-provided evidence: open-rppg's frame
    # statistics count forward-filled frames and cannot be trusted for it.
    return {
        "bpm": bpm,
        "signal_quality": sqi,
        "latency": latency,
        "hrv": {key: _to_float(value) for key, value in (result.get("hrv") or {}).items()},
        "samples": [],
        "sample_rate_hz": None,
        "detector": "open-rppg-facephys",
        "face_present": None,
        "face_confidence": None,
    }


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
