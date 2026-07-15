from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import AudioEvidence, ServiceAnalyzeResponse
from ml_service.config import settings
from ml_service.core.checks import score_audio
from ml_service.services.common import read_upload, service_response, unavailable_check


class AudioService:
    name = "audio"

    async def analyze_audio(
        self,
        file: UploadFile,
        *,
        phrase_expected: str | None,
        phrase_transcribed: str | None,
    ) -> ServiceAnalyzeResponse:
        # phrase_transcribed from the client is accepted for API compatibility
        # but never trusted: the transcript is produced by server-side ASR.
        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await read_upload(file))
            tmp.flush()
            duration = await run_in_threadpool(_probe_duration, Path(tmp.name))
            result, error_message = await run_in_threadpool(_run_audio_model, Path(tmp.name))
            transcript = await run_in_threadpool(_run_asr, Path(tmp.name))

        ai_probability = result.get("ai_probability") if result else None
        evidence = AudioEvidence(
            phrase_expected=phrase_expected,
            phrase_transcribed=transcript,
            ai_probability=ai_probability,
            duration_seconds=duration,
            detector=(result or {}).get("detector") if ai_probability is not None else "unavailable",
        )
        if ai_probability is None:
            check = unavailable_check("audio", 0.20, error_message)
            return service_response(self.name, evidence, check, message=error_message)
        check = score_audio(evidence, challenge=None)
        return service_response(self.name, evidence, check)


def _run_audio_model(audio_path: Path) -> tuple[dict | None, str]:
    model_path = Path(settings.audio_model_path)
    if not model_path.exists():
        return None, "audio anti-spoof model is not configured"
    try:
        from ml_service.adapters.audio_adapter import AudioModelAdapter
        result = AudioModelAdapter(model_path=model_path).predict(audio_path)
    except Exception as exc:
        return None, f"audio anti-spoof inference failed: {type(exc).__name__}"
    return result, ""


def _run_asr(audio_path: Path) -> str | None:
    """Server-side transcript; None when ASR cannot run (scored as unverified)."""
    model_path = Path(settings.asr_model_path)
    if not model_path.exists():
        return None
    try:
        from ml_service.adapters.asr_adapter import WhisperAsrAdapter
        return WhisperAsrAdapter(
            model_path=model_path,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
            cpu_threads=settings.asr_cpu_threads,
        ).transcribe(audio_path)
    except Exception:
        return None


def warm_asr_model() -> None:
    """Load the local ASR model before the first verification request."""
    model_path = Path(settings.asr_model_path)
    if not model_path.exists():
        return
    try:
        from ml_service.adapters.asr_adapter import _load_model
        _load_model(
            str(model_path),
            settings.asr_device,
            settings.asr_compute_type,
            settings.asr_cpu_threads,
        )
    except Exception:
        # The request path turns an unavailable ASR model into an explicit
        # unverified result; startup must remain available for other checks.
        return


def _probe_duration(audio_path: Path) -> float | None:
    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    try:
        import wave
        with wave.open(str(audio_path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except Exception:
        return None
