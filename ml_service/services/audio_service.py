from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import AudioEvidence, ServiceAnalyzeResponse
from ml_service.config import settings
from ml_service.core.checks import score_audio
from ml_service.services.common import service_response, unavailable_check


class AudioService:
    name = "audio"

    async def analyze_audio(
        self,
        file: UploadFile,
        *,
        phrase_expected: str | None,
        phrase_transcribed: str | None,
    ) -> ServiceAnalyzeResponse:
        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            duration = await run_in_threadpool(_probe_duration, Path(tmp.name))
            ai_probability, error_message = await run_in_threadpool(_run_audio_model, Path(tmp.name))

        evidence = AudioEvidence(
            phrase_expected=phrase_expected,
            phrase_transcribed=phrase_transcribed,
            ai_probability=ai_probability,
            duration_seconds=duration,
            detector="audio-cnn" if ai_probability is not None else "unavailable",
        )
        check = score_audio(evidence, challenge=None)
        message = None if ai_probability is not None else error_message
        if ai_probability is None:
            check = unavailable_check("audio", 0.20, error_message)
        return service_response(self.name, evidence, check, message=message)


def _run_audio_model(audio_path: Path) -> tuple[float | None, str]:
    model_path = Path(settings.audio_model_path)
    if not model_path.exists():
        return None, "audio anti-spoof model is not configured"
    try:
        from ml_service.adapters.audio_adapter import AudioModelAdapter
        result = AudioModelAdapter(model_path=model_path).predict(audio_path)
    except Exception as exc:
        return None, f"audio anti-spoof inference failed: {type(exc).__name__}"
    return result.get("ai_probability"), ""


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
