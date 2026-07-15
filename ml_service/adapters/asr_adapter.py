"""Whisper ASR adapter for verifying the spoken challenge phrase.

Uses the transformers pipeline (torch + transformers already ship in the
image) with a local snapshot of whisper-tiny.en — no HuggingFace download at
runtime. A quiet clip is short-circuited to an empty transcript: whisper
hallucinates words on silence, and silence must fail the phrase check.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

# Below this RMS (float32 full-scale) the clip carries no speech at all;
# normal speech sits around 0.02-0.1.
SILENCE_RMS = 0.003

_ASR_LOCK = threading.Lock()


class WhisperAsrAdapter:
    def __init__(self, *, model_path: Path) -> None:
        self.model_path = model_path

    def transcribe(self, audio_path: Path) -> str:
        import numpy as np
        import soundfile as sf

        from ml_service.adapters.audio_adapter import _as_soundfile_input

        wav_path = _as_soundfile_input(audio_path, 16_000)
        try:
            samples, rate = sf.read(wav_path, dtype="float32", always_2d=True)
        finally:
            if wav_path != audio_path:
                wav_path.unlink(missing_ok=True)
        mono = samples.mean(axis=1)
        if not len(mono) or float(np.sqrt(np.mean(mono**2))) < SILENCE_RMS:
            return ""

        pipe = _load_pipeline(str(self.model_path))
        with _ASR_LOCK:
            result = pipe({"array": mono, "sampling_rate": rate})
        return str(result.get("text", "")).strip()


@lru_cache(maxsize=1)
def _load_pipeline(model_path: str):
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=model_path, device="cpu")
