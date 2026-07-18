"""Low-latency Faster-Whisper ASR for verifying the spoken challenge phrase.

The local CTranslate2 ``faster-whisper-medium`` model is loaded once per
process. It never downloads a model during a verification request. A quiet
clip is short-circuited to an empty transcript: Whisper hallucinates words on
silence, and silence must fail the phrase check.
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
    def __init__(self, *, model_path: Path, device: str = "cpu", compute_type: str = "int8", cpu_threads: int = 4) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads

    def transcribe(self, audio_path: Path) -> str:
        import numpy as np
        import soundfile as sf

        from ml_service.adapters.audio_adapter import _as_soundfile_input

        wav_path = _as_soundfile_input(audio_path, 16_000)
        try:
            samples, rate = sf.read(wav_path, dtype="float32", always_2d=True)
            mono = samples.mean(axis=1)
            if not len(mono) or float(np.sqrt(np.mean(mono**2))) < SILENCE_RMS:
                return ""

            model = _load_model(str(self.model_path), self.device, self.compute_type, self.cpu_threads)
            with _ASR_LOCK:
                segments, _ = model.transcribe(
                    str(wav_path),
                    language="ru",
                    task="transcribe",
                    # Anti-hallucination recipe. Greedy (beam_size=1) plus the
                    # default temperature fallback made Whisper emit a canned
                    # phrase ("Я не могу это сделать") on quiet speech it could
                    # not resolve. Beam search + a single temperature=0 kill that
                    # resampling cascade; the thresholds drop low-confidence /
                    # repetitive decodes instead of returning them.
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    # Trim non-speech first (Silero VAD ships with faster-whisper,
                    # onnxruntime is in the image). A permissive threshold keeps
                    # quiet speech that the default 0.5 would drop and leave
                    # Whisper hallucinating on the noise that remains.
                    vad_filter=True,
                    vad_parameters={"threshold": 0.3, "min_silence_duration_ms": 500},
                )
                # ``segments`` is a generator; consume it under the lock so a
                # concurrent request cannot run inference through the same model.
                # Belt-and-braces: skip any segment Whisper itself flags as
                # probably-silence — that is where the canned hallucinations live.
                return " ".join(
                    segment.text.strip()
                    for segment in segments
                    if segment.no_speech_prob < 0.6
                ).strip()
        finally:
            if wav_path != audio_path:
                wav_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str, compute_type: str, cpu_threads: int):
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
        local_files_only=True,
    )
