"""WavLM anti-spoof adapter over the vendored deepfake_audio inference code.

The checkpoint carries every weight plus its training config and EER
threshold; the encoder skeleton is built from the vendored wavlm-base-plus
config so no HuggingFace download happens at load time.
"""
from __future__ import annotations

import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

# Formats libsndfile reads directly; anything else (browser webm/opus, m4a)
# is transcoded with the ffmpeg already shipped in the image.
_SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg"}

_PREDICT_LOCK = threading.Lock()


class AudioModelAdapter:
    def __init__(self, *, model_path: Path, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device

    def predict(self, audio_path: Path) -> dict[str, Any]:
        from deepfake_audio.predict import predict_file

        model, checkpoint, device = _load_model(str(self.model_path), self.device)
        config = checkpoint.get("config", {})
        sample_rate = int(config.get("sample_rate", 16_000))
        clip_seconds = float(config.get("clip_seconds", 5.0))
        threshold = float(checkpoint.get("metrics", {}).get("eer_threshold", 0.5))

        wav_path = _as_soundfile_input(audio_path, sample_rate)
        try:
            with _PREDICT_LOCK:
                result = predict_file(
                    model,
                    wav_path,
                    device,
                    sample_rate,
                    clip_seconds,
                    hop_seconds=2.5,
                    batch_size=4,
                )
        finally:
            if wav_path != audio_path:
                wav_path.unlink(missing_ok=True)

        return {
            "ai_probability": float(result["fake_score_mean"]),
            "fake_score_max": float(result["fake_score_max"]),
            "windows": int(result["windows"]),
            "threshold": threshold,
            "duration_seconds": float(result["duration_seconds"]),
            "detector": "audio-wavlm-all4",
        }


@lru_cache(maxsize=1)
def _load_model(model_path: str, device_str: str):
    import torch
    from transformers import WavLMConfig

    import deepfake_audio
    from deepfake_audio.model import WavLMDeepfakeClassifier

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config_dir = Path(deepfake_audio.__file__).resolve().parent / "wavlm_config"
    wavlm_config = WavLMConfig.from_pretrained(config_dir)
    model = WavLMDeepfakeClassifier(config=wavlm_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint, device


def _as_soundfile_input(audio_path: Path, sample_rate: int) -> Path:
    if audio_path.suffix.lower() in _SOUNDFILE_SUFFIXES:
        return audio_path
    wav_path = audio_path.with_suffix(".decoded.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path
