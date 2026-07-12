from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


class AudioModelAdapter:
    def __init__(self, *, model_path: Path, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device

    def predict(self, audio_path: Path) -> dict[str, Any]:
        from neiro_model.audio.infer import predict

        args = SimpleNamespace(
            audio=str(audio_path),
            model=str(self.model_path),
            device=self.device,
            ai_class=1,
            threshold=0.5,
            hop_length=160,
            log_mel="natural",
            sample_rate=16000,
            clip_seconds=8.0,
            stride_seconds=None,
        )
        return predict(args)
