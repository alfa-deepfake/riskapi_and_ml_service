"""Per-detector scoring functions, one module each.

Each returns a CheckScore for its signal. Import them from here so callers keep
a single entry point (``from ml_service.core.checks import score_audio``).
"""

from ml_service.core.checks.active_light import score_active_light
from ml_service.core.checks.audio import score_audio
from ml_service.core.checks.classifier import score_classifier
from ml_service.core.checks.gesture import score_gesture
from ml_service.core.checks.rppg import score_rppg

__all__ = [
    "score_active_light",
    "score_audio",
    "score_classifier",
    "score_gesture",
    "score_rppg",
]
