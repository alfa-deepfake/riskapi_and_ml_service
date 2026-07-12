from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ml_service.api.schemas import CheckScore, ScoreRequest
from ml_service.config import Settings
from ml_service.core.checks import score_active_light, score_audio, score_classifier, score_gesture, score_rppg


@dataclass(frozen=True)
class DetectionContext:
    settings: Settings


class SignalDetector(Protocol):
    """Single cascade stage.

    Add future models by implementing this protocol and registering the class in
    `default_detector_registry`. The scorer stays unchanged.
    """

    name: str

    def evaluate(self, request: ScoreRequest, context: DetectionContext) -> CheckScore:
        ...


class ClassifierDetector:
    name = "classifier"

    def evaluate(self, request: ScoreRequest, context: DetectionContext) -> CheckScore:
        return score_classifier(request.evidence.classifier)


class ActiveLightDetector:
    name = "active_light"

    def evaluate(self, request: ScoreRequest, context: DetectionContext) -> CheckScore:
        return score_active_light(request.evidence.active_light, request.challenge, context.settings)


class RppgDetector:
    name = "rppg"

    def evaluate(self, request: ScoreRequest, context: DetectionContext) -> CheckScore:
        return score_rppg(request.evidence.rppg, context.settings)


class GestureDetector:
    name = "gesture"

    def evaluate(self, request: ScoreRequest, context: DetectionContext) -> CheckScore:
        return score_gesture(request.evidence.gesture, request.challenge)


class AudioDetector:
    name = "audio"

    def evaluate(self, request: ScoreRequest, context: DetectionContext) -> CheckScore:
        return score_audio(request.evidence.audio, request.challenge)


class DetectorRegistry:
    def __init__(self, detectors: list[SignalDetector] | None = None) -> None:
        self._detectors: dict[str, SignalDetector] = {}
        for detector in detectors or []:
            self.register(detector)

    def register(self, detector: SignalDetector) -> None:
        if detector.name in self._detectors:
            raise ValueError(f"Detector already registered: {detector.name}")
        self._detectors[detector.name] = detector

    def evaluate_all(self, request: ScoreRequest, context: DetectionContext) -> list[CheckScore]:
        return [detector.evaluate(request, context) for detector in self._detectors.values()]

    @property
    def names(self) -> list[str]:
        return list(self._detectors)


def default_detector_registry() -> DetectorRegistry:
    return DetectorRegistry(
        [
            ClassifierDetector(),
            ActiveLightDetector(),
            RppgDetector(),
            GestureDetector(),
            AudioDetector(),
        ]
    )
