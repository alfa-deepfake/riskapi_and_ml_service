from ml_service.api.schemas import CheckScore, ScoreRequest
from ml_service.config import settings
from ml_service.core.detectors import DetectionContext, DetectorRegistry
from ml_service.core.scoring import CascadeScorer


class DummyDetector:
    name = "future_audio_model"

    def evaluate(self, request: ScoreRequest, context: DetectionContext) -> CheckScore:
        return CheckScore(
            name=self.name,
            status="passed",
            risk=0.1,
            confidence=0.9,
            weight=0.5,
            reason="future model plugged into registry",
        )


def test_scorer_accepts_custom_detector_registry():
    registry = DetectorRegistry([DummyDetector()])

    result = CascadeScorer(settings, registry=registry).score(ScoreRequest(uid="u1"))

    assert [check.name for check in result.checks] == ["future_audio_model"]
    assert result.decision == "allow"
