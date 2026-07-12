import math

from ml_service.api.schemas import RppgEvidence
from ml_service.config import settings
from ml_service.core.checks import score_rppg


def test_rppg_uses_sliding_window_stability():
    sample_rate = 10.0
    samples = [
        100.0 + 8.0 * math.sin(index / sample_rate * 2.0 * math.pi * 1.2)
        for index in range(120)
    ]

    score = score_rppg(
        RppgEvidence(samples=samples, sample_rate_hz=sample_rate, window_seconds=4.0, face_present=True),
        settings=settings,
    )

    assert score.status == "passed"
    assert score.details["stability"] is not None
    assert score.details["sample_count"] == 120
