from ml_service.api.schemas import ActiveLightEvidence
from ml_service.config import settings
from ml_service.core.checks import score_active_light


def test_active_light_tolerates_one_step_camera_lag():
    expected = [0, 255, 0, 255, 0, 255]
    observed_with_lag = [128, 0, 255, 0, 255, 0]

    score = score_active_light(
        ActiveLightEvidence(expected_luma=expected, observed_face_luma=observed_with_lag, face_present=True),
        challenge=None,
        settings=settings,
    )

    assert score.status == "passed"
    assert score.details["lag"] == 1
