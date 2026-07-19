from __future__ import annotations

from ml_service.api.schemas import ActiveLightEvidence, CheckScore
from ml_service.config import Settings
from ml_service.core.challenge import ChallengePlan
from ml_service.core.checks._common import challenge_luma, skipped
from ml_service.core.math_utils import best_lagged_correlation, clamp01


def score_active_light(
    evidence: ActiveLightEvidence | None,
    challenge: ChallengePlan | None,
    settings: Settings,
) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return skipped("active_light", 0.22)
    if evidence is None:
        return CheckScore(
            name="active_light",
            status="unknown",
            risk=0.50,
            confidence=0.0,
            weight=0.22,
            reason="active light evidence is missing",
        )

    if evidence.face_present is False:
        return CheckScore(
            name="active_light",
            status="failed",
            risk=0.95,
            confidence=clamp01(evidence.face_confidence or 0.8),
            weight=0.22,
            reason="active light cannot pass without a detected face",
            details={"face_present": evidence.face_present, "face_confidence": evidence.face_confidence},
        )

    if evidence.detector == "face-flashing-frame-pairs" and evidence.verifier_score is not None:
        temporal = evidence.temporal_correlation if evidence.temporal_correlation is not None else evidence.best_correlation
        passed = bool(
            evidence.pair_count is not None
            and evidence.pair_count >= settings.active_light_min_pair_count
            and evidence.verifier_score >= settings.active_light_min_correlation
            and temporal is not None
            and temporal >= settings.active_light_min_temporal_correlation
            and evidence.spatial_contrast is not None
            and evidence.spatial_contrast >= settings.active_light_min_spatial_contrast
            and evidence.response_snr is not None
            and evidence.response_snr >= settings.active_light_min_response_snr
            and evidence.color_cosine is not None
            and evidence.color_cosine >= settings.active_light_min_color_cosine
        )
        risk = clamp01(1.0 - evidence.verifier_score) if passed else max(clamp01(1.0 - evidence.verifier_score), 0.7)
        return CheckScore(
            name="active_light",
            status="passed" if passed else "failed",
            risk=risk,
            confidence=clamp01(evidence.verifier_score),
            weight=0.22,
            reason="face flashing frame-pair verifier evaluated",
            details={
                "detector": evidence.detector,
                "pair_count": evidence.pair_count,
                "verifier_score": evidence.verifier_score,
                "temporal_correlation": evidence.temporal_correlation,
                "best_correlation": evidence.best_correlation,
                "spatial_contrast": evidence.spatial_contrast,
                "spatial_center_error": evidence.spatial_center_error,
                "response_snr": evidence.response_snr,
                "response_magnitude": evidence.response_magnitude,
                "color_cosine": evidence.color_cosine,
            },
        )

    expected = evidence.expected_luma or challenge_luma(challenge)
    observed = evidence.observed_face_luma
    if len(expected) < 3 or len(observed) < 3:
        return CheckScore(
            name="active_light",
            status="unknown",
            risk=0.55,
            confidence=0.0,
            weight=0.22,
            reason="not enough active light samples",
            details={"expected_samples": len(expected), "observed_samples": len(observed)},
        )

    corr, lag = best_lagged_correlation(expected, observed, max_lag=1)
    observed_peak = max(observed)
    observed_span = observed_peak - min(observed)
    contrast = observed_span / max(observed_peak, 1e-6)
    if corr is None:
        return CheckScore(
            name="active_light",
            status="unknown",
            risk=0.60,
            confidence=0.0,
            weight=0.22,
            reason="active light correlation is undefined",
        )

    corr_score = clamp01((corr + 1.0) / 2.0)
    contrast_score = clamp01(contrast / max(settings.active_light_min_contrast, 1e-6))
    live_score = 0.75 * corr_score + 0.25 * min(1.0, contrast_score)
    risk = clamp01(1.0 - live_score)
    passed = corr >= settings.active_light_min_correlation and contrast >= settings.active_light_min_contrast
    return CheckScore(
        name="active_light",
        status="passed" if passed else "failed",
        risk=risk,
        confidence=clamp01(abs(corr) * 0.8 + min(contrast, 1.0) * 0.2),
        weight=0.22,
        reason="face luminance response compared with screen challenge",
        details={
            "correlation": corr,
            "lag": lag,
            "contrast": contrast,
            "expected_samples": len(expected),
            "observed_samples": len(observed),
        },
    )
