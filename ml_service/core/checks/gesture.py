from __future__ import annotations

from ml_service.api.schemas import CheckScore, GestureEvidence
from ml_service.core.challenge import ChallengePlan
from ml_service.core.checks._common import challenge_payload, skipped
from ml_service.core.math_utils import clamp01


def score_gesture(evidence: GestureEvidence | None, challenge: ChallengePlan | None) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return skipped("gesture", 0.15)
    # The session challenge is authoritative: the client must not be able to
    # redefine which gesture was expected. Client value is a fallback for the
    # challenge-less direct scoring endpoint only.
    expected = challenge_payload(challenge, "gesture", "expected_action")
    if expected is None:
        expected = evidence.expected_action if evidence else None
    if evidence is None or not expected or not evidence.observed_action:
        return CheckScore(name="gesture", status="unknown", risk=0.45, confidence=0.0, weight=0.15, reason="gesture evidence is missing")
    if evidence.detector in (None, "manual"):
        return CheckScore(
            name="gesture",
            status="unknown",
            risk=0.65,
            confidence=0.0,
            weight=0.15,
            reason="gesture requires a real detector, manual confirmation is not accepted",
            details=_gesture_details(evidence, expected),
        )
    if evidence.face_present is False:
        return CheckScore(
            name="gesture",
            status="failed",
            risk=0.95,
            confidence=0.8,
            weight=0.15,
            reason="gesture cannot pass without a detected face/body target",
            details=_gesture_details(evidence, expected),
        )

    matched = evidence.observed_action == expected
    confidence = evidence.confidence if evidence.confidence is not None else 0.8 if matched else 0.4
    risk = 1.0 - confidence if matched else 0.85
    return CheckScore(
        name="gesture",
        status="passed" if matched and confidence >= 0.5 else "failed",
        risk=clamp01(risk),
        confidence=clamp01(confidence),
        weight=0.15,
        reason="gesture challenge response evaluated",
        details=_gesture_details(evidence, expected),
    )


def _gesture_details(evidence: GestureEvidence, expected: str) -> dict:
    return {
        "expected_action": expected,
        "observed_action": evidence.observed_action,
        "detector": evidence.detector,
        "face_present": evidence.face_present,
        "frame_count": evidence.frame_count,
        "best_distance": evidence.best_distance,
    }
