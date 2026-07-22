from __future__ import annotations

from datetime import datetime, timezone

from ml_service.api.schemas import CheckScore, ScoreRequest, ScoreResponse
from ml_service.config import Settings
from ml_service.core.checks import score_active_light, score_audio, score_classifier, score_gesture, score_rppg
from ml_service.core.math_utils import clamp01

# Challenge-response checks are scored apart from the passive signal average:
# passing them must not dilute a bad video score, and failing BOTH together is
# not a technical glitch — it means a pre-recorded stream or disabled devices.
CHALLENGE_CHECKS = ("gesture", "audio")


class CascadeScorer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def score(self, request: ScoreRequest) -> ScoreResponse:
        checks = self._evaluate(request)
        risk_score = _weighted_risk(checks)
        confidence = _weighted_confidence(checks)
        if _challenge_pair_failed(checks):
            risk_score = 1.0
        decision = _decision(
            checks,
            risk_score,
            allow_threshold=self._settings.decision_allow_threshold,
            deny_threshold=self._settings.decision_deny_threshold,
        )
        return ScoreResponse(
            uid=request.uid,
            check_id=request.check_id,
            decision=decision,
            risk_score=risk_score,
            confidence=confidence,
            checks=checks,
            factors=_factors(checks),
            created_at=datetime.now(timezone.utc),
        )

    def _evaluate(self, request: ScoreRequest) -> list[CheckScore]:
        return [
            score_classifier(request.evidence.classifier),
            score_active_light(request.evidence.active_light, request.challenge, self._settings),
            score_rppg(request.evidence.rppg, self._settings),
            score_gesture(request.evidence.gesture, request.challenge),
            score_audio(request.evidence.audio, request.challenge),
        ]


def _challenge_pair_failed(checks: list[CheckScore]) -> bool:
    statuses = {check.name: check.status for check in checks}
    return all(statuses.get(name) == "failed" for name in CHALLENGE_CHECKS)


def _weighted_risk(checks: list[CheckScore]) -> float:
    active = [check for check in checks if check.status != "skipped" and check.name not in CHALLENGE_CHECKS]
    total_weight = sum(check.weight for check in active)
    if total_weight <= 0.0:
        return 0.5
    return clamp01(sum(check.risk * check.weight for check in active) / total_weight)


def _weighted_confidence(checks: list[CheckScore]) -> float:
    active = [check for check in checks if check.status not in ("skipped", "unknown") and check.name not in CHALLENGE_CHECKS]
    total_weight = sum(check.weight for check in active)
    if total_weight <= 0.0:
        return 0.0
    return clamp01(sum(check.confidence * check.weight for check in active) / total_weight)


def _decision(checks: list[CheckScore], risk_score: float, *, allow_threshold: float, deny_threshold: float):
    # Deny is driven by the averaged weighted risk of the passive checks (the
    # challenge pair is scored apart; both failing forces risk 1.0 upstream): an
    # individual check can fail for benign reasons (poor webcam/mic, slow link),
    # so one bad signal sends the session to review, while consistently bad
    # signals push the average over the deny threshold. A failed or missing
    # liveness signal still blocks "allow" — it must not be averaged away.
    if risk_score >= deny_threshold:
        return "deny"
    # The classifier is hard too: a session must not be allowed when the
    # primary deepfake detector produced no verdict at all.
    hard_checks = {"classifier", "active_light", "rppg", "gesture", "audio"}
    if any(check.status == "failed" for check in checks):
        return "review"
    if any(check.status in ("unknown", "skipped") for check in checks if check.name in hard_checks):
        return "review"
    if risk_score <= allow_threshold:
        return "allow"
    return "review"


def _factors(checks: list[CheckScore]) -> list[str]:
    factors: list[str] = []
    if _challenge_pair_failed(checks):
        factors.append("challenge: gesture and audio both failed — pre-recorded stream or disabled devices suspected")
    for check in checks:
        if check.status == "failed":
            factors.append(f"{check.name}: {check.reason}")
        elif check.status == "unknown":
            factors.append(f"{check.name}: insufficient evidence")
    return factors
