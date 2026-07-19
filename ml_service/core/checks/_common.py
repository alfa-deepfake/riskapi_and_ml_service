from __future__ import annotations

from ml_service.api.schemas import CheckScore
from ml_service.core.challenge import ChallengePlan


def skipped(name: str, weight: float) -> CheckScore:
    return CheckScore(
        name=name,
        status="skipped",
        risk=0.50,
        confidence=0.0,
        weight=weight,
        reason="check skipped in test mode",
    )


def challenge_payload(challenge: ChallengePlan | None, step_type: str, key: str):
    if challenge is None:
        return None
    for step in challenge.steps:
        if step.type == step_type:
            return step.payload.get(key)
    return None


def challenge_luma(challenge: ChallengePlan | None) -> list[float]:
    sequence = challenge_payload(challenge, "active_light", "luma_sequence")
    if not isinstance(sequence, list):
        return []
    return [float(value) for value in sequence]
