from __future__ import annotations

from ml_service.api.schemas import AudioEvidence, CheckScore
from ml_service.core.challenge import ChallengePlan
from ml_service.core.checks._common import challenge_payload, skipped
from ml_service.core.math_utils import clamp01, levenshtein_ratio, phrase_word_match_fraction


def score_audio(evidence: AudioEvidence | None, challenge: ChallengePlan | None) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return skipped("audio", 0.20)
    # The session challenge is authoritative: the client must not be able to
    # substitute its own "expected" phrase for the random one it was issued.
    expected_phrase = challenge_payload(challenge, "audio_phrase", "phrase")
    if expected_phrase is None:
        expected_phrase = evidence.phrase_expected if evidence else None
    if evidence is None:
        return CheckScore(name="audio", status="unknown", risk=0.45, confidence=0.0, weight=0.20, reason="audio evidence is missing")
    if evidence.ai_probability is None:
        return CheckScore(
            name="audio",
            status="unknown",
            risk=0.60,
            confidence=0.0,
            weight=0.20,
            reason="audio anti-spoof model result is required",
            details={
                "expected_phrase": expected_phrase,
                "phrase_transcribed": evidence.phrase_transcribed,
                "detector": evidence.detector,
                "duration_seconds": evidence.duration_seconds,
            },
        )

    # The transcript comes from server-side ASR. None means the phrase could
    # not be verified at all — that must not read as a pass; an empty string
    # is verified silence and scores ratio 0 (fails the phrase).
    if expected_phrase and evidence.phrase_transcribed is None:
        return CheckScore(
            name="audio",
            status="unknown",
            risk=0.60,
            confidence=0.0,
            weight=0.20,
            reason="audio phrase transcript is unavailable",
            details={
                "expected_phrase": expected_phrase,
                "ai_probability": evidence.ai_probability,
                "duration_seconds": evidence.duration_seconds,
                "detector": evidence.detector,
            },
        )
    phrase_ratio = None
    word_fraction = None
    if expected_phrase:
        phrase_ratio = levenshtein_ratio(expected_phrase, evidence.phrase_transcribed)
        word_fraction = phrase_word_match_fraction(expected_phrase, evidence.phrase_transcribed)
    # Full-string ratio 0.78 cannot survive ASR dropping one word of three on a
    # poor microphone; 2-of-3 fuzzy word matches keep the challenge binding
    # while tolerating exactly that.
    phrase_ok = phrase_ratio is None or phrase_ratio >= 0.78 or word_fraction >= 2 / 3
    ai_risk = evidence.ai_probability
    speaker_bonus = evidence.speaker_match_probability if evidence.speaker_match_probability is not None else 0.50
    phrase_risk = 0.0 if phrase_ok else 0.70
    risk = clamp01(0.55 * ai_risk + 0.25 * phrase_risk + 0.20 * (1.0 - speaker_bonus))
    confidence_parts = [value for value in (evidence.ai_probability, evidence.speaker_match_probability, phrase_ratio) if value is not None]
    confidence = sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.0
    status = "failed" if ai_risk >= 0.70 or not phrase_ok else "passed"
    return CheckScore(
        name="audio",
        status=status,
        risk=risk,
        confidence=clamp01(confidence),
        weight=0.20,
        reason="audio challenge and synthetic speech signals evaluated",
        details={
            "expected_phrase": expected_phrase,
            "phrase_ratio": phrase_ratio,
            "phrase_word_fraction": word_fraction,
            "ai_probability": evidence.ai_probability,
            "speaker_match_probability": evidence.speaker_match_probability,
            "duration_seconds": evidence.duration_seconds,
        },
    )
