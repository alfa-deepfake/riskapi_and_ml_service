"""Regression tests for the 2026-07-18 bug-fix batch."""
from ml_service.api.schemas import AudioEvidence, GestureEvidence
from ml_service.config import settings
from ml_service.core.challenge import generate_challenge
from ml_service.core.checks import score_audio, score_gesture
from ml_service.core.math_utils import _normalize_phrase, levenshtein_ratio
from ml_service.core.scoring import _decision
from ml_service.api.schemas import CheckScore
from ml_service.services.common import safe_suffix


def test_russian_phrase_normalization_ignores_punctuation_and_yo():
    # Whisper punctuates and spells ё/е freely; neither should hurt the match.
    assert levenshtein_ratio("банк сигнал река", "Банк, сигнал, река.") == 1.0
    assert levenshtein_ratio("ёлка", "елка") == 1.0
    assert _normalize_phrase("  Привет,  МИР! ") == "привет мир"


def test_safe_suffix_rejects_pathological_client_filenames():
    assert safe_suffix("clip.webm", ".webm") == ".webm"
    assert safe_suffix("clip.WAV", ".webm") == ".WAV"
    # NUL byte, 5000-char ext, missing name -> fall back, never crash tempfile
    assert safe_suffix("x." + "a" * 5000, ".webm") == ".webm"
    assert safe_suffix("bad\x00.webm", ".webm") == ".webm"
    assert safe_suffix(None, ".webm") == ".webm"
    assert safe_suffix("noext", ".webm") == ".webm"


def test_gesture_uses_session_challenge_not_client_expected_action():
    challenge = generate_challenge(seed=3)
    expected = next(s for s in challenge.steps if s.type == "gesture").payload["expected_action"]
    other = "touch_nose" if expected == "touch_mouth" else "touch_mouth"
    # Client claims it was asked for `other` and performed `other` — must not pass.
    check = score_gesture(
        GestureEvidence(
            expected_action=other, observed_action=other,
            confidence=0.9, detector="hands-pose-face-mesh", face_present=True,
        ),
        challenge=challenge,
    )
    assert check.status == "failed"


def test_audio_uses_session_challenge_not_client_phrase():
    challenge = generate_challenge(seed=5)
    check = score_audio(
        AudioEvidence(
            phrase_expected="произвольная своя фраза",
            phrase_transcribed="произвольная своя фраза",
            ai_probability=0.02,
        ),
        challenge=challenge,
    )
    # Transcript matched the client's own phrase, not the issued challenge phrase.
    assert check.details["phrase_ratio"] < 0.78


def test_missing_classifier_verdict_is_not_allowed():
    passing = [
        CheckScore(name=n, status="passed", risk=0.1, confidence=0.9, weight=0.2, reason="ok")
        for n in ("active_light", "rppg", "gesture", "audio")
    ]
    missing = CheckScore(
        name="classifier", status="unknown", risk=0.45, confidence=0.0, weight=0.25, reason="missing",
    )
    assert _decision([missing, *passing], 0.2, allow_threshold=0.35, deny_threshold=0.72) == "review"
