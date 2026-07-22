from ml_service.api.schemas import (
    ActiveLightEvidence,
    AudioEvidence,
    ClassifierEvidence,
    EvidenceBundle,
    GestureEvidence,
    RppgEvidence,
    ScoreRequest,
)
from ml_service.config import settings
from ml_service.core.challenge import generate_challenge
from ml_service.core.scoring import CascadeScorer
from ml_service.core.checks import score_active_light, score_rppg


def test_low_risk_evidence_is_allowed():
    challenge = generate_challenge(seed=7)
    light = next(step for step in challenge.steps if step.type == "active_light")
    gesture = next(step for step in challenge.steps if step.type == "gesture")
    audio = next(step for step in challenge.steps if step.type == "audio_phrase")

    result = CascadeScorer(settings).score(
        ScoreRequest(
            uid="u1",
            check_id="c1",
            challenge=challenge,
            evidence=EvidenceBundle(
                classifier=ClassifierEvidence(fake_probability=0.05, confidence=0.95, frame_count=24, face_present=True),
                active_light=ActiveLightEvidence(
                    expected_luma=light.payload["luma_sequence"],
                    observed_face_luma=light.payload["luma_sequence"],
                    face_present=True,
                ),
                rppg=RppgEvidence(
                    samples=[100.0, 105.8, 109.5, 109.5, 105.8, 100.0, 94.2, 90.5, 90.5, 94.2] * 12,
                    sample_rate_hz=10,
                    window_seconds=4,
                    face_present=True,
                ),
                gesture=GestureEvidence(
                    expected_action=gesture.payload["expected_action"],
                    observed_action=gesture.payload["expected_action"],
                    confidence=0.9,
                    detector="unit-test-detector",
                    face_present=True,
                ),
                audio=AudioEvidence(
                    phrase_expected=audio.payload["phrase"],
                    phrase_transcribed=audio.payload["phrase"],
                    ai_probability=0.04,
                    speaker_match_probability=0.91,
                ),
            ),
        )
    )

    assert result.decision == "allow"
    assert result.risk_score < 0.35
    assert not [check for check in result.checks if check.status == "failed"]


def test_failed_classifier_blocks_allow_despite_passing_liveness():
    challenge = generate_challenge(seed=13)
    light = next(step for step in challenge.steps if step.type == "active_light")
    gesture = next(step for step in challenge.steps if step.type == "gesture")
    audio = next(step for step in challenge.steps if step.type == "audio_phrase")

    result = CascadeScorer(settings).score(
        ScoreRequest(
            uid="u1",
            check_id="c1",
            challenge=challenge,
            evidence=EvidenceBundle(
                classifier=ClassifierEvidence(fake_probability=0.95, confidence=0.95, face_present=True),
                active_light=ActiveLightEvidence(
                    expected_luma=light.payload["luma_sequence"],
                    observed_face_luma=light.payload["luma_sequence"],
                    face_present=True,
                ),
                rppg=RppgEvidence(
                    samples=[100.0, 105.8, 109.5, 109.5, 105.8, 100.0, 94.2, 90.5, 90.5, 94.2] * 12,
                    sample_rate_hz=10,
                    window_seconds=4,
                    face_present=True,
                ),
                gesture=GestureEvidence(
                    expected_action=gesture.payload["expected_action"],
                    observed_action=gesture.payload["expected_action"],
                    confidence=0.9,
                    detector="unit-test-detector",
                    face_present=True,
                ),
                audio=AudioEvidence(
                    phrase_expected=audio.payload["phrase"],
                    phrase_transcribed=audio.payload["phrase"],
                    ai_probability=0.04,
                    speaker_match_probability=0.91,
                ),
            ),
        )
    )

    # A flagged deepfake must never be averaged away into "allow". With clean
    # liveness signals the averaged risk lands in the review band — a single
    # failed check escalates to review, only a consistently bad set denies.
    assert result.decision == "review"


def test_gesture_and_audio_both_failed_force_max_risk():
    challenge = generate_challenge(seed=21)
    light = next(step for step in challenge.steps if step.type == "active_light")
    gesture = next(step for step in challenge.steps if step.type == "gesture")
    audio = next(step for step in challenge.steps if step.type == "audio_phrase")

    result = CascadeScorer(settings).score(
        ScoreRequest(
            uid="u1",
            check_id="c1",
            challenge=challenge,
            evidence=EvidenceBundle(
                classifier=ClassifierEvidence(fake_probability=0.05, confidence=0.95, frame_count=24, face_present=True),
                active_light=ActiveLightEvidence(
                    expected_luma=light.payload["luma_sequence"],
                    observed_face_luma=light.payload["luma_sequence"],
                    face_present=True,
                ),
                rppg=RppgEvidence(bpm=72.0, signal_quality=0.82, detector="python-rppg", face_present=True),
                gesture=GestureEvidence(
                    expected_action=gesture.payload["expected_action"],
                    observed_action="__wrong_action__",
                    confidence=0.9,
                    detector="unit-test-detector",
                    face_present=True,
                ),
                audio=AudioEvidence(
                    phrase_expected=audio.payload["phrase"],
                    phrase_transcribed="",
                    ai_probability=0.95,
                ),
            ),
        )
    )

    # Both challenge checks failing together is a pre-recorded stream or
    # disabled devices, never a benign glitch — clean passive signals must
    # not average it away.
    assert result.risk_score == 1.0
    assert result.decision == "deny"
    assert any("gesture and audio both failed" in factor for factor in result.factors)


def test_deepfake_video_not_diluted_by_perfect_gesture_and_audio():
    challenge = generate_challenge(seed=23)
    gesture = next(step for step in challenge.steps if step.type == "gesture")
    audio = next(step for step in challenge.steps if step.type == "audio_phrase")

    result = CascadeScorer(settings).score(
        ScoreRequest(
            uid="u1",
            check_id="c1",
            challenge=challenge,
            evidence=EvidenceBundle(
                classifier=ClassifierEvidence(fake_probability=0.95, confidence=0.95, face_present=True),
                active_light=ActiveLightEvidence(
                    expected_luma=[0, 255, 0, 255, 0, 255],
                    observed_face_luma=[120, 121, 120, 121, 120, 121],
                    face_present=True,
                ),
                rppg=RppgEvidence(bpm=53.33, signal_quality=0.60, detector="rppg-toolbox-pos", face_present=True),
                gesture=GestureEvidence(
                    expected_action=gesture.payload["expected_action"],
                    observed_action=gesture.payload["expected_action"],
                    confidence=0.95,
                    detector="unit-test-detector",
                    face_present=True,
                ),
                audio=AudioEvidence(
                    phrase_expected=audio.payload["phrase"],
                    phrase_transcribed=audio.payload["phrase"],
                    ai_probability=0.02,
                    speaker_match_probability=0.95,
                ),
            ),
        )
    )

    # Perfectly faked gesture/audio must not pull a failing video stack down
    # to a medium score: the challenge pair is out of the weighted average.
    # The same evidence scored ~0.35 when the pair was still averaged in.
    assert result.decision != "allow"
    assert result.risk_score >= 0.5


def test_high_risk_classifier_and_audio_are_denied():
    result = CascadeScorer(settings).score(
        ScoreRequest(
            uid="u1",
            check_id="c1",
            evidence=EvidenceBundle(
                classifier=ClassifierEvidence(fake_probability=0.98, confidence=0.98),
                audio=AudioEvidence(ai_probability=0.95, speaker_match_probability=0.20),
            ),
        )
    )

    assert result.decision in {"review", "deny"}
    assert result.risk_score >= 0.5
    assert "classifier" in " ".join(result.factors)


def test_no_face_liveness_is_denied_not_allowed():
    challenge = generate_challenge(seed=11)
    light = next(step for step in challenge.steps if step.type == "active_light")
    gesture = next(step for step in challenge.steps if step.type == "gesture")

    result = CascadeScorer(settings).score(
        ScoreRequest(
            uid="ceiling",
            check_id="ceiling-check",
            challenge=challenge,
            evidence=EvidenceBundle(
                classifier=ClassifierEvidence(fake_probability=0.05, confidence=0.95, face_present=False),
                active_light=ActiveLightEvidence(
                    expected_luma=light.payload["luma_sequence"],
                    observed_face_luma=light.payload["luma_sequence"],
                    face_present=False,
                ),
                rppg=RppgEvidence(
                    samples=[100.0, 105.0, 110.0, 105.0, 100.0, 95.0, 90.0, 95.0] * 15,
                    sample_rate_hz=10,
                    face_present=False,
                ),
                gesture=GestureEvidence(
                    expected_action=gesture.payload["expected_action"],
                    observed_action=gesture.payload["expected_action"],
                    confidence=0.9,
                    detector="manual",
                    face_present=False,
                ),
            ),
        )
    )

    assert result.decision == "deny"
    assert {check.name: check.status for check in result.checks}["rppg"] == "failed"
    assert {check.name: check.status for check in result.checks}["gesture"] != "passed"


def test_active_light_face_flashing_verifier_passes_with_reference_metrics():
    check = score_active_light(
        ActiveLightEvidence(
            detector="face-flashing-frame-pairs",
            verifier_score=0.91,
            pair_count=8,
            temporal_correlation=0.86,
            best_correlation=0.91,
            spatial_contrast=0.18,
            spatial_center_error=0.08,
            response_snr=0.21,
            response_magnitude=15.0,
            color_cosine=0.82,
            face_present=True,
            face_confidence=1.0,
        ),
        challenge=None,
        settings=settings,
    )

    assert check.status == "passed"
    assert check.reason == "face flashing frame-pair verifier evaluated"
    assert check.details["detector"] == "face-flashing-frame-pairs"


def test_active_light_frame_pairs_reject_low_temporal_even_when_verifier_high():
    check = score_active_light(
        ActiveLightEvidence(
            detector="face-flashing-frame-pairs",
            verifier_score=0.90,
            pair_count=8,
            temporal_correlation=0.47,
            best_correlation=0.47,
            spatial_contrast=2.38,
            spatial_center_error=0.077,
            response_snr=0.13,
            response_magnitude=2.0,
            color_cosine=0.93,
            face_present=True,
        ),
        challenge=None,
        settings=settings,
    )

    assert check.status == "failed"
    assert check.risk >= 0.5


def test_active_light_face_flashing_verifier_fails_without_pairs():
    check = score_active_light(
        ActiveLightEvidence(
            detector="face-flashing-frame-pairs",
            verifier_score=0.2,
            pair_count=0,
            temporal_correlation=0.0,
            best_correlation=0.0,
            spatial_contrast=0.0,
            response_snr=0.0,
            color_cosine=0.0,
            face_present=False,
            face_confidence=0.0,
        ),
        challenge=None,
        settings=settings,
    )

    assert check.status == "failed"
    assert check.risk >= 0.9


def test_python_rppg_passes_from_reference_model_output_without_raw_samples():
    check = score_rppg(
        RppgEvidence(
            bpm=72.0,
            signal_quality=0.82,
            latency=1.1,
            hrv={"sdnn": 42.0},
            detector="python-rppg",
            face_present=True,
        ),
        settings=settings,
    )

    assert check.status == "passed"
    assert check.details["detector"] == "python-rppg"
    assert check.details["bpm"] == 72.0


def test_rppg_toolbox_output_uses_model_scoring_path():
    check = score_rppg(
        RppgEvidence(
            bpm=72.0,
            signal_quality=0.82,
            detector="rppg-toolbox-pos",
            face_present=True,
        ),
        settings=settings,
    )

    assert check.status == "passed"
    assert check.details["detector"] == "rppg-toolbox-pos"


def test_active_light_flat_observed_fails_contrast_gate():
    check = score_active_light(
        ActiveLightEvidence(
            expected_luma=[0, 255, 0, 255, 0, 255],
            observed_face_luma=[120, 121, 120, 121, 120, 121],
            face_present=True,
        ),
        challenge=None,
        settings=settings,
    )

    assert check.status == "failed"
    assert check.details["contrast"] < settings.active_light_min_contrast


def test_rppg_face_present_none_is_not_auto_denied():
    check = score_rppg(
        RppgEvidence(
            bpm=72.0,
            signal_quality=0.82,
            detector="python-rppg",
            face_present=None,
        ),
        settings=settings,
    )

    assert check.status == "passed"


def test_gesture_edge_confidence_matched_passes():
    from ml_service.api.schemas import GestureEvidence
    from ml_service.core.checks import score_gesture

    check = score_gesture(
        GestureEvidence(
            expected_action="touch_nose",
            observed_action="touch_nose",
            confidence=0.5,
            detector="hands-pose-face-mesh",
            face_present=True,
        ),
        challenge=None,
    )

    assert check.status == "passed"


def test_python_rppg_edge_bpm_requires_higher_sqi():
    borderline = score_rppg(
        RppgEvidence(bpm=53.33, signal_quality=0.60, detector="rppg-toolbox-pos", face_present=True),
        settings=settings,
    )
    assert borderline.status == "failed"

    clean = score_rppg(
        RppgEvidence(bpm=53.33, signal_quality=0.70, detector="rppg-toolbox-pos", face_present=True),
        settings=settings,
    )
    assert clean.status == "passed"


def test_python_rppg_low_quality_is_unknown_not_fallback_failed():
    check = score_rppg(
        RppgEvidence(
            bpm=72.0,
            signal_quality=0.2,
            detector="python-rppg",
            face_present=True,
        ),
        settings=settings,
    )

    assert check.status == "unknown"
    assert check.reason == "rPPG signal quality is too low"
