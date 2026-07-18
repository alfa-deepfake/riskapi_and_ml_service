from __future__ import annotations

from ml_service.api.schemas import (
    ActiveLightEvidence,
    AudioEvidence,
    CheckScore,
    ClassifierEvidence,
    GestureEvidence,
    RppgEvidence,
)
from ml_service.config import Settings
from ml_service.core.challenge import ChallengePlan
from ml_service.core.math_utils import best_lagged_correlation, clamp01, levenshtein_ratio


def score_classifier(evidence: ClassifierEvidence | None) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return _skipped("classifier", 0.25)
    if evidence is not None and evidence.face_present is False:
        return CheckScore(
            name="classifier",
            status="failed",
            risk=0.95,
            confidence=clamp01(evidence.face_confidence or evidence.confidence or 0.8),
            weight=0.25,
            reason="frame classifier cannot pass without a detected face",
            details={"face_present": evidence.face_present, "face_confidence": evidence.face_confidence},
        )
    if evidence is None or evidence.fake_probability is None:
        return CheckScore(
            name="classifier",
            status="unknown",
            risk=0.45,
            confidence=0.0,
            weight=0.25,
            reason="frame classifier evidence is missing",
        )

    risk = clamp01(evidence.fake_probability)
    confidence = evidence.confidence if evidence.confidence is not None else max(risk, 1.0 - risk)
    fail_threshold = evidence.threshold if evidence.threshold is not None else 0.70
    details = {
        "model_name": evidence.model_name,
        "frame_count": evidence.frame_count,
        "fake_probability": evidence.fake_probability,
        "threshold": fail_threshold,
        "feature_count": evidence.feature_count,
        "preprocessing": evidence.preprocessing,
        "face_size_px": evidence.face_size_px,
        "condition": evidence.condition,
        "low_info": evidence.low_info,
        "cnn_probability": evidence.cnn_probability,
        "tree_probability": evidence.tree_probability,
        "upsample_diff": evidence.upsample_diff,
    }
    if evidence.model_scores is not None:
        details["model_scores"] = evidence.model_scores

    # v16 REJECT policy: AI restoration/upscaling on the input (GFPGAN and
    # kin) hides swap traces and is itself disallowed for a bank check.
    if evidence.condition == "restored":
        return CheckScore(
            name="classifier",
            status="failed",
            risk=max(risk, 0.85),
            confidence=clamp01(confidence),
            weight=0.25,
            reason="AI restoration/upscaling detected on the input — rejected",
            details=details,
        )

    # v16 forensic override: on a low-detail input (source face <180px or
    # wholly upscaled) the noise-CNN modality is physically blind and drags
    # the fused score toward REAL — when the trees still fire (mean >= t_susp,
    # 0.75 measured held-out: TPR 95.0->95.1%, FPR 3.56->3.66%) a REAL verdict
    # is not issued. This replaced the v15 withhold-FAKE gate: the v15b CNN
    # retrain fixed the false-FAKE modes the withhold guarded against, so a
    # fused FAKE on low-detail input now stands (annotated via details).
    tree_mean = evidence.tree_probability
    t_susp = evidence.t_susp if evidence.t_susp is not None else 0.75
    if evidence.low_info and risk < fail_threshold and tree_mean is not None and tree_mean >= t_susp:
        return CheckScore(
            name="classifier",
            status="failed",
            risk=max(risk, clamp01(tree_mean)),
            confidence=clamp01(tree_mean),
            weight=0.25,
            reason=f"forensic override: trees {tree_mean:.2f} on low-detail input — REAL verdict withheld",
            details=details,
        )

    return CheckScore(
        name="classifier",
        status="failed" if risk >= fail_threshold else "passed",
        risk=risk,
        confidence=clamp01(confidence),
        weight=0.25,
        reason="deepfake classifier probability evaluated",
        details=details,
    )


def score_active_light(
    evidence: ActiveLightEvidence | None,
    challenge: ChallengePlan | None,
    settings: Settings,
) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return _skipped("active_light", 0.22)
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

    expected = evidence.expected_luma or _challenge_luma(challenge)
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


def score_rppg(evidence: RppgEvidence | None, settings: Settings) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return _skipped("rppg", 0.18)
    if evidence is None:
        return CheckScore(name="rppg", status="unknown", risk=0.50, confidence=0.0, weight=0.18, reason="rPPG evidence is missing")

    if evidence.face_present is False:
        return CheckScore(
            name="rppg",
            status="failed",
            risk=0.95,
            confidence=clamp01(evidence.face_confidence or 0.8),
            weight=0.18,
            reason="rPPG cannot pass without a detected face ROI",
            details={"face_present": evidence.face_present, "face_confidence": evidence.face_confidence},
        )
    # Model-output evidence (bpm + signal quality, no raw samples) is scored on
    # the model path regardless of the exact detector label the runtime reports.
    if (
        evidence.detector == "python-rppg"
        or str(evidence.detector or "").startswith(("rppg-toolbox", "open-rppg"))
        or (evidence.signal_quality is not None and not evidence.samples)
    ):
        return _score_python_rppg(evidence)
    if not evidence.samples or not evidence.sample_rate_hz:
        return CheckScore(
            name="rppg",
            status="unknown",
            risk=0.65,
            confidence=0.0,
            weight=0.18,
            reason="rPPG raw samples are required",
        )
    if len(evidence.samples) < int(evidence.sample_rate_hz * 8):
        return CheckScore(
            name="rppg",
            status="unknown",
            risk=0.65,
            confidence=0.0,
            weight=0.18,
            reason="rPPG sample window is too short",
            details={"sample_count": len(evidence.samples), "sample_rate_hz": evidence.sample_rate_hz},
        )

    quality = _estimate_signal_quality(evidence.samples)
    bpm = _estimate_bpm(evidence.samples, evidence.sample_rate_hz)
    stability = _estimate_window_stability(
        evidence.samples,
        evidence.sample_rate_hz,
        evidence.window_seconds or 4.0,
    )

    if quality is None:
        return CheckScore(name="rppg", status="unknown", risk=0.55, confidence=0.0, weight=0.18, reason="rPPG signal quality is unavailable")

    physiological = bpm is not None and 45.0 <= bpm <= 140.0
    stable = stability is None or stability >= 0.35
    passed = quality >= settings.rppg_min_signal_quality and physiological and stable
    stability_score = 0.5 if stability is None else stability
    risk = clamp01(1.0 - (0.50 * quality + 0.30 * (1.0 if physiological else 0.0) + 0.20 * stability_score))
    return CheckScore(
        name="rppg",
        status="passed" if passed else "failed",
        risk=risk,
        confidence=clamp01(0.70 * quality + 0.30 * stability_score),
        weight=0.18,
        reason="physiological pulse signal evaluated with sliding-window stability",
        details={
            "bpm": bpm,
            "signal_quality": quality,
            "stability": stability,
            "sample_count": len(evidence.samples),
            "duration_seconds": len(evidence.samples) / evidence.sample_rate_hz if evidence.sample_rate_hz else None,
            "window_seconds": evidence.window_seconds,
            "detector": evidence.detector,
        },
    )


def _score_python_rppg(evidence: RppgEvidence) -> CheckScore:
    sqi = evidence.signal_quality
    bpm = evidence.bpm
    details = {
        "bpm": bpm,
        "signal_quality": sqi,
        "latency": evidence.latency,
        "hrv": evidence.hrv,
        "detector": evidence.detector,
    }
    if sqi is None:
        return CheckScore(
            name="rppg",
            status="unknown",
            risk=0.55,
            confidence=0.0,
            weight=0.18,
            reason="rPPG signal quality is unavailable",
            details=details,
        )
    if sqi < 0.35:
        return CheckScore(
            name="rppg",
            status="unknown",
            risk=0.60,
            confidence=clamp01(sqi),
            weight=0.18,
            reason="rPPG signal quality is too low",
            details=details,
        )

    physiological = bpm is not None and 45.0 <= bpm <= 140.0
    comfort = bpm is not None and 55.0 <= bpm <= 115.0
    min_sqi = 0.50 if comfort else 0.65
    passed = physiological and sqi >= min_sqi
    risk = clamp01(1.0 - (0.70 * sqi + 0.30 * (1.0 if physiological else 0.0)))
    return CheckScore(
        name="rppg",
        status="passed" if passed else "failed",
        risk=risk,
        confidence=clamp01(sqi),
        weight=0.18,
        reason="rPPG model heart-rate and signal-quality output evaluated",
        details=details,
    )


def score_gesture(evidence: GestureEvidence | None, challenge: ChallengePlan | None) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return _skipped("gesture", 0.15)
    # The session challenge is authoritative: the client must not be able to
    # redefine which gesture was expected. Client value is a fallback for the
    # challenge-less direct scoring endpoint only.
    expected = _challenge_payload(challenge, "gesture", "expected_action")
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


def score_audio(evidence: AudioEvidence | None, challenge: ChallengePlan | None) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return _skipped("audio", 0.20)
    # The session challenge is authoritative: the client must not be able to
    # substitute its own "expected" phrase for the random one it was issued.
    expected_phrase = _challenge_payload(challenge, "audio_phrase", "phrase")
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
    if expected_phrase:
        phrase_ratio = levenshtein_ratio(expected_phrase, evidence.phrase_transcribed)
    phrase_ok = phrase_ratio is None or phrase_ratio >= 0.78
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
            "ai_probability": evidence.ai_probability,
            "speaker_match_probability": evidence.speaker_match_probability,
            "duration_seconds": evidence.duration_seconds,
        },
    )


def _challenge_luma(challenge: ChallengePlan | None) -> list[float]:
    if challenge is None:
        return []
    sequence = _challenge_payload(challenge, "active_light", "luma_sequence")
    if not isinstance(sequence, list):
        return []
    return [float(value) for value in sequence]


def _challenge_payload(challenge: ChallengePlan | None, step_type: str, key: str):
    if challenge is None:
        return None
    for step in challenge.steps:
        if step.type == step_type:
            return step.payload.get(key)
    return None


def _estimate_signal_quality(samples: list[float]) -> float | None:
    if len(samples) < 10:
        return None
    mean = sum(samples) / len(samples)
    centered = [value - mean for value in samples]
    peak_to_peak = max(centered) - min(centered)
    noise = sum(abs(centered[index] - centered[index - 1]) for index in range(1, len(centered))) / (len(centered) - 1)
    return clamp01(peak_to_peak / (peak_to_peak + noise + 1e-6))


def _estimate_bpm(samples: list[float], sample_rate_hz: float) -> float | None:
    if len(samples) < 20:
        return None
    mean = sum(samples) / len(samples)
    centered = [value - mean for value in samples]
    crossings = 0
    for previous, current in zip(centered, centered[1:]):
        if previous <= 0.0 < current:
            crossings += 1
    duration_minutes = len(samples) / sample_rate_hz / 60.0
    if duration_minutes <= 0.0:
        return None
    return crossings / duration_minutes


def _estimate_window_stability(samples: list[float], sample_rate_hz: float, window_seconds: float) -> float | None:
    window_size = max(8, int(round(sample_rate_hz * window_seconds)))
    if len(samples) < window_size * 2:
        return None

    stride = max(1, window_size // 2)
    qualities = []
    for start in range(0, len(samples) - window_size + 1, stride):
        quality = _estimate_signal_quality(samples[start : start + window_size])
        if quality is not None:
            qualities.append(quality)
    if len(qualities) < 2:
        return None

    mean_quality = sum(qualities) / len(qualities)
    max_jump = max(abs(current - previous) for previous, current in zip(qualities, qualities[1:]))
    # High-quality windows with low jumps mean the physiological signal was not abruptly changing.
    return clamp01(mean_quality * (1.0 - max_jump))


def _skipped(name: str, weight: float) -> CheckScore:
    return CheckScore(
        name=name,
        status="skipped",
        risk=0.50,
        confidence=0.0,
        weight=weight,
        reason="check skipped in test mode",
    )
