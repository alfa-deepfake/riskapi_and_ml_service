from __future__ import annotations

from ml_service.api.schemas import CheckScore, RppgEvidence
from ml_service.config import Settings
from ml_service.core.checks._common import skipped
from ml_service.core.math_utils import clamp01


def score_rppg(evidence: RppgEvidence | None, settings: Settings) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return skipped("rppg", 0.18)
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
    if evidence.ensemble:
        details["ensemble"] = evidence.ensemble
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
