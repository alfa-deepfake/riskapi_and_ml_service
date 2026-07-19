from __future__ import annotations

from ml_service.api.schemas import CheckScore, ClassifierEvidence
from ml_service.core.checks._common import skipped
from ml_service.core.math_utils import clamp01


def score_classifier(evidence: ClassifierEvidence | None) -> CheckScore:
    if evidence is not None and evidence.skipped:
        return skipped("classifier", 0.25)
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
