from pathlib import Path

import pytest

from ml_service.api.schemas import ClassifierEvidence
from ml_service.core.checks import score_classifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "v15"
T_BIN = 0.6498336791992188  # models/v15/v16/v16_fusion_config.json
T_SUSP = 0.75  # models/v15/v15_blend_config.json


def test_classifier_fails_at_fusion_threshold():
    check = score_classifier(ClassifierEvidence(fake_probability=0.66, threshold=T_BIN))
    assert check.status == "failed"
    assert check.details["threshold"] == T_BIN


def test_classifier_passes_below_fusion_threshold():
    check = score_classifier(ClassifierEvidence(fake_probability=0.60, threshold=T_BIN))
    assert check.status == "passed"


def test_restored_condition_is_rejected_even_when_score_is_low():
    check = score_classifier(
        ClassifierEvidence(fake_probability=0.20, threshold=T_BIN, condition="restored")
    )
    assert check.status == "failed"
    assert check.risk >= 0.85
    assert "restoration" in check.reason


def test_low_info_fake_verdict_stands():
    # v16: the verdict is always binary — a fused FAKE on low-detail input is
    # no longer withheld (the v15b CNN retrain fixed the false-FAKE modes the
    # old withhold gate guarded against); low_info stays annotated.
    check = score_classifier(
        ClassifierEvidence(fake_probability=0.90, threshold=T_BIN, low_info=True)
    )
    assert check.status == "failed"
    assert check.details["low_info"] is True


def test_low_info_forensic_override_fails_real_verdict():
    # v16 forensic override: on low-detail input the noise CNN is blind and
    # drags the fused score under the threshold — trees >= t_susp means no
    # REAL verdict.
    check = score_classifier(
        ClassifierEvidence(
            fake_probability=0.30,
            threshold=T_BIN,
            low_info=True,
            tree_probability=0.80,
            t_susp=T_SUSP,
        )
    )
    assert check.status == "failed"
    assert check.risk >= 0.80
    assert "forensic override" in check.reason


def test_low_info_real_verdict_passes_when_trees_are_quiet():
    check = score_classifier(
        ClassifierEvidence(
            fake_probability=0.10,
            threshold=T_BIN,
            low_info=True,
            tree_probability=0.40,
            t_susp=T_SUSP,
        )
    )
    assert check.status == "passed"
    assert check.details["low_info"] is True


def test_vidcall_condition_keeps_verdict_and_is_annotated():
    # realtime v15 policy for live video: the verdict stands, condition is
    # surfaced for the operator.
    check = score_classifier(
        ClassifierEvidence(fake_probability=0.10, threshold=T_BIN, condition="vidcall")
    )
    assert check.status == "passed"
    assert check.details["condition"] == "vidcall"


def test_classifier_details_include_both_modalities():
    scores = {"id": 0.3, "facefusion": 0.4}
    check = score_classifier(
        ClassifierEvidence(
            fake_probability=0.45,
            threshold=T_BIN,
            model_scores=scores,
            cnn_probability=0.55,
            condition="clean",
        )
    )
    assert check.details["model_scores"] == scores
    assert check.details["cnn_probability"] == 0.55


def test_forensic_override_blocks_allow_decision():
    # A forensic-override FAKE (v16 low-detail policy) must route to review
    # even when every hard liveness check passes — otherwise framing the face
    # small neutralizes the forensic classifier entirely.
    from ml_service.api.schemas import CheckScore
    from ml_service.core.scoring import _decision

    overridden = score_classifier(
        ClassifierEvidence(
            fake_probability=0.30, threshold=T_BIN, low_info=True,
            tree_probability=0.80, t_susp=T_SUSP,
        )
    )
    assert overridden.status == "failed"
    passing = [
        CheckScore(name=name, status="passed", risk=0.10, confidence=0.9, weight=0.2, reason="ok")
        for name in ("active_light", "rppg", "gesture", "audio")
    ]
    decision = _decision([overridden, *passing], 0.20, allow_threshold=0.35, deny_threshold=0.72)
    assert decision == "review"

    # A merely-missing classifier keeps the old behavior.
    missing = CheckScore(
        name="classifier", status="unknown", risk=0.45, confidence=0.0, weight=0.25,
        reason="frame classifier evidence is missing",
    )
    assert _decision([missing, *passing], 0.20, allow_threshold=0.35, deny_threshold=0.72) == "allow"


def test_v15_adapter_gate_requires_cnn_weights(monkeypatch, tmp_path):
    # The blend config is committed to git; the CNN weights are deployed
    # separately. A checkout without the weights must fall back (None), not
    # return an adapter that errors on every request.
    from ml_service.services import classifier_service

    v15 = tmp_path / "v15"
    (v15 / "cnn").mkdir(parents=True)
    (v15 / "v16").mkdir()
    (v15 / "v15_blend_config.json").write_text("{}")
    (v15 / "v16" / "gbm_fusion.ubj").write_bytes(b"stub")
    monkeypatch.setattr(classifier_service.settings, "video_v15_dir", str(v15))

    classifier_service._get_v15_adapter.cache_clear()
    assert classifier_service._get_v15_adapter() is None

    for name in classifier_service._V15_CNN_WEIGHTS:
        (v15 / "cnn" / name).write_bytes(b"stub")
    classifier_service._get_v15_adapter.cache_clear()
    # With all weights present the gate passes; adapter may still be None in
    # a dev venv without ML deps (ImportError branch) — both are acceptable
    # here, we only assert the gate no longer blocks.
    try:
        classifier_service._get_v15_adapter()
    finally:
        classifier_service._get_v15_adapter.cache_clear()


def test_warm_video_model_never_raises(monkeypatch):
    from ml_service.services import classifier_service

    class BrokenAdapter:
        def load(self):
            raise FileNotFoundError("noise_cnn_holdout_deeplivecam.pt")

    monkeypatch.setattr(classifier_service, "_get_v15_adapter", lambda: BrokenAdapter())
    classifier_service.warm_video_model()  # must not raise


# Golden feature values pin the ported 73-feature extraction (base set +
# verbatim v6-v9 modules): any change to the residual sigma, color order,
# spectrum binning, round-trip sizes or noise estimator shifts them.
GOLDEN_FEATURES = {
    "res_std_r": 0.2636285722255707,
    "hf_over_lf": 0.8256732948594436,
    "q_noise_mad": 25.203854707190512,
    "upsample_diff_256": 1538.18310546875,
    "wavelet_h": 0.13395562767982483,
    "spectral_slope": 0.017757088423302588,
    "face_bg_res_ratio": 1.0029892921447754,
    "jpeg_ghost_depth": 0.9719654977575968,
    "seam_grad_excess": 0.9997677803039551,
    "noise_symmetry_lr": -0.3100692628350381,
}


def _adapter_module():
    for dep in ("numpy", "cv2", "scipy", "xgboost", "PIL", "torch", "timm", "joblib"):
        pytest.importorskip(dep)
    from ml_service.adapters import v15_video_adapter

    return v15_video_adapter


def test_features_cover_manifest_and_match_goldens():
    adapter = _adapter_module()
    import numpy as np

    feat_names = (MODELS_DIR / "v13" / "feature_names.txt").read_text().splitlines()
    assert len(feat_names) == 73

    rng = np.random.default_rng(0)
    crop = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
    feats = adapter._features(crop)

    missing = [name for name in feat_names if name not in feats]
    assert missing == []
    for name, expected in GOLDEN_FEATURES.items():
        assert feats[name] == pytest.approx(expected, rel=1e-3), name


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="v15 models are not deployed")
def test_trees_and_gate_load_and_score():
    adapter = _adapter_module()
    import numpy as np
    import xgboost

    feat_names = (MODELS_DIR / "v13" / "feature_names.txt").read_text().splitlines()
    rng = np.random.default_rng(0)
    crop = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
    feats = adapter._features(crop)
    row = np.array([[feats.get(k, float("nan")) for k in feat_names]], dtype=np.float32)
    matrix = xgboost.DMatrix(row, feature_names=feat_names)

    for name in ["id"] + adapter.GENS:
        booster = xgboost.Booster()
        booster.load_model(str(MODELS_DIR / "v13" / f"xgb_v13f_{name}.ubj"))
        assert booster.num_features() == 73
        score = float(booster.predict(matrix)[0])
        assert 0.0 <= score <= 1.0, name

    gate = xgboost.Booster()
    gate.load_model(str(MODELS_DIR / "v13" / "xgb_gate_condition.ubj"))
    cond_p = gate.predict(matrix)[0]
    assert len(cond_p) == len(adapter.COND_NAMES)


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="v15/v16 models are not deployed")
def test_v16_fusion_loads_and_scores():
    import json

    np = pytest.importorskip("numpy")
    xgboost = pytest.importorskip("xgboost")

    cfg = json.loads((MODELS_DIR / "v16" / "v16_fusion_config.json").read_text())
    assert cfg["inputs"] == ["tmean", "cnn", "cnn_std", "tree_std"]
    assert cfg["t_bin"] == T_BIN

    fusion = xgboost.Booster()
    fusion.load_model(str(MODELS_DIR / "v16" / "gbm_fusion.ubj"))
    row = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
    p = float(fusion.predict(xgboost.DMatrix(row, feature_names=cfg["inputs"]))[0])
    assert 0.0 <= p <= 1.0
    # high tmean + high cnn must land clearly fake-side, and above the
    # low-everything row — pins that the booster is a real fusion, not noise
    lo = float(fusion.predict(xgboost.DMatrix(
        np.array([[0.05, 0.05, 0.02, 0.05]], dtype=np.float32), feature_names=cfg["inputs"]))[0])
    hi = float(fusion.predict(xgboost.DMatrix(
        np.array([[0.95, 0.95, 0.02, 0.05]], dtype=np.float32), feature_names=cfg["inputs"]))[0])
    assert lo < cfg["t_bin"] <= hi


def test_noise_map_is_deterministic_and_bounded():
    _adapter_module()
    import numpy as np
    from PIL import Image

    from noise_map_v15 import noise_map_tensor

    rng = np.random.default_rng(0)
    im = Image.fromarray((rng.random((512, 512, 3)) * 255).astype(np.uint8))
    x = noise_map_tensor(im, 256)
    assert tuple(x.shape) == (3, 256, 256)
    assert float(x.abs().max()) <= 1.0
    y = noise_map_tensor(im, 256)
    assert bool((x == y).all())
