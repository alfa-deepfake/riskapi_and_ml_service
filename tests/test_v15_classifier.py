from pathlib import Path

import pytest

from ml_service.api.schemas import ClassifierEvidence
from ml_service.core.checks import score_classifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "v15"
T_BIN = 0.5940805446666666  # models/v15/v15_blend_config.json


def test_classifier_fails_at_blend_threshold():
    check = score_classifier(ClassifierEvidence(fake_probability=0.60, threshold=T_BIN))
    assert check.status == "failed"
    assert check.details["threshold"] == T_BIN


def test_classifier_passes_below_blend_threshold():
    check = score_classifier(ClassifierEvidence(fake_probability=0.55, threshold=T_BIN))
    assert check.status == "passed"


def test_restored_condition_is_rejected_even_when_score_is_low():
    check = score_classifier(
        ClassifierEvidence(fake_probability=0.20, threshold=T_BIN, condition="restored")
    )
    assert check.status == "failed"
    assert check.risk >= 0.85
    assert "restoration" in check.reason


def test_low_info_fake_verdict_is_withheld():
    # A small/upscaled face fabricates the fake signature — never confidently
    # accuse on it (the v15 asymmetric gate).
    check = score_classifier(
        ClassifierEvidence(fake_probability=0.90, threshold=T_BIN, low_info=True)
    )
    assert check.status == "unknown"
    assert check.details["low_info"] is True


def test_low_info_real_verdict_passes_annotated():
    check = score_classifier(
        ClassifierEvidence(fake_probability=0.10, threshold=T_BIN, low_info=True)
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
