from pathlib import Path

import pytest

from ml_service.api.schemas import ClassifierEvidence
from ml_service.core.checks import score_classifier
from ml_service.core.math_utils import mean_without_lone_dissenter

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "xgb"


def test_lone_fake_dissenter_is_dropped():
    scores = {"a": 0.9, "b": 0.10, "c": 0.20, "d": 0.15, "e": 0.10, "f": 0.05}
    mean, dropped = mean_without_lone_dissenter(scores, 0.45)
    assert dropped == ["a"]
    assert mean == pytest.approx((0.10 + 0.20 + 0.15 + 0.10 + 0.05) / 5)


def test_lone_real_dissenter_is_dropped():
    scores = {"a": 0.05, "b": 0.80, "c": 0.90, "d": 0.75, "e": 0.60, "f": 0.95}
    mean, dropped = mean_without_lone_dissenter(scores, 0.45)
    assert dropped == ["a"]
    assert mean == pytest.approx((0.80 + 0.90 + 0.75 + 0.60 + 0.95) / 5)


def test_split_vote_keeps_all_models():
    scores = {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.2, "e": 0.7, "f": 0.6}
    mean, dropped = mean_without_lone_dissenter(scores, 0.45)
    assert dropped == []
    assert mean == pytest.approx(sum(scores.values()) / 6)


def test_tie_vote_keeps_all_models():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.1, "e": 0.2, "f": 0.3}
    _, dropped = mean_without_lone_dissenter(scores, 0.45)
    assert dropped == []


def test_unanimous_vote_keeps_all_models():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.9, "f": 0.5}
    mean, dropped = mean_without_lone_dissenter(scores, 0.45)
    assert dropped == []
    assert mean == pytest.approx(sum(scores.values()) / 6)


def test_two_model_disagreement_keeps_both():
    _, dropped = mean_without_lone_dissenter({"a": 0.9, "b": 0.1}, 0.45)
    assert dropped == []


def test_empty_scores_raise():
    with pytest.raises(ValueError):
        mean_without_lone_dissenter({}, 0.45)


def test_classifier_fails_at_evidence_threshold():
    check = score_classifier(ClassifierEvidence(fake_probability=0.50, threshold=0.45))
    assert check.status == "failed"
    assert check.details["threshold"] == 0.45


def test_classifier_passes_below_evidence_threshold():
    check = score_classifier(ClassifierEvidence(fake_probability=0.40, threshold=0.45))
    assert check.status == "passed"


def test_classifier_keeps_070_cutoff_without_threshold():
    check = score_classifier(ClassifierEvidence(fake_probability=0.50))
    assert check.status == "passed"
    check = score_classifier(ClassifierEvidence(fake_probability=0.70))
    assert check.status == "failed"


def test_classifier_details_include_model_scores():
    scores = {"id": 0.9, "loo_facefusion": 0.8}
    check = score_classifier(
        ClassifierEvidence(fake_probability=0.85, threshold=0.45, model_scores=scores, dropped_models=[])
    )
    assert check.details["model_scores"] == scores
    assert check.details["dropped_models"] == []


# Golden feature values pin the ported extraction pipeline: any change to the
# residual sigma, color order, spectrum binning, or noise estimator shifts them.
GOLDEN_FEATURES = {
    "res_std_r": 0.2636285722255707,
    "res_kurt_g": -1.0802498780854162,
    "res_corr_rg": 0.00031346922622769346,
    "hf_over_lf": 0.8256732948594436,
    "sp_bin_3": 1.8115601945826005,
    "q_noise_mad": 25.203854707190512,
    "q_contrast": 49.23039245605469,
    "q_jpeg_est": 85.0,
    "mean_r": 0.49874386191368103,
}


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="XGB models are not deployed")
def test_features_and_models_end_to_end():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    pytest.importorskip("xgboost")
    pytest.importorskip("mediapipe")
    pytest.importorskip("PIL")
    import xgboost

    from ml_service.adapters.xgb_video_adapter import MODEL_FILES, _features

    feat_names = (MODELS_DIR / "feature_names.txt").read_text().splitlines()
    rng = np.random.default_rng(0)
    crop = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
    feats = _features(crop)

    missing = [name for name in feat_names if name not in feats]
    assert missing == []
    for name, expected in GOLDEN_FEATURES.items():
        assert feats[name] == pytest.approx(expected, rel=1e-3), name

    row = np.asarray([[feats[name] for name in feat_names]], dtype=np.float32)
    matrix = xgboost.DMatrix(row)
    for name, filename in MODEL_FILES:
        path = MODELS_DIR / filename
        if not path.exists():
            continue
        booster = xgboost.Booster()
        booster.load_model(str(path))
        score = float(booster.predict(matrix)[0])
        assert 0.0 <= score <= 1.0, name
