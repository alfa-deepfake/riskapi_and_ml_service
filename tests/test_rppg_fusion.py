import pytest

from ml_service.services.rppg_service import _fuse_candidates


def candidate(source, bpm, sqi):
    return {"source": source, "bpm": bpm, "sqi": sqi, "raw": {}}


def test_agreeing_estimators_boost_quality():
    fused = _fuse_candidates(
        [candidate("a", 71.0, 0.42), candidate("b", 74.0, 0.40), candidate("c", 76.0, 0.38)]
    )
    assert fused["agreeing"] == 3
    assert fused["bpm"] == 74.0
    assert fused["sqi"] == pytest.approx(0.42 + 0.08 * 2)


def test_disagreeing_estimators_fall_back_to_best_single():
    fused = _fuse_candidates([candidate("a", 60.0, 0.55), candidate("b", 120.0, 0.30)])
    assert fused["agreeing"] == 1
    assert fused["bpm"] == 60.0
    assert fused["sqi"] == 0.55  # no consensus, no bonus


def test_no_valid_candidates_keeps_best_sqi_for_diagnostics():
    fused = _fuse_candidates([candidate("a", None, 0.9), candidate("b", None, None)])
    assert fused["bpm"] is None
    assert fused["sqi"] == 0.9
    assert fused["agreeing"] == 0


def test_pos_agreement_rescues_a_weak_model_reading():
    # A dim-room clip: the learned model alone sits under the 0.5 pass floor,
    # but the classical POS estimate agreeing on the BPM lifts it over.
    fused = _fuse_candidates([candidate("model", 68.0, 0.44), candidate("pos", 66.0, 0.52)])
    assert fused["agreeing"] == 2
    assert fused["bpm"] == 67.0
    assert fused["sqi"] == pytest.approx(0.60)


def test_quality_bonus_is_capped_at_one():
    fused = _fuse_candidates(
        [candidate("a", 70.0, 0.99), candidate("b", 70.0, 0.98), candidate("c", 70.0, 0.97)]
    )
    assert fused["sqi"] == 1.0
