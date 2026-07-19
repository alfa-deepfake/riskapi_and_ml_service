import pytest

np = pytest.importorskip("numpy")

from face_flashing.active_light import ActiveLightLivenessVerifier, LightPair, active_light_result_to_dict
from face_flashing.challenges import Challenge


def _challenge(kind, background, lighting=None, pair_index=0):
    return Challenge.from_dict(
        {
            "kind": kind,
            "background_rgb": list(background),
            "lighting_rgb": list(lighting) if lighting else None,
            "stripe_top": 0 if lighting else None,
            "stripe_bottom": 720 if lighting else None,
            "width": 1280,
            "height": 720,
            "pair_index": pair_index,
        }
    )


def _base_face(rng):
    face = rng.normal(120.0, 12.0, size=(64, 64, 3))
    return np.clip(face, 0, 255)


def _make_pairs(responsive: bool, n_pairs: int = 8):
    rng = np.random.default_rng(7)
    directions = [1 if i % 2 == 0 else -1 for i in range(n_pairs)]  # white/black alternation
    # Reflection is strongest around the face center — light falls off radially.
    ys, xs = np.mgrid[0:64, 0:64]
    falloff = np.exp(-(((ys - 32) / 30.0) ** 2 + ((xs - 32) / 30.0) ** 2))
    pairs = []
    for index, direction in enumerate(directions):
        background_color = (0, 0, 0) if direction > 0 else (255, 255, 255)
        lighting_color = (255, 255, 255) if direction > 0 else (0, 0, 0)
        base = _base_face(rng)
        if responsive:
            response = direction * 40.0 * falloff[..., None]
        else:
            response = rng.normal(0.0, 1.0, size=base.shape)
        lit = np.clip(base + response, 0, 255)
        pairs.append(
            LightPair(
                background_challenge=_challenge("background", background_color, pair_index=index),
                background_rgb=base.astype(np.uint8),
                lighting_challenge=_challenge("lighting", background_color, lighting_color, pair_index=index),
                lighting_rgb=lit.astype(np.uint8),
            )
        )
    return pairs


def test_responsive_face_passes_all_thresholds():
    result = ActiveLightLivenessVerifier().verify(_make_pairs(responsive=True))
    # Thresholds mirror ml_service Settings defaults for score_active_light.
    assert result.pair_count == 8
    assert result.score >= 0.55
    assert result.zero_lag_correlation >= 0.65
    assert result.median_contrast >= 0.025
    assert result.median_response_snr >= 0.04
    assert result.mean_color_cosine >= 0.15
    assert result.median_center_error <= 0.5


def test_colored_flash_pairs_pass_thresholds():
    # Unnatural saturated flashes on black backgrounds: the expected luma
    # sequence varies by hue (magenta 105, cyan 172, yellow 226, white 255),
    # and the reflected tint must match the flash color.
    colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 255, 255)] * 2
    rng = np.random.default_rng(11)
    ys, xs = np.mgrid[0:64, 0:64]
    falloff = np.exp(-(((ys - 32) / 30.0) ** 2 + ((xs - 32) / 30.0) ** 2))
    pairs = []
    for index, color in enumerate(colors):
        base = _base_face(rng)
        response = 40.0 * falloff[..., None] * (np.asarray(color, dtype=np.float64) / 255.0)
        lit = np.clip(base + response, 0, 255)
        pairs.append(
            LightPair(
                background_challenge=_challenge("background", (0, 0, 0), pair_index=index),
                background_rgb=base.astype(np.uint8),
                lighting_challenge=_challenge("lighting", (0, 0, 0), color, pair_index=index),
                lighting_rgb=lit.astype(np.uint8),
            )
        )
    result = ActiveLightLivenessVerifier().verify(pairs)
    assert result.pair_count == len(colors)
    assert result.score >= 0.55
    assert result.zero_lag_correlation >= 0.65
    assert result.median_response_snr >= 0.04
    assert result.mean_color_cosine >= 0.15


def test_unresponsive_face_fails():
    result = ActiveLightLivenessVerifier().verify(_make_pairs(responsive=False))
    assert result.score < 0.4
    assert abs(result.zero_lag_correlation) < 0.65


def test_result_dict_matches_service_contract():
    result = ActiveLightLivenessVerifier().verify(_make_pairs(responsive=True))
    payload = active_light_result_to_dict(result)
    assert set(payload) == {
        "score",
        "pair_count",
        "temporal",
        "spatial",
        "median_response_snr",
        "median_response_magnitude",
        "mean_color_cosine",
    }
    assert set(payload["temporal"]) == {"zero_lag_correlation", "best_correlation", "best_lag"}
    assert set(payload["spatial"]) == {"median_contrast", "median_center_error"}


def test_empty_pairs_score_zero():
    result = ActiveLightLivenessVerifier().verify([])
    assert result.pair_count == 0
    assert result.score == 0.0


def test_constant_direction_gives_zero_correlation():
    pairs = _make_pairs(responsive=True)
    # Force every pair to flash the same way: constant expected sequence.
    same = [p for p in pairs if p.lighting_challenge.mean_screen_luma() > p.background_challenge.mean_screen_luma()]
    result = ActiveLightLivenessVerifier().verify(same)
    assert result.zero_lag_correlation == 0.0
