import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from ml_service.services.rppg_pos import estimate_from_traces  # noqa: E402


def synthetic_trace(bpm, *, seconds=18.0, fps=30.0, amplitude=0.5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / fps)
    pulse = np.sin(2.0 * np.pi * (bpm / 60.0) * t)
    # rPPG-like chromaticity: pulse modulates green strongest, red/blue weaker.
    rgb = np.stack(
        [
            90.0 + 0.3 * amplitude * pulse + noise * rng.standard_normal(t.size),
            110.0 + 1.0 * amplitude * pulse + noise * rng.standard_normal(t.size),
            80.0 + 0.2 * amplitude * pulse + noise * rng.standard_normal(t.size),
        ],
        axis=1,
    )
    return t, rgb


def test_pos_recovers_bpm_with_region_consensus():
    # noise=0.35 models a dim room after spatial pooling: sensor noise of a few
    # luma units per pixel, averaged over a few thousand skin pixels.
    t, forehead = synthetic_trace(72, noise=0.35, seed=1)
    _, left_cheek = synthetic_trace(72, noise=0.35, seed=2)
    _, right_cheek = synthetic_trace(72, noise=0.35, seed=3)

    result = estimate_from_traces(
        {"forehead": forehead, "left_cheek": left_cheek, "right_cheek": right_cheek}, t
    )

    assert result is not None
    assert result.agreeing >= 2
    assert result.bpm == pytest.approx(72.0, abs=3.0)
    assert result.quality >= 0.5


def test_pos_noise_only_gives_no_confident_pulse():
    t, a = synthetic_trace(0, amplitude=0.0, noise=1.0, seed=4)
    _, b = synthetic_trace(0, amplitude=0.0, noise=1.0, seed=5)
    _, c = synthetic_trace(0, amplitude=0.0, noise=1.0, seed=6)

    result = estimate_from_traces({"forehead": a, "left_cheek": b, "right_cheek": c}, t)

    assert result is not None
    assert result.bpm is None or result.quality < 0.5


def test_pos_single_region_signal_is_never_pass_capable_alone():
    # One periodic patch could be a rendering artifact — without a second
    # agreeing region the quality must stay under the 0.5 pass floor.
    t, forehead = synthetic_trace(72, noise=0.1, seed=1)
    _, left_cheek = synthetic_trace(0, amplitude=0.0, noise=1.0, seed=5)
    _, right_cheek = synthetic_trace(0, amplitude=0.0, noise=1.0, seed=6)

    result = estimate_from_traces(
        {"forehead": forehead, "left_cheek": left_cheek, "right_cheek": right_cheek}, t
    )

    assert result is not None
    assert result.agreeing <= 1
    assert result.quality <= 0.45


def test_pos_rejects_too_short_traces():
    t, trace = synthetic_trace(72, seconds=5.0)
    assert estimate_from_traces({"forehead": trace}, t) is None
