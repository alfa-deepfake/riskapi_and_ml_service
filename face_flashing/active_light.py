from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_flashing.challenges import LUMA_WEIGHTS, Challenge

_EPS = 1e-6


@dataclass
class LightPair:
    background_challenge: Challenge
    background_rgb: np.ndarray  # face crop, HxWx3 uint8
    lighting_challenge: Challenge
    lighting_rgb: np.ndarray


@dataclass
class PairMetrics:
    pair_index: int
    expected_delta_luma: float  # screen-side luma change, [-255, 255]
    response_magnitude: float  # observed mean face luma change
    response_snr: float
    spatial_contrast: float
    center_error: float
    color_cosine: float


@dataclass
class ActiveLightResult:
    score: float
    pair_count: int
    zero_lag_correlation: float
    best_correlation: float
    best_lag: int
    median_contrast: float
    median_center_error: float
    median_response_snr: float
    median_response_magnitude: float
    mean_color_cosine: float


class ActiveLightLivenessVerifier:
    """Checks that the face actually reflects the randomized screen flashes.

    Per pair: the face luma must move in the flash direction (SNR, magnitude),
    the reflected color must match the screen color delta (cosine), and the
    response must have spatial structure (a flat global gain looks like
    auto-exposure, not like light on a 3D face). Across pairs: the observed
    response sequence must correlate with the random challenge sequence —
    that is the anti-replay core.
    """

    def verify(self, pairs: list[LightPair]) -> ActiveLightResult:
        metrics = [self._pair_metrics(pair) for pair in pairs]
        metrics = [metric for metric in metrics if metric is not None]
        if not metrics:
            return ActiveLightResult(
                score=0.0,
                pair_count=0,
                zero_lag_correlation=0.0,
                best_correlation=0.0,
                best_lag=0,
                median_contrast=0.0,
                median_center_error=1.0,
                median_response_snr=0.0,
                median_response_magnitude=0.0,
                mean_color_cosine=0.0,
            )

        metrics.sort(key=lambda metric: metric.pair_index)
        expected = np.array([metric.expected_delta_luma for metric in metrics], dtype=np.float64)
        observed = np.array([metric.response_magnitude for metric in metrics], dtype=np.float64)
        zero_lag = _pearson(expected, observed)
        best_corr, best_lag = zero_lag, 0
        for lag in (-1, 1):
            lagged = _pearson(expected[max(0, lag) : len(expected) + min(0, lag)],
                              observed[max(0, -lag) : len(observed) + min(0, -lag)])
            if lagged > best_corr:
                best_corr, best_lag = lagged, lag

        median_snr = float(np.median([metric.response_snr for metric in metrics]))
        median_contrast = float(np.median([metric.spatial_contrast for metric in metrics]))
        median_center_error = float(np.median([metric.center_error for metric in metrics]))
        median_magnitude = float(np.median([abs(metric.response_magnitude) for metric in metrics]))
        mean_cosine = float(np.mean([metric.color_cosine for metric in metrics]))

        score = float(
            np.clip(
                0.45 * max(0.0, zero_lag)
                + 0.20 * min(1.0, median_snr / 0.30)
                + 0.20 * max(0.0, mean_cosine)
                + 0.15 * min(1.0, median_contrast / 0.10),
                0.0,
                1.0,
            )
        )
        return ActiveLightResult(
            score=score,
            pair_count=len(metrics),
            zero_lag_correlation=zero_lag,
            best_correlation=best_corr,
            best_lag=best_lag,
            median_contrast=median_contrast,
            median_center_error=median_center_error,
            median_response_snr=median_snr,
            median_response_magnitude=median_magnitude,
            mean_color_cosine=mean_cosine,
        )

    def _pair_metrics(self, pair: LightPair) -> PairMetrics | None:
        background = np.asarray(pair.background_rgb, dtype=np.float32)
        lighting = np.asarray(pair.lighting_rgb, dtype=np.float32)
        if background.shape != lighting.shape or background.ndim != 3:
            return None

        expected_rgb = np.array(pair.lighting_challenge.mean_screen_rgb(), dtype=np.float64) - np.array(
            pair.background_challenge.mean_screen_rgb(), dtype=np.float64
        )
        expected_delta_luma = float(np.dot(LUMA_WEIGHTS, expected_rgb))

        diff_rgb = lighting - background
        diff_luma = diff_rgb @ np.asarray(LUMA_WEIGHTS, dtype=np.float32)
        magnitude = float(diff_luma.mean())
        snr = abs(magnitude) / (float(diff_luma.std()) + _EPS)

        # Response map aligned with the expected direction, so "contrast" and
        # "center" describe the reflection regardless of flash polarity.
        aligned = diff_luma * (1.0 if expected_delta_luma >= 0 else -1.0)
        p10, p90 = np.percentile(aligned, (10.0, 90.0))
        spatial_contrast = float(p90 - p10) / 255.0

        height, width = aligned.shape
        weights = np.clip(aligned, 0.0, None)
        total = float(weights.sum())
        if total > _EPS:
            ys, xs = np.mgrid[0:height, 0:width]
            center_y = float((weights * ys).sum() / total)
            center_x = float((weights * xs).sum() / total)
            half_diagonal = 0.5 * float(np.hypot(height, width))
            center_error = float(
                np.hypot(center_y - (height - 1) / 2.0, center_x - (width - 1) / 2.0) / half_diagonal
            )
        else:
            center_error = 1.0

        observed_rgb = diff_rgb.reshape(-1, 3).mean(axis=0).astype(np.float64)
        norm = np.linalg.norm(observed_rgb) * np.linalg.norm(expected_rgb)
        color_cosine = float(np.dot(observed_rgb, expected_rgb) / norm) if norm > _EPS else 0.0

        return PairMetrics(
            pair_index=pair.lighting_challenge.pair_index,
            expected_delta_luma=expected_delta_luma,
            response_magnitude=magnitude,
            response_snr=snr,
            spatial_contrast=spatial_contrast,
            center_error=center_error,
            color_cosine=color_cosine,
        )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    if float(np.std(left)) < _EPS or float(np.std(right)) < _EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def active_light_result_to_dict(result: ActiveLightResult) -> dict:
    return {
        "score": result.score,
        "pair_count": result.pair_count,
        "temporal": {
            "zero_lag_correlation": result.zero_lag_correlation,
            "best_correlation": result.best_correlation,
            "best_lag": result.best_lag,
        },
        "spatial": {
            "median_contrast": result.median_contrast,
            "median_center_error": result.median_center_error,
        },
        "median_response_snr": result.median_response_snr,
        "median_response_magnitude": result.median_response_magnitude,
        "mean_color_cosine": result.mean_color_cosine,
    }
