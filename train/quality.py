"""Quality features. Extracted separately from deepfake features so we can:
- concat them into the 28-feature set for XGBoost
- learn correlation with quality label
- diagnose whether quality alone predicts fake (bad — proves confusion)

All are cheap OpenCV ops except brisque; brisque is optional (skipped if opencv-contrib not installed).
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import median_abs_deviation


def _gray(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im


def blur_lap(im: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(im), cv2.CV_64F).var())


def blur_tenengrad(im: np.ndarray) -> float:
    g = _gray(im).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def blur_sml(im: np.ndarray) -> float:
    """Sum-Modified-Laplacian: |Lx| + |Ly|, more robust than variance to noise."""
    g = _gray(im).astype(np.float32)
    lx = cv2.filter2D(g, cv2.CV_32F, np.array([[-1, 2, -1]], dtype=np.float32))
    ly = cv2.filter2D(g, cv2.CV_32F, np.array([[-1], [2], [-1]], dtype=np.float32))
    return float((np.abs(lx) + np.abs(ly)).mean())


def noise_mad(im: np.ndarray) -> float:
    """Donoho-like noise estimate: MAD of wavelet HH1 sub-band, robust to structure."""
    g = _gray(im).astype(np.float32)
    # simple 1-level Haar: HH = quarter-image high-freq diagonal
    ll = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2]) / 4
    hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 4
    mad = median_abs_deviation(hh.ravel())
    return float(mad / 0.6745)  # convert to gaussian σ estimate


def contrast_rms(im: np.ndarray) -> float:
    return float(_gray(im).std())


def chroma_energy(im_rgb: np.ndarray) -> float:
    """Energy in the color (U/V) channels — low means chroma-subsampled."""
    if im_rgb.ndim != 3:
        return 0.0
    yuv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YUV)
    return float(yuv[..., 1:].std())


def jpeg_q_est(im_pil: Image.Image, fast: bool = True) -> float:
    """Estimate JPEG quality. Fast mode probes just 3 levels; full mode probes 8."""
    arr = np.asarray(im_pil.convert("RGB")).astype(np.int16)
    levels = (85, 60, 35) if fast else (95, 85, 75, 65, 55, 45, 35, 25)
    best_q, best_diff = 100, 1e9
    for q in levels:
        buf = io.BytesIO()
        im_pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        rt = np.asarray(Image.open(buf).convert("RGB")).astype(np.int16)
        diff = np.abs(arr - rt).mean()
        if diff < best_diff:
            best_diff, best_q = diff, q
        if diff < 0.5:
            return float(q)
    return float(best_q)


def dct_flatness(im: np.ndarray) -> float:
    """A cheap proxy for compression: mean AC energy in 8x8 DCT blocks.
    Highly compressed images have concentrated energy in low-freq DC → flatness ↓."""
    g = _gray(im).astype(np.float32) - 128
    h, w = g.shape
    h8, w8 = h - h % 8, w - w % 8
    g = g[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    d = np.stack([cv2.dct(b) for b in g])
    ac = d[:, 1:, 1:].reshape(len(d), -1)
    return float(np.abs(ac).mean())


try:  # opencv-contrib provides BRISQUE
    _brisque = cv2.quality.QualityBRISQUE_create(
        "/dev/null", "/dev/null"  # silently disabled unless config paths provided
    )
except Exception:
    _brisque = None


def brisque_score(im: np.ndarray) -> float:
    if _brisque is None:
        return float("nan")
    try:
        score = _brisque.compute(im)[0][0]
        return float(score)
    except Exception:
        return float("nan")


def quality_features(im_pil: Image.Image) -> dict:
    im = np.asarray(im_pil.convert("RGB"))
    return {
        "q_blur_lap":    blur_lap(im),
        "q_blur_sml":    blur_sml(im),
        "q_blur_teneng": blur_tenengrad(im),
        "q_noise_mad":   noise_mad(im),
        "q_contrast":    contrast_rms(im),
        "q_chroma":      chroma_energy(im),
        "q_dct_ac":      dct_flatness(im),
        "q_jpeg_est":    jpeg_q_est(im_pil),
        "q_brisque":     brisque_score(im),
    }


# ponytail: self-check on a synthetic gradient
def _demo() -> None:
    import numpy as np
    g = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    im = Image.fromarray(g)
    f = quality_features(im)
    assert len(f) == 9, f.keys()
    print("quality features:", {k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()})


if __name__ == "__main__":
    _demo()
