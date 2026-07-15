"""Frame-level XGBoost forensic-feature ensemble for uploaded challenge video.

Ports the training repo's ``infer.py``: 6 XGBoost models (one in-distribution,
five leave-one-generator-out) score a face crop on 37 forensic/quality
features. Per model the score is averaged across sampled frames, then the
per-model scores are combined with :func:`mean_without_lone_dissenter` — if a
single model votes against all the others it is ignored.

Heavy imports live at module level; the module itself is imported lazily by
``classifier_service`` and skipped when ML dependencies are absent.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import xgboost
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, median_abs_deviation, skew

from ml_service.core.math_utils import mean_without_lone_dissenter

# Ordered names → filenames, mirrors the training repo's infer.py. Missing
# files are skipped so a partial model drop still scores.
MODEL_FILES = [
    ("id", "xgb_id.ubj"),
    ("loo_deeplivecam", "xgb_loo_deeplivecam.ubj"),
    ("loo_facefusion", "xgb_loo_facefusion.ubj"),
    ("loo_visomaster", "xgb_loo_visomaster.ubj"),
    ("loo_inswapper128", "xgb_loo_inswapper128.ubj"),
    ("loo_reswapper", "xgb_loo_reswapper.ubj"),
]

FACE_CROP_SIZE = 512
FFT_SIZE = 256
RESIDUAL_SIGMA = 1.2
CROP_MARGIN = 1.6
MIN_CROP_SIDE = 32


class XgbVideoEnsembleAdapter:
    def __init__(
        self,
        *,
        models_dir: Path,
        threshold: float = 0.45,
        max_inferences: int = 12,
        infer_every: int = 5,
    ) -> None:
        self.models_dir = models_dir
        self.threshold = threshold
        self.max_inferences = max(1, max_inferences)
        self.infer_every = max(1, infer_every)
        self._loaded: tuple[dict[str, xgboost.Booster], list[str]] | None = None

    def predict(self, video_path: Path) -> dict[str, Any]:
        boosters, feat_names = self._load()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open uploaded video: {video_path}")

        frame_count = 0
        examined = 0
        rows: list[list[float]] = []
        try:
            with mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            ) as detector:
                while len(rows) < self.max_inferences:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_count += 1
                    if frame_count % self.infer_every != 0:
                        continue
                    # Cap the scan so a long face-less upload cannot hold a
                    # threadpool thread by running detection on every frame.
                    if examined >= self.max_inferences * 8:
                        break
                    examined += 1
                    crop = _face_crop(frame, detector)
                    if crop is None:
                        continue
                    feats = _features(crop)
                    rows.append([feats.get(name, float("nan")) for name in feat_names])
        finally:
            cap.release()

        result: dict[str, Any] = {
            "threshold": self.threshold,
            "model_name": f"xgb-forensic-ensemble-{len(boosters)}",
            "frame_count": frame_count,
            "face_present": bool(rows),
            "face_confidence": len(rows) / examined if examined else 0.0,
            "sampled_frames": len(rows),
        }
        if not rows:
            result.update({"fake_probability": None, "confidence": 0.0})
            return result

        matrix = xgboost.DMatrix(np.asarray(rows, dtype=np.float32))
        model_scores = {name: float(np.mean(booster.predict(matrix))) for name, booster in boosters.items()}
        fake_probability, dropped = mean_without_lone_dissenter(model_scores, self.threshold)
        confidence = fake_probability if fake_probability >= self.threshold else 1.0 - fake_probability
        result.update(
            {
                "fake_probability": float(fake_probability),
                "confidence": float(confidence),
                "model_scores": model_scores,
                "dropped_models": dropped,
            }
        )
        return result

    def _load(self) -> tuple[dict[str, xgboost.Booster], list[str]]:
        if self._loaded is None:
            names_path = self.models_dir / "feature_names.txt"
            if not names_path.exists():
                raise FileNotFoundError(f"Feature names not found: {names_path}")
            feat_names = names_path.read_text().splitlines()
            boosters: dict[str, xgboost.Booster] = {}
            for name, filename in MODEL_FILES:
                path = self.models_dir / filename
                if not path.exists():
                    continue
                booster = xgboost.Booster()
                booster.load_model(str(path))
                boosters[name] = booster
            if not boosters:
                raise FileNotFoundError(f"No XGBoost models found in {self.models_dir}")
            self._loaded = (boosters, feat_names)
        return self._loaded


def _face_crop(frame_bgr: np.ndarray, detector: Any) -> np.ndarray | None:
    """Largest-face square RGB crop resized to FACE_CROP_SIZE.

    ponytail: plain bbox crop with a fixed margin; the training-time aligner
    (train/face_crop.py) is not in this repo. Swap this function for the real
    aligner when it lands to remove train/serve skew.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if not results.detections:
        return None
    box = max(
        results.detections,
        key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height,
    ).location_data.relative_bounding_box
    height, width = rgb.shape[:2]
    center_x = (box.xmin + box.width / 2.0) * width
    center_y = (box.ymin + box.height / 2.0) * height
    side = int(round(max(box.width * width, box.height * height) * CROP_MARGIN))
    side = min(side, width, height)
    if side < MIN_CROP_SIDE:
        return None
    x0 = max(0, min(int(round(center_x - side / 2.0)), width - side))
    y0 = max(0, min(int(round(center_y - side / 2.0)), height - side))
    crop = rgb[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=cv2.INTER_AREA)


def _features(crop_rgb: np.ndarray) -> dict[str, float]:
    """37 forensic + quality features; ported verbatim from infer.py."""
    rgb = np.asarray(crop_rgb, dtype=np.float32) / 255.0
    im_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    feats = _forensic_features(rgb)
    feats.update(_quality_features(im_pil))
    feats["jpeg_q"] = 0  # no round-trip augmentation at inference
    return feats


def _residual(rgb: np.ndarray) -> np.ndarray:
    return np.stack([rgb[..., i] - gaussian_filter(rgb[..., i], RESIDUAL_SIGMA) for i in range(3)], axis=-1)


def _radial_spectrum(gray: np.ndarray) -> np.ndarray:
    src = Image.fromarray((gray * 255).astype(np.uint8)).resize((FFT_SIZE, FFT_SIZE), Image.BILINEAR)
    g = np.asarray(src, dtype=np.float32) / 255.0
    g = g - g.mean()
    win = np.hanning(FFT_SIZE)[:, None] * np.hanning(FFT_SIZE)[None, :]
    F = np.fft.fftshift(np.fft.fft2(g * win))
    mag = np.log1p(np.abs(F))
    cy, cx = FFT_SIZE // 2, FFT_SIZE // 2
    y, x = np.indices(mag.shape)
    r = np.round(np.hypot(x - cx, y - cy)).astype(np.int32)
    nbin = FFT_SIZE // 2
    tbin = np.bincount(r.ravel(), mag.ravel(), minlength=nbin + 1)[:nbin]
    nr = np.bincount(r.ravel(), minlength=nbin + 1)[:nbin]
    return tbin / np.maximum(nr, 1)


def _forensic_features(rgb: np.ndarray) -> dict[str, float]:
    gray = rgb.mean(-1)
    res = _residual(rgb)
    feats: dict[str, float] = {}
    for i, ch in enumerate("rgb"):
        r = res[..., i].ravel()
        feats[f"res_std_{ch}"] = float(r.std())
        feats[f"res_kurt_{ch}"] = float(kurtosis(r, fisher=True))
        feats[f"res_skew_{ch}"] = float(skew(r))
    r = res.reshape(-1, 3)
    r = r - r.mean(0)
    cov = (r.T @ r) / max(1, r.shape[0] - 1)
    diag = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(diag, diag)
    feats["res_corr_rg"] = float(corr[0, 1])
    feats["res_corr_rb"] = float(corr[0, 2])
    feats["res_corr_gb"] = float(corr[1, 2])
    for i, ch in enumerate("rgb"):
        feats[f"mean_{ch}"] = float(rgb[..., i].mean())
        feats[f"std_{ch}"] = float(rgb[..., i].std())
    feats["lap_var"] = float(res.mean(-1).var())
    sp = _radial_spectrum(gray)
    feats["hf_over_lf"] = float(sp[len(sp) // 2 :].mean() / max(1e-9, sp[: len(sp) // 4].mean()))
    for i, b in enumerate(np.array_split(sp, 8)):
        feats[f"sp_bin_{i}"] = float(b.mean())
    return feats


def _quality_features(im_pil: Image.Image) -> dict[str, float]:
    im = np.asarray(im_pil.convert("RGB"))
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # blur measures
    lap_var = float(cv2.Laplacian(g.astype(np.uint8), cv2.CV_64F).var())
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.sqrt(gx * gx + gy * gy).mean())
    lx = cv2.filter2D(g, cv2.CV_32F, np.array([[-1, 2, -1]], dtype=np.float32))
    ly = cv2.filter2D(g, cv2.CV_32F, np.array([[-1], [2], [-1]], dtype=np.float32))
    sml = float((np.abs(lx) + np.abs(ly)).mean())
    # noise (Donoho HH1)
    ll_hh_hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 4
    noise_mad = float(median_abs_deviation(ll_hh_hh.ravel()) / 0.6745)
    # contrast
    contrast = float(g.std())
    # chroma
    yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    chroma = float(yuv[..., 1:].std())
    # DCT AC energy
    g0 = g - 128
    h, w = g0.shape
    h8, w8 = h - h % 8, w - w % 8
    blocks = g0[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    d = np.stack([cv2.dct(b) for b in blocks])
    dct_ac = float(np.abs(d[:, 1:, 1:].reshape(len(d), -1)).mean())
    # jpeg quality estimate (fast: 3 probes)
    arr = np.asarray(im_pil.convert("RGB")).astype(np.int16)
    best_q, best_diff = 100, 1e9
    for q in (85, 60, 35):
        buf = io.BytesIO()
        im_pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        rt = np.asarray(Image.open(buf).convert("RGB")).astype(np.int16)
        diff = np.abs(arr - rt).mean()
        if diff < best_diff:
            best_diff, best_q = diff, q
        if diff < 0.5:
            best_q = q
            break
    jpeg_est = float(best_q)
    return {
        "q_blur_lap": lap_var,
        "q_blur_sml": sml,
        "q_blur_teneng": tenengrad,
        "q_noise_mad": noise_mad,
        "q_contrast": contrast,
        "q_chroma": chroma,
        "q_dct_ac": dct_ac,
        "q_jpeg_est": jpeg_est,
        "q_brisque": float("nan"),
    }
