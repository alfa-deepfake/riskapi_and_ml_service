"""v15 two-modality deepfake ensemble for uploaded challenge video.

Ports the training repo's v15 release (see its infer_v15.py / realtime_v15.py):
per sampled frame an InsightFace-aligned 512px crop is scored by

  1. 6 XGBoost v13 trees on 73 forensic/quality features,
  2. a Noise-CNN (5 ConvNeXt-Tiny folds on a 256px residual map, temperature
     scaled, logistic-calibrated),

blended ``p = w_tree*mean(trees) + w_cnn*cnn`` and median-smoothed across
frames (the realtime pipeline's video treatment). A 5-class condition gate
classifies the input (clean/degraded/restored/vidcall/vidcall_ff) and the
asymmetric low-info gate (face < 180px source or wholly-upscaled input) marks
frames whose fake signature may be fabricated by our own resampling; the
verdict policy on both lives in ``ml_service.core.checks.score_classifier``.

Heavy imports live at module level; the module itself is imported lazily by
``classifier_service`` and skipped when ML dependencies are absent.
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
import xgboost
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, median_abs_deviation, skew

# The training-repo modules cross-import by bare name (features_v8 does
# `from features_v6 import ...`), exactly as in the release bundle — keep them
# verbatim and give them their directory on sys.path instead of rewriting.
_TRAIN_DIR = str(Path(__file__).resolve().parents[2] / "train")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from features_v6 import candidates_v6  # noqa: E402
from features_v7 import candidates_v7  # noqa: E402
from features_v8 import candidates_v8  # noqa: E402
from features_v9 import candidates_v9  # noqa: E402
from model_def import make_model  # noqa: E402
from noise_map_v15 import noise_map_tensor  # noqa: E402

FACE_CROP_SIZE = 512
FFT_SIZE = 256
RESIDUAL_SIGMA = 1.2

GENS = ["deeplivecam", "facefusion", "visomaster", "inswapper128", "reswapper"]
COND_NAMES = ["clean", "degraded", "restored", "vidcall", "vidcall_ff"]

# Verbatim from the release bundle's infer.py: faces smaller than this get
# upscaled >2.8x by the 512 crop, and wholly-upscaled inputs carry no signal
# above their native resolution — both fabricate the generator HF-loss
# signature, so confident FAKE verdicts on such input are withheld.
MIN_FACE_PX = 180
MIN_UPSAMPLE_DIFF = 0.4


class V15VideoAdapter:
    def __init__(
        self,
        *,
        models_dir: Path,
        max_inferences: int = 12,
        infer_every: int = 5,
    ) -> None:
        self.models_dir = models_dir
        self.max_inferences = max(1, max_inferences)
        self.infer_every = max(1, infer_every)
        self._state: dict[str, Any] | None = None

    def load(self) -> None:
        """Eagerly load all models (the CNN folds take ~25s on CPU)."""
        self._load()

    def predict(self, video_path: Path) -> dict[str, Any]:
        st = self._load()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open uploaded video: {video_path}")

        frame_count = 0
        examined = 0
        blend_ps: list[float] = []
        tree_rows: list[dict[str, float]] = []
        cnn_ps: list[float] = []
        conds: list[str] = []
        face_sizes: list[float] = []
        upsample_diffs: list[float] = []
        try:
            while len(blend_ps) < self.max_inferences:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                if frame_count % self.infer_every != 0:
                    continue
                # Cap the scan so a long face-less upload cannot hold a
                # threadpool thread by running alignment on every frame.
                if examined >= self.max_inferences * 8:
                    break
                examined += 1
                crop, face_size = _face_crop(frame)
                if crop is None:
                    continue
                scored = self._score_crop(st, crop)
                blend_ps.append(scored["p"])
                tree_rows.append(scored["trees"])
                cnn_ps.append(scored["cnn_p"])
                conds.append(scored["cond"])
                upsample_diffs.append(scored["upsample_diff"])
                if face_size is not None:
                    face_sizes.append(face_size)
        finally:
            cap.release()

        result: dict[str, Any] = {
            "threshold": st["blend"]["t_bin"],
            "model_name": "v15-xgb6+noise-cnn5",
            "frame_count": frame_count,
            "face_present": bool(blend_ps),
            "face_confidence": len(blend_ps) / examined if examined else 0.0,
            "sampled_frames": len(blend_ps),
            "feature_count": len(st["feat_names"]),
            "preprocessing": "insightface-buffalo_l-norm_crop-512+noise-map-256",
            "face_size_px": float(np.median(face_sizes)) if face_sizes else None,
        }
        if not blend_ps:
            result.update({"fake_probability": None, "confidence": 0.0})
            return result

        p = float(np.median(blend_ps))
        face_px = result["face_size_px"]
        upsample_diff = float(np.median(upsample_diffs))
        low_info = (face_px is not None and face_px < MIN_FACE_PX) or upsample_diff < MIN_UPSAMPLE_DIFF
        borderline = st["blend"]["t_lo"] < p < st["blend"]["t_hi"]
        result.update(
            {
                "fake_probability": p,
                # inside the grey band neither verdict is confident
                "confidence": 0.5 if borderline else max(p, 1.0 - p),
                "model_scores": {
                    name: float(np.mean([row[name] for row in tree_rows])) for name in ["id"] + GENS
                },
                "cnn_probability": float(np.median(cnn_ps)),
                "condition": Counter(conds).most_common(1)[0][0],
                "low_info": low_info,
                "upsample_diff": upsample_diff,
            }
        )
        return result

    def _score_crop(self, st: dict[str, Any], crop_rgb: np.ndarray) -> dict[str, Any]:
        feats = _features(crop_rgb)
        row = np.array(
            [[feats.get(name, float("nan")) for name in st["feat_names"]]], dtype=np.float32
        )
        matrix = xgboost.DMatrix(row, feature_names=st["feat_names"])
        trees = {name: float(booster.predict(matrix)[0]) for name, booster in st["trees"].items()}
        cond_p = st["gate"].predict(matrix)[0]
        cnn_p = self._cnn_score(st, Image.fromarray(crop_rgb))
        p = st["blend"]["w_tree"] * float(np.mean(list(trees.values()))) + st["blend"]["w_cnn"] * cnn_p
        return {
            "p": p,
            "trees": trees,
            "cnn_p": cnn_p,
            "cond": COND_NAMES[int(np.argmax(cond_p))],
            "upsample_diff": feats.get("upsample_diff_256", 99.0),
        }

    @staticmethod
    def _cnn_score(st: dict[str, Any], im_pil: Image.Image) -> float:
        x = noise_map_tensor(im_pil, st["cnn_size"]).unsqueeze(0)
        ps = []
        with torch.no_grad():
            for model, temperature in st["cnn_folds"]:
                logit = float(model(x).flatten()[0]) / temperature
                ps.append(1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40))))
        mean_p = float(np.mean(ps))
        logit = np.log(np.clip(mean_p, 1e-6, 1 - 1e-6) / np.clip(1 - mean_p, 1e-6, 1))
        return float(st["calibrator"].predict_proba(np.array([[logit]]))[:, 1][0])

    def _load(self) -> dict[str, Any]:
        if self._state is not None:
            return self._state
        v13_dir = self.models_dir / "v13"
        cnn_dir = self.models_dir / "cnn"
        feat_names = (v13_dir / "feature_names.txt").read_text().splitlines()
        tree_prefix = json.loads((v13_dir / "v13_config.json").read_text())["tree_prefix"]
        blend = json.loads((self.models_dir / "v15_blend_config.json").read_text())

        trees: dict[str, xgboost.Booster] = {}
        for name in ["id"] + GENS:
            booster = xgboost.Booster()
            booster.load_model(str(v13_dir / f"xgb_{tree_prefix}_{name}.ubj"))
            if booster.num_features() != len(feat_names):
                raise ValueError(
                    f"xgb_{tree_prefix}_{name} expects {booster.num_features()} features, "
                    f"manifest has {len(feat_names)}"
                )
            trees[name] = booster
        gate = xgboost.Booster()
        gate.load_model(str(v13_dir / "xgb_gate_condition.ubj"))

        cnn_cfg = json.loads((cnn_dir / "noise_cnn_metrics.json").read_text())
        folds = []
        for held in GENS:
            ckpt = torch.load(
                cnn_dir / f"noise_cnn_holdout_{held}.pt", map_location="cpu", weights_only=True
            )
            model = make_model(False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            folds.append((model, float(cnn_cfg["folds"][held]["temperature"])))
        calibrator = joblib.load(cnn_dir / "noise_cnn_ensemble_calibrator.joblib")

        self._state = {
            "feat_names": feat_names,
            "trees": trees,
            "gate": gate,
            "blend": blend,
            "cnn_folds": folds,
            "calibrator": calibrator,
            "cnn_size": int(cnn_cfg["size"]),
        }
        return self._state


def _face_crop(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """Apply the exact InsightFace alignment used to train the v15 models."""
    import face_crop

    crop_bgr = face_crop.crop_face_bgr(frame_bgr, size=FACE_CROP_SIZE)
    if crop_bgr is None:
        return None, None
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), face_crop.last_face_px


def _features(crop_rgb: np.ndarray) -> dict[str, float]:
    """The 73 forensic + quality features expected by the v13 trees and gate.

    Base set is verbatim from the training repo's infer.py; v6–v9 candidates
    come from the verbatim ``train/features_v*.py`` modules.
    """
    rgb = np.asarray(crop_rgb, dtype=np.float32) / 255.0
    im_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    feats = _forensic_features(rgb)
    feats.update(_quality_features(im_pil))
    feats["jpeg_q"] = 0  # no round-trip augmentation at inference
    feats.update(candidates_v6(rgb))
    feats.update(candidates_v7(rgb))
    feats.update(candidates_v8(rgb))
    feats.update(candidates_v9(rgb))
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
