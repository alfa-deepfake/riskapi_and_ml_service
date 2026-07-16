"""Predict real/fake for image(s) using an ensemble of 6 XGBoost models.

- xgb_id        : trained in-distribution (all 5 generators seen)
- xgb_loo_dlc   : trained without DeepLiveCam samples
- xgb_loo_ff    : trained without FaceFusion samples
- xgb_loo_viso  : trained without VisoMaster samples
- xgb_loo_ins   : trained without inswapper128 samples
- xgb_loo_rs    : trained without ReSwapper samples

Each model votes independently — all 6 scores are printed. No averaging: caller
decides how to combine (majority, unanimous, weighted). Useful for finding
disagreement patterns during evaluation.

Usage: python infer.py path/to/image.png [more.png ...]
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, median_abs_deviation, skew

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
FEAT_NAMES = (MODELS_DIR / "feature_names.txt").read_text().splitlines()

# Detect + align face once per image; all forensic/quality features are computed
# on that 512x512 crop. Same op at train time — no distribution shift.
sys.path.insert(0, str(ROOT / "train"))
from face_crop import crop_from_path  # noqa: E402

# ordered short names → filenames; keep short so headers stay readable
MODEL_FILES = [
    ("id",   "xgb_id.ubj"),
    ("-dlc", "xgb_loo_deeplivecam.ubj"),
    ("-ff",  "xgb_loo_facefusion.ubj"),
    ("-viso","xgb_loo_visomaster.ubj"),
    ("-ins", "xgb_loo_inswapper128.ubj"),
    ("-rs",  "xgb_loo_reswapper.ubj"),
]

FACE_CROP_SIZE = 512
FFT_SIZE = 256
RESIDUAL_SIGMA = 1.2


def load_gray_rgb(path: Path) -> np.ndarray:
    """Face-crop the image and return as (H,W,3) float in [0,1]."""
    im = crop_from_path(Path(path), size=FACE_CROP_SIZE)
    if im is None:
        raise RuntimeError("no face detected in image")
    return np.asarray(im, dtype=np.float32) / 255.0


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
    r = res.reshape(-1, 3); r = r - r.mean(0)
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
    feats["hf_over_lf"] = float(sp[len(sp) // 2:].mean() / max(1e-9, sp[: len(sp) // 4].mean()))
    for i, b in enumerate(np.array_split(sp, 8)):
        feats[f"sp_bin_{i}"] = float(b.mean())
    return feats


def _quality_features(im_pil: Image.Image) -> dict[str, float]:
    """9 quality features: matches train/quality.py."""
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
    h, w = g0.shape; h8, w8 = h - h % 8, w - w % 8
    blocks = g0[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    d = np.stack([cv2.dct(b) for b in blocks])
    dct_ac = float(np.abs(d[:, 1:, 1:].reshape(len(d), -1)).mean())
    # jpeg quality estimate (fast: 3 probes)
    arr = np.asarray(im_pil.convert("RGB")).astype(np.int16)
    best_q, best_diff = 100, 1e9
    for q in (85, 60, 35):
        buf = io.BytesIO(); im_pil.save(buf, format="JPEG", quality=q); buf.seek(0)
        rt = np.asarray(Image.open(buf).convert("RGB")).astype(np.int16)
        diff = np.abs(arr - rt).mean()
        if diff < best_diff:
            best_diff, best_q = diff, q
        if diff < 0.5:
            best_q = q; break
    jpeg_est = float(best_q)
    return {
        "q_blur_lap": lap_var, "q_blur_sml": sml, "q_blur_teneng": tenengrad,
        "q_noise_mad": noise_mad, "q_contrast": contrast, "q_chroma": chroma,
        "q_dct_ac": dct_ac, "q_jpeg_est": jpeg_est, "q_brisque": float("nan"),
    }


def features_for(img_path: Path) -> dict[str, float]:
    rgb = load_gray_rgb(img_path)  # face-aligned 512x512
    im_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    f = _forensic_features(rgb)
    f.update(_quality_features(im_pil))
    f["jpeg_q"] = 0  # no round-trip augmentation at inference
    # v6-v9 candidate features (74-feature production set; v10 was rejected —
    # see TRAINING_JOURNAL). Vector is assembled by FEAT_NAMES lookup.
    from features_v6 import candidates_v6
    from features_v7 import candidates_v7
    from features_v8 import candidates_v8
    from features_v9 import candidates_v9
    f.update(candidates_v6(rgb))
    f.update(candidates_v7(rgb))
    f.update(candidates_v8(rgb))
    f.update(candidates_v9(rgb))
    return f


_boosters: dict[str, xgb.Booster] = {}


def _models() -> dict[str, xgb.Booster]:
    if not _boosters:
        for name, fn in MODEL_FILES:
            path = MODELS_DIR / fn
            if not path.exists():
                continue
            b = xgb.Booster(); b.load_model(str(path))
            _boosters[name] = b
    return _boosters


_last_feats: dict[str, float] = {}


def predict(img_path: Path) -> dict[str, float]:
    """Return {model_name: p_fake} for every loaded model."""
    global _last_feats
    feats = features_for(img_path)
    _last_feats = feats
    x = xgb.DMatrix(np.array([[feats.get(k, float("nan")) for k in FEAT_NAMES]], dtype=np.float32))
    return {name: float(b.predict(x)[0]) for name, b in _models().items()}


# faces smaller than this get upscaled >2.8x by the 512 crop — the top of the
# spectrum is then fabricated by our own resampling, so verdicts are unreliable.
# ponytail: fixed cutoff; make quality-adaptive if the bank needs finer policy
MIN_FACE_PX = 180

# wholly-upscaled images (pre-resized before reaching us) carry no signal above
# their native resolution — indistinguishable from generator HF-loss. Dataset
# fakes measure >=4.8 here (swap keeps native background); genuine upscales <0.4.
MIN_UPSAMPLE_DIFF = 0.4


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python infer.py <image_path> [more.png ...]", file=sys.stderr)
        sys.exit(1)
    import face_crop as _fc
    header = "path".ljust(60) + "  ".join(n.rjust(6) for n, _ in MODEL_FILES)
    print(header)
    print("-" * len(header))
    for p in sys.argv[1:]:
        try:
            scores = predict(Path(p))
            row = Path(p).name.ljust(60) + "  ".join(f"{scores.get(n, float('nan')):6.3f}" for n, _ in MODEL_FILES)
            verdicts = sum(1 for n, _ in MODEL_FILES if scores.get(n, 0) > 0.5)
            face_px = _fc.last_face_px
            up_diff = _last_feats.get("upsample_diff_256", 99.0)
            caveat = None
            if face_px is not None and face_px < MIN_FACE_PX:
                caveat = f"face {face_px:.0f}px < {MIN_FACE_PX}px"
            elif up_diff < MIN_UPSAMPLE_DIFF:
                caveat = f"upscaled image, detail {up_diff:.2f} < {MIN_UPSAMPLE_DIFF}"
            # asymmetric gate: low-info inputs must not produce a confident FAKE
            # (false accusation is the expensive error), but a REAL verdict may
            # pass with a warning — the bank can still escalate on the caveat
            if caveat and verdicts >= 5:
                print(f"{row}  → UNSURE ({caveat} — signal unreliable; raw votes {verdicts}/6)")
            elif caveat:
                print(f"{row}  → {verdicts}/6 vote fake  [warning: {caveat}]")
            else:
                print(f"{row}  → {verdicts}/6 vote fake")
        except Exception as e:
            print(f"{p}\tERROR\t{e}", file=sys.stderr)
