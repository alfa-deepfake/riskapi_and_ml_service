"""v6 features — adds 7 candidates to v5 (38 → 45), designed to be orthogonal.

Candidates targeting known-signal artifacts:
  1. res_std_s3        — mid-freq residual (σ=3), fills band gap above σ=1.2
  2. chroma_res_std    — Cb+Cr residual std, catches chroma-mis-reconstruction (GANs learn Y-heavy)
  3. azim_std          — angular FFT profile std, directional GAN upsampling grids
  4. face_bg_res_ratio — face-region vs background residual ratio; swap only modifies face
  5. local_var_cv      — CV of per-cell residual variance; texture homogeneity
  6. res_autocorr_mean — lag-1 autocorr of residual (real=white noise, upsampled=correlated)
  7. dct_ac_kurt       — kurtosis of 8x8 DCT AC coefficients (double-compression signature)
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, skew

from data import GEN_ID, collect, apply_degradation
from face_crop import crop_from_path
from quality import quality_features

FACE_CROP_SIZE = 512
FFT_SIZE = 256
RESIDUAL_SIGMA = 1.2


def load_gray_rgb(path: Path, jpeg_q: int | None = None) -> np.ndarray | None:
    im = crop_from_path(Path(path), size=FACE_CROP_SIZE)
    if im is None:
        return None
    if jpeg_q is not None:
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=jpeg_q); buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def _residual(rgb, sigma=RESIDUAL_SIGMA):
    return np.stack([rgb[..., c] - gaussian_filter(rgb[..., c], sigma) for c in range(3)], -1)


def _radial_spectrum(gray):
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
    return tbin / np.maximum(nr, 1), mag


def base_v5(rgb):
    """Existing 38 v5 features (residual+FFT+quality, minus quality which merged separately)."""
    gray = rgb.mean(-1)
    res = _residual(rgb)
    f = {}
    for i, ch in enumerate("rgb"):
        r = res[..., i].ravel()
        f[f"res_std_{ch}"] = float(r.std())
        f[f"res_kurt_{ch}"] = float(kurtosis(r, fisher=True))
        f[f"res_skew_{ch}"] = float(skew(r))
    r = res.reshape(-1, 3); r = r - r.mean(0)
    cov = (r.T @ r) / max(1, r.shape[0] - 1)
    diag = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(diag, diag)
    f["res_corr_rg"] = float(corr[0, 1]); f["res_corr_rb"] = float(corr[0, 2]); f["res_corr_gb"] = float(corr[1, 2])
    for i, ch in enumerate("rgb"):
        f[f"mean_{ch}"] = float(rgb[..., i].mean())
        f[f"std_{ch}"] = float(rgb[..., i].std())
    f["lap_var"] = float(res.mean(-1).var())
    sp, _ = _radial_spectrum(gray)
    f["hf_over_lf"] = float(sp[len(sp) // 2:].mean() / max(1e-9, sp[: len(sp) // 4].mean()))
    for i, b in enumerate(np.array_split(sp, 8)):
        f[f"sp_bin_{i}"] = float(b.mean())
    return f


def candidates_v6(rgb):
    """7 new candidates."""
    gray = rgb.mean(-1)
    f = {}

    # 1. multi-scale residual σ=3 (mid-freq band)
    lp3 = np.stack([gaussian_filter(rgb[..., c], 3.0) for c in range(3)], -1)
    res3 = rgb - lp3
    f["res_std_s3"] = float(res3.std())

    # 2. chroma residual std (Cb + Cr)
    yuv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    lp_c = np.stack([gaussian_filter(yuv[..., c], 1.2) for c in (1, 2)], -1)
    res_c = yuv[..., 1:] - lp_c
    f["chroma_res_std"] = float(res_c.std())

    # 3. angular FFT profile std (12 bins, mid-freq annulus)
    src = cv2.resize(gray.astype(np.float32), (FFT_SIZE, FFT_SIZE))
    src = src - src.mean()
    win = np.hanning(FFT_SIZE)[:, None] * np.hanning(FFT_SIZE)[None, :]
    F = np.fft.fftshift(np.fft.fft2(src * win))
    mag = np.log1p(np.abs(F))
    cy, cx = FFT_SIZE // 2, FFT_SIZE // 2
    yy, xx = np.indices(mag.shape)
    r = np.hypot(xx - cx, yy - cy)
    theta = np.arctan2(yy - cy, xx - cx)
    keep = (r > FFT_SIZE * 0.15) & (r < FFT_SIZE * 0.4)
    ang_bins = 12
    bins = ((theta[keep] + np.pi) / (2 * np.pi) * ang_bins).astype(int) % ang_bins
    prof = np.bincount(bins, mag[keep], minlength=ang_bins) / np.maximum(np.bincount(bins, minlength=ang_bins), 1)
    f["azim_std"] = float(prof.std())

    # 4. face-vs-background residual std ratio
    res_g = gray - gaussian_filter(gray, 1.2)
    h, w = res_g.shape
    yy, xx = np.indices((h, w))
    rr = np.hypot(xx - w / 2, yy - h / 2)
    face = rr < min(h, w) * 0.35
    bg = rr > min(h, w) * 0.45
    f["face_bg_res_ratio"] = float(res_g[face].std() / max(res_g[bg].std(), 1e-9))

    # 5. local variance CV (24×24 cells on residual)
    cell = 24
    trimmed = res_g[:h // cell * cell, :w // cell * cell]
    cells = trimmed.reshape(h // cell, cell, w // cell, cell)
    v = cells.var(axis=(1, 3)).ravel()
    f["local_var_cv"] = float(v.std() / max(v.mean(), 1e-12))

    # 6. lag-1 residual autocorrelation (mean of x + y)
    ax = float(np.corrcoef(res_g[:, :-1].ravel(), res_g[:, 1:].ravel())[0, 1])
    ay = float(np.corrcoef(res_g[:-1, :].ravel(), res_g[1:, :].ravel())[0, 1])
    f["res_autocorr_mean"] = float((ax + ay) / 2)

    # 7. DCT AC kurtosis (subsampled 2000 blocks for speed)
    g255 = (gray * 255).astype(np.float32) - 128
    h2, w2 = g255.shape; h8, w8 = h2 - h2 % 8, w2 - w2 % 8
    blocks = g255[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    if len(blocks) > 2000:
        idx = np.linspace(0, len(blocks) - 1, 2000).astype(int)
        blocks = blocks[idx]
    d = np.stack([cv2.dct(b) for b in blocks])
    ac = np.abs(d[:, 1:, 1:]).ravel()
    f["dct_ac_kurt"] = float(kurtosis(ac))

    return f


DEGRADE_CONFIGS = [("orig", 0.0, 0), ("med_a", 0.5, 1), ("med_b", 0.5, 2), ("heavy", 0.9, 3)]


def _work(item):
    path, label, gen = item
    rows = []
    face_pil = crop_from_path(Path(path), size=FACE_CROP_SIZE)
    if face_pil is None:
        return [{"error": "no face", "path": str(path), "aug": "none"}]
    for aug_name, severity, seed in DEGRADE_CONFIGS:
        try:
            if severity > 0:
                rng = random.Random(hash((str(path), seed)) & 0xffff)
                im, applied = apply_degradation(face_pil, severity, rng)
            else:
                im, applied = face_pil, {}
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            f = base_v5(rgb)
            f.update(candidates_v6(rgb))
            f.update(quality_features(im))
            f["label"] = label; f["gen_id"] = gen
            f["path"] = str(path); f["aug"] = aug_name
            f["jpeg_q"] = applied.get("jpeg_q", 0)
            rows.append(f)
        except Exception as e:
            rows.append({"error": str(e), "path": str(path), "aug": aug_name})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = collect()
    if args.limit:
        rng = random.Random(0); rng.shuffle(items); items = items[:args.limit]
    print(f"processing {len(items)} images...", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, batch in enumerate(ex.map(_work, items, chunksize=8)):
            rows.extend(batch)
            if (i + 1) % 500 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(items)}  {dt:.0f}s  ({(i+1)/dt:.1f} img/s)  rows={len(rows)}", flush=True)

    keys = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  rows={len(rows)}  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
