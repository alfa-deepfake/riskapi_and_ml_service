"""v7 features — v6 (45) + 10 more candidates = 55 total.

New candidates (each targets a distinct physical artifact):
  8.  res_cross_scale_corr — corr(res σ1.2, res σ3): white sensor noise decorrelates
                             across scales, GAN structure correlates
  9.  benford_dct_chi2     — first-digit Benford deviation of DCT AC coeffs
  10. noise_rg_ratio       — per-channel noise floor ratio R/G (Bayer signature)
  11. noise_bg_ratio       — per-channel noise floor ratio B/G
  12. azim_peak_ratio      — max/mean of angular FFT profile (upsampling grid peaks)
  13. res_entropy          — Shannon entropy of quantized residual
  14. blockiness           — 8x8 block-boundary gradient excess (codec grid)
  15. face_bg_autocorr_diff— lag-1 autocorr inside face minus outside
  16. res_p99_over_std     — residual tail ratio (mask-boundary spikes vs Gaussian)
  17. spectral_slope       — log-log slope of radial spectrum (1/f deviation)
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis

from face_crop import crop_from_path
from features_v6 import base_v5, candidates_v6, FACE_CROP_SIZE, FFT_SIZE, DEGRADE_CONFIGS


def _mad_noise(ch: np.ndarray) -> float:
    """Donoho noise floor via HH1 quad differences."""
    hh = (ch[0::2, 0::2] - ch[0::2, 1::2] - ch[1::2, 0::2] + ch[1::2, 1::2]) / 4
    return float(np.median(np.abs(hh - np.median(hh))) / 0.6745 + 1e-12)


def candidates_v7(rgb: np.ndarray) -> dict:
    gray = rgb.mean(-1)
    f = {}

    res12 = gray - gaussian_filter(gray, 1.2)
    res3 = gray - gaussian_filter(gray, 3.0)

    # 8. cross-scale residual correlation
    f["res_cross_scale_corr"] = float(np.corrcoef(res12.ravel(), res3.ravel())[0, 1])

    # 9. Benford chi2 on DCT AC coefficients (subsample blocks)
    g255 = (gray * 255).astype(np.float32) - 128
    h, w = g255.shape; h8, w8 = h - h % 8, w - w % 8
    blocks = g255[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    if len(blocks) > 1500:
        blocks = blocks[np.linspace(0, len(blocks) - 1, 1500).astype(int)]
    d = np.stack([cv2.dct(b) for b in blocks])
    ac = np.abs(d[:, 1:, 1:]).ravel()
    ac = ac[ac > 1e-3]
    if len(ac) > 100:
        first_digit = (ac / 10 ** np.floor(np.log10(ac))).astype(int).clip(1, 9)
        obs = np.bincount(first_digit, minlength=10)[1:10].astype(float)
        obs /= obs.sum()
        benford = np.log10(1 + 1 / np.arange(1, 10))
        f["benford_dct_chi2"] = float(((obs - benford) ** 2 / benford).sum())
    else:
        f["benford_dct_chi2"] = 0.0

    # 10-11. per-channel noise floor ratios (Bayer signature)
    nr = _mad_noise(rgb[..., 0] * 255)
    ng = _mad_noise(rgb[..., 1] * 255)
    nb = _mad_noise(rgb[..., 2] * 255)
    f["noise_rg_ratio"] = float(nr / ng)
    f["noise_bg_ratio"] = float(nb / ng)

    # 12. azimuthal peak ratio (same annulus as azim_std)
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
    f["azim_peak_ratio"] = float(prof.max() / max(prof.mean(), 1e-9))

    # 13. residual entropy (64-bin histogram of clipped residual)
    q = np.clip((res12 + 0.125) / 0.25 * 64, 0, 63).astype(int)
    p = np.bincount(q.ravel(), minlength=64).astype(float)
    p = p / p.sum()
    p = p[p > 0]
    f["res_entropy"] = float(-(p * np.log2(p)).sum())

    # 14. blockiness — gradient excess at 8-multiples
    dx = np.abs(np.diff(gray, axis=1)); dy = np.abs(np.diff(gray, axis=0))
    bx = dx[:, 7::8].mean() if dx.shape[1] >= 8 else 0.0
    by = dy[7::8, :].mean() if dy.shape[0] >= 8 else 0.0
    f["blockiness"] = float((bx + by) / 2 - (dx.mean() + dy.mean()) / 2)

    # 15. face vs bg lag-1 autocorr difference
    h2, w2 = res12.shape
    yy2, xx2 = np.indices((h2, w2))
    rr = np.hypot(xx2 - w2 / 2, yy2 - h2 / 2)
    face_m = rr < min(h2, w2) * 0.35
    bg_m = rr > min(h2, w2) * 0.45
    def _ac(mask):
        m = mask[:, :-1] & mask[:, 1:]
        a = res12[:, :-1][m]; b = res12[:, 1:][m]
        if len(a) < 100: return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    f["face_bg_autocorr_diff"] = _ac(face_m) - _ac(bg_m)

    # 16. residual tail ratio
    absr = np.abs(res12).ravel()
    f["res_p99_over_std"] = float(np.percentile(absr, 99) / max(absr.std(), 1e-9))

    # 17. spectral slope (log-log fit, skip DC bins)
    cyc, mag2 = None, None
    rbin = np.round(r).astype(np.int32)
    nbin = FFT_SIZE // 2
    tb = np.bincount(rbin.ravel(), mag.ravel(), minlength=nbin + 1)[:nbin]
    nrr = np.bincount(rbin.ravel(), minlength=nbin + 1)[:nbin]
    sp = tb / np.maximum(nrr, 1)
    xs = np.arange(4, nbin); ys = sp[4:]
    f["spectral_slope"] = float(np.polyfit(np.log(xs), ys, 1)[0])

    return f


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
            f.update(candidates_v7(rgb))
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
    ap.add_argument("--workers", type=int, default=12)
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
    with open(out, "w", newline="") as f2:
        w = csv.DictWriter(f2, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  rows={len(rows)}  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
