"""v8 features — v7 (55) + 10 more = 65 total.

New candidates:
  18. wavelet_h/v/d       — Haar sub-band mean |energy| (3 feats; directional HF)
  19. cfa_periodicity     — Bayer/CFA 2x2 demosaic peak strength in FFT
  20. ela_residual_std    — error-level analysis: re-JPEG Q90 residual std
  21. noise_symmetry_lr   — residual-std correlation between mirrored face halves
  22. upsample_diff_128   — bicubic 128 round-trip MSE (native res of inswapper)
  23. upsample_diff_256   — bicubic 256 round-trip MSE (native res of reswapper)
  24. grad_orient_entropy — entropy of gradient-orientation histogram
  25. res_std_s6          — sigma=6 residual std (completes scale ladder 1.2/3/6)
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

from face_crop import crop_from_path
from features_v6 import base_v5, candidates_v6, FACE_CROP_SIZE, DEGRADE_CONFIGS
from features_v7 import candidates_v7


def candidates_v8(rgb: np.ndarray) -> dict:
    gray = rgb.mean(-1)
    f = {}

    # 18. Haar wavelet sub-band energies
    a = gray[: gray.shape[0] // 2 * 2, : gray.shape[1] // 2 * 2]
    q00, q01 = a[0::2, 0::2], a[0::2, 1::2]
    q10, q11 = a[1::2, 0::2], a[1::2, 1::2]
    f["wavelet_h"] = float(np.mean(np.abs((q00 + q01 - q10 - q11) / 2)))
    f["wavelet_v"] = float(np.mean(np.abs((q00 - q01 + q10 - q11) / 2)))
    f["wavelet_d"] = float(np.mean(np.abs((q00 - q01 - q10 + q11) / 2)))

    # 19. CFA periodicity — energy at (N/2, N/2)-type peaks vs local background
    res = gray - gaussian_filter(gray, 1.2)
    n = 256
    src = cv2.resize(res.astype(np.float32), (n, n))
    F = np.abs(np.fft.fftshift(np.fft.fft2(src)))
    c = n // 2
    peaks = [F[c, 0], F[0, c], F[0, 0], F[c, n - 1], F[n - 1, c]]
    ring = F[c - 8 : c + 8, c - 8 : c + 8].mean()
    f["cfa_periodicity"] = float(np.mean(peaks) / max(ring, 1e-9))

    # 20. ELA — re-JPEG at Q90, residual std
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90); buf.seek(0)
    rt = np.asarray(Image.open(buf), dtype=np.float32) / 255.0
    f["ela_residual_std"] = float((rgb - rt).std())

    # 21. noise symmetry left-right (residual std per row-band, corr across mirror)
    h, w = res.shape
    left = res[:, : w // 2]
    right = res[:, w - w // 2 :][:, ::-1]
    bands = 16
    ls = [left[i * h // bands : (i + 1) * h // bands].std() for i in range(bands)]
    rs = [right[i * h // bands : (i + 1) * h // bands].std() for i in range(bands)]
    f["noise_symmetry_lr"] = float(np.corrcoef(ls, rs)[0, 1])

    # 22-23. bicubic round-trip MSE at generator-native resolutions
    g8 = (gray * 255).astype(np.float32)
    for tgt in (128, 256):
        low = cv2.resize(g8, (tgt, tgt), interpolation=cv2.INTER_CUBIC)
        back = cv2.resize(low, (g8.shape[1], g8.shape[0]), interpolation=cv2.INTER_CUBIC)
        f[f"upsample_diff_{tgt}"] = float(((g8 - back) ** 2).mean())

    # 24. gradient orientation entropy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    keep = mag > np.percentile(mag, 75)
    hist = np.histogram(ang[keep], bins=36, range=(-np.pi, np.pi), weights=mag[keep])[0]
    p = hist / max(hist.sum(), 1e-9)
    p = p[p > 0]
    f["grad_orient_entropy"] = float(-(p * np.log2(p)).sum())

    # 25. sigma=6 residual std
    f["res_std_s6"] = float((gray - gaussian_filter(gray, 6.0)).std())

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
            f.update(candidates_v8(rgb))
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
        rng = random.Random(0); rng.shuffle(items); items = items[: args.limit]
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
