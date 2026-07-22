"""v9 features — v8 (65) + 9 more = 74 total.

New candidates (ratio-family + seam + double-JPEG focus):
  26. face_bg_hf_ratio    — HF spectral energy face vs bg (spectral sibling of #3/#4 winners)
  27. face_bg_kurt_diff   — residual kurtosis inside face minus outside
  28. eyes_mouth_res_ratio— residual std eyes-band / mouth-band (GFPGAN over-sharpens eyes)
  29. seam_grad_excess    — gradient excess in blend-seam annulus vs inside face
  30. jpeg_ghost_q        — argmin-Q of re-JPEG residual sweep (double-JPEG primary Q)
  31. jpeg_ghost_depth    — min/mean ratio of that sweep (ghost strength)
  32. res_std_ratio_s12_s3— band ratio sigma1.2/sigma3 (degradation-robust shape)
  33. color_cast_ab       — LAB ab-plane shift magnitude (GAN color drift)
  34. sp16_hf_ratio       — 16-bin radial split: bin15/bin11 ratio (fine HF slope)
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
from scipy.stats import kurtosis

from data import collect, apply_degradation
from face_crop import crop_from_path
from quality import quality_features
from features_v6 import base_v5, candidates_v6, FACE_CROP_SIZE, FFT_SIZE, DEGRADE_CONFIGS
from features_v7 import candidates_v7
from features_v8 import candidates_v8


def candidates_v9(rgb: np.ndarray) -> dict:
    gray = rgb.mean(-1)
    res = gray - gaussian_filter(gray, 1.2)
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    rr = np.hypot(xx - w / 2, yy - h / 2)
    face_m = rr < min(h, w) * 0.35
    bg_m = rr > min(h, w) * 0.45
    f = {}

    # 26. face vs bg HF spectral energy ratio (per-region FFT on 128 crops)
    def _hf_energy(region_mask):
        ys, xs = np.where(region_mask)
        sub = res[ys.min():ys.max()+1, xs.min():xs.max()+1]
        n = 128
        s = cv2.resize(sub.astype(np.float32), (n, n))
        F = np.abs(np.fft.fftshift(np.fft.fft2(s * (np.hanning(n)[:, None] * np.hanning(n)[None, :]))))
        c = n // 2
        yy2, xx2 = np.indices(F.shape)
        r2 = np.hypot(xx2 - c, yy2 - c)
        return float(F[r2 > n * 0.25].mean())
    f["face_bg_hf_ratio"] = _hf_energy(face_m) / max(_hf_energy(bg_m), 1e-9)

    # 27. kurtosis diff face vs bg
    f["face_bg_kurt_diff"] = float(kurtosis(res[face_m].ravel()) - kurtosis(res[bg_m].ravel()))

    # 28. eyes vs mouth residual std (fixed zones — crop is aligned)
    eyes = res[int(h * 0.30):int(h * 0.45), int(w * 0.2):int(w * 0.8)]
    mouth = res[int(h * 0.62):int(h * 0.80), int(w * 0.3):int(w * 0.7)]
    f["eyes_mouth_res_ratio"] = float(eyes.std() / max(mouth.std(), 1e-9))

    # 29. seam gradient excess in blend annulus
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    seam_m = (rr > min(h, w) * 0.35) & (rr < min(h, w) * 0.45)
    f["seam_grad_excess"] = float(gmag[seam_m].mean() / max(gmag[face_m].mean(), 1e-9))

    # 30-31. JPEG ghost sweep
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    arr = np.asarray(im, dtype=np.float32)
    diffs = []
    qs = list(range(60, 100, 5))
    for q in qs:
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q); buf.seek(0)
        rt = np.asarray(Image.open(buf), dtype=np.float32)
        diffs.append(float(np.abs(arr - rt).mean()))
    diffs = np.array(diffs)
    f["jpeg_ghost_q"] = float(qs[int(diffs.argmin())])
    f["jpeg_ghost_depth"] = float(diffs.min() / max(diffs.mean(), 1e-9))

    # 32. scale-ladder band ratio
    res3 = gray - gaussian_filter(gray, 3.0)
    f["res_std_ratio_s12_s3"] = float(res.std() / max(res3.std(), 1e-9))

    # 33. LAB ab color cast
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    f["color_cast_ab"] = float(np.hypot(lab[..., 1].mean() - 128, lab[..., 2].mean() - 128))

    # 34. fine 16-bin HF ratio
    n = FFT_SIZE
    src = cv2.resize(gray.astype(np.float32), (n, n)); src -= src.mean()
    win = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    F = np.fft.fftshift(np.fft.fft2(src * win))
    mag = np.log1p(np.abs(F))
    cy, cx = n // 2, n // 2
    yy3, xx3 = np.indices(mag.shape)
    r3 = np.round(np.hypot(xx3 - cx, yy3 - cy)).astype(np.int32)
    nbin = n // 2
    tb = np.bincount(r3.ravel(), mag.ravel(), minlength=nbin + 1)[:nbin]
    nr = np.bincount(r3.ravel(), minlength=nbin + 1)[:nbin]
    sp = tb / np.maximum(nr, 1)
    b16 = np.array_split(sp, 16)
    f["sp16_hf_ratio"] = float(b16[15].mean() / max(b16[11].mean(), 1e-9))

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
            f.update(candidates_v9(rgb))
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
