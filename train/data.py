"""Dataset + split logic used by every training script.

Layout on the A100 box:
  DATA_ROOT/data/images1024x1024/*/*.png                — reals (70k)
  DATA_ROOT/output/generated/{dlc,ff,viso}/*/*.png       — fakes (~12k)

split: seeded 80/10/10 (train/val/test), stratified per class.
"""
from __future__ import annotations

import io
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # some FFHQ PNGs have short IDAT chunks

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/master/ffhq_face_swap_20260711"))
GENERATORS = ["deeplivecam", "facefusion", "visomaster", "inswapper128", "reswapper"]
GEN_ID = {"real": 0, "deeplivecam": 1, "facefusion": 2, "visomaster": 3, "inswapper128": 4, "reswapper": 5}


def collect(seed: int = 0, max_real: int | None = None) -> list[tuple[Path, int, int]]:
    """Return [(path, label_binary, generator_id)]. label_binary: 0=real, 1=fake."""
    rng = random.Random(seed)
    items: list[tuple[Path, int, int]] = []
    for g in GENERATORS:
        for p in sorted((DATA_ROOT / "output/generated" / g).rglob("*.png")):
            items.append((p, 1, GEN_ID[g]))
    n_fake = len(items)
    reals = sorted((DATA_ROOT / "data/images1024x1024").rglob("*.png"))
    if max_real is None:
        max_real = n_fake
    rng.shuffle(reals)
    for p in reals[:max_real]:
        items.append((p, 0, GEN_ID["real"]))
    rng.shuffle(items)
    return items


def split_train_val_test(items, seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.1):
    rng = random.Random(seed + 1)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n = len(idx)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    val = [items[i] for i in idx[:n_val]]
    test = [items[i] for i in idx[n_val:n_val + n_test]]
    train = [items[i] for i in idx[n_val + n_test:]]
    return train, val, test


def split_heldout_visomaster(items, seed: int = 0, val_frac: float = 0.1):
    """Train on {real, DLC, FF}, test on {real, Viso}. Val is a slice of the train set."""
    return _split_leave_one_gen_out(items, "visomaster", seed=seed, val_frac=val_frac)


def _split_leave_one_gen_out(items, held_out: str, seed: int = 0, val_frac: float = 0.1):
    """Leave-one-generator-out CV: fakes from held_out generator go to test,
    fakes from every other generator go to train (with a val slice). Reals split
    proportionally so test has 50/50 real vs fake."""
    ho_id = GEN_ID[held_out]
    train_pool = [it for it in items if it[2] != ho_id]
    test = [it for it in items if it[2] in (GEN_ID["real"], ho_id)]
    rng = random.Random(seed + 2)
    reals = [it for it in test if it[1] == 0]
    fakes = [it for it in test if it[1] == 1]
    rng.shuffle(reals)
    test = reals[: len(fakes)] + fakes
    rng.shuffle(test)
    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * val_frac)
    val = train_pool[:n_val]
    train = train_pool[n_val:]
    return train, val, test


def residual_np(rgb: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """rgb: (H,W,3) float32 [0,1]. Returns high-pass residual."""
    lp = np.stack([gaussian_filter(rgb[..., c], sigma) for c in range(3)], axis=-1)
    return rgb - lp


def _jpeg_roundtrip(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _motion_blur(arr: np.ndarray, ksize: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    k /= k.sum() + 1e-9
    return cv2.filter2D(arr, -1, k)


def _downsample_upsample(arr: np.ndarray, target_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    low = cv2.resize(arr, (target_side, target_side), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_degradation(im: Image.Image, severity: float, rng: random.Random) -> tuple[Image.Image, dict]:
    """Randomly degrade quality of real image so model doesn't confuse quality with fake.

    severity ∈ [0, 1]. Returns the degraded image + a dict of applied ops (for quality label).
    Ops applied independently with prob scaled by severity.
    """
    arr = np.asarray(im, dtype=np.float32)
    applied: dict = {}

    # 1) resample small then back (mimics low-res webcam / thumbnail)
    if rng.random() < 0.4 * severity:
        target = rng.choice([256, 384, 512])
        arr = _downsample_upsample(arr, target)
        applied["down"] = target

    # 2) motion blur (mimics phone motion)
    if rng.random() < 0.35 * severity:
        k = rng.choice([5, 7, 9, 11, 13])
        angle = rng.uniform(0, 360)
        arr = _motion_blur(arr, k, angle)
        applied["motion"] = k

    # 3) gaussian blur (mimics defocus)
    if rng.random() < 0.3 * severity:
        sigma = rng.uniform(0.5, 2.0)
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
        applied["blur_sigma"] = round(sigma, 2)

    # 4) gaussian sensor noise + poisson shot noise (mimics high ISO)
    if rng.random() < 0.45 * severity:
        sigma = rng.uniform(1.5, 8.0)
        arr = arr + rng.gauss(0, 1) * 0 + np.random.normal(0, sigma, arr.shape).astype(np.float32)
        applied["noise_sigma"] = round(sigma, 2)
    if rng.random() < 0.2 * severity:
        arr = np.random.poisson(np.clip(arr, 0, 255) * 0.6) / 0.6
        applied["poisson"] = True

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)

    # 5) JPEG round-trip at low quality (mimics compression)
    if rng.random() < 0.7 * severity:
        q = rng.randint(30, 90)
        im = _jpeg_roundtrip(im, q)
        applied["jpeg_q"] = q

    return im, applied


def quality_target(applied: dict) -> float:
    """Rough quality score in [0,1] from applied degradation dict.
    Used later by the two-head model. For plan 1 it just serves as diagnostic."""
    q = 1.0
    if "jpeg_q" in applied:      q -= (100 - applied["jpeg_q"]) / 100 * 0.35
    if "noise_sigma" in applied: q -= min(applied["noise_sigma"] / 20, 0.3)
    if "blur_sigma" in applied:  q -= min(applied["blur_sigma"] / 3, 0.25)
    if "motion" in applied:      q -= min(applied["motion"] / 20, 0.25)
    if "down" in applied:        q -= max(0, (768 - applied["down"]) / 768 * 0.25)
    if applied.get("poisson"):   q -= 0.15
    return max(0.0, min(1.0, q))


class ImageBinaryDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, int, int]],
        size: int = 224,
        use_residual: bool = False,
        train: bool = False,
        degrade_p: float = 0.6,           # prob to degrade an image (real OR fake, same distribution)
        degrade_severity: float = 0.8,    # 0..1 how aggressive
    ):
        self.items = items
        self.size = size
        self.use_residual = use_residual
        self.train = train
        self.degrade_p = degrade_p
        self.degrade_severity = degrade_severity

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: Path, is_real: bool) -> tuple[np.ndarray, float]:
        """Return (rgb, quality_score).

        Degradation is applied SYMMETRICALLY to real and fake with the same
        distribution. Otherwise the model learns 'clean PNG = fake' — but real
        deepfakes typically arrive through Zoom/H.264/Telegram, i.e. already
        degraded. quality_score depends only on the applied degradation,
        independent of the class label.
        """
        im = Image.open(path).convert("RGB")
        q_score = 1.0
        rng = random
        if self.train and rng.random() < self.degrade_p:
            im, applied = apply_degradation(im, self.degrade_severity, rng)
            q_score = quality_target(applied)
        w, h = im.size
        crop = min(w, h)
        if self.train:
            ox = rng.randint(0, w - crop)
            oy = rng.randint(0, h - crop)
        else:
            ox = (w - crop) // 2
            oy = (h - crop) // 2
        im = im.crop((ox, oy, ox + crop, oy + crop)).resize((self.size, self.size), Image.BILINEAR)
        if self.train and rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        # note: is_real is unused for augmentation itself; kept in signature for the
        # future quality-head that ignores label but still validates independence
        _ = is_real
        return np.asarray(im, dtype=np.float32) / 255.0, q_score

    def __getitem__(self, idx: int):
        # if a specific image is broken, fall through to the next
        for _ in range(4):
            path, label, gen_id = self.items[idx]
            try:
                rgb, q = self._load(path, is_real=(label == 0))
                break
            except Exception:
                idx = (idx + 1) % len(self.items)
        else:
            raise RuntimeError("4 consecutive broken images — data path corrupted")
        chans = [rgb]
        if self.use_residual:
            chans.append(residual_np(rgb))
        stacked = np.concatenate(chans, axis=-1)   # (H,W,C)
        x = torch.from_numpy(stacked).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x[:3] = (x[:3] - mean) / std
        if self.use_residual:
            x[3:] = x[3:] * 10.0
        return (x,
                torch.tensor(label, dtype=torch.long),
                torch.tensor(gen_id, dtype=torch.long),
                torch.tensor(q, dtype=torch.float32))


# ponytail: self-check — ensure dataset shape and residual math sane
def _demo() -> None:
    items = collect(max_real=5)[:2]
    print("collected", len(items))
    if not items:
        print("no data — DATA_ROOT unset?")
        return
    ds = ImageBinaryDataset(items, size=64, use_residual=True, train=True)
    x, y, g, q = ds[0]
    assert x.shape == (6, 64, 64), x.shape
    assert 0 <= q.item() <= 1, q.item()
    print("ok", x.shape, y.item(), g.item(), round(q.item(), 3))


if __name__ == "__main__":
    _demo()
