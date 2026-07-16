"""Deterministic three-channel forensic noise map for aligned face crops."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


NOISE_CHANNELS = (
    "luma_gaussian_residual",
    "luma_laplacian_residual",
    "chroma_gaussian_residual",
)
NOISE_SCALES = (0.025, 0.12, 0.005)
NOISE_CLIP = 4.0


def noise_map_np(image: Image.Image, size: int = 256) -> np.ndarray:
    """Return a fixed-scale CHW float32 noise map in [-1, 1].

    Fixed scaling is intentional: per-image normalization would erase absolute
    residual strength, which carries camera, codec, denoising and synthesis
    information.
    """
    rgb = np.asarray(
        image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_low = cv2.GaussianBlur(y, (0, 0), sigmaX=1.15, sigmaY=1.15)
    luma_residual = (y - y_low) / NOISE_SCALES[0]

    laplacian_kernel = np.asarray(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    laplacian_residual = cv2.filter2D(y, cv2.CV_32F, laplacian_kernel) / NOISE_SCALES[1]

    cr_residual = cr - cv2.GaussianBlur(cr, (0, 0), sigmaX=1.15, sigmaY=1.15)
    cb_residual = cb - cv2.GaussianBlur(cb, (0, 0), sigmaX=1.15, sigmaY=1.15)
    chroma_residual = ((cr_residual + cb_residual) * 0.5) / NOISE_SCALES[2]

    stacked = np.stack((luma_residual, laplacian_residual, chroma_residual), axis=0)
    return (np.clip(stacked, -NOISE_CLIP, NOISE_CLIP) / NOISE_CLIP).astype(np.float32)


def noise_map_tensor(image: Image.Image, size: int = 256) -> torch.Tensor:
    return torch.from_numpy(noise_map_np(image, size)).contiguous()


def preview_image(noise_map: np.ndarray) -> Image.Image:
    """Encode the three signed channels as RGB solely for visual inspection."""
    hwc = np.moveaxis(noise_map, 0, -1)
    return Image.fromarray(np.rint((hwc + 1.0) * 127.5).clip(0, 255).astype(np.uint8), "RGB")
