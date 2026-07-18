"""Face crop utility. Detect + align a single 512x512 face from any image.

Used to normalize inputs across sources (FFHQ, phone photos, Telegram JPEG etc).
Same tool at train time and inference time — no distribution shift.
"""
from __future__ import annotations

import contextlib
import io as _io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


_APP = None


def _app():
    global _APP
    if _APP is None:
        with contextlib.redirect_stdout(_io.StringIO()):
            from insightface.app import FaceAnalysis
            _APP = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            # 640 is SCRFD's native input — better recall than 512 at both
            # scale extremes (tiny faces in wide 4K frames, big close-ups).
            _APP.prepare(ctx_id=0, det_size=(640, 640))
    return _APP


def _detect_faces(img_bgr: np.ndarray) -> list:
    """Detect with retries covering the camera-resolution extremes.

    SCRFD resizes the frame aspect-fit into det_size, so what matters is the
    face size *relative to the frame*: a close-up selfie face (>~80% of the
    frame) overshoots the anchor range, a far-away face in a wide 4K frame
    undershoots it. Retry zoomed out (padded border) for the former and on
    the center crop for the latter (the UI guides the face into a centered
    oval), mapping geometry back to source coordinates either way.
    """
    app = _app()
    faces = app.get(img_bgr)
    if faces:
        return faces
    height, width = img_bgr.shape[:2]
    pad_y, pad_x = height // 3, width // 3
    padded = cv2.copyMakeBorder(img_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT)
    faces = app.get(padded)
    if faces:
        for face in faces:
            face.bbox = face.bbox - np.array([pad_x, pad_y, pad_x, pad_y], dtype=face.bbox.dtype)
            face.kps = face.kps - np.array([[pad_x, pad_y]], dtype=face.kps.dtype)
        return faces
    y0, x0 = height // 4, width // 4
    faces = app.get(np.ascontiguousarray(img_bgr[y0:height - y0, x0:width - x0]))
    for face in faces:
        face.bbox = face.bbox + np.array([x0, y0, x0, y0], dtype=face.bbox.dtype)
        face.kps = face.kps + np.array([[x0, y0]], dtype=face.kps.dtype)
    return faces


last_face_px: float | None = None  # max side of the last detected bbox (source pixels)


def crop_face_bgr(img_bgr: np.ndarray, size: int = 512) -> np.ndarray | None:
    """Return aligned face crop as BGR uint8, or None if no face found.

    Side effect: sets module-level `last_face_px` to the detected bbox max side —
    faces much smaller than `size` get upscaled by norm_crop, which fabricates
    the HF-loss signature; callers should gate verdicts on it.
    """
    global last_face_px
    faces = _detect_faces(img_bgr)
    if not faces:
        last_face_px = None
        return None
    from insightface.utils import face_align
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    last_face_px = float(max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]))
    return face_align.norm_crop(img_bgr, face.kps, image_size=size)


def crop_face_pil(img_pil: Image.Image, size: int = 512) -> Image.Image | None:
    bgr = cv2.cvtColor(np.asarray(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    cropped = crop_face_bgr(bgr, size)
    if cropped is None:
        return None
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def crop_from_path(path: Path, size: int = 512) -> Image.Image | None:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None
    cropped = crop_face_bgr(img_bgr, size)
    return None if cropped is None else Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    # ponytail: self-check — crops a sample image if given as arg
    import sys
    if len(sys.argv) < 2:
        print("usage: python face_crop.py <image> [out.png]")
        sys.exit(1)
    im = crop_from_path(Path(sys.argv[1]))
    assert im is not None, "no face detected"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/face_crop_test.png"
    im.save(out)
    print("saved", out, im.size)
