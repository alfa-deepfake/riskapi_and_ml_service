from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

FACE_CROP_SIZE = 256
CROP_MARGIN = 1.4
MIN_CROP_SIDE = 24


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image_bgr[..., ::-1])


@dataclass
class ExtractedFace:
    image_rgb: np.ndarray


class FaceExtractor:
    """Square face crop via mediapipe.

    ``allow_reuse=True`` falls back to the previous frame's box when detection
    fails — a full-screen white flash can blow out the face enough to break
    detection on the lighting frame while the crop location is unchanged.
    """

    def __init__(self, *, crop_size: int = FACE_CROP_SIZE, min_confidence: float = 0.5) -> None:
        self.crop_size = crop_size
        # Short-range model (0) is tuned for selfie distance; the full-range
        # model (1) covers faces beyond ~2m / small in wide-FOV frames. Try
        # short-range first, fall back to full-range on a miss.
        self._detectors = [
            mp.solutions.face_detection.FaceDetection(
                model_selection=selection, min_detection_confidence=min_confidence
            )
            for selection in (0, 1)
        ]
        self._last_box: tuple[int, int, int, int] | None = None

    def close(self) -> None:
        """Release the mediapipe graphs — GC does not free their native memory."""
        for detector in self._detectors:
            detector.close()

    def extract(self, image_rgb: np.ndarray, *, allow_reuse: bool = False) -> ExtractedFace | None:
        box = self._detect_box(image_rgb)
        if box is None:
            if not (allow_reuse and self._last_box is not None):
                return None
            box = self._last_box
        self._last_box = box
        x0, y0, side, _ = box
        crop = image_rgb[y0 : y0 + side, x0 : x0 + side]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        return ExtractedFace(image_rgb=crop)

    def _detect_box(self, image_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        results = None
        for detector in self._detectors:
            results = detector.process(image_rgb)
            if results.detections:
                break
        if not results or not results.detections:
            return None
        best = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        ).location_data.relative_bounding_box
        height, width = image_rgb.shape[:2]
        center_x = (best.xmin + best.width / 2.0) * width
        center_y = (best.ymin + best.height / 2.0) * height
        side = int(round(max(best.width * width, best.height * height) * CROP_MARGIN))
        side = min(side, width, height)
        if side < MIN_CROP_SIDE:
            return None
        x0 = max(0, min(int(round(center_x - side / 2.0)), width - side))
        y0 = max(0, min(int(round(center_y - side / 2.0)), height - side))
        return (x0, y0, side, side)
