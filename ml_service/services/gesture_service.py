from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import GestureEvidence, ServiceAnalyzeResponse
from ml_service.core.checks import score_gesture
from ml_service.services.common import read_upload, safe_suffix, service_response, unavailable_check


class GestureService:
    name = "gesture"

    async def analyze_video(
        self,
        file: UploadFile,
        *,
        expected_action: str,
        face_present: bool | None,
    ) -> ServiceAnalyzeResponse:
        suffix = safe_suffix(file.filename, ".webm")
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await read_upload(file))
            tmp.flush()
            try:
                # Mediapipe processes up to max_frames synchronously — keep it off the event loop.
                result = await run_in_threadpool(_run_touch_detector, Path(tmp.name), expected_action)
            except Exception as exc:
                # Mediapipe/cv2 raise arbitrary exception types mid-clip; a
                # degraded check must never become an HTTP 500.
                reason = str(exc) if isinstance(exc, RuntimeError) else f"gesture inference failed: {type(exc).__name__}"
                evidence = GestureEvidence(expected_action=expected_action, detector="hands-pose-face-mesh", face_present=face_present)
                check = unavailable_check("gesture", 0.15, reason)
                return service_response(self.name, evidence, check, message=reason)

        evidence = GestureEvidence(
            expected_action=expected_action,
            observed_action=result["observed_action"],
            confidence=result["confidence"],
            detector="hands-pose-face-mesh",
            face_present=result["face_present"] if result["face_present"] is not None else face_present,
            frame_count=result["frame_count"],
            best_distance=result["best_distance"],
        )
        check = score_gesture(evidence, challenge=None)
        return service_response(self.name, evidence, check)


class PoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    visibility: float = 1.0


# Touch acceptance radius as a fraction of the outer inter-ocular distance, so
# "touching" means the same physical distance whether the face is near or far
# from the camera. The clamps keep a degenerate landmark scale from making the
# check impossible (tiny face) or trivial (huge face); the default applies
# until a face scale has been seen. 0.70 × a typical inter-ocular reading
# reproduces the old fixed 0.075 at usual webcam framing.
TOUCH_THRESHOLD_IOD = 0.70
TOUCH_THRESHOLD_MIN = 0.05
TOUCH_THRESHOLD_MAX = 0.14
TOUCH_THRESHOLD_DEFAULT = 0.075
# Net touching frames to confirm. Misses decay the score instead of resetting
# it: a hand right in front of a face flickers in and out of detection, and a
# single dropped frame mid-touch must not void the whole hold.
TOUCH_CONFIRM_SCORE = 4.0
TOUCH_MISS_DECAY = 1.0
# A real touch usually occludes the very landmarks that define the target —
# covering the nose with a palm makes the face mesh vanish at the moment of
# contact. Keep the last real target alive this many frames (~1.5s at 30fps)
# so the covering hand is measured against where the face just was.
TARGET_MEMORY_FRAMES = 45


@dataclass
class TouchTracker:
    """Per-frame touch decision state, independent of mediapipe/cv2.

    Feed it one update per processed frame: the target point actually detected
    on that frame (or None), every candidate hand landmark, and the current
    inter-ocular distance when known. Any hand landmark counts as a touch
    point — a fingertip of any finger, a knuckle, or the palm all qualify.
    """

    score: float = 0.0
    confirmed: bool = False
    target_frames: int = 0
    best_distance: float | None = None
    best_ratio: float | None = None
    _target: Point | None = field(default=None, repr=False)
    _target_age: int = field(default=0, repr=False)
    _iod: float | None = field(default=None, repr=False)

    def update(self, target: Point | None, hand_points: list[Point], iod: float | None) -> None:
        if iod is not None and iod > 1e-6:
            self._iod = iod
        if target is not None:
            self._target = target
            self._target_age = 0
            self.target_frames += 1
        elif self._target is not None:
            self._target_age += 1
            if self._target_age > TARGET_MEMORY_FRAMES:
                self._target = None
        if self._target is None or not hand_points:
            self.score = max(0.0, self.score - TOUCH_MISS_DECAY)
            return

        threshold = self.threshold
        distance = min(math.hypot(p.x - self._target.x, p.y - self._target.y) for p in hand_points)
        if self.best_distance is None or distance < self.best_distance:
            self.best_distance = distance
        ratio = distance / threshold
        if self.best_ratio is None or ratio < self.best_ratio:
            self.best_ratio = ratio
        if distance <= threshold:
            self.score += 1.0
            if self.score >= TOUCH_CONFIRM_SCORE:
                self.confirmed = True
        else:
            self.score = max(0.0, self.score - TOUCH_MISS_DECAY)

    @property
    def threshold(self) -> float:
        if self._iod is None:
            return TOUCH_THRESHOLD_DEFAULT
        return min(TOUCH_THRESHOLD_MAX, max(TOUCH_THRESHOLD_MIN, TOUCH_THRESHOLD_IOD * self._iod))

    @property
    def confidence(self) -> float:
        if self.best_ratio is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.best_ratio / 2.0))


def _run_touch_detector(
    video_path: Path,
    expected_action: str,
    *,
    max_frames: int = 240,
) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("gesture runtime requires opencv-python") from exc
    mp = _import_mediapipe()

    target = TARGETS.get(expected_action.removeprefix("touch_"))
    if target is None:
        raise RuntimeError(f"unsupported gesture action: {expected_action}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("could not open gesture video")

    # max_frames is calibrated for ~30fps (8s of clip). A 60fps camera would
    # otherwise hit the cap at 4s and lose late touches — subsample instead so
    # the cap always covers the same wall-clock span.
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = max(1, round(fps / 30.0)) if fps > 0 else 1

    tracker = TouchTracker()
    frame_count = 0
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    try:
        with (
            # static_image_mode: full detection on every frame instead of
            # detect-then-track. Tracking loses the hand exactly when it moves
            # fast toward the face, and coasts on a stale ROI afterwards; the
            # per-frame detector never does. Detection floors are deliberately
            # low (dim rooms) — the leaky touch accumulator and the face-frame
            # ratio guard absorb the extra flicker a loose detector produces.
            mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.35) as hands,
            mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.40) as pose,
            mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.40) as face_mesh,
        ):
            read_index = 0
            while frame_count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                read_index += 1
                if (read_index - 1) % stride != 0:
                    continue
                frame_count += 1
                rgb = _enhance_if_dark(cv2, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rgb.flags.writeable = False
                face_results = face_mesh.process(rgb)
                target_point = target.from_face(face_results)
                if target_point is None:
                    # Pose is the expensive fallback — consulted only on frames
                    # where the face mesh has no target (occlusion, bad light,
                    # face partly out of frame).
                    target_point = target.from_pose(pose.process(rgb))
                hand_points = _iter_hand_points(hands.process(rgb))
                tracker.update(target_point, hand_points, _interocular_distance(face_results))
                if tracker.confirmed:
                    break
    finally:
        cap.release()

    # A confirmed touch already required real target detections within the
    # memory window, so it is presence on its own; the early break truncates
    # frame_count, which would otherwise starve the ratio guard. The ratio
    # guard only applies to the non-confirmed path, which runs the full clip.
    face_present = _gesture_face_present(tracker.confirmed, tracker.target_frames, frame_count)
    return {
        "observed_action": expected_action if tracker.confirmed else "not_completed",
        "confidence": tracker.confidence,
        "face_present": face_present,
        "frame_count": frame_count,
        "best_distance": tracker.best_distance,
    }


# Mediapipe detectors go blind well before a scene is truly dark. CLAHE on the
# luma plane restores local contrast in dim rooms; brighter clips pass through
# untouched so normal captures keep their native statistics.
DARK_FRAME_MEAN_LUMA = 70.0


def _enhance_if_dark(cv2, rgb):
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    luma = ycrcb[:, :, 0]
    if float(luma.mean()) >= DARK_FRAME_MEAN_LUMA:
        return rgb
    ycrcb[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(luma)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ponytail: face must persist across the clip, not flicker on one frame. A single
# spurious face-mesh hit on a wall/background must not mark the subject "present".
# Tune these two if real gestures that briefly occlude the face start failing.
FACE_MIN_FRAMES = 8
FACE_MIN_RATIO = 0.30


def _face_reliably_present(face_frames: int, processed_frames: int) -> bool:
    if processed_frames <= 0:
        return False
    return face_frames >= FACE_MIN_FRAMES and face_frames >= FACE_MIN_RATIO * processed_frames


def _gesture_face_present(confirmed: bool, face_frames: int, processed_frames: int) -> bool:
    # A confirmed touch requires real face/pose target detections at most
    # TARGET_MEMORY_FRAMES before the hold, so it is presence on its own.
    # Otherwise fall back to the ratio guard over the (fully processed) clip.
    return confirmed or _face_reliably_present(face_frames, processed_frames)


def _import_mediapipe():
    sounddevice_stub = types.ModuleType("sounddevice")

    class InputStream:
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("sounddevice is disabled in this CV-only service")

    sounddevice_stub.InputStream = InputStream
    sys.modules.setdefault("sounddevice", sounddevice_stub)
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("gesture runtime requires mediapipe") from exc
    return mp


def _pose_point(pose_results: object, landmark_id: PoseLandmark, min_visibility: float = 0.45) -> Point | None:
    if not pose_results or not pose_results.pose_landmarks:
        return None
    landmark = pose_results.pose_landmarks.landmark[int(landmark_id)]
    visibility = getattr(landmark, "visibility", 1.0)
    if visibility < min_visibility:
        return None
    return Point(landmark.x, landmark.y, visibility)


def _face_point(face_results: object, landmark_ids: tuple[int, ...]) -> Point | None:
    if not face_results or not face_results.multi_face_landmarks:
        return None
    landmarks = face_results.multi_face_landmarks[0].landmark
    points = [landmarks[index] for index in landmark_ids]
    return Point(sum(point.x for point in points) / len(points), sum(point.y for point in points) / len(points))


def _midpoint(left: Point | None, right: Point | None) -> Point | None:
    if left is None or right is None:
        return None
    return Point((left.x + right.x) / 2.0, (left.y + right.y) / 2.0, min(left.visibility, right.visibility))


# Outer eye corners: 33 is on the subject's right eye, 263 on the left.
_IOD_LANDMARKS = (33, 263)


def _interocular_distance(face_results: object) -> float | None:
    if not face_results or not face_results.multi_face_landmarks:
        return None
    landmarks = face_results.multi_face_landmarks[0].landmark
    right, left = (landmarks[index] for index in _IOD_LANDMARKS)
    return math.hypot(left.x - right.x, left.y - right.y)


@dataclass(frozen=True)
class TargetSpec:
    face_ids: tuple[int, ...]
    pose_fallback: Callable[[object], Point | None] | None = None

    def from_face(self, face_results: object) -> Point | None:
        return _face_point(face_results, self.face_ids)

    def from_pose(self, pose_results: object) -> Point | None:
        return self.pose_fallback(pose_results) if self.pose_fallback else None


# Face-mesh landmark ids are anatomical (the uploaded stream is not mirrored):
# 362/263 is the subject's LEFT eye, 33/133 the RIGHT. Each target carries a
# pose-model fallback for frames where the face mesh finds nothing.
TARGETS: dict[str, TargetSpec] = {
    "nose": TargetSpec((1, 4, 5), lambda pose_results: _pose_point(pose_results, PoseLandmark.NOSE)),
    "mouth": TargetSpec(
        (13, 14, 78, 308),
        lambda pose_results: _midpoint(
            _pose_point(pose_results, PoseLandmark.MOUTH_LEFT),
            _pose_point(pose_results, PoseLandmark.MOUTH_RIGHT),
        ),
    ),
    "left_eye": TargetSpec((362, 263), lambda pose_results: _pose_point(pose_results, PoseLandmark.LEFT_EYE)),
    "right_eye": TargetSpec((33, 133), lambda pose_results: _pose_point(pose_results, PoseLandmark.RIGHT_EYE)),
}


def _iter_hand_points(hand_results: object) -> list[Point]:
    # Every hand landmark is a candidate touch point: people point with any
    # finger, with a knuckle, or cover the target with the whole palm — the
    # palm-side landmarks are the ones that land on the face in that case.
    if not hand_results or not hand_results.multi_hand_landmarks:
        return []
    return [
        Point(landmark.x, landmark.y)
        for hand_landmarks in hand_results.multi_hand_landmarks
        for landmark in hand_landmarks.landmark
    ]
