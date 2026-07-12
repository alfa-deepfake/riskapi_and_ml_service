from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import UploadFile

from ml_service.api.schemas import GestureEvidence, ServiceAnalyzeResponse
from ml_service.core.checks import score_gesture
from ml_service.services.common import service_response, unavailable_check


class GestureService:
    name = "gesture"

    async def analyze_video(
        self,
        file: UploadFile,
        *,
        expected_action: str,
        face_present: bool | None,
    ) -> ServiceAnalyzeResponse:
        suffix = Path(file.filename or "gesture.webm").suffix or ".webm"
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            try:
                result = _run_touch_detector(Path(tmp.name), expected_action)
            except RuntimeError as exc:
                evidence = GestureEvidence(expected_action=expected_action, detector="hands-pose-face-mesh", face_present=face_present)
                check = unavailable_check("gesture", 0.15, str(exc))
                return service_response(self.name, evidence, check, message=str(exc))

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
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10


class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class Target:
    name: str
    resolver: Callable[[object, object], Point | None]


@dataclass(frozen=True)
class FingerTrace:
    landmarks: tuple[Point, ...]

    @property
    def tip(self) -> Point:
        return self.landmarks[-1]


FINGER_CHAINS: tuple[tuple[HandLandmark, ...], ...] = (
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    (HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    (HandLandmark.WRIST, HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
)


def _run_touch_detector(
    video_path: Path,
    expected_action: str,
    *,
    threshold: float = 0.075,
    hold_frames: int = 4,
    max_frames: int = 180,
) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("gesture runtime requires opencv-python") from exc
    mp = _import_mediapipe()

    target = TARGETS.get(_target_name_from_action(expected_action))
    if target is None:
        raise RuntimeError(f"unsupported gesture action: {expected_action}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("could not open gesture video")

    face_seen = False
    best_distance: float | None = None
    touch_streak = 0
    confirmed = False
    frame_count = 0
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    with (
        mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.55, min_tracking_confidence=0.55) as hands,
        mp_pose.Pose(model_complexity=1, min_detection_confidence=0.55, min_tracking_confidence=0.55) as pose,
        mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.55, min_tracking_confidence=0.55) as face_mesh,
    ):
        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands.process(rgb)
            pose_results = pose.process(rgb)
            face_results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            target_point = target.resolver(pose_results, face_results)
            if target_point is not None:
                face_seen = True
            fingers = _iter_fingers(hand_results)
            if target_point is None or not fingers:
                touch_streak = 0
                continue

            closest_distance = min(_normalized_distance(finger.tip, target_point) for finger in fingers)
            best_distance = closest_distance if best_distance is None else min(best_distance, closest_distance)
            touching = closest_distance <= threshold
            touch_streak = touch_streak + 1 if touching else 0
            if touch_streak >= hold_frames:
                confirmed = True
                break
    cap.release()

    confidence = 0.0 if best_distance is None else max(0.0, min(1.0, 1.0 - best_distance / max(threshold * 2.0, 1e-6)))
    return {
        "observed_action": expected_action if confirmed else "not_completed",
        "confidence": confidence,
        "face_present": face_seen,
        "frame_count": frame_count,
        "best_distance": best_distance,
    }


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


def _pose_target(landmark_id: PoseLandmark) -> Callable[[object, object], Point | None]:
    return lambda pose_results, _face_results: _pose_point(pose_results, landmark_id)


def _face_target(*landmark_ids: int) -> Callable[[object, object], Point | None]:
    return lambda _pose_results, face_results: _face_point(face_results, landmark_ids)


def _mouth_target(pose_results: object, face_results: object) -> Point | None:
    return _face_point(face_results, (13, 14, 78, 308)) or _midpoint(
        _pose_point(pose_results, PoseLandmark.MOUTH_LEFT),
        _pose_point(pose_results, PoseLandmark.MOUTH_RIGHT),
    )


def _nose_target(pose_results: object, face_results: object) -> Point | None:
    return _face_point(face_results, (1, 4, 5)) or _pose_point(pose_results, PoseLandmark.NOSE)


TARGETS: dict[str, Target] = {
    "nose": Target("nose", _nose_target),
    "mouth": Target("mouth", _mouth_target),
    "left_eye": Target("left_eye", _face_target(33, 133)),
    "right_eye": Target("right_eye", _face_target(362, 263)),
}


def _target_name_from_action(action: str) -> str:
    mapping = {
        "touch_mouth": "mouth",
        "touch_nose": "nose",
        "touch_left_eye": "left_eye",
        "touch_right_eye": "right_eye",
    }
    return mapping.get(action, action.removeprefix("touch_"))


def _iter_fingers(hand_results: object) -> list[FingerTrace]:
    if not hand_results or not hand_results.multi_hand_landmarks:
        return []
    fingers: list[FingerTrace] = []
    for hand_landmarks in hand_results.multi_hand_landmarks:
        for chain in FINGER_CHAINS:
            fingers.append(
                FingerTrace(
                    tuple(
                        Point(
                            hand_landmarks.landmark[int(landmark_id)].x,
                            hand_landmarks.landmark[int(landmark_id)].y,
                        )
                        for landmark_id in chain
                    )
                )
            )
    return fingers


def _normalized_distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
