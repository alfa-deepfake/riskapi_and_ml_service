import pytest

from ml_service.services.gesture_service import (
    TARGET_MEMORY_FRAMES,
    Point,
    TouchTracker,
)

NOSE = Point(0.5, 0.5)
IOD = 0.10  # typical webcam framing → threshold 0.7 * 0.10 = 0.07
ON_TARGET = [Point(0.505, 0.503)]
FAR_AWAY = [Point(0.9, 0.9)]


def test_touch_confirms_after_net_hold():
    tracker = TouchTracker()
    for _ in range(4):
        tracker.update(NOSE, ON_TARGET, IOD)
    assert tracker.confirmed
    assert tracker.confidence >= 0.5


def test_single_dropped_detection_does_not_reset_the_hold():
    # A hand right in front of a face flickers in and out of detection; one
    # missed frame mid-touch must decay the hold, not void it.
    tracker = TouchTracker()
    for _ in range(3):
        tracker.update(NOSE, ON_TARGET, IOD)
    tracker.update(NOSE, [], IOD)
    for _ in range(2):
        tracker.update(NOSE, ON_TARGET, IOD)
    assert tracker.confirmed


def test_palm_covering_the_face_counts_via_remembered_target():
    # Covering the nose with a palm hides the face mesh at the moment of
    # contact — the hand is then measured against where the face just was.
    tracker = TouchTracker()
    tracker.update(NOSE, [], IOD)
    for _ in range(4):
        tracker.update(None, ON_TARGET, None)
    assert tracker.confirmed


def test_remembered_target_expires():
    tracker = TouchTracker()
    tracker.update(NOSE, [], IOD)
    for _ in range(TARGET_MEMORY_FRAMES + 1):
        tracker.update(None, [], None)
    tracker.update(None, ON_TARGET, None)
    assert not tracker.confirmed
    assert tracker.score == 0.0


def test_hand_far_from_target_never_confirms():
    tracker = TouchTracker()
    for _ in range(30):
        tracker.update(NOSE, FAR_AWAY, IOD)
    assert not tracker.confirmed
    assert tracker.confidence < 0.5


def test_threshold_scales_with_face_size():
    close = TouchTracker()
    close.update(NOSE, [], 0.19)
    assert close.threshold == pytest.approx(0.133)  # 0.7 × iod, under the max clamp

    far = TouchTracker()
    far.update(NOSE, [], 0.05)
    assert far.threshold == 0.05  # clamped to the floor

    unknown = TouchTracker()
    assert unknown.threshold == 0.075  # no face scale seen yet → legacy default
