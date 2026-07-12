from ml_service.services.gesture_service import (
    _face_reliably_present,
    _gesture_face_present,
)


def test_wall_flicker_is_not_a_present_face():
    # A blank wall that spuriously detects a "face" on a couple of frames must not count.
    assert _face_reliably_present(face_frames=3, processed_frames=49) is False
    assert _face_reliably_present(face_frames=8, processed_frames=100) is False  # 8% of frames


def test_subject_in_frame_is_present():
    assert _face_reliably_present(face_frames=40, processed_frames=49) is True
    assert _face_reliably_present(face_frames=8, processed_frames=10) is True


def test_empty_clip_is_not_present():
    assert _face_reliably_present(face_frames=0, processed_frames=0) is False


def test_confirmed_quick_gesture_is_present_despite_truncated_clip():
    # Regression: a real gesture confirmed within the first few frames breaks the
    # loop early, truncating processed_frames below FACE_MIN_FRAMES. It must NOT be
    # rejected — the sustained touch streak already proves the face was tracked.
    assert _gesture_face_present(confirmed=True, face_frames=4, processed_frames=5) is True


def test_unconfirmed_wall_flicker_is_still_rejected():
    # No confirmed touch → fall back to the ratio guard over the full clip.
    assert _gesture_face_present(confirmed=False, face_frames=3, processed_frames=180) is False
    assert _gesture_face_present(confirmed=False, face_frames=40, processed_frames=49) is True
