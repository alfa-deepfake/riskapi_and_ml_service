from ml_service.services.gesture_service import _face_reliably_present


def test_wall_flicker_is_not_a_present_face():
    # A blank wall that spuriously detects a "face" on a couple of frames must not count.
    assert _face_reliably_present(face_frames=3, processed_frames=49) is False
    assert _face_reliably_present(face_frames=8, processed_frames=100) is False  # 8% of frames


def test_subject_in_frame_is_present():
    assert _face_reliably_present(face_frames=40, processed_frames=49) is True
    assert _face_reliably_present(face_frames=8, processed_frames=10) is True


def test_empty_clip_is_not_present():
    assert _face_reliably_present(face_frames=0, processed_frames=0) is False
