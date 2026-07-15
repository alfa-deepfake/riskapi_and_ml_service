"""Face-flashing active-light liveness verifier.

The screen shows random background/lighting frame pairs; a live face reflects
each flash, so its brightness tracks the challenge sequence. Replayed video or
a virtual camera cannot follow randomly chosen flashes.
"""

from face_flashing.active_light import (
    ActiveLightLivenessVerifier,
    ActiveLightResult,
    LightPair,
    active_light_result_to_dict,
)
from face_flashing.challenges import Challenge
from face_flashing.face import ExtractedFace, FaceExtractor, bgr_to_rgb

__all__ = [
    "ActiveLightLivenessVerifier",
    "ActiveLightResult",
    "Challenge",
    "ExtractedFace",
    "FaceExtractor",
    "LightPair",
    "active_light_result_to_dict",
    "bgr_to_rgb",
]
