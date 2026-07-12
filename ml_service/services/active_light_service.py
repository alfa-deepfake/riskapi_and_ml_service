from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import UploadFile

from ml_service.api.schemas import ActiveLightAnalyzeRequest, ActiveLightEvidence, ServiceAnalyzeResponse
from ml_service.config import Settings
from ml_service.core.checks import score_active_light
from ml_service.services.common import service_response, unavailable_check


class ActiveLightService:
    name = "active_light"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(self, payload: ActiveLightAnalyzeRequest) -> ServiceAnalyzeResponse:
        evidence = ActiveLightEvidence(
            expected_luma=payload.expected_luma,
            observed_face_luma=payload.observed_face_luma,
            detector="luma-correlation",
            face_present=payload.face_present,
            face_confidence=payload.face_confidence,
        )
        check = score_active_light(evidence, challenge=None, settings=self._settings)
        return service_response(self.name, evidence, check)

    async def analyze_frame_pairs(self, *, manifest: str, files: list[UploadFile]) -> ServiceAnalyzeResponse:
        try:
            result = await _run_face_flashing_verifier(manifest=manifest, files=files)
        except RuntimeError as exc:
            evidence = ActiveLightEvidence(detector="face-flashing-frame-pairs", face_present=False)
            check = unavailable_check("active_light", 0.22, str(exc))
            return service_response(self.name, evidence, check, message=str(exc))

        evidence = ActiveLightEvidence(
            detector="face-flashing-frame-pairs",
            verifier_score=result["score"],
            pair_count=result["pair_count"],
            temporal_correlation=result["temporal"]["zero_lag_correlation"],
            best_correlation=result["temporal"]["best_correlation"],
            spatial_contrast=result["spatial"]["median_contrast"],
            spatial_center_error=result["spatial"]["median_center_error"],
            response_snr=result["median_response_snr"],
            response_magnitude=result["median_response_magnitude"],
            color_cosine=result["mean_color_cosine"],
            face_present=result["pair_count"] > 0,
            face_confidence=1.0 if result["pair_count"] > 0 else 0.0,
        )
        check = score_active_light(evidence, challenge=None, settings=self._settings)
        return service_response(self.name, evidence, check)


async def _run_face_flashing_verifier(*, manifest: str, files: list[UploadFile]) -> dict:
    _ensure_repo_root_on_path()
    try:
        import cv2
        from face_flashing.active_light import ActiveLightLivenessVerifier, LightPair, active_light_result_to_dict
        from face_flashing.challenges import Challenge
        from face_flashing.face import FaceExtractor, bgr_to_rgb
    except Exception as exc:
        raise RuntimeError(f"face_flashing runtime is unavailable: {type(exc).__name__}") from exc

    try:
        data = json.loads(manifest)
        pairs_spec = data["pairs"]
    except Exception as exc:
        raise RuntimeError("invalid face_flashing manifest") from exc
    if not isinstance(pairs_spec, list) or not pairs_spec:
        raise RuntimeError("face_flashing manifest has no pairs")

    uploads = {file.filename: file for file in files if file.filename}
    extractor = FaceExtractor()
    light_pairs = []
    with TemporaryDirectory() as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        saved: dict[str, Path] = {}
        for name, upload in uploads.items():
            path = tmp_dir / Path(name).name
            path.write_bytes(await upload.read())
            saved[name] = path

        for pair in pairs_spec:
            background_name = pair.get("background_file")
            lighting_name = pair.get("lighting_file")
            if background_name not in saved or lighting_name not in saved:
                continue
            background_bgr = cv2.imread(str(saved[background_name]), cv2.IMREAD_COLOR)
            lighting_bgr = cv2.imread(str(saved[lighting_name]), cv2.IMREAD_COLOR)
            if background_bgr is None or lighting_bgr is None:
                continue
            background_face = extractor.extract(bgr_to_rgb(background_bgr), allow_reuse=False)
            lighting_face = extractor.extract(bgr_to_rgb(lighting_bgr), allow_reuse=True)
            if background_face is None or lighting_face is None:
                continue
            light_pairs.append(
                LightPair(
                    background_challenge=Challenge.from_dict(pair["background_challenge"]),
                    background_rgb=background_face.image_rgb,
                    lighting_challenge=Challenge.from_dict(pair["lighting_challenge"]),
                    lighting_rgb=lighting_face.image_rgb,
                )
            )

    if not light_pairs:
        raise RuntimeError("no valid face frame pairs for face_flashing verifier")
    result = ActiveLightLivenessVerifier().verify(light_pairs)
    return active_light_result_to_dict(result, include_pairs=False)


def _ensure_repo_root_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
