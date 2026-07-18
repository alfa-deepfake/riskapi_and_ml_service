from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import ActiveLightAnalyzeRequest, ActiveLightEvidence, ServiceAnalyzeResponse
from ml_service.config import Settings
from ml_service.core.checks import score_active_light
from ml_service.services.common import read_upload, service_response, unavailable_check


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
        except HTTPException:
            raise  # read_upload's 413 is a real client error, not a degraded check
        except Exception as exc:
            # Mediapipe/cv2/numpy raise arbitrary types on malformed frames; a
            # degraded check must never become an HTTP 500.
            reason = str(exc) if isinstance(exc, RuntimeError) else f"active light verification failed: {type(exc).__name__}"
            # face_present=False only when frames were readable but held no
            # face — that is a legitimate liveness fail, scored as such. Any
            # other breakage is "unknown": this evidence is re-scored at final
            # submission, and a fabricated False would hard-fail the user at
            # risk 0.95 for a server-side hiccup.
            if "no valid face frame pairs" in reason:
                evidence = ActiveLightEvidence(detector="face-flashing-frame-pairs", face_present=False)
                check = score_active_light(evidence, challenge=None, settings=self._settings)
            else:
                evidence = ActiveLightEvidence(detector="face-flashing-frame-pairs")
                check = unavailable_check("active_light", 0.22, reason)
            return service_response(self.name, evidence, check, message=reason)

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
    # The challenge issues 8 pairs (16 frames). Cap hard: an unbounded pairs
    # list would spool tens of GB to temp disk and run 1000s of mediapipe
    # extractions per request (CPU/disk DoS).
    pairs_spec = pairs_spec[:16]
    files = files[:32]

    uploads = {file.filename: file for file in files if file.filename}

    def build_pairs(saved: dict[str, Path]) -> list:
        extractor = FaceExtractor()
        try:
            return _extract_pairs(extractor, saved)
        finally:
            extractor.close()

    def _extract_pairs(extractor: FaceExtractor, saved: dict[str, Path]) -> list:
        light_pairs = []
        for pair in pairs_spec:
            background_name = pair.get("background_file")
            lighting_name = pair.get("lighting_file")
            if background_name not in saved or lighting_name not in saved:
                continue
            background_bgr = cv2.imread(str(saved[background_name]), cv2.IMREAD_COLOR)
            lighting_bgr = cv2.imread(str(saved[lighting_name]), cv2.IMREAD_COLOR)
            if background_bgr is None or lighting_bgr is None:
                continue
            try:
                bg_challenge = Challenge.from_dict(pair["background_challenge"])
                light_challenge = Challenge.from_dict(pair["lighting_challenge"])
            except Exception:
                # Client-supplied manifest — a malformed pair is skipped, not a 500.
                continue
            # A full-screen WHITE frame can blow out the face enough to break
            # detection. Polarity is randomized, so the blown frame is the
            # background on white-bg pairs and the lighting on black-bg pairs.
            # Detect on the darker frame first (allow_reuse=False seeds the box),
            # then reuse that box on the brighter frame — otherwise ~half of a
            # genuine user's pairs drop and the temporal correlation collapses.
            bg_darker = bg_challenge.mean_screen_luma() <= light_challenge.mean_screen_luma()
            if bg_darker:
                background_face = extractor.extract(bgr_to_rgb(background_bgr), allow_reuse=False)
                lighting_face = extractor.extract(bgr_to_rgb(lighting_bgr), allow_reuse=True)
            else:
                lighting_face = extractor.extract(bgr_to_rgb(lighting_bgr), allow_reuse=False)
                background_face = extractor.extract(bgr_to_rgb(background_bgr), allow_reuse=True)
            if background_face is None or lighting_face is None:
                continue
            light_pairs.append(
                LightPair(
                    background_challenge=bg_challenge,
                    background_rgb=background_face.image_rgb,
                    lighting_challenge=light_challenge,
                    lighting_rgb=lighting_face.image_rgb,
                )
            )
        return light_pairs

    with TemporaryDirectory() as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        saved: dict[str, Path] = {}
        for name, upload in uploads.items():
            path = tmp_dir / Path(name).name
            path.write_bytes(await read_upload(upload))
            saved[name] = path
        # Face extraction over every uploaded frame is CPU-heavy — keep it off the event loop.
        light_pairs = await run_in_threadpool(build_pairs, saved)

    if not light_pairs:
        raise RuntimeError("no valid face frame pairs for face_flashing verifier")
    result = await run_in_threadpool(ActiveLightLivenessVerifier().verify, light_pairs)
    return active_light_result_to_dict(result, include_pairs=False)
