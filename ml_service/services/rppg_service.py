from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ml_service.api.schemas import RppgAnalyzeRequest, RppgEvidence, ServiceAnalyzeResponse
from ml_service.config import Settings
from ml_service.core.checks import score_rppg
from ml_service.services.common import read_upload, safe_suffix, service_response, unavailable_check


class RppgService:
    name = "rppg"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze_samples(self, payload: RppgAnalyzeRequest) -> ServiceAnalyzeResponse:
        evidence = RppgEvidence(
            samples=payload.samples,
            sample_rate_hz=payload.sample_rate_hz,
            window_seconds=payload.window_seconds,
            detector="browser-luma-samples",
            face_present=payload.face_present,
            face_confidence=payload.face_confidence,
        )
        check = score_rppg(evidence, self._settings)
        return service_response(self.name, evidence, check)

    async def analyze_video(self, file: UploadFile, *, face_present: bool | None, face_confidence: float | None) -> ServiceAnalyzeResponse:
        suffix = safe_suffix(file.filename, ".webm")
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await read_upload(file))
            tmp.flush()
            try:
                result = await run_in_threadpool(_run_rppg_runtime, Path(tmp.name))
            except Exception as exc:
                # _run_rppg_runtime wraps known failures in RuntimeError; anything
                # else is still a degraded check, never an HTTP 500.
                reason = str(exc) if isinstance(exc, RuntimeError) else f"rPPG inference failed: {type(exc).__name__}"
                evidence = RppgEvidence(face_present=face_present, face_confidence=face_confidence)
                check = unavailable_check("rppg", 0.18, reason)
                return service_response(self.name, evidence, check, message=reason)

        evidence = RppgEvidence(
            bpm=result.get("bpm"),
            signal_quality=result.get("signal_quality"),
            latency=result.get("latency"),
            hrv=result.get("hrv", {}),
            samples=result.get("samples", []),
            sample_rate_hz=result.get("sample_rate_hz"),
            detector=result.get("detector"),
            ensemble=result.get("ensemble", {}),
            face_present=result.get("face_present") if result.get("face_present") is not None else face_present,
            face_confidence=result.get("face_confidence") if result.get("face_confidence") is not None else face_confidence,
        )
        check = score_rppg(evidence, self._settings)
        return service_response(self.name, evidence, check)


_MODEL_LOCK = threading.Lock()

# Three independent architectures over the same clip: they degrade differently
# in sensor noise, and several estimators agreeing on one physiological BPM is
# far stronger evidence than any single model's SQI — that agreement is what
# rescues genuine dim-room clips without also rescuing deepfakes.
ENSEMBLE_MODELS = ("FacePhys.rlap", "PhysNet.rlap", "TSCAN.rlap")
# A first read this clean needs no second opinion — skip the rest of the
# ensemble so well-lit sessions keep today's latency.
FAST_PATH_SQI = 0.60
# Below this fused quality the classical POS estimator gets a vote too.
POS_RESCUE_SQI = 0.50
# Estimators within this BPM window are counted as agreeing; each extra
# agreeing estimator adds a quality bonus on top of the best member's SQI.
AGREEMENT_BPM = 7.0
CONSENSUS_BONUS = 0.08


@lru_cache(maxsize=None)
def _rppg_model(name: str):
    # ~1min to build each (session + weights bundled in the open-rppg wheel):
    # cache for the process lifetime and warm off the request path at startup.
    import rppg

    return rppg.Model(name)


# Ensemble members that finished building. A live request never waits out a
# cold ~1min model build for a *secondary* opinion — those join the ensemble
# as the background warmup completes; the primary model builds on demand.
_WARMED_MODELS: set[str] = set()


def warm_rppg_model() -> None:
    """Preload the rPPG ensemble in the background; never raises."""
    for name in ENSEMBLE_MODELS:
        try:
            _rppg_model(name)
        except Exception:
            continue
        _WARMED_MODELS.add(name)


def _run_rppg_runtime(video_path: Path) -> dict:
    primary, *secondary = ENSEMBLE_MODELS
    available = [primary, *[name for name in secondary if name in _WARMED_MODELS]]
    candidates: list[dict] = []
    failure: Exception | None = None
    dependency_missing: Exception | None = None
    for name in available:
        try:
            model = _rppg_model(name)
        except ImportError as exc:
            dependency_missing = exc
            break
        except Exception as exc:
            failure = exc
            continue
        try:
            # Each model is a stateful stream processor — one video at a time.
            with _MODEL_LOCK:
                result = model.process_video(str(video_path)) or {}
        except Exception as exc:
            failure = exc
            continue
        bpm = _to_float(result.get("hr"))
        if bpm is not None and not (20.0 <= bpm <= 220.0):
            bpm = None
        sqi = _to_float(result.get("SQI"))
        if sqi is not None:
            sqi = max(0.0, min(1.0, sqi))
        candidates.append({"source": name, "bpm": bpm, "sqi": sqi, "raw": result})
        if sqi is not None and sqi >= FAST_PATH_SQI and bpm is not None and 45.0 <= bpm <= 140.0:
            break
    if dependency_missing is not None:
        raise RuntimeError("rPPG runtime dependency is missing: pip install open-rppg") from dependency_missing
    if not candidates:
        raise RuntimeError(f"rPPG runtime failed: {type(failure).__name__}" if failure else "rPPG runtime produced no result")

    fused = _fuse_candidates(candidates)
    pos_ran = False
    if fused["bpm"] is None or fused["sqi"] is None or fused["sqi"] < POS_RESCUE_SQI:
        pos = _pos_candidate(video_path)
        if pos is not None:
            pos_ran = True
            candidates.append(pos)
            fused = _fuse_candidates(candidates)

    raw = _best_raw(candidates, fused["sources"])
    latency = _to_float(raw.get("latency"))
    if latency is not None and latency < 0:
        # schema requires latency >= 0; a junk model value must not 500 the check
        latency = None
    # Face presence stays with the caller-provided evidence: open-rppg's frame
    # statistics count forward-filled frames and cannot be trusted for it.
    return {
        "bpm": fused["bpm"],
        "signal_quality": fused["sqi"],
        "latency": latency,
        "hrv": {key: _to_float(value) for key, value in (raw.get("hrv") or {}).items()},
        "samples": [],
        "sample_rate_hz": None,
        "detector": "open-rppg-ensemble+pos" if pos_ran else "open-rppg-ensemble",
        "ensemble": _ensemble_readings(candidates, fused),
        "face_present": None,
        "face_confidence": None,
    }


def _fuse_candidates(candidates: list[dict]) -> dict:
    """Cross-estimator consensus: the largest BPM cluster wins; agreement
    between independent estimators boosts quality above any single SQI."""
    valid = [c for c in candidates if c["bpm"] is not None and c["sqi"] is not None]
    if not valid:
        sqis = [c["sqi"] for c in candidates if c["sqi"] is not None]
        return {"bpm": None, "sqi": max(sqis) if sqis else None, "sources": [], "agreeing": 0}

    cluster: list[dict] = []
    for anchor in valid:
        members = [c for c in valid if abs(c["bpm"] - anchor["bpm"]) <= AGREEMENT_BPM]
        if len(members) > len(cluster) or (
            len(members) == len(cluster) and max(m["sqi"] for m in members) > max(m["sqi"] for m in cluster)
        ):
            cluster = members

    best_sqi = max(member["sqi"] for member in cluster)
    sqi = min(1.0, best_sqi + CONSENSUS_BONUS * (len(cluster) - 1))
    return {
        "bpm": float(median(member["bpm"] for member in cluster)),
        "sqi": sqi,
        "sources": [member["source"] for member in cluster],
        "agreeing": len(cluster),
    }


def _pos_candidate(video_path: Path) -> dict | None:
    """Classical multi-region POS estimate; an optional second opinion that
    must never sink the check when its own runtime is unavailable."""
    try:
        from ml_service.services.rppg_pos import estimate_pulse_pos

        result = estimate_pulse_pos(video_path)
    except Exception:
        return None
    if result is None or result.bpm is None:
        return None
    return {"source": "pos", "bpm": result.bpm, "sqi": result.quality, "raw": {}, "pos": result}


def _best_raw(candidates: list[dict], winning_sources: list[str]) -> dict:
    """hrv/latency come from the strongest model inside the winning cluster."""
    members = [c for c in candidates if c["source"] in winning_sources and c["raw"]]
    pool = members or [c for c in candidates if c["raw"]]
    if not pool:
        return {}
    return max(pool, key=lambda c: c["sqi"] if c["sqi"] is not None else -1.0)["raw"]


def _ensemble_readings(candidates: list[dict], fused: dict) -> dict:
    readings: dict[str, float | None] = {"agreeing_estimators": float(fused["agreeing"])}
    for candidate in candidates:
        readings[f"{candidate['source']}_bpm"] = candidate["bpm"]
        readings[f"{candidate['source']}_sqi"] = candidate["sqi"]
        pos = candidate.get("pos")
        if pos is not None:
            readings["pos_agreeing_regions"] = float(pos.agreeing)
            for estimate in pos.estimates:
                readings[f"pos_{estimate.region}_bpm"] = estimate.bpm
                readings[f"pos_{estimate.region}_snr_db"] = estimate.snr_db
    return readings


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
