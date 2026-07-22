"""Classical POS rPPG estimator (Wang et al. 2017) with multi-region consensus.

Second opinion for the learned rPPG ensemble, aimed at dim rooms: spatial mean
pooling over large skin regions recovers a pulse well below per-pixel sensor
noise, and cross-region BPM agreement separates a real perfused face (forehead
and both cheeks pulse at the same frequency) from noise or a synthesized face
(which do not agree). Camera noise is spatially uncorrelated; blood volume is
not — that asymmetry is the whole trick.

The signal math (`estimate_from_traces`) is numpy/scipy only and unit-testable
with synthetic traces; `estimate_pulse_pos` adds the cv2 + mediapipe frame
reader in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch

TARGET_FPS = 30.0
# 45–141 bpm — matches the physiological band the rPPG check scores against.
BAND_HZ = (0.75, 2.35)
# SNR is judged over a wider band so the first harmonic of a fast pulse still
# counts as signal instead of noise (de Haan & Jeanne 2013). The signal windows
# must cover the Welch mainlobe (~±0.23 Hz at nperseg 256 / 30 fps), otherwise
# a clean tone's own spectral leakage is counted as noise.
SNR_BAND_HZ = (0.6, 4.8)
PEAK_WINDOW_HZ = 0.25
HARMONIC_WINDOW_HZ = 0.35
# Must stay below the frontend's 8s capture: the usable face trace of an 8s
# clip lands just under 8s after detection warmup and dropped frames.
MIN_TRACE_SECONDS = 6.0
MAX_FRAMES = 1200
# Every Nth frame runs face detection; the box is forward-filled in between —
# the subject holds still during the pulse step, RGB means are per-frame.
DETECT_EVERY = 3
# Face-box-relative skin patches (x0, x1, y0, y1): forehead + both cheeks,
# avoiding eyes, brows, nose and mouth.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "forehead": (0.25, 0.75, 0.08, 0.28),
    "left_cheek": (0.14, 0.40, 0.42, 0.68),
    "right_cheek": (0.60, 0.86, 0.42, 0.68),
}
# ponytail: consensus/quality calibration, chosen to sit slightly below the
# ensemble SQI scale (0.5 = pass-capable). Retune against labelled dim-room
# clips if POS starts over- or under-riding the learned models.
AGREEMENT_BPM = 6.0
MIN_SNR_DB = 2.0


@dataclass(frozen=True)
class RegionEstimate:
    region: str
    bpm: float
    snr_db: float


@dataclass(frozen=True)
class PosResult:
    bpm: float | None
    quality: float
    agreeing: int
    estimates: list[RegionEstimate]


def estimate_pulse_pos(video_path: Path) -> PosResult | None:
    """Full pipeline over a recorded clip; None when no usable face track."""
    traces, timestamps = _extract_face_traces(video_path)
    if traces is None:
        return None
    return estimate_from_traces(traces, timestamps)


def estimate_from_traces(traces: dict[str, np.ndarray], timestamps: np.ndarray) -> PosResult | None:
    """POS + consensus over per-region mean-RGB traces of shape (N, 3)."""
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.size < 2 or timestamps[-1] - timestamps[0] < MIN_TRACE_SECONDS:
        return None
    grid = np.arange(timestamps[0], timestamps[-1], 1.0 / TARGET_FPS)

    estimates: list[RegionEstimate] = []
    for region, rgb in traces.items():
        rgb = np.asarray(rgb, dtype=float)
        if rgb.shape[0] != timestamps.size:
            continue
        # Browser recordings have jittery frame spacing — resample each channel
        # onto a uniform grid before any frequency-domain step.
        uniform = np.column_stack([np.interp(grid, timestamps, rgb[:, channel]) for channel in range(3)])
        pulse = _pos_signal(uniform, TARGET_FPS)
        estimate = _bpm_and_snr(pulse, TARGET_FPS)
        if estimate is not None:
            estimates.append(RegionEstimate(region, *estimate))

    usable = [estimate for estimate in estimates if estimate.snr_db >= MIN_SNR_DB]
    cluster = _largest_bpm_cluster(usable)
    if len(cluster) >= 2:
        bpms = sorted(estimate.bpm for estimate in cluster)
        snr = float(np.median([estimate.snr_db for estimate in cluster]))
        return PosResult(
            bpm=float(np.median(bpms)),
            quality=_clamp01(0.50 + 0.02 * snr),
            agreeing=len(cluster),
            estimates=estimates,
        )
    if usable:
        best = max(usable, key=lambda estimate: estimate.snr_db)
        # A single agreeing region is never pass-capable on its own (capped
        # under the 0.5 floor): one periodic patch could be a rendering
        # artifact. It can still pass by agreeing with a learned model's BPM
        # in the ensemble fusion — that agreement is the evidence.
        return PosResult(
            bpm=best.bpm,
            quality=min(0.45, _clamp01(0.30 + 0.02 * best.snr_db)),
            agreeing=1,
            estimates=estimates,
        )
    return PosResult(bpm=None, quality=0.15, agreeing=0, estimates=estimates)


def _largest_bpm_cluster(estimates: list[RegionEstimate]) -> list[RegionEstimate]:
    best: list[RegionEstimate] = []
    for anchor in estimates:
        cluster = [estimate for estimate in estimates if abs(estimate.bpm - anchor.bpm) <= AGREEMENT_BPM]
        if len(cluster) > len(best):
            best = cluster
    return best


def _pos_signal(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Plane-orthogonal-to-skin projection with overlap-add (Wang et al. 2017)."""
    window = max(2, int(round(1.6 * fps)))
    total = rgb.shape[0]
    pulse = np.zeros(total)
    for start in range(0, total - window + 1):
        block = rgb[start : start + window]
        mean = block.mean(axis=0)
        if np.any(mean <= 1e-6):
            continue
        normalized = block / mean
        s1 = normalized[:, 1] - normalized[:, 2]
        s2 = normalized[:, 1] + normalized[:, 2] - 2.0 * normalized[:, 0]
        deviation = s2.std()
        projected = s1 + (s1.std() / deviation) * s2 if deviation > 1e-9 else s1
        pulse[start : start + window] += projected - projected.mean()
    return pulse


def _bpm_and_snr(pulse: np.ndarray, fps: float) -> tuple[float, float] | None:
    if pulse.size < int(fps * MIN_TRACE_SECONDS) or float(np.ptp(pulse)) <= 0.0:
        return None
    low, high = BAND_HZ
    coefficients = butter(3, [low, high], fs=fps, btype="band")
    filtered = filtfilt(*coefficients, pulse)
    frequencies, power = welch(filtered, fs=fps, nperseg=min(pulse.size, 256), nfft=16384)

    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band) or float(power[band].sum()) <= 0.0:
        return None
    peak_hz = float(frequencies[band][np.argmax(power[band])])

    snr_band = (frequencies >= SNR_BAND_HZ[0]) & (frequencies <= SNR_BAND_HZ[1])
    signal_mask = snr_band & (
        (np.abs(frequencies - peak_hz) <= PEAK_WINDOW_HZ)
        | (np.abs(frequencies - 2.0 * peak_hz) <= HARMONIC_WINDOW_HZ)
    )
    signal_power = float(power[signal_mask].sum())
    noise_power = float(power[snr_band & ~signal_mask].sum())
    if noise_power <= 0.0:
        return None
    return peak_hz * 60.0, 10.0 * float(np.log10(signal_power / noise_power))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_face_traces(video_path: Path) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
    import cv2

    from ml_service.services.gesture_service import _import_mediapipe

    mp = _import_mediapipe()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, np.empty(0)

    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    samples: dict[str, list[np.ndarray]] = {region: [] for region in REGIONS}
    timestamps: list[float] = []
    box: tuple[int, int, int, int] | None = None
    frame_index = 0
    try:
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.35, min_tracking_confidence=0.35
        ) as face_mesh:
            while frame_index < MAX_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = position_ms / 1000.0 if position_ms and position_ms > 0 else frame_index / fps
                if timestamps and timestamp <= timestamps[-1]:
                    timestamp = timestamps[-1] + 1.0 / fps
                if frame_index % DETECT_EVERY == 0 or box is None:
                    detected = _face_box(face_mesh, cv2, frame)
                    box = detected or box
                frame_index += 1
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).astype(np.float32)
                height, width = rgb.shape[:2]
                if height < 20 or width < 20:
                    continue
                timestamps.append(timestamp)
                for region, (rx0, rx1, ry0, ry1) in REGIONS.items():
                    patch = rgb[int(ry0 * height) : int(ry1 * height), int(rx0 * width) : int(rx1 * width)]
                    samples[region].append(patch.reshape(-1, 3).mean(axis=0))
    finally:
        cap.release()

    if len(timestamps) < 2:
        # Span/coverage is judged by estimate_from_traces; this only rejects
        # clips where no face box was ever established.
        return None, np.empty(0)
    traces = {region: np.vstack(values) for region, values in samples.items() if values}
    return traces, np.asarray(timestamps)


def _face_box(face_mesh, cv2, frame) -> tuple[int, int, int, int] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    height, width = frame.shape[:2]
    xs = [landmark.x for landmark in results.multi_face_landmarks[0].landmark]
    ys = [landmark.y for landmark in results.multi_face_landmarks[0].landmark]
    x0 = max(0, int(min(xs) * width))
    x1 = min(width, int(max(xs) * width))
    y0 = max(0, int(min(ys) * height))
    y1 = min(height, int(max(ys) * height))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return x0, y0, x1, y1
