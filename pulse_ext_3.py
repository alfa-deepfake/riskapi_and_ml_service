import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import rppg


def fourcc_to_str(fourcc_value):
    fourcc_int = int(fourcc_value)
    return "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))


def record_png_frames_then_ffv1(
    camera_index=0,
    duration_sec=30,
    target_fps=30,
    width=640,
    height=480,
    warmup_sec=2,
):
    device_path = f"/dev/video{camera_index}"
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть {device_path}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    print("Camera reports:")
    print(f"  CAP FOURCC: {fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))}")
    print(f"  CAP FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"  CAP WIDTH: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  CAP HEIGHT: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    print(f"\nПрогрев камеры: {warmup_sec} сек...")
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_sec:
        cap.read()

    tmp_dir = Path(tempfile.mkdtemp())
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nЗапись PNG-кадров {device_path}: {duration_sec} сек")
    start = time.time()
    frame_idx = 0

    while time.time() - start < duration_sec:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame = cv2.resize(frame, (width, height))
        cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.png"), frame)
        frame_idx += 1

    actual_duration = time.time() - start
    cap.release()

    if frame_idx == 0:
        raise RuntimeError("Не записано ни одного кадра")

    real_fps = frame_idx / actual_duration
    video_path = tmp_dir / "camera_capture_ffv1.mkv"

    print("\nЗапись завершена:")
    print(f"  Кадров: {frame_idx}")
    print(f"  Длительность: {actual_duration:.2f} сек")
    print(f"  Фактический FPS: {real_fps:.4f}")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{real_fps:.6f}",
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "ffv1",
        "-level",
        "3",
        str(video_path),
    ]

    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print(f"  Видео: {video_path}")

    if real_fps < 24:
        print(
            "\nWARNING: камера реально отдаёт FPS ниже 24. "
            "Для rPPG попробуй яркий свет или 320x240."
        )

    return str(video_path), {
        "device": device_path,
        "frames_count": frame_idx,
        "actual_duration_sec": actual_duration,
        "real_fps": real_fps,
        "width": width,
        "height": height,
        "video_path": str(video_path),
    }


def to_float_safe(value):
    try:
        return float(value)
    except Exception:
        return None


def interpret_rppg(result):
    hr = to_float_safe(result.get("hr"))
    sqi = to_float_safe(result.get("SQI"))

    if sqi is None:
        return {
            "reliable": False,
            "passed": None,
            "reason": "missing_signal_quality",
        }

    if sqi < 0.35:
        return {
            "reliable": False,
            "passed": None,
            "reason": "low_rppg_signal_quality",
        }

    if hr is not None and 45 <= hr <= 140 and sqi >= 0.50:
        return {
            "reliable": True,
            "passed": True,
            "reason": "stable_physiological_signal_detected",
        }

    return {
        "reliable": True,
        "passed": False,
        "reason": "unstable_or_implausible_heart_rate",
    }


def main():
    video_path, capture_info = record_png_frames_then_ffv1(
        camera_index=0,
        duration_sec=30,
        target_fps=30,
        width=640,
        height=480,
        warmup_sec=2,
    )

    model = rppg.Model()
    result = model.process_video(video_path)

    output = {
        "module": "rppg",
        "method": "open-rppg-default",
        "video_path": video_path,
        "capture": capture_info,
        "hr_bpm": to_float_safe(result.get("hr")),
        "signal_quality": to_float_safe(result.get("SQI")),
        "latency": to_float_safe(result.get("latency")),
        "hrv": {k: to_float_safe(v) for k, v in result.get("hrv", {}).items()},
        **interpret_rppg(result),
    }

    print("\nRPPG result:")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
