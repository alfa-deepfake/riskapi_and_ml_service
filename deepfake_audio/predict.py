# Vendored from the deepfake_audio_inference drop (2026-07-11), unchanged.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .model import WavLMDeepfakeClassifier

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def audio_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    raise FileNotFoundError(f"Audio path not found: {path}")


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).mean(dim=1)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform


def make_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    clip_seconds: float,
    hop_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip_samples = int(sample_rate * clip_seconds)
    hop_samples = max(1, int(sample_rate * hop_seconds))
    if waveform.numel() <= clip_samples:
        valid = waveform.numel()
        padded = F.pad(waveform, (0, clip_samples - waveform.numel()))
        mask = torch.zeros(clip_samples, dtype=torch.long)
        mask[:valid] = 1
        return padded.unsqueeze(0), mask.unsqueeze(0)

    starts = list(range(0, waveform.numel() - clip_samples + 1, hop_samples))
    if starts[-1] != waveform.numel() - clip_samples:
        starts.append(waveform.numel() - clip_samples)
    windows = [waveform[start : start + clip_samples] for start in starts]
    masks = [torch.ones(clip_samples, dtype=torch.long) for _ in starts]
    return torch.stack(windows), torch.stack(masks)


@torch.no_grad()
def predict_file(
    model: WavLMDeepfakeClassifier,
    path: str | Path,
    device: torch.device,
    sample_rate: int,
    clip_seconds: float,
    hop_seconds: float,
    batch_size: int,
) -> dict[str, object]:
    waveform = load_audio(path, sample_rate)
    windows, masks = make_windows(waveform, sample_rate, clip_seconds, hop_seconds)
    scores = []
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size].to(device)
        batch_mask = masks[start : start + batch_size].to(device)
        logits = model(batch, batch_mask)
        scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return {
        "path": str(Path(path).resolve()),
        "duration_seconds": round(waveform.numel() / sample_rate, 3),
        "windows": len(scores),
        "fake_score_mean": sum(scores) / len(scores),
        "fake_score_max": max(scores),
        "window_scores": scores,
    }


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[WavLMDeepfakeClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = WavLMDeepfakeClassifier(config.get("model_name", "microsoft/wavlm-base-plus"))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether custom audio is a deepfake")
    parser.add_argument("--checkpoint", default="outputs/wavlm_internal_full_all4/best.pt")
    parser.add_argument("--audio", required=True, help="Audio file or directory")
    parser.add_argument("--output", help="Optional CSV output path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, help="Fake threshold. Defaults to checkpoint EER threshold or 0.5")
    parser.add_argument("--score-mode", choices=["mean", "max"], default="mean")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--clip-seconds", type=float)
    parser.add_argument("--hop-seconds", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print JSON lines instead of compact text")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = checkpoint.get("config", {})
    sample_rate = args.sample_rate or int(config.get("sample_rate", 16_000))
    clip_seconds = args.clip_seconds or float(config.get("clip_seconds", 5.0))
    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("metrics", {}).get("eer_threshold", 0.5))

    paths = audio_paths(args.audio)
    results = []
    for path in paths:
        result = predict_file(
            model,
            path,
            device,
            sample_rate,
            clip_seconds,
            args.hop_seconds,
            args.batch_size,
        )
        score_key = f"fake_score_{args.score_mode}"
        score = float(result[score_key])
        result["score_mode"] = args.score_mode
        result["threshold"] = threshold
        result["prediction"] = "fake" if score >= threshold else "real"
        results.append(result)
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(
                f"{result['prediction']:>4} score={score:.4f} "
                f"mean={result['fake_score_mean']:.4f} max={result['fake_score_max']:.4f} "
                f"windows={result['windows']} path={result['path']}"
            )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "path",
                    "duration_seconds",
                    "windows",
                    "fake_score_mean",
                    "fake_score_max",
                    "score_mode",
                    "threshold",
                    "prediction",
                ],
            )
            writer.writeheader()
            for result in results:
                row = dict(result)
                row.pop("window_scores", None)
                writer.writerow(row)


if __name__ == "__main__":
    main()
