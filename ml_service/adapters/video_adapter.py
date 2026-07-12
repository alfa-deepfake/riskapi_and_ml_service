from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any


class VideoModelAdapter:
    """Batch adapter around the existing CLIP/ConvNeXt video inference code."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        convnext_checkpoint_path: Path | None = None,
        device: str = "auto",
        max_inferences: int = 12,
        infer_every: int = 5,
        face_fallback: str = "skip",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.convnext_checkpoint_path = convnext_checkpoint_path
        self.device = device
        self.max_inferences = max(1, max_inferences)
        self.infer_every = max(1, infer_every)
        self.face_fallback = face_fallback
        self._loaded: tuple[Any, Any | None, dict[str, int], float, Any, Any] | None = None

    def predict(self, video_path: Path) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Video checkpoint not found: {self.checkpoint_path}")
        if self.convnext_checkpoint_path is not None and not self.convnext_checkpoint_path.exists():
            raise FileNotFoundError(f"ConvNeXt checkpoint not found: {self.convnext_checkpoint_path}")

        try:
            import cv2
            from neiro_model import video_infer
        except ImportError as exc:
            raise RuntimeError(f"Video inference dependency is missing: {exc}") from exc

        clip_model, convnext_model, label_to_id, threshold, device, torch = self._load_models(video_infer)
        fake_id = int(label_to_id.get("fake", 1))
        face_detector = video_infer.make_face_detector(True, "auto", 0.35)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open uploaded video: {video_path}")

        frame_count = 0
        face_hits = 0
        probabilities: list[float] = []
        input_modes: list[str] = []
        try:
            while len(probabilities) < self.max_inferences:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                if frame_count % self.infer_every != 0:
                    continue

                face_img, _face_box = video_infer.largest_face_crop(frame, face_detector, 0.18)
                input_mode = "face"
                if face_img is None:
                    if self.face_fallback == "skip":
                        continue
                    face_img, _face_box, input_mode = video_infer.fallback_face_crop(frame, self.face_fallback)
                if face_img is None:
                    continue

                face_hits += 1
                rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                clip_pixel_values = video_infer.preprocess_face(
                    rgb,
                    torch,
                    device,
                    False,
                    video_infer.CLIP_MEAN,
                    video_infer.CLIP_STD,
                )
                with torch.inference_mode():
                    clip_logits = clip_model(clip_pixel_values)
                    clip_probs = torch.softmax(clip_logits.float(), dim=-1)[0]
                    prob_fake = float(clip_probs[fake_id].detach().cpu())

                    if convnext_model is not None:
                        convnext_pixel_values = video_infer.preprocess_face(
                            rgb,
                            torch,
                            device,
                            False,
                            video_infer.IMAGENET_MEAN,
                            video_infer.IMAGENET_STD,
                        )
                        convnext_logits = convnext_model(convnext_pixel_values)
                        convnext_probs = torch.softmax(convnext_logits.float(), dim=-1)[0]
                        convnext_prob_fake = float(convnext_probs[fake_id].detach().cpu())
                        prob_fake = 0.75 * prob_fake + 0.25 * convnext_prob_fake

                probabilities.append(prob_fake)
                input_modes.append(input_mode)
        finally:
            cap.release()
            if face_detector is not None:
                face_detector.close()

        if not probabilities:
            return {
                "fake_probability": None,
                "confidence": 0.0,
                "threshold": threshold,
                "model_name": self._model_name(),
                "frame_count": frame_count,
                "face_present": False,
                "face_confidence": 0.0,
                "device": str(device),
                "input_modes": input_modes,
            }

        fake_probability = sum(probabilities) / len(probabilities)
        confidence = fake_probability if fake_probability >= threshold else 1.0 - fake_probability
        return {
            "fake_probability": float(fake_probability),
            "confidence": float(confidence),
            "threshold": threshold,
            "model_name": self._model_name(),
            "frame_count": frame_count,
            "face_present": True,
            "face_confidence": min(1.0, face_hits / max(1, len(probabilities))),
            "device": str(device),
            "sampled_frames": len(probabilities),
            "input_modes": input_modes,
        }

    def _load_models(self, video_infer: Any) -> tuple[Any, Any | None, dict[str, int], float, Any, Any]:
        if self._loaded is None:
            args = Namespace(
                checkpoint=self.checkpoint_path,
                checkpoint_zip=Path("__missing_checkpoint_zip__.zip"),
                convnext_checkpoint=self.convnext_checkpoint_path,
                device=self.device,
                threshold=None,
                clip_weight=1.0,
                convnext_weight=0.0 if self.convnext_checkpoint_path is None else 0.25,
                fp16=False,
            )
            if args.clip_weight + args.convnext_weight <= 0:
                raise RuntimeError("Video classifier blend weights must be positive")
            weight_sum = args.clip_weight + args.convnext_weight
            args.clip_weight /= weight_sum
            args.convnext_weight /= weight_sum
            self._loaded = video_infer.load_models(args)
        return self._loaded

    def _model_name(self) -> str:
        if self.convnext_checkpoint_path is None:
            return "clip-vit-b16-deepfake"
        return "clip-vit-b16-deepfake+convnext"
