from __future__ import annotations

from dataclasses import dataclass

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class Challenge:
    """One screen frame of the flashing challenge (see ml_service challenge.py)."""

    kind: str
    background_rgb: tuple[int, int, int]
    lighting_rgb: tuple[int, int, int] | None = None
    stripe_top: int | None = None
    stripe_bottom: int | None = None
    height: int = 720
    pair_index: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Challenge":
        def rgb(value):
            return tuple(int(c) for c in value) if value is not None else None

        return cls(
            kind=str(data.get("kind", "background")),
            background_rgb=rgb(data.get("background_rgb")) or (0, 0, 0),
            lighting_rgb=rgb(data.get("lighting_rgb")),
            stripe_top=data.get("stripe_top"),
            stripe_bottom=data.get("stripe_bottom"),
            height=int(data.get("height", 720)),
            pair_index=int(data.get("pair_index", 0)),
        )

    def mean_screen_rgb(self) -> tuple[float, float, float]:
        """Average color the screen emits for this frame (stripe blended in)."""
        background = self.background_rgb
        if self.kind != "lighting" or self.lighting_rgb is None:
            return tuple(float(c) for c in background)
        top = self.stripe_top if self.stripe_top is not None else 0
        bottom = self.stripe_bottom if self.stripe_bottom is not None else self.height
        fraction = max(0.0, min(1.0, (bottom - top) / max(1, self.height)))
        return tuple(
            float(bg) * (1.0 - fraction) + float(light) * fraction
            for bg, light in zip(background, self.lighting_rgb)
        )

    def mean_screen_luma(self) -> float:
        return sum(w * c for w, c in zip(LUMA_WEIGHTS, self.mean_screen_rgb()))
