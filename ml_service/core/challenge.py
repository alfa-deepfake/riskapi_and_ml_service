from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


ChallengeType = Literal["active_light", "gesture", "audio_phrase"]


class ChallengeStep(BaseModel):
    step_id: str
    type: ChallengeType
    prompt: str
    payload: dict = Field(default_factory=dict)
    duration_ms: int = Field(default=1500, ge=100)


class ChallengePlan(BaseModel):
    challenge_id: str
    issued_at: datetime
    steps: list[ChallengeStep]


GESTURES = (
    ("touch_mouth", "коснитесь губ"),
    ("touch_nose", "коснитесь носа"),
)

# Multi-syllable, phonetically distinct words: short words ("банк", "река")
# are exactly what ASR drops on a poor microphone, and near-homophones would
# defeat the per-word fuzzy match.
AUDIO_WORDS = (
    "апельсин",
    "барабан",
    "библиотека",
    "виноград",
    "горизонт",
    "карандаш",
    "капитан",
    "космонавт",
    "крокодил",
    "лестница",
    "магазин",
    "молоток",
    "океан",
    "паровоз",
    "пирамида",
    "телефон",
    "фотография",
    "черепаха",
)

# Shown until the client requests a fresh phrase right before recording; the
# real phrase must not sit in the challenge payload for the whole session.
AUDIO_PROMPT_PLACEHOLDER = "фраза появится перед записью"


def generate_audio_phrase(rng: random.Random | None = None) -> str:
    rng = rng or random.Random(uuid.uuid4().int)
    return " ".join(rng.sample(AUDIO_WORDS, 3))


def generate_challenge(seed: int | None = None) -> ChallengePlan:
    rng = random.Random(seed if seed is not None else uuid.uuid4().int)
    gesture_id, gesture_prompt = rng.choice(GESTURES)
    phrase = generate_audio_phrase(rng)
    light_sequence = [rng.choice((0, 255)) for _ in range(10)]
    if len(set(light_sequence)) == 1:
        light_sequence[-1] = 255 - light_sequence[-1]
    face_flash_pairs = _generate_face_flash_pairs(rng)

    return ChallengePlan(
        challenge_id=str(uuid.uuid4()),
        issued_at=datetime.now(timezone.utc),
        steps=[
            ChallengeStep(
                step_id=str(uuid.uuid4()),
                type="active_light",
                prompt="следуйте за вспышками экрана",
                payload={
                    "luma_sequence": light_sequence,
                    "colors": ["black", "white"],
                    "face_flash_pairs": face_flash_pairs,
                },
                duration_ms=4500,
            ),
            ChallengeStep(
                step_id=str(uuid.uuid4()),
                type="gesture",
                prompt=gesture_prompt,
                payload={"expected_action": gesture_id},
                # 7s, not 5: reading the prompt, raising the hand and holding
                # the touch was routinely clipped mid-gesture at 5s.
                duration_ms=7000,
            ),
            ChallengeStep(
                step_id=str(uuid.uuid4()),
                type="audio_phrase",
                # payload.phrase is server-side only (final scoring verifies
                # against it); SessionResponse strips it, so clients see the
                # phrase only via the TTL'd issue endpoint.
                prompt=AUDIO_PROMPT_PLACEHOLDER,
                payload={"phrase": phrase},
                duration_ms=6000,
            ),
        ],
    )


def _generate_face_flash_pairs(rng: random.Random, *, n_pairs: int = 8) -> list[dict]:
    width = 1280
    height = 720
    black = (0, 0, 0)
    white = (255, 255, 255)
    # Saturated flash colors no ambient lighting produces: a face reflection
    # matching them (color_cosine) can't come from the environment, and the
    # random hue adds challenge entropy. All keep ≥40% of white's luma so the
    # temporal luma correlation still has signal; pure blue is out — skin
    # reflects it poorly.
    colored = ((255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0))
    backgrounds = [rng.choice((black, white)) for _ in range(n_pairs)]
    # All pairs flashing the same direction makes the temporal correlation
    # degenerate (constant expected sequence) — force both directions in.
    if len(set(backgrounds)) == 1 and n_pairs > 1:
        backgrounds[-1] = black if backgrounds[-1] == white else white
    # Dark pairs flash white or an unnatural color; bright pairs flash to black
    # (a colored flash on white would darken the screen and muddy the hue).
    lightings = [rng.choice((white,) + colored) if bg == black else black for bg in backgrounds]
    if black in backgrounds and not any(light in colored for light in lightings):
        lightings[backgrounds.index(black)] = rng.choice(colored)
    pairs = []
    for pair_index in range(n_pairs):
        background_rgb = backgrounds[pair_index]
        lighting_rgb = lightings[pair_index]
        stripe_top = 0
        stripe_bottom = height
        pairs.append(
            {
                "background": {
                    "index": pair_index * 2,
                    "pair_index": pair_index,
                    "kind": "background",
                    "background_rgb": list(background_rgb),
                    "lighting_rgb": None,
                    "stripe_top": None,
                    "stripe_bottom": None,
                    "width": width,
                    "height": height,
                    # ≥0.25s per phase → ≤2 flashes/s, under the WCAG 2.3.1
                    # photosensitivity limit of 3 flashes/s.
                    "period_seconds": 0.25,
                },
                "lighting": {
                    "index": pair_index * 2 + 1,
                    "pair_index": pair_index,
                    "kind": "lighting",
                    "background_rgb": list(background_rgb),
                    "lighting_rgb": list(lighting_rgb),
                    "stripe_top": stripe_top,
                    "stripe_bottom": stripe_bottom,
                    "width": width,
                    "height": height,
                    # ≥0.25s per phase → ≤2 flashes/s, under the WCAG 2.3.1
                    # photosensitivity limit of 3 flashes/s.
                    "period_seconds": 0.25,
                },
            }
        )
    return pairs
