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
    ("touch_mouth", "touch your lips"),
    ("touch_nose", "touch your nose"),
)

AUDIO_WORDS = ("bank", "signal", "river", "credit", "winter", "orange", "vector", "client")


def generate_challenge(seed: int | None = None) -> ChallengePlan:
    rng = random.Random(seed if seed is not None else uuid.uuid4().int)
    gesture_id, gesture_prompt = rng.choice(GESTURES)
    phrase = " ".join(rng.sample(AUDIO_WORDS, 3))
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
                prompt="follow screen flashes",
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
                duration_ms=5000,
            ),
            ChallengeStep(
                step_id=str(uuid.uuid4()),
                type="audio_phrase",
                prompt=phrase,
                payload={"phrase": phrase},
                duration_ms=4000,
            ),
        ],
    )


def _generate_face_flash_pairs(rng: random.Random, *, n_pairs: int = 8) -> list[dict]:
    width = 1280
    height = 720
    palette = ((0, 0, 0), (255, 255, 255))
    pairs = []
    for pair_index in range(n_pairs):
        background_rgb = rng.choice(palette)
        lighting_rgb = rng.choice([color for color in palette if color != background_rgb])
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
                    "period_seconds": 0.12,
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
                    "period_seconds": 0.12,
                },
            }
        )
    return pairs
