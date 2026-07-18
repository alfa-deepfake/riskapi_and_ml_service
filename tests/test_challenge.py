from ml_service.core.challenge import generate_challenge


def test_generate_challenge_contains_required_modalities():
    challenge = generate_challenge(seed=42)
    active_light = next(step for step in challenge.steps if step.type == "active_light")

    assert {step.type for step in challenge.steps} == {"active_light", "gesture", "audio_phrase"}
    assert len(challenge.steps[0].payload["luma_sequence"]) >= 3
    assert len(active_light.payload["face_flash_pairs"]) >= 4
    pair = active_light.payload["face_flash_pairs"][0]
    assert pair["background"]["kind"] == "background"
    assert pair["lighting"]["kind"] == "lighting"
    assert pair["lighting"]["lighting_rgb"] is not None
    assert pair["lighting"]["stripe_top"] is not None
    monochrome = ([0, 0, 0], [255, 255, 255])
    colored = ([255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 255, 0])
    lightings = []
    for pair in active_light.payload["face_flash_pairs"]:
        assert pair["background"]["background_rgb"] in monochrome
        assert pair["lighting"]["background_rgb"] in monochrome
        assert pair["lighting"]["lighting_rgb"] in monochrome + colored
        assert pair["lighting"]["stripe_top"] == 0
        assert pair["lighting"]["stripe_bottom"] == pair["lighting"]["height"]
        # Photosensitivity floor: ≤2 flashes/s (WCAG 2.3.1 limit is 3/s).
        assert pair["lighting"]["period_seconds"] >= 0.25
        lightings.append(pair["lighting"]["lighting_rgb"])
    # Every session gets at least one unnatural colored flash (color evidence)
    # and both flash directions (non-degenerate temporal correlation).
    assert any(light in colored for light in lightings)
    backgrounds = {tuple(pair["background"]["background_rgb"]) for pair in active_light.payload["face_flash_pairs"]}
    assert backgrounds == {(0, 0, 0), (255, 255, 255)}
    assert challenge.steps[1].payload["expected_action"]
    assert challenge.steps[2].payload["phrase"]
