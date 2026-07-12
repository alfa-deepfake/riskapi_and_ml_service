import pytest
from httpx import ASGITransport, AsyncClient

from ml_service.main import app


@pytest.mark.anyio
async def test_session_challenge_and_evidence_flow():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post(
        "/v1/sessions",
        json={"uid": "u-api", "check_id": "c-api", "scenario": "video_call"},
        )
        assert created.status_code == 201
        session = created.json()
        assert session["challenge"]["steps"]

        challenge = await client.get(f"/v1/sessions/{session['session_id']}/challenge")
        assert challenge.status_code == 200

        light = next(step for step in session["challenge"]["steps"] if step["type"] == "active_light")
        gesture = next(step for step in session["challenge"]["steps"] if step["type"] == "gesture")
        audio = next(step for step in session["challenge"]["steps"] if step["type"] == "audio_phrase")

        scored = await client.post(
            f"/v1/sessions/{session['session_id']}/evidence",
            json={
                "uid": "u-api",
                "check_id": "c-api",
                "evidence": {
                    "classifier": {"fake_probability": 0.1, "confidence": 0.9, "face_present": True},
                    "active_light": {
                        "expected_luma": light["payload"]["luma_sequence"],
                        "observed_face_luma": light["payload"]["luma_sequence"],
                        "face_present": True,
                    },
                    "rppg": {
                        "samples": [100.0, 105.8, 109.5, 109.5, 105.8, 100.0, 94.2, 90.5, 90.5, 94.2] * 12,
                        "sample_rate_hz": 10,
                        "window_seconds": 4,
                        "face_present": True,
                    },
                    "gesture": {
                        "expected_action": gesture["payload"]["expected_action"],
                        "observed_action": gesture["payload"]["expected_action"],
                        "confidence": 0.86,
                        "detector": "api-test-detector",
                        "face_present": True,
                    },
                    "audio": {
                        "phrase_expected": audio["payload"]["phrase"],
                        "phrase_transcribed": audio["payload"]["phrase"],
                        "ai_probability": 0.08,
                        "speaker_match_probability": 0.88,
                    },
                },
            },
        )
        assert scored.status_code == 200
        score = scored.json()
        assert score["decision"] == "allow"
        assert {check["name"] for check in score["checks"]} == {
            "classifier",
            "active_light",
            "rppg",
            "gesture",
            "audio",
        }

        # The challenge is one-time: a scored session cannot be replayed.
        replay = await client.post(
            f"/v1/sessions/{session['session_id']}/evidence",
            json={"uid": "u-api", "check_id": "c-api", "evidence": {}},
        )
        assert replay.status_code == 404
