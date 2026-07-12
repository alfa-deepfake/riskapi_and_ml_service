from __future__ import annotations

import json
import sys
import time
from urllib import request


ML_API = "http://localhost:8100"


def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{ML_API}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    uid = f"smoke-user-{int(time.time())}"
    check_id = f"smoke-check-{int(time.time())}"
    session = post_json(
        "/v1/sessions",
        {"uid": uid, "check_id": check_id, "scenario": "compose_smoke"},
    )
    steps = session["challenge"]["steps"]
    light = next(step for step in steps if step["type"] == "active_light")
    gesture = next(step for step in steps if step["type"] == "gesture")
    audio = next(step for step in steps if step["type"] == "audio_phrase")

    score = post_json(
        f"/v1/sessions/{session['session_id']}/evidence",
        {
            "uid": uid,
            "check_id": check_id,
            "evidence": {
                "classifier": {
                    "fake_probability": 0.07,
                    "confidence": 0.9,
                    "model_name": "compose-smoke",
                    "frame_count": len(light["payload"]["luma_sequence"]),
                    "face_present": True,
                    "face_confidence": 0.9,
                },
                "active_light": {
                    "expected_luma": light["payload"]["luma_sequence"],
                    "observed_face_luma": light["payload"]["luma_sequence"],
                    "face_present": True,
                    "face_confidence": 0.9,
                },
                "rppg": {
                    "samples": [100.0, 105.8, 109.5, 109.5, 105.8, 100.0, 94.2, 90.5, 90.5, 94.2] * 12,
                    "sample_rate_hz": 10,
                    "window_seconds": 4,
                    "face_present": True,
                    "face_confidence": 0.9,
                },
                "gesture": {
                    "expected_action": gesture["payload"]["expected_action"],
                    "observed_action": gesture["payload"]["expected_action"],
                    "confidence": 0.88,
                    "detector": "compose-smoke-detector",
                    "face_present": True,
                },
                "audio": {
                    "phrase_expected": audio["payload"]["phrase"],
                    "phrase_transcribed": audio["payload"]["phrase"],
                    "ai_probability": 0.08,
                    "speaker_match_probability": 0.86,
                    "duration_seconds": 3.0,
                },
            },
        },
    )
    print(json.dumps({"uid": uid, "check_id": check_id, "decision": score["decision"], "risk_score": score["risk_score"]}, indent=2))
    return 0 if score["decision"] == "allow" else 1


if __name__ == "__main__":
    raise SystemExit(main())
