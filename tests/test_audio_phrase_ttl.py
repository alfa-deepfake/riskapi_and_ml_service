import pytest
from httpx import ASGITransport, AsyncClient

from ml_service.api.schemas import AudioEvidence
from ml_service.config import settings
from ml_service.core.challenge import AUDIO_PROMPT_PLACEHOLDER, AUDIO_WORDS
from ml_service.core.checks import score_audio
from ml_service.core.math_utils import phrase_word_match_fraction
from ml_service.main import app
from ml_service.services import audio_service


def test_word_match_tolerates_one_dropped_word():
    expected = "апельсин крокодил лестница"
    assert phrase_word_match_fraction(expected, "апельсин лестница") == pytest.approx(2 / 3)
    check = score_audio(
        AudioEvidence(phrase_expected=expected, phrase_transcribed="апельсин лестница", ai_probability=0.1),
        challenge=None,
    )
    assert check.status == "passed"
    assert check.details["phrase_word_fraction"] == pytest.approx(2 / 3)


def test_word_match_rejects_wrong_words():
    expected = "апельсин крокодил лестница"
    assert phrase_word_match_fraction(expected, "молоток телефон океан") == 0.0
    check = score_audio(
        AudioEvidence(phrase_expected=expected, phrase_transcribed="молоток телефон океан", ai_probability=0.1),
        challenge=None,
    )
    assert check.status == "failed"


def test_word_match_rejects_vocabulary_recitation():
    # Reciting the whole vocabulary must not sweep up 2-of-3 matches.
    expected = "апельсин крокодил лестница"
    assert phrase_word_match_fraction(expected, " ".join(AUDIO_WORDS)) == 0.0


def _mock_audio_models(monkeypatch, transcript_holder):
    monkeypatch.setattr(
        audio_service,
        "_run_audio_model",
        lambda _path: ({"ai_probability": 0.12, "detector": "audio-wavlm-all4"}, ""),
    )
    monkeypatch.setattr(audio_service, "_run_asr", lambda _path: transcript_holder["value"])
    monkeypatch.setattr(audio_service, "_probe_duration", lambda _path: 3.0)


async def _create_session(client):
    created = await client.post(
        "/v1/sessions",
        json={"uid": "u-ttl", "check_id": "c-ttl", "scenario": "video_call"},
    )
    assert created.status_code == 201
    return created.json()


@pytest.mark.anyio
async def test_fresh_phrase_flow_and_single_submission(monkeypatch):
    transcript = {"value": ""}
    _mock_audio_models(monkeypatch, transcript)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        session = await _create_session(client)
        session_id = session["session_id"]
        audio_step = next(s for s in session["challenge"]["steps"] if s["type"] == "audio_phrase")
        # The real phrase is not disclosed in the session prompt.
        assert audio_step["prompt"] == AUDIO_PROMPT_PLACEHOLDER

        # Submitting before any phrase was issued is an explicit unknown.
        early = await client.post(
            f"/v1/sessions/{session_id}/audio/analyze",
            files={"file": ("speech.webm", b"not-audio", "audio/webm")},
        )
        assert early.json()["message"] == "phrase_expired"

        issued = await client.post(f"/v1/sessions/{session_id}/audio/phrase")
        assert issued.status_code == 200
        body = issued.json()
        assert len(body["phrase"].split()) == 3
        assert body["attempts_left"] == settings.audio_max_attempts - 1

        # The rotated phrase never rides in the challenge payload — the client
        # gets it from the issue response (the prompt mirrors it for UI only).
        challenge = (await client.get(f"/v1/sessions/{session_id}/challenge")).json()
        step = next(s for s in challenge["challenge"]["steps"] if s["type"] == "audio_phrase")
        assert "phrase" not in step["payload"]
        assert step["prompt"] == body["phrase"]

        transcript["value"] = body["phrase"]
        analyzed = await client.post(
            f"/v1/sessions/{session_id}/audio/analyze",
            files={"file": ("speech.webm", b"not-audio", "audio/webm")},
        )
        assert analyzed.json()["status"] == "passed"

        # The phrase is single-submission.
        again = await client.post(
            f"/v1/sessions/{session_id}/audio/analyze",
            files={"file": ("speech.webm", b"not-audio", "audio/webm")},
        )
        assert again.json()["message"] == "phrase_expired"

        # Final scoring uses the server-held analysis, not the client bundle:
        # the client claims a terrible ai_probability, the check still passes.
        scored = await client.post(
            f"/v1/sessions/{session_id}/evidence",
            json={
                "uid": "u-ttl",
                "check_id": "c-ttl",
                "evidence": {"audio": {"ai_probability": 0.99, "phrase_transcribed": "чужая фраза"}},
            },
        )
        audio_check = next(c for c in scored.json()["checks"] if c["name"] == "audio")
        assert audio_check["status"] == "passed"
        assert audio_check["details"]["ai_probability"] == 0.12


@pytest.mark.anyio
async def test_expired_phrase_is_rejected(monkeypatch):
    transcript = {"value": "что-нибудь"}
    _mock_audio_models(monkeypatch, transcript)
    monkeypatch.setattr(settings, "audio_phrase_ttl_seconds", -1)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        session = await _create_session(client)
        session_id = session["session_id"]
        issued = await client.post(f"/v1/sessions/{session_id}/audio/phrase")
        assert issued.status_code == 200
        analyzed = await client.post(
            f"/v1/sessions/{session_id}/audio/analyze",
            files={"file": ("speech.webm", b"not-audio", "audio/webm")},
        )
        body = analyzed.json()
        assert body["message"] == "phrase_expired"
        assert body["status"] == "unknown"


@pytest.mark.anyio
async def test_forged_evidence_without_issuing_a_phrase_fails_the_phrase(monkeypatch):
    # The client skips /audio/phrase and submits self-scored audio evidence.
    # The session phrase is never disclosed (stripped from responses), so the
    # forged transcript is checked against a phrase the client cannot know.
    # ponytail: secrecy-only defense — the 18-word vocabulary is public, so a
    # guess passes ~5% of tries; the unconditional server-held override is the
    # upgrade path if that ever matters.
    from ml_service.core import challenge as challenge_module

    monkeypatch.setattr(challenge_module, "generate_audio_phrase", lambda rng=None: "телефон пирамида горизонт")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        session = await _create_session(client)
        audio_step = next(s for s in session["challenge"]["steps"] if s["type"] == "audio_phrase")
        assert "phrase" not in audio_step["payload"]
        assert audio_step["prompt"] == AUDIO_PROMPT_PLACEHOLDER

        scored = await client.post(
            f"/v1/sessions/{session['session_id']}/evidence",
            json={
                "uid": "u-ttl",
                "check_id": "c-ttl",
                "evidence": {
                    "audio": {
                        "phrase_expected": "апельсин крокодил лестница",
                        "phrase_transcribed": "апельсин крокодил лестница",
                        "ai_probability": 0.02,
                        "speaker_match_probability": 1.0,
                    }
                },
            },
        )
        audio_check = next(c for c in scored.json()["checks"] if c["name"] == "audio")
        assert audio_check["status"] == "failed"
        assert scored.json()["decision"] != "allow"


@pytest.mark.anyio
async def test_attempts_are_limited_and_forged_evidence_is_ignored():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        session = await _create_session(client)
        session_id = session["session_id"]
        for _ in range(settings.audio_max_attempts):
            assert (await client.post(f"/v1/sessions/{session_id}/audio/phrase")).status_code == 200
        exhausted = await client.post(f"/v1/sessions/{session_id}/audio/phrase")
        assert exhausted.status_code == 409

        # A phrase was issued but never analyzed server-side: perfect forged
        # audio evidence must not count.
        scored = await client.post(
            f"/v1/sessions/{session_id}/evidence",
            json={
                "uid": "u-ttl",
                "check_id": "c-ttl",
                "evidence": {
                    "audio": {
                        "phrase_expected": "апельсин крокодил лестница",
                        "phrase_transcribed": "апельсин крокодил лестница",
                        "ai_probability": 0.02,
                    }
                },
            },
        )
        audio_check = next(c for c in scored.json()["checks"] if c["name"] == "audio")
        assert audio_check["status"] == "unknown"
        assert audio_check["reason"] == "audio evidence is missing"
        assert scored.json()["decision"] != "allow"
