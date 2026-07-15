from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from ml_service.api.schemas import AudioEvidence
from ml_service.core.checks import score_audio
from ml_service.main import app
from ml_service.services import audio_service

AUDIO_MODEL = Path(__file__).resolve().parent.parent / "models" / "audio" / "wavlm_all4_best.pt"
ASR_MODEL = Path(__file__).resolve().parent.parent / "models" / "asr" / "faster-whisper-medium"


async def _post_audio(monkeypatch, *, asr_result, form_extra=None):
    def fake_run_audio_model(_audio_path):
        return {"ai_probability": 0.12, "threshold": 0.42, "detector": "audio-wavlm-all4"}, ""

    monkeypatch.setattr(audio_service, "_run_audio_model", fake_run_audio_model)
    monkeypatch.setattr(audio_service, "_run_asr", lambda _path: asr_result)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(
            "/v1/services/audio/analyze",
            files={"file": ("speech.webm", b"not-a-real-audio", "audio/webm")},
            data={"phrase_expected": "orange client river", **(form_extra or {})},
        )


@pytest.mark.anyio
async def test_audio_passes_when_server_hears_the_phrase(monkeypatch):
    response = await _post_audio(monkeypatch, asr_result="orange client river")
    body = response.json()
    assert body["evidence"]["ai_probability"] == 0.12
    assert body["evidence"]["phrase_transcribed"] == "orange client river"
    assert body["check"]["status"] == "passed"


@pytest.mark.anyio
async def test_silence_fails_even_if_client_claims_the_phrase(monkeypatch):
    # The client-supplied transcript must be ignored: server ASR heard nothing.
    response = await _post_audio(
        monkeypatch,
        asr_result="",
        form_extra={"phrase_transcribed": "orange client river"},
    )
    body = response.json()
    assert body["evidence"]["phrase_transcribed"] == ""
    assert body["check"]["status"] == "failed"
    assert body["check"]["details"]["phrase_ratio"] == 0.0


@pytest.mark.anyio
async def test_missing_asr_scores_unknown(monkeypatch):
    response = await _post_audio(monkeypatch, asr_result=None)
    body = response.json()
    assert body["check"]["status"] == "unknown"
    assert body["check"]["reason"] == "audio phrase transcript is unavailable"


def test_score_audio_requires_transcript_when_phrase_expected():
    check = score_audio(AudioEvidence(phrase_expected="orange client river", ai_probability=0.1), challenge=None)
    assert check.status == "unknown"
    check = score_audio(
        AudioEvidence(phrase_expected="orange client river", phrase_transcribed="", ai_probability=0.1),
        challenge=None,
    )
    assert check.status == "failed"


@pytest.mark.anyio
async def test_audio_endpoint_unavailable_without_model(monkeypatch):
    monkeypatch.setattr(
        audio_service,
        "_run_audio_model",
        lambda _path: (None, "audio anti-spoof model is not configured"),
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/audio/analyze",
            files={"file": ("speech.webm", b"not-a-real-audio", "audio/webm")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert body["evidence"]["detector"] == "unavailable"


@pytest.mark.skipif(not ASR_MODEL.exists(), reason="Faster-Whisper model is not deployed")
def test_faster_whisper_silence_gate_and_pipeline(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("faster_whisper")
    sf = pytest.importorskip("soundfile")

    from ml_service.adapters.asr_adapter import WhisperAsrAdapter

    adapter = WhisperAsrAdapter(model_path=ASR_MODEL)

    silence = tmp_path / "silence.wav"
    sf.write(silence, np.zeros(16_000, dtype=np.float32), 16_000)
    assert adapter.transcribe(silence) == ""

    noise = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    sf.write(noise, (rng.normal(0, 0.05, 32_000)).astype(np.float32), 16_000)
    transcript = adapter.transcribe(noise)
    assert isinstance(transcript, str)


@pytest.mark.skipif(not AUDIO_MODEL.exists(), reason="WavLM checkpoint is not deployed")
def test_wavlm_checkpoint_end_to_end(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sf = pytest.importorskip("soundfile")

    from ml_service.adapters.audio_adapter import AudioModelAdapter

    rate = 16_000
    t = np.arange(rate * 3) / rate
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, tone, rate)

    result = AudioModelAdapter(model_path=AUDIO_MODEL).predict(wav)
    assert 0.0 <= result["ai_probability"] <= 1.0
    assert result["detector"] == "audio-wavlm-all4"
    assert result["windows"] >= 1
    assert 0.0 < result["threshold"] < 1.0
