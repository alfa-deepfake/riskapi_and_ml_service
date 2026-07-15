from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from ml_service.main import app
from ml_service.services import audio_service

AUDIO_MODEL = Path(__file__).resolve().parent.parent / "models" / "audio" / "wavlm_all4_best.pt"


@pytest.mark.anyio
async def test_audio_endpoint_uses_wavlm_adapter_result(monkeypatch):
    def fake_run_audio_model(_audio_path):
        return {"ai_probability": 0.12, "threshold": 0.42, "detector": "audio-wavlm-all4"}, ""

    monkeypatch.setattr(audio_service, "_run_audio_model", fake_run_audio_model)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/audio/analyze",
            files={"file": ("speech.webm", b"not-a-real-audio", "audio/webm")},
            data={"phrase_expected": "один два три", "phrase_transcribed": "один два три"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["evidence"]["ai_probability"] == 0.12
    assert body["evidence"]["detector"] == "audio-wavlm-all4"
    assert body["check"]["status"] == "passed"


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
