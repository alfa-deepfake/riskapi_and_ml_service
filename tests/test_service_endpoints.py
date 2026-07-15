import pytest
from httpx import ASGITransport, AsyncClient

from ml_service.api.schemas import ClassifierEvidence
from ml_service.core.checks import score_classifier
from ml_service.main import app
import ml_service.services.active_light_service as active_light_module
from ml_service.services import classifier_service, rppg_service


def test_classifier_fails_without_face():
    check = score_classifier(ClassifierEvidence(face_present=False, face_confidence=0.91))

    assert check.status == "failed"
    assert check.risk >= 0.9


@pytest.mark.anyio
async def test_classifier_endpoint_uses_video_adapter_result(monkeypatch):
    def fake_run_video_model(_video_path):
        return {
            "fake_probability": 0.12,
            "confidence": 0.88,
            "model_name": "test-video-model",
            "frame_count": 25,
            "face_present": True,
            "face_confidence": 1.0,
        }

    monkeypatch.setattr(classifier_service, "_run_video_model", fake_run_video_model)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/classifier/analyze-video",
            files={"file": ("sample.webm", b"not-a-real-video", "video/webm")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["evidence"]["fake_probability"] == 0.12
    assert body["evidence"]["model_name"] == "test-video-model"


@pytest.mark.anyio
async def test_classifier_endpoint_returns_unknown_when_model_missing(monkeypatch):
    monkeypatch.setattr(classifier_service, "_run_video_model", lambda _video_path: None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/classifier/analyze-video",
            files={"file": ("sample.webm", b"not-a-real-video", "video/webm")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert body["check"]["reason"] == "video classifier model is not configured"


@pytest.mark.anyio
async def test_classifier_endpoint_returns_unknown_when_adapter_fails(monkeypatch):
    def broken_run_video_model(_video_path):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(classifier_service, "_run_video_model", broken_run_video_model)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/classifier/analyze-video",
            files={"file": ("sample.webm", b"not-a-real-video", "video/webm")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert body["check"]["reason"] == "video classifier inference failed: RuntimeError"


@pytest.mark.anyio
async def test_active_light_frame_pair_endpoint_uses_reference_verifier(monkeypatch):
    async def fake_run_face_flashing_verifier(*, manifest, files):
        assert "pairs" in manifest
        assert len(files) == 2
        return {
            "passed": True,
            "score": 0.93,
            "pair_count": 8,
            "temporal": {"zero_lag_correlation": 0.89, "best_correlation": 0.91},
            "spatial": {"median_contrast": 0.16, "median_center_error": 0.07},
            "median_response_snr": 0.22,
            "median_response_magnitude": 14.0,
            "mean_color_cosine": 0.84,
        }

    monkeypatch.setattr(active_light_module, "_run_face_flashing_verifier", fake_run_face_flashing_verifier)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/active-light/analyze-frame-pairs",
            data={"manifest": '{"pairs": []}'},
            files=[
                ("files", ("active_light_bg_0.png", b"bg", "image/png")),
                ("files", ("active_light_light_0.png", b"light", "image/png")),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["evidence"]["detector"] == "face-flashing-frame-pairs"
    assert body["evidence"]["verifier_score"] == 0.93


@pytest.mark.anyio
async def test_rppg_endpoint_uses_reference_runtime_output(monkeypatch):
    def fake_run_rppg_runtime(_video_path):
        return {
            "bpm": 73.0,
            "signal_quality": 0.86,
            "latency": 1.2,
            "hrv": {"sdnn": 40.0},
            "samples": [],
            "sample_rate_hz": None,
            "detector": "python-rppg",
            "face_present": True,
            "face_confidence": None,
        }

    monkeypatch.setattr(rppg_service, "_run_rppg_runtime", fake_run_rppg_runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/rppg/analyze-video",
            files={"file": ("pulse.webm", b"not-a-real-video", "video/webm")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["evidence"]["detector"] == "python-rppg"
    assert body["evidence"]["bpm"] == 73.0


def _stub_rppg_model(monkeypatch, process_video):
    import sys
    import types

    from ml_service.services import rppg_service as module

    class FakeModel:
        def process_video(self, path):
            return process_video(path)

    stub = types.ModuleType("rppg")
    stub.Model = FakeModel
    monkeypatch.setitem(sys.modules, "rppg", stub)
    module._rppg_model.cache_clear()
    return module


@pytest.mark.anyio
async def test_rppg_runtime_clamps_out_of_range_bpm(monkeypatch):
    module = _stub_rppg_model(
        monkeypatch,
        lambda _path: {"hr": 350.0, "SQI": 1.5, "hrv": {}, "latency": 0.0},
    )
    try:
        from pathlib import Path
        result = module._run_rppg_runtime(Path("/tmp/does-not-exist.webm"))
    finally:
        module._rppg_model.cache_clear()
    assert result["bpm"] is None
    assert result["signal_quality"] == 1.0
    assert result["detector"] == "open-rppg-facephys"
    assert result["face_present"] is None


@pytest.mark.anyio
async def test_rppg_runtime_handles_no_signal_result(monkeypatch):
    # open-rppg returns None when no pulse signal was captured at all
    module = _stub_rppg_model(monkeypatch, lambda _path: None)
    try:
        from pathlib import Path
        result = module._run_rppg_runtime(Path("/tmp/does-not-exist.webm"))
    finally:
        module._rppg_model.cache_clear()
    assert result["bpm"] is None
    assert result["signal_quality"] is None
    assert result["detector"] == "open-rppg-facephys"


@pytest.mark.anyio
async def test_rppg_runtime_wraps_generic_error_as_runtime_error(monkeypatch):
    def boom(_path):
        raise ValueError("bad video")

    module = _stub_rppg_model(monkeypatch, boom)
    try:
        from pathlib import Path
        with pytest.raises(RuntimeError, match="rPPG runtime failed"):
            module._run_rppg_runtime(Path("/tmp/does-not-exist.webm"))
    finally:
        module._rppg_model.cache_clear()


@pytest.mark.anyio
async def test_rppg_endpoint_returns_unknown_when_reference_runtime_missing(monkeypatch):
    def missing_rppg_runtime(_video_path):
        raise RuntimeError("rPPG runtime dependency is missing: pip install rppg")

    monkeypatch.setattr(rppg_service, "_run_rppg_runtime", missing_rppg_runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/services/rppg/analyze-video",
            files={"file": ("pulse.webm", b"not-a-real-video", "video/webm")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert "rPPG runtime dependency is missing" in body["message"]
