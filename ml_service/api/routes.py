from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ml_service.api.schemas import (
    ActiveLightAnalyzeRequest,
    AudioEvidence,
    AudioPhraseResponse,
    EvidenceRequest,
    HealthResponse,
    RppgAnalyzeRequest,
    ScoreRequest,
    ScoreResponse,
    ServiceAnalyzeResponse,
    SessionCreateRequest,
    SessionResponse,
)
from ml_service.config import settings
from ml_service.core.challenge import generate_audio_phrase
from ml_service.core.challenge_store import ChallengeStore
from ml_service.core.risk_api import RiskApiClient
from ml_service.core.scoring import CascadeScorer
from ml_service.services.active_light_service import ActiveLightService
from ml_service.services.audio_service import AudioService
from ml_service.services.classifier_service import ClassifierService
from ml_service.services.gesture_service import GestureService
from ml_service.services.rppg_service import RppgService


router = APIRouter()
store = ChallengeStore(ttl_seconds=settings.challenge_ttl_seconds)
scorer = CascadeScorer(settings=settings)
active_light_service = ActiveLightService(settings=settings)
rppg_service = RppgService(settings=settings)
gesture_service = GestureService()
audio_service = AudioService()
classifier_service = ClassifierService()
risk_client = RiskApiClient(
    base_url=settings.risk_api_url,
    enabled=settings.enable_risk_api_callbacks,
    timeout_seconds=settings.callback_timeout_seconds,
)


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name, time=datetime.now(timezone.utc))


@router.post("/v1/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED, tags=["sessions"])
async def create_session(payload: SessionCreateRequest) -> SessionResponse:
    session = store.create(uid=payload.uid, check_id=payload.check_id, scenario=payload.scenario)
    await risk_client.send_status(payload.check_id, uid=payload.uid, status="started", message="ML session created")
    return SessionResponse.from_session(session)


@router.get("/v1/sessions/{session_id}/challenge", response_model=SessionResponse, tags=["sessions"])
async def get_challenge(session_id: str) -> SessionResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена или истекла")
    return SessionResponse.from_session(session)


def _audio_step(session):
    return next(step for step in session.challenge.steps if step.type == "audio_phrase")


@router.post("/v1/sessions/{session_id}/audio/phrase", response_model=AudioPhraseResponse, tags=["services"])
async def issue_audio_phrase(session_id: str) -> AudioPhraseResponse:
    """A fresh random phrase with a short TTL, issued right before recording.

    Issuing rotates the phrase stored in the session challenge (so final
    scoring always verifies against the latest one) and voids any previous
    server-held audio analysis — a client cannot keep the best of several
    attempts."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена или истекла")
    if session.audio_attempts >= settings.audio_max_attempts:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Попытки аудио-проверки исчерпаны")
    phrase = generate_audio_phrase()
    session.audio_attempts += 1
    session.audio_phrase_issued_at = datetime.now(timezone.utc)
    session.audio_phrase_consumed = False
    session.audio_result_evidence = None
    step = _audio_step(session)
    step.prompt = phrase
    step.payload["phrase"] = phrase
    return AudioPhraseResponse(
        phrase=phrase,
        ttl_seconds=settings.audio_phrase_ttl_seconds,
        attempts_left=settings.audio_max_attempts - session.audio_attempts,
    )


@router.post("/v1/sessions/{session_id}/audio/analyze", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_session_audio(session_id: str, file: UploadFile = File(...)) -> ServiceAnalyzeResponse:
    """Session-bound audio analysis: the expected phrase comes from the session
    (never from the client), must be fresh (TTL) and is single-submission."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена или истекла")
    issued_at = session.audio_phrase_issued_at
    fresh = (
        issued_at is not None
        and (datetime.now(timezone.utc) - issued_at).total_seconds() <= settings.audio_phrase_ttl_seconds
    )
    if not fresh or session.audio_phrase_consumed:
        return audio_service.expired_response()
    session.audio_phrase_consumed = True
    response = await audio_service.analyze_audio(
        file,
        phrase_expected=_audio_step(session).payload["phrase"],
        phrase_transcribed=None,
    )
    session.audio_result_evidence = response.evidence
    return response


@router.post("/v1/sessions/{session_id}/evidence", response_model=ScoreResponse, tags=["scoring"])
async def score_session_evidence(session_id: str, payload: EvidenceRequest) -> ScoreResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена или истекла")
    if payload.uid != session.uid or payload.check_id != session.check_id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Данные не соответствуют владельцу сессии")

    # Once a fresh phrase was issued, the client's audio evidence is ignored:
    # only the server-held analysis (made against that phrase, within its TTL)
    # counts. No analysis on record → the audio check scores as missing.
    if session.audio_phrase_issued_at is not None:
        stored = session.audio_result_evidence
        payload.evidence.audio = AudioEvidence(**stored) if stored is not None else None

    result = scorer.score(payload.to_score_request(session.challenge))
    # A challenge is one-time: once scored, the same session/evidence cannot be
    # replayed for another attempt at the same challenge.
    store.pop(session_id)
    await risk_client.send_status(payload.check_id, uid=payload.uid, status="in_progress", message="ML evidence scored")
    await risk_client.send_result(payload.check_id, uid=payload.uid, score=result.model_dump(mode="json"))
    await risk_client.send_status(
        payload.check_id,
        uid=payload.uid,
        status="finished",
        success=result.decision != "deny",
        message=f"ML decision: {result.decision}",
    )
    return result


@router.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
async def score_direct(payload: ScoreRequest) -> ScoreResponse:
    result = scorer.score(payload)
    if payload.check_id:
        await risk_client.send_result(payload.check_id, uid=payload.uid, score=result.model_dump(mode="json"))
    return result


@router.post("/v1/services/active-light/analyze", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_active_light(payload: ActiveLightAnalyzeRequest) -> ServiceAnalyzeResponse:
    return active_light_service.analyze(payload)


@router.post("/v1/services/active-light/analyze-frame-pairs", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_active_light_frame_pairs(
    manifest: str = Form(...),
    files: list[UploadFile] = File(...),
) -> ServiceAnalyzeResponse:
    return await active_light_service.analyze_frame_pairs(manifest=manifest, files=files)


@router.post("/v1/services/rppg/analyze", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_rppg(payload: RppgAnalyzeRequest) -> ServiceAnalyzeResponse:
    return rppg_service.analyze_samples(payload)


@router.post("/v1/services/rppg/analyze-video", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_rppg_video(
    file: UploadFile = File(...),
    face_present: bool | None = Form(default=None),
    face_confidence: float | None = Form(default=None),
) -> ServiceAnalyzeResponse:
    return await rppg_service.analyze_video(file, face_present=face_present, face_confidence=face_confidence)


@router.post("/v1/services/gesture/analyze-video", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_gesture_video(
    file: UploadFile = File(...),
    expected_action: str = Form(...),
    face_present: bool | None = Form(default=None),
) -> ServiceAnalyzeResponse:
    return await gesture_service.analyze_video(file, expected_action=expected_action, face_present=face_present)


@router.post("/v1/services/audio/analyze", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_audio(
    file: UploadFile = File(...),
    phrase_expected: str | None = Form(default=None),
    phrase_transcribed: str | None = Form(default=None),
) -> ServiceAnalyzeResponse:
    return await audio_service.analyze_audio(
        file,
        phrase_expected=phrase_expected,
        phrase_transcribed=phrase_transcribed,
    )


@router.post("/v1/services/classifier/analyze-video", response_model=ServiceAnalyzeResponse, tags=["services"])
async def analyze_classifier_video(
    file: UploadFile = File(...),
    face_present: bool | None = Form(default=None),
    face_confidence: float | None = Form(default=None),
) -> ServiceAnalyzeResponse:
    return await classifier_service.analyze_video(file, face_present=face_present, face_confidence=face_confidence)
