from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ml_service.core.challenge import ChallengePlan
from ml_service.core.challenge_store import SessionRecord


Decision = Literal["allow", "review", "deny"]
CheckStatus = Literal["passed", "failed", "unknown", "skipped"]


class HealthResponse(BaseModel):
    status: str
    service: str
    time: datetime


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    uid: str = Field(..., min_length=1)
    check_id: str = Field(..., min_length=1)
    scenario: str = Field(default="video_verification", min_length=1)


class SessionResponse(BaseModel):
    session_id: str
    uid: str
    check_id: str
    scenario: str
    created_at: datetime
    expires_at: datetime
    challenge: ChallengePlan

    @classmethod
    def from_session(cls, session: SessionRecord) -> "SessionResponse":
        return cls(**session.model_dump())


class ClassifierEvidence(BaseModel):
    skipped: bool = False
    fake_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model_name: str | None = None
    model_scores: dict[str, float] | None = None
    dropped_models: list[str] | None = None
    frame_count: int | None = Field(default=None, ge=0)
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ActiveLightEvidence(BaseModel):
    skipped: bool = False
    expected_luma: list[float] = Field(default_factory=list)
    observed_face_luma: list[float] = Field(default_factory=list)
    observed_background_luma: list[float] = Field(default_factory=list)
    frame_timestamps_ms: list[float] = Field(default_factory=list)
    detector: str | None = None
    verifier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    pair_count: int | None = Field(default=None, ge=0)
    temporal_correlation: float | None = None
    best_correlation: float | None = None
    spatial_contrast: float | None = None
    spatial_center_error: float | None = None
    response_snr: float | None = None
    response_magnitude: float | None = None
    color_cosine: float | None = None
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RppgEvidence(BaseModel):
    skipped: bool = False
    bpm: float | None = Field(default=None, ge=20.0, le=220.0)
    signal_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    latency: float | None = Field(default=None, ge=0.0)
    hrv: dict[str, float | None] = Field(default_factory=dict)
    samples: list[float] = Field(default_factory=list)
    sample_rate_hz: float | None = Field(default=None, gt=0.0)
    window_seconds: float | None = Field(default=None, gt=0.0)
    detector: str | None = None
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GestureEvidence(BaseModel):
    skipped: bool = False
    expected_action: str | None = None
    observed_action: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    completed_at_ms: float | None = Field(default=None, ge=0.0)
    detector: str | None = None
    face_present: bool | None = None
    frame_count: int | None = Field(default=None, ge=0)
    best_distance: float | None = Field(default=None, ge=0.0)


class AudioEvidence(BaseModel):
    skipped: bool = False
    phrase_expected: str | None = None
    phrase_transcribed: str | None = None
    ai_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker_match_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    detector: str | None = None


class EvidenceBundle(BaseModel):
    classifier: ClassifierEvidence | None = None
    active_light: ActiveLightEvidence | None = None
    rppg: RppgEvidence | None = None
    gesture: GestureEvidence | None = None
    audio: AudioEvidence | None = None


class ScoreRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    uid: str = Field(..., min_length=1)
    check_id: str | None = Field(default=None, min_length=1)
    challenge: ChallengePlan | None = None
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    uid: str = Field(..., min_length=1)
    check_id: str = Field(..., min_length=1)
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)

    def to_score_request(self, challenge: ChallengePlan) -> ScoreRequest:
        return ScoreRequest(uid=self.uid, check_id=self.check_id, challenge=challenge, evidence=self.evidence)


class CheckScore(BaseModel):
    name: str
    status: CheckStatus
    risk: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    reason: str
    details: dict = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    uid: str
    check_id: str | None
    decision: Decision
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    checks: list[CheckScore]
    factors: list[str]
    created_at: datetime


class ServiceAnalyzeResponse(BaseModel):
    service: str
    status: CheckStatus
    evidence: dict
    check: CheckScore
    message: str | None = None


class ActiveLightAnalyzeRequest(BaseModel):
    expected_luma: list[float] = Field(default_factory=list)
    observed_face_luma: list[float] = Field(default_factory=list)
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RppgAnalyzeRequest(BaseModel):
    samples: list[float] = Field(default_factory=list)
    sample_rate_hz: float = Field(default=10.0, gt=0.0)
    window_seconds: float = Field(default=4.0, gt=0.0)
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
