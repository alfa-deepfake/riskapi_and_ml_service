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

    uid: str = Field(..., min_length=1, max_length=200)
    check_id: str = Field(..., min_length=1, max_length=200)
    scenario: str = Field(default="video_verification", min_length=1, max_length=200)


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
        data = session.model_dump()
        # The audio phrase is disclosed only by the TTL'd issue endpoint;
        # session/challenge responses must never carry it.
        for step in data["challenge"]["steps"]:
            step["payload"].pop("phrase", None)
        return cls(**data)


class AudioPhraseResponse(BaseModel):
    phrase: str
    ttl_seconds: int
    attempts_left: int


class ClassifierEvidence(BaseModel):
    skipped: bool = False
    fake_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model_name: str | None = None
    model_scores: dict[str, float] | None = None
    cnn_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    tree_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    t_susp: float | None = Field(default=None, ge=0.0, le=1.0)
    condition: str | None = None
    low_info: bool | None = None
    upsample_diff: float | None = Field(default=None, ge=0.0)
    frame_count: int | None = Field(default=None, ge=0)
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_count: int | None = Field(default=None, ge=1)
    preprocessing: str | None = None
    face_size_px: float | None = Field(default=None, gt=0.0)


class ActiveLightEvidence(BaseModel):
    skipped: bool = False
    expected_luma: list[float] = Field(default_factory=list, max_length=1_000)
    observed_face_luma: list[float] = Field(default_factory=list, max_length=1_000)
    observed_background_luma: list[float] = Field(default_factory=list, max_length=1_000)
    frame_timestamps_ms: list[float] = Field(default_factory=list, max_length=1_000)
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
    # 2 minutes at 100 Hz with headroom — enough for any real capture, small
    # enough that the O(n^2/stride) stability scan stays cheap.
    samples: list[float] = Field(default_factory=list, max_length=20_000)
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
    # Challenge phrases are ~30 chars; the cap keeps attacker-supplied strings
    # out of the O(n*m) levenshtein that runs on the event loop.
    phrase_expected: str | None = Field(default=None, max_length=500)
    phrase_transcribed: str | None = Field(default=None, max_length=2000)
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

    uid: str = Field(..., min_length=1, max_length=200)
    check_id: str | None = Field(default=None, min_length=1, max_length=200)
    challenge: ChallengePlan | None = None
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    uid: str = Field(..., min_length=1, max_length=200)
    check_id: str = Field(..., min_length=1, max_length=200)
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
    expected_luma: list[float] = Field(default_factory=list, max_length=1_000)
    observed_face_luma: list[float] = Field(default_factory=list, max_length=1_000)
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RppgAnalyzeRequest(BaseModel):
    samples: list[float] = Field(default_factory=list, max_length=20_000)
    sample_rate_hz: float = Field(default=10.0, gt=0.0)
    window_seconds: float = Field(default=4.0, gt=0.0)
    face_present: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
