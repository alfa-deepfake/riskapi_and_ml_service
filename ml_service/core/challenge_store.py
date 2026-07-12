from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from ml_service.core.challenge import ChallengePlan, generate_challenge


class SessionRecord(BaseModel):
    session_id: str
    uid: str
    check_id: str
    scenario: str
    created_at: datetime
    expires_at: datetime
    challenge: ChallengePlan


class ChallengeStore:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._sessions: dict[str, SessionRecord] = {}

    def create(self, *, uid: str, check_id: str, scenario: str) -> SessionRecord:
        self._collect_expired()
        now = datetime.now(timezone.utc)
        session_id = str(uuid.uuid4())
        session = SessionRecord(
            session_id=session_id,
            uid=uid,
            check_id=check_id,
            scenario=scenario,
            created_at=now,
            expires_at=now + self._ttl,
            challenge=generate_challenge(),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionRecord | None:
        self._collect_expired()
        return self._sessions.get(session_id)

    def pop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _collect_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [session_id for session_id, session in self._sessions.items() if session.expires_at <= now]
        for session_id in expired:
            self._sessions.pop(session_id, None)
