from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import httpx


RiskStatus = Literal["started", "in_progress", "finished"]


class RiskApiClient:
    def __init__(self, *, base_url: str | None, enabled: bool, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._enabled = enabled and bool(self._base_url)
        self._timeout = timeout_seconds

    async def send_status(
        self,
        check_id: str,
        *,
        uid: str,
        status: RiskStatus,
        success: bool | None = None,
        message: str | None = None,
    ) -> None:
        if not self._enabled:
            return
        payload = {
            "uid": uid,
            "status": status,
            "success": success,
            "message": message,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._post(f"/checks/{check_id}/status", payload)

    async def send_result(self, check_id: str, *, uid: str, score: dict[str, Any]) -> None:
        if not self._enabled:
            return
        await self._post(f"/checks/{check_id}/result", {"uid": uid, "score": score})

    async def _post(self, path: str, payload: dict[str, Any]) -> None:
        assert self._base_url is not None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}{path}", json=payload)
                response.raise_for_status()
        except httpx.HTTPError:
            # ML decision must be returned even when antifraud storage is temporarily unavailable.
            return
