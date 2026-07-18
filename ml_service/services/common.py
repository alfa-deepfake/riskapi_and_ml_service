from __future__ import annotations

import re

from fastapi import HTTPException, UploadFile

from ml_service.api.schemas import CheckScore, ServiceAnalyzeResponse


# Short webm/png artifacts are a few MB; anything near this cap is not a valid
# challenge recording. Guards service memory, not the network — pair with a
# request-size limit on the reverse proxy in real deployments.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def safe_suffix(filename: str | None, default: str) -> str:
    """A tempfile suffix derived from a client filename, never longer than a
    real extension. A NUL byte or a 5000-char extension would otherwise crash
    NamedTemporaryFile with an OSError/ValueError (HTTP 500)."""
    suffix = ""
    if filename:
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        if re.fullmatch(r"[A-Za-z0-9]{1,8}", ext):
            suffix = "." + ext
    return suffix or default


async def read_upload(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Загруженный файл превышает {max_bytes // (1024 * 1024)} МБ",
        )
    return data


def service_response(service: str, evidence, check: CheckScore, message: str | None = None) -> ServiceAnalyzeResponse:
    return ServiceAnalyzeResponse(
        service=service,
        status=check.status,
        evidence=evidence.model_dump(mode="json"),
        check=check,
        message=message,
    )


def unavailable_check(name: str, weight: float, reason: str) -> CheckScore:
    return CheckScore(
        name=name,
        status="unknown",
        risk=0.65,
        confidence=0.0,
        weight=weight,
        reason=reason,
    )
