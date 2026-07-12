from __future__ import annotations

from fastapi import HTTPException, UploadFile

from ml_service.api.schemas import CheckScore, ServiceAnalyzeResponse


# Short webm/png artifacts are a few MB; anything near this cap is not a valid
# challenge recording. Guards service memory, not the network — pair with a
# request-size limit on the reverse proxy in real deployments.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


async def read_upload(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"uploaded file exceeds {max_bytes} bytes",
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
