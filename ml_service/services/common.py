from __future__ import annotations

from ml_service.api.schemas import CheckScore, ServiceAnalyzeResponse


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
