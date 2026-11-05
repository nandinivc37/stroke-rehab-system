"""Health and readiness routes for the NARS API."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Check API process health")
def health_check():
    """Return process health without touching the database or ML models."""
    return {"status": "ok", "service": "nars-api", "api_version": "v1"}
