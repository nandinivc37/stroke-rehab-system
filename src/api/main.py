"""FastAPI application entry point for NARS."""

from fastapi import Depends, FastAPI

from src.api.dependencies import get_current_user
from src.api.v1.auth import router as auth_router
from src.api.v1.health import router as health_router
from src.api.v1.patients import router as patients_router
from src.api.v1.assessments import router as assessments_router
from src.api.v1.recommendations import router as recommendations_router
from src.api.v1.sessions import router as sessions_router
from src.api.v1.adaptation import router as adaptation_router
from src.api.v1.progress import router as progress_router

app = FastAPI(
    title="NARS API",
    description=(
        "API backend for the NARS vision-based stroke rehabilitation "
        "prototype. Clinical validation is not established."
    ),
    version="0.1.0",
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(patients_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(assessments_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(recommendations_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(sessions_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(adaptation_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(progress_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])


@app.get("/", tags=["health"], include_in_schema=False)
def root():
    return {
        "service": "NARS API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.on_event("startup")
def initialize_authentication():
    from src.services.auth_service import bootstrap_admin_from_environment, ensure_auth_schema

    ensure_auth_schema()
    bootstrap_admin_from_environment()
