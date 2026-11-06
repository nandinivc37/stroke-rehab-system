"""FastAPI application entry point for NARS."""

from fastapi import Depends, FastAPI

from src.api.dependencies import get_current_user
from src.database import db
from src.database.models import Base
from src.database.sqlalchemy_support import create_database_engine, create_session_factory
from src.api.v1.auth import router as auth_router
from src.api.v1.health import router as health_router
from src.api.v1.patients import router as patients_router
from src.api.v1.assessments import router as assessments_router
from src.api.v1.recommendations import router as recommendations_router
from src.api.v1.sessions import router as sessions_router
from src.api.v1.sessions import stream_router as session_stream_router
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
# WebSocket streaming authenticates with its first message (browser clients
# cannot attach arbitrary Authorization headers to the native WebSocket API).
app.include_router(session_stream_router, prefix="/api/v1")
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
def initialize_database_services():
    from src.services.auth_service import (
        bootstrap_admin_from_environment,
        ensure_auth_schema,
    )

    # The desktop application's legacy SQLite schema is retained. API ORM
    # repositories use NARS_DATABASE_URL when configured, otherwise that DB.
    db.initialize_database()
    engine = create_database_engine()
    app.state.database_engine = engine
    app.state.database_session_factory = create_session_factory(engine)

    # SQLite remains the local development/test database. Create any missing
    # ORM tables alongside the legacy schema; deployed PostgreSQL databases
    # must be upgraded explicitly through Alembic.
    if engine.dialect.name == "sqlite":
        Base.metadata.create_all(engine)

    ensure_auth_schema(engine)
    from src.database.repositories.live_session_repository import (
        mark_active_sessions_interrupted,
    )
    recovery_session = app.state.database_session_factory()
    try:
        # Analyzer state is process-local for both SQLite and PostgreSQL.
        # A restart must never expose a formerly ACTIVE session as resumable.
        mark_active_sessions_interrupted(recovery_session)
    finally:
        recovery_session.close()
    bootstrap_admin_from_environment(app.state.database_session_factory)


@app.on_event("shutdown")
def dispose_database_engine():
    engine = getattr(app.state, "database_engine", None)
    if engine is not None:
        engine.dispose()
