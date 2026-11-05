"""FastAPI application entry point for NARS."""

from fastapi import FastAPI

from src.api.v1.health import router as health_router
from src.api.v1.patients import router as patients_router

app = FastAPI(
    title="NARS API",
    description=(
        "API backend for the NARS vision-based stroke rehabilitation "
        "prototype. Clinical validation is not established."
    ),
    version="0.1.0",
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(patients_router, prefix="/api/v1")


@app.get("/", tags=["health"], include_in_schema=False)
def root():
    return {
        "service": "NARS API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
