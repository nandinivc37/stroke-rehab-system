"""Smoke tests for the initial NARS API foundation."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_service_status():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "nars-api",
        "api_version": "v1",
    }


def test_root_endpoint_points_to_api_docs_and_health():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
    assert response.json()["health"] == "/api/v1/health"
