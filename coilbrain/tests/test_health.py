"""Tests for CoilBrain API endpoints."""

from fastapi.testclient import TestClient

from coilbrain.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    """GET /health should return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body() -> None:
    """GET /health should return the expected JSON payload."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "coilbrain"
