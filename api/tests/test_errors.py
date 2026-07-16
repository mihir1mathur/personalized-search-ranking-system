"""Tests for error handling and the uniform error contract."""

from __future__ import annotations

ERROR_KEYS = {"error", "detail", "status_code", "request_id"}


def test_service_not_ready_returns_503(degraded_client):
    resp = degraded_client.post("/ltr-search", json={"query": "shoes"})
    assert resp.status_code == 503
    body = resp.json()
    assert ERROR_KEYS.issubset(body.keys())
    assert body["error"] == "ModelNotReady"
    assert body["status_code"] == 503


def test_unknown_route_returns_404_uniform(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert ERROR_KEYS.issubset(body.keys())
    assert body["error"] == "HTTPError"


def test_error_includes_request_id(client):
    resp = client.post("/search", json={"query": "shoes", "method": "nope"})
    assert resp.status_code == 422
    assert resp.json()["request_id"] is not None


def test_validation_error_body_shape(client):
    resp = client.post("/search", json={"query": ""})
    body = resp.json()
    assert ERROR_KEYS.issubset(body.keys())
    assert body["status_code"] == 422


def test_method_not_allowed(client):
    # /health only supports GET; a POST should be a clean 405.
    resp = client.post("/health")
    assert resp.status_code == 405
    assert resp.json()["error"] == "HTTPError"
