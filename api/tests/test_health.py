"""Tests for /health, /version, and the root banner."""

from __future__ import annotations


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["corpus_size"] == 5
    assert any(c["name"] == "ltr" for c in body["components"])


def test_health_degraded_returns_503(degraded_client):
    resp = degraded_client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["ready"] is False


def test_version(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]
    assert "Learning-to-Rank" in body["pipeline"]
    assert "ltr" in body["supported_methods"]


def test_root_banner(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "/search" in body["endpoints"]
    assert body["docs"].endswith("/docs")


def test_openapi_and_docs_available(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_request_id_header_present(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert "x-process-time-ms" in resp.headers
