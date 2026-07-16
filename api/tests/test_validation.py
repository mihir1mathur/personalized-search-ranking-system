"""Tests for input validation (query, top_k, method, weights, depth)."""

from __future__ import annotations

import pytest


def test_empty_query_rejected(client):
    # Pydantic min_length rejects "" at the edge (422).
    resp = client.post("/search", json={"query": "", "method": "tfidf"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "ValidationError"


def test_whitespace_query_rejected(client):
    resp = client.post("/search", json={"query": "    ", "method": "tfidf"})
    assert resp.status_code == 422


def test_unsupported_method_rejected(client):
    resp = client.post("/search", json={"query": "shoes", "method": "magic"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "UnsupportedMethod"
    assert "magic" in body["detail"]


def test_top_k_zero_rejected(client):
    resp = client.post("/search", json={"query": "shoes", "top_k": 0})
    assert resp.status_code == 422


def test_top_k_over_max_rejected(client):
    resp = client.post("/search", json={"query": "shoes", "top_k": 100000,
                                        "method": "hybrid"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "InvalidParameter"


def test_query_too_long_rejected(client):
    # Longer than pydantic's max_length -> 422 ValidationError at the edge.
    resp = client.post("/search", json={"query": "x" * 5000, "method": "tfidf"})
    assert resp.status_code == 422


@pytest.mark.parametrize("bad", [-0.5, 1.5])
def test_hybrid_weight_out_of_range_rejected(client, bad):
    resp = client.post("/hybrid-search", json={"query": "cable", "alpha": bad})
    assert resp.status_code == 422


def test_candidate_depth_over_pool_rejected(client):
    # candidate_pool_size defaults to 100; ask for more -> InvalidParameter.
    resp = client.post("/rerank", json={"query": "cable", "candidate_depth": 100000})
    assert resp.status_code == 422
    assert resp.json()["error"] == "InvalidParameter"


def test_missing_query_field_rejected(client):
    resp = client.post("/search", json={"method": "tfidf"})
    assert resp.status_code == 422
