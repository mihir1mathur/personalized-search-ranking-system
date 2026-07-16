"""Tests for the search endpoints and the response contract."""

from __future__ import annotations

import pytest

RESULT_KEYS = {"rank", "product_id", "title", "score", "confidence"}


def _assert_envelope(body, method):
    assert body["ranking_method"] == method
    assert body["query"]
    assert body["processing_time_ms"] >= 0.0
    assert isinstance(body["results"], list)
    assert body["count"] == len(body["results"])
    for item in body["results"]:
        assert RESULT_KEYS.issubset(item.keys())
        assert 0.0 <= item["confidence"] <= 1.0
        assert item["rank"] >= 1


@pytest.mark.parametrize("method", ["tfidf", "bm25", "embedding", "hybrid", "rerank", "ltr"])
def test_search_all_methods(client, method):
    resp = client.post("/search", json={"query": "wireless headphones",
                                         "top_k": 3, "method": method})
    assert resp.status_code == 200
    body = resp.json()
    _assert_envelope(body, method)
    assert body["count"] <= 3
    # ranks are contiguous starting at 1
    assert [r["rank"] for r in body["results"]] == list(range(1, body["count"] + 1))


def test_search_get_form(client):
    resp = client.get("/search", params={"q": "laptop", "top_k": 2, "method": "bm25"})
    assert resp.status_code == 200
    _assert_envelope(resp.json(), "bm25")


def test_hybrid_search_with_weights(client):
    resp = client.post("/hybrid-search",
                        json={"query": "usb cable", "top_k": 4,
                              "alpha": 0.3, "beta": 0.7})
    assert resp.status_code == 200
    _assert_envelope(resp.json(), "hybrid")


def test_rerank_endpoint(client):
    resp = client.post("/rerank", json={"query": "iphone charger", "top_k": 3})
    assert resp.status_code == 200
    _assert_envelope(resp.json(), "rerank")


def test_ltr_endpoint(client):
    resp = client.post("/ltr-search", json={"query": "running shoes", "top_k": 5})
    assert resp.status_code == 200
    _assert_envelope(resp.json(), "ltr")


def test_default_top_k_applied(client):
    resp = client.post("/search", json={"query": "tv"})
    assert resp.status_code == 200
    body = resp.json()
    # default_top_k is 10, but the fake corpus only has 5 products.
    assert body["top_k"] == 10
    assert body["count"] == 5


def test_query_is_stripped(client):
    resp = client.post("/search", json={"query": "  padded query  ", "method": "tfidf"})
    assert resp.status_code == 200
    assert resp.json()["query"] == "padded query"


def test_cache_second_call_is_flagged(client):
    payload = {"query": "cache me", "top_k": 3, "method": "ltr"}
    first = client.post("/ltr-search",
                        json={"query": "cache me", "top_k": 3}).json()
    second = client.post("/ltr-search",
                         json={"query": "cache me", "top_k": 3}).json()
    assert first["cached"] is False
    assert second["cached"] is True
    # Same ranking either way.
    assert [r["product_id"] for r in first["results"]] == \
           [r["product_id"] for r in second["results"]]
