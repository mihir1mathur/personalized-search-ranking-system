"""
Frontend smoke tests (no running server, no Streamlit runtime needed).
======================================================================

These exercise the frontend's pure logic in `utils.py`:
  * the six methods map to the correct backend endpoint + payload,
  * network/HTTP failures become friendly ApiError messages (no traceback),
  * the Week 6 benchmark reports parse into usable numbers,
  * formatting helpers behave,
  * the modules import cleanly.

The HTTP layer is monkeypatched, so nothing here needs the backend running.
Run with:  python -m pytest frontend/tests -q
"""

from __future__ import annotations

import types

import pytest

import utils


class _FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


# --------------------------------------------------------------------------
# Method mapping
# --------------------------------------------------------------------------
def test_six_methods_present():
    assert utils.METHOD_LABELS == [
        "TF-IDF", "BM25", "Embeddings", "Hybrid", "CrossEncoder", "LTR"]


@pytest.mark.parametrize("label,endpoint,key", [
    ("TF-IDF", "/search", "tfidf"),
    ("BM25", "/search", "bm25"),
    ("Embeddings", "/search", "embedding"),
    ("Hybrid", "/hybrid-search", "hybrid"),
    ("CrossEncoder", "/rerank", "rerank"),
    ("LTR", "/ltr-search", "ltr"),
])
def test_method_routing(monkeypatch, label, endpoint, key):
    captured = {}

    def fake_request(method, url, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        return _FakeResp(200, {"query": "q", "ranking_method": key,
                               "results": [], "count": 0,
                               "processing_time_ms": 1.0, "cached": False})

    monkeypatch.setattr(utils.requests, "request", fake_request)
    resp = utils.search("q", label, top_k=5, alpha=0.3, beta=0.7, candidate_depth=40)

    assert resp["ranking_method"] == key
    assert captured["url"].endswith(endpoint)
    assert captured["json"]["query"] == "q"
    assert captured["json"]["top_k"] == 5
    if endpoint == "/search":
        assert captured["json"]["method"] == key
    elif endpoint == "/hybrid-search":
        assert captured["json"]["alpha"] == 0.3 and captured["json"]["beta"] == 0.7
    else:
        assert captured["json"]["candidate_depth"] == 40


def test_unknown_method_raises():
    with pytest.raises(utils.ApiError):
        utils.search("q", "NotAMethod")


# --------------------------------------------------------------------------
# Friendly error handling
# --------------------------------------------------------------------------
def test_connection_error_is_friendly(monkeypatch):
    def boom(*a, **k):
        raise utils.requests.exceptions.ConnectionError()
    monkeypatch.setattr(utils.requests, "request", boom)
    with pytest.raises(utils.ApiError) as e:
        utils.search("q", "LTR")
    assert "Cannot reach the backend" in str(e.value)


def test_timeout_is_friendly(monkeypatch):
    def slow(*a, **k):
        raise utils.requests.exceptions.Timeout()
    monkeypatch.setattr(utils.requests, "request", slow)
    with pytest.raises(utils.ApiError) as e:
        utils.search("q", "LTR")
    assert "too long" in str(e.value)


def test_http_error_uses_backend_detail(monkeypatch):
    def bad(*a, **k):
        return _FakeResp(422, {"error": "UnsupportedMethod",
                               "detail": "method 'x' is not supported"})
    monkeypatch.setattr(utils.requests, "request", bad)
    with pytest.raises(utils.ApiError) as e:
        utils.search("q", "LTR")
    assert "not supported" in str(e.value)


def test_backend_url_env_override(monkeypatch):
    monkeypatch.setenv("SEARCH_API_URL", "http://example.com:9000/")
    assert utils.backend_url() == "http://example.com:9000"


def test_backend_reachable_false(monkeypatch):
    def boom(*a, **k):
        raise utils.requests.exceptions.ConnectionError()
    monkeypatch.setattr(utils.requests, "get", boom)
    assert utils.backend_reachable() is False


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def test_confidence_pct():
    assert utils.confidence_pct(0.589) == "58.9%"


def test_fmt_ms():
    assert utils.fmt_ms(42.7) == "42.7 ms"
    assert utils.fmt_ms(2014.1) == "2.01 s"


# --------------------------------------------------------------------------
# Benchmark report parsing (uses the real results/ files if present)
# --------------------------------------------------------------------------
def test_parse_latency_report_real_files():
    data = utils.parse_latency_report()
    if not data:
        pytest.skip("Week 6 latency report not present")
    assert "per_method" in data
    # The full pipeline method should be parsed with a cold latency.
    assert "ltr" in data["per_method"]
    assert data["per_method"]["ltr"].get("cold", 0) > 0
    assert "ltr" in data["averages"]


def test_parse_memory_report_real_files():
    mem = utils.parse_memory_report()
    if not mem:
        pytest.skip("Week 6 memory report not present")
    assert mem.get("corpus", 0) > 0
    assert mem.get("rss_after_mb", 0) > 0


def test_report_files_present_shape():
    present = utils.report_files_present()
    assert set(present.keys()) == {
        "week6_latency.txt", "week6_memory.txt",
        "week6_api_benchmark.txt", "week6_api_examples.txt"}


# --------------------------------------------------------------------------
# Modules import cleanly (no import-time Streamlit calls)
# --------------------------------------------------------------------------
def test_components_imports():
    import components
    assert hasattr(components, "result_card")
    assert hasattr(components, "hero")


def test_generate_assets_imports():
    import generate_assets
    assert isinstance(generate_assets.FALLBACK_RESULT, dict)
    assert generate_assets.FALLBACK_RESULT["results"]
