"""
utils.py  --  frontend helpers: backend API client + report parsing.
====================================================================

This module is the ONLY place the frontend talks to the outside world. It:

  * knows the backend base URL (configurable via the SEARCH_API_URL env var),
  * maps the six user-facing ranking methods onto the correct backend endpoint
    (so the UI never duplicates any search/ranking logic -- it only calls the
    Week 6 REST API),
  * turns network / HTTP failures into FRIENDLY messages (never a raw Python
    traceback), via the ApiError exception,
  * parses the Week 6 benchmark text reports in results/ into plain dicts the
    Benchmark page can chart.

Nothing here imports the backend Python code; the frontend is a pure HTTP
client, exactly like a real web app would be.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def backend_url() -> str:
    """Base URL of the FastAPI backend (override with SEARCH_API_URL)."""
    return os.environ.get("SEARCH_API_URL", DEFAULT_BACKEND_URL).rstrip("/")


# The six user-facing methods, in display order, each mapped to the backend
# endpoint that serves it. tfidf/bm25/embedding share the generic /search
# endpoint; hybrid/rerank/ltr use their dedicated endpoints so per-request
# tuning parameters (alpha/beta/candidate_depth) can be passed through.
METHODS: Dict[str, Dict[str, str]] = {
    "TF-IDF":       {"key": "tfidf",     "endpoint": "/search",        "stage": "Lexical retrieval"},
    "BM25":         {"key": "bm25",      "endpoint": "/search",        "stage": "Lexical retrieval"},
    "Embeddings":   {"key": "embedding", "endpoint": "/search",        "stage": "Semantic retrieval"},
    "Hybrid":       {"key": "hybrid",    "endpoint": "/hybrid-search", "stage": "BM25 + embedding fusion"},
    "CrossEncoder": {"key": "rerank",    "endpoint": "/rerank",        "stage": "Two-stage re-ranking"},
    "LTR":          {"key": "ltr",       "endpoint": "/ltr-search",    "stage": "Full learned pipeline"},
}
METHOD_LABELS: List[str] = list(METHODS.keys())


class ApiError(Exception):
    """A user-friendly error the UI can show verbatim (no traceback leaks)."""


# --------------------------------------------------------------------------
# Low-level request helper
# --------------------------------------------------------------------------
def _request(method: str, path: str, *, json: Optional[dict] = None,
             timeout: float = 30.0) -> Any:
    url = f"{backend_url()}{path}"
    try:
        resp = requests.request(method, url, json=json, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise ApiError(
            f"Cannot reach the backend at {backend_url()}. "
            "Is it running? Start it with:  uvicorn api.main:app")
    except requests.exceptions.Timeout:
        raise ApiError(
            f"The backend took too long to respond (> {timeout:.0f}s). "
            "Cross-encoder / LTR calls can be slow on CPU for a cold query.")
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Network error talking to the backend: {exc}")

    if resp.status_code >= 400:
        # The backend returns a uniform error body {error, detail, ...}.
        detail = None
        try:
            body = resp.json()
            detail = body.get("detail") or body.get("error")
        except Exception:
            detail = resp.text[:200] if resp.text else None
        raise ApiError(detail or f"Backend returned HTTP {resp.status_code}.")

    try:
        return resp.json()
    except ValueError:
        raise ApiError("Backend returned a non-JSON response.")


# --------------------------------------------------------------------------
# Public API-client functions (used by the pages)
# --------------------------------------------------------------------------
def search(query: str, method_label: str, top_k: int = 10,
           alpha: Optional[float] = None, beta: Optional[float] = None,
           candidate_depth: Optional[int] = None,
           timeout: float = 60.0) -> Dict[str, Any]:
    """
    Run a search via the backend, routing to the right endpoint for the chosen
    method. Returns the response envelope dict. Raises ApiError on any failure.
    """
    if method_label not in METHODS:
        raise ApiError(f"Unknown ranking method: {method_label}")
    spec = METHODS[method_label]
    key, endpoint = spec["key"], spec["endpoint"]

    payload: Dict[str, Any] = {"query": query, "top_k": top_k}
    if endpoint == "/search":
        payload["method"] = key
    elif endpoint == "/hybrid-search":
        if alpha is not None:
            payload["alpha"] = alpha
        if beta is not None:
            payload["beta"] = beta
    else:  # /rerank or /ltr-search
        if candidate_depth is not None:
            payload["candidate_depth"] = candidate_depth

    return _request("POST", endpoint, json=payload, timeout=timeout)


def get_health(timeout: float = 5.0) -> Dict[str, Any]:
    return _request("GET", "/health", timeout=timeout)


def get_version(timeout: float = 5.0) -> Dict[str, Any]:
    return _request("GET", "/version", timeout=timeout)


def backend_reachable(timeout: float = 3.0) -> bool:
    """True if /health responds at all (even 503). Never raises."""
    try:
        requests.get(f"{backend_url()}/health", timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def confidence_pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def fmt_ms(value: float) -> str:
    v = float(value)
    if v < 1000:
        return f"{v:.1f} ms"
    return f"{v / 1000:.2f} s"


# --------------------------------------------------------------------------
# Week 6 benchmark report parsing (read-only; files live in results/)
# --------------------------------------------------------------------------
def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def parse_latency_report() -> Dict[str, Any]:
    """
    Parse results/week6_latency.txt into:
      {"per_method": {method: {"cold": mean, "warm": mean}}, "averages": {...}}
    Returns {} if the file is missing (page degrades gracefully).
    """
    text = _read(RESULTS_DIR / "week6_latency.txt")
    if not text:
        return {}
    per_method: Dict[str, Dict[str, float]] = {}
    # Rows like: | tfidf      | cold  |   296.67 |   290.99 | ...
    row = re.compile(
        r"\|\s*([a-z0-9\-]+)\s*\|\s*(cold|warm)\s*\|\s*([\d.]+)\s*\|")
    for m in row.finditer(text):
        method, phase, mean = m.group(1), m.group(2), float(m.group(3))
        per_method.setdefault(method, {})[phase] = mean

    averages: Dict[str, float] = {}
    patterns = {
        "search": r"Average search time[^:]*:\s*([\d.]+)\s*ms",
        "rerank": r"Average rerank time[^:]*:\s*([\d.]+)\s*ms",
        "ltr": r"Average LTR time[^:]*:\s*([\d.]+)\s*ms",
        "cached": r"Average cached[^:]*:\s*([\d.]+)\s*ms",
    }
    for name, pat in patterns.items():
        found = re.search(pat, text)
        if found:
            averages[name] = float(found.group(1))
    return {"per_method": per_method, "averages": averages}


def parse_memory_report() -> Dict[str, Any]:
    """Parse results/week6_memory.txt into a flat dict of numbers."""
    text = _read(RESULTS_DIR / "week6_memory.txt")
    if not text:
        return {}
    out: Dict[str, Any] = {}
    grabs = {
        "rss_before_mb": r"RSS before load\s*:\s*([\d.]+)\s*MB",
        "rss_after_mb": r"RSS after load[^:]*:\s*([\d.]+)\s*MB",
        "rss_peak_mb": r"RSS peak[^:]*:\s*([\d.]+)\s*MB",
        "load_footprint_mb": r"Load footprint\s*:\s*([\d.]+)\s*MB",
        "startup_s": r"Startup[^:]*:\s*([\d.]+)\s*s",
        "logical_cpus": r"Logical CPUs\s*:\s*(\d+)",
        "physical_cpus": r"Physical CPUs\s*:\s*(\d+)",
    }
    for name, pat in grabs.items():
        found = re.search(pat, text)
        if found:
            val = found.group(1)
            out[name] = float(val) if "." in val else int(val)
    found = re.search(r"Corpus indexed\s*:\s*([\d,]+)", text)
    if found:
        out["corpus"] = int(found.group(1).replace(",", ""))
    return out


def parse_benchmark_summary() -> Dict[str, Any]:
    """Parse results/week6_api_benchmark.txt for the cache-speedup headline."""
    text = _read(RESULTS_DIR / "week6_api_benchmark.txt")
    if not text:
        return {}
    out: Dict[str, Any] = {}
    found = re.search(r"~\s*([\d,]+)x faster", text)
    if found:
        out["cache_speedup_x"] = int(found.group(1).replace(",", ""))
    return out


def benchmark_examples_text() -> Optional[str]:
    """Raw text of the Week 6 request/response examples (for display)."""
    return _read(RESULTS_DIR / "week6_api_examples.txt")


def report_files_present() -> Dict[str, bool]:
    return {
        "week6_latency.txt": (RESULTS_DIR / "week6_latency.txt").exists(),
        "week6_memory.txt": (RESULTS_DIR / "week6_memory.txt").exists(),
        "week6_api_benchmark.txt": (RESULTS_DIR / "week6_api_benchmark.txt").exists(),
        "week6_api_examples.txt": (RESULTS_DIR / "week6_api_examples.txt").exists(),
    }
