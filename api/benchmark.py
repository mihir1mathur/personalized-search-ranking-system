"""
benchmark.py  --  measure the API's latency, memory, and CPU, and write the
Week 6 performance reports.
===========================================================================

Run it with (offline-friendly)::

    HF_HUB_OFFLINE=1 python -m api.benchmark

WHAT IT MEASURES
----------------
It builds the REAL :class:`SearchService` (loading every Week 0-5 model/cache
exactly as the running API does), then times a fixed set of representative
queries across every method:

  * COLD latency  -- cache cleared before the call (worst case).
  * WARM latency  -- the same call again, served from the query cache.
  * per-stage averages: average search time, average rerank time, average LTR
    time (the numbers the task asks for).

It also records process memory (RSS before/after load) and CPU info via psutil.

OUTPUTS (all NEW Week 6 files; nothing from Weeks 0-5 is touched):
  results/week6_api_benchmark.txt   full report
  results/week6_latency.txt         latency tables
  results/week6_memory.txt          memory report
  results/week6_api_examples.txt    example queries + JSON responses + curl

It does NOT retrain, re-index, or regenerate any Week 0-5 output, and it opens
the Week 5 cross-encoder cache read-only.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

from api.config.settings import get_settings
from api.services.search_service import SearchService
from api.utils.logging_config import configure_logging

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None

# A small, representative query set (typos, synonyms, exact terms, multi-word).
BENCH_QUERIES = [
    "wireless noise cancelling headphones",
    "iphone charger cable",
    "running shoes for men",
    "stainless steel water bottle",
    "4k smart tv",
    "laptop backpack waterproof",
    "holiday pajamas",
    "bluetooth speaker",
]

SINGLE_METHODS = ["tfidf", "bm25", "embedding", "hybrid"]
PIPELINE_METHODS = ["rerank", "ltr"]


def _rss_mb() -> float:
    if _PROC is None:
        return 0.0
    return _PROC.memory_info().rss / (1024 * 1024)


def _time_call(fn: Callable[[], dict]) -> tuple[float, dict]:
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000.0, result


def _run_method(service: SearchService, method: str,
                queries: List[str]) -> Dict[str, List[float]]:
    """Return cold + warm latencies (ms) for a method across the queries."""
    cold, warm = [], []
    for q in queries:
        service.cache.clear()  # force a cold computation
        cold_ms, _ = _time_call(lambda: service.search(q, top_k=10, method=method))
        warm_ms, _ = _time_call(lambda: service.search(q, top_k=10, method=method))
        cold.append(cold_ms)
        warm.append(warm_ms)
    return {"cold": cold, "warm": warm}


def _fmt_stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    return (f"mean={statistics.mean(values):8.2f}  "
            f"median={statistics.median(values):8.2f}  "
            f"min={min(values):8.2f}  max={max(values):8.2f}")


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    results_dir = settings.project_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print("WEEK 6: API PERFORMANCE BENCHMARK")
    print("=" * 78)

    rss_before = _rss_mb()
    t_load0 = time.perf_counter()
    service = SearchService(settings)
    service.load()
    load_secs = time.perf_counter() - t_load0
    rss_after = _rss_mb()

    if not service.ready:
        print("Service is DEGRADED; aborting benchmark. Health:")
        print(json.dumps(service.health(), indent=2))
        return

    print(f"Loaded in {load_secs:.1f}s | corpus={len(service.product_ids):,} "
          f"products | RSS {rss_before:.0f} -> {rss_after:.0f} MB")

    # ------------------------------------------------------------------ latency
    per_method: Dict[str, Dict[str, List[float]]] = {}
    rss_peak = max(rss_before, rss_after)
    for method in SINGLE_METHODS + PIPELINE_METHODS:
        print(f"  benchmarking {method} ...")
        per_method[method] = _run_method(service, method, BENCH_QUERIES)
        rss_peak = max(rss_peak, _rss_mb())  # true running max, never < steady

    # Aggregate the task's requested averages.
    avg_search = statistics.mean(
        [statistics.mean(per_method[m]["cold"]) for m in SINGLE_METHODS])
    avg_rerank = statistics.mean(per_method["rerank"]["cold"])
    avg_ltr = statistics.mean(per_method["ltr"]["cold"])
    all_warm = [v for m in per_method for v in per_method[m]["warm"]]
    avg_cached = statistics.mean(all_warm)

    cpu_count = (psutil.cpu_count(logical=True) if psutil else os.cpu_count())
    cpu_count_phys = (psutil.cpu_count(logical=False) if psutil else None)

    # ---------------------------------------------------------------- write files
    _write_latency(results_dir, per_method, avg_search, avg_rerank, avg_ltr, avg_cached)
    _write_memory(results_dir, rss_before, rss_after, rss_peak, load_secs,
                  len(service.product_ids), cpu_count, cpu_count_phys)
    _write_benchmark(results_dir, service, per_method, avg_search, avg_rerank,
                     avg_ltr, avg_cached, load_secs, rss_after, rss_peak,
                     cpu_count, cpu_count_phys)
    _write_examples(results_dir, service)

    print("\nWEEK 6 BENCHMARK COMPLETE. Wrote:")
    for name in ("week6_api_benchmark.txt", "week6_latency.txt",
                 "week6_memory.txt", "week6_api_examples.txt"):
        print(f"  results/{name}")


def _bar() -> str:
    return "=" * 78


def _write_latency(results_dir: Path, per_method, avg_search, avg_rerank,
                   avg_ltr, avg_cached) -> None:
    lines = [_bar(), "WEEK 6: API LATENCY REPORT",
             "Project 3: Personalized Search Ranking System", _bar(), "",
             f"Queries per method : {len(BENCH_QUERIES)}",
             "Top-K              : 10",
             "COLD = cache cleared before the call (full compute).",
             "WARM = identical repeat call served from the query cache.",
             "All numbers are milliseconds (ms).", ""]
    lines.append("PER-METHOD LATENCY (ms)")
    lines.append("-" * 78)
    lines.append(f"| {'Method':<10} | {'phase':<5} | {'mean':>8} | {'median':>8} "
                 f"| {'min':>8} | {'max':>8} |")
    lines.append("| " + "-" * 10 + " | " + "-" * 5 + " | " + "-" * 8 + " | "
                 + "-" * 8 + " | " + "-" * 8 + " | " + "-" * 8 + " |")
    for method, phases in per_method.items():
        for phase in ("cold", "warm"):
            v = phases[phase]
            lines.append(f"| {method:<10} | {phase:<5} | {statistics.mean(v):8.2f} "
                         f"| {statistics.median(v):8.2f} | {min(v):8.2f} "
                         f"| {max(v):8.2f} |")
    lines += ["", "TASK-REQUESTED AVERAGES (cold unless noted)", "-" * 78,
              f"  Average search time (tfidf/bm25/embedding/hybrid): {avg_search:8.2f} ms",
              f"  Average rerank time (hybrid + cross-encoder)     : {avg_rerank:8.2f} ms",
              f"  Average LTR time (full pipeline)                 : {avg_ltr:8.2f} ms",
              f"  Average cached (warm) response                   : {avg_cached:8.2f} ms",
              ""]
    (results_dir / "week6_latency.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_memory(results_dir: Path, rss_before, rss_after, rss_peak, load_secs,
                  corpus, cpu_count, cpu_count_phys) -> None:
    lines = [_bar(), "WEEK 6: API MEMORY & CPU REPORT",
             "Project 3: Personalized Search Ranking System", _bar(), "",
             "Resident set size (RSS) = actual physical RAM used by the process.",
             "", "MEMORY", "-" * 78,
             f"  RSS before load          : {rss_before:8.0f} MB",
             f"  RSS after load (steady)  : {rss_after:8.0f} MB",
             f"  RSS peak during bench    : {rss_peak:8.0f} MB",
             f"  Load footprint           : {rss_after - rss_before:8.0f} MB",
             f"  Startup (model load) time: {load_secs:8.1f} s",
             f"  Corpus indexed           : {corpus:,} products", "",
             "CPU", "-" * 78,
             f"  Logical CPUs             : {cpu_count}",
             f"  Physical CPUs            : {cpu_count_phys}",
             "  Inference device         : cpu", "",
             "NOTES", "-" * 78,
             "  * Product embeddings are loaded from the cached Week 2 .npy files,",
             "    so the 48k-product catalog is never re-encoded at startup.",
             "  * The dominant resident memory is the product embedding matrix and",
             "    the transformer weights (embedding + cross-encoder models).",
             "  * The LTR model is < 1 MB and negligible.", ""]
    (results_dir / "week6_memory.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_benchmark(results_dir: Path, service, per_method, avg_search,
                     avg_rerank, avg_ltr, avg_cached, load_secs, rss_after,
                     rss_peak, cpu_count, cpu_count_phys) -> None:
    stats = service.cache_stats()
    lines = [_bar(), "WEEK 6: API BENCHMARK SUMMARY",
             "Project 3: Personalized Search Ranking System", _bar(), "",
             "SETUP", "-" * 78,
             f"  Corpus (unique products) : {len(service.product_ids):,}",
             f"  Startup time             : {load_secs:.1f} s",
             f"  Steady-state RSS         : {rss_after:.0f} MB "
             f"(peak {rss_peak:.0f} MB)",
             f"  CPUs (logical/physical)  : {cpu_count}/{cpu_count_phys}",
             f"  Queries benchmarked      : {len(BENCH_QUERIES)} per method", "",
             "HEADLINE LATENCY (cold, mean ms)", "-" * 78]
    for method in SINGLE_METHODS + PIPELINE_METHODS:
        lines.append(f"  {method:<10} : {statistics.mean(per_method[method]['cold']):8.2f} ms")
    lines += ["", "AVERAGES REQUESTED BY THE TASK", "-" * 78,
              f"  Average search time  : {avg_search:8.2f} ms  "
              "(tfidf/bm25/embedding/hybrid)",
              f"  Average rerank time  : {avg_rerank:8.2f} ms  (cross-encoder stage)",
              f"  Average LTR time     : {avg_ltr:8.2f} ms  (full pipeline)",
              f"  Average cached call  : {avg_cached:8.2f} ms  (query-cache hit)", "",
              "CACHE EFFECTIVENESS", "-" * 78,
              f"  A warm (cached) response is ~{(avg_ltr / avg_cached):.0f}x faster "
              "than a cold LTR call.",
              f"  Cache stats after benchmark: {stats}", "",
              "INTERPRETATION", "-" * 78,
              "  * Single-stage retrieval (tfidf/bm25/embedding) and hybrid fusion",
              "    are fast: they compare precomputed representations.",
              "  * rerank/ltr are dominated by the cross-encoder transformer pass",
              "    over the candidate shortlist -- the expected, well-understood cost.",
              "  * The query cache turns repeated popular queries into microsecond",
              "    responses, which is where most production traffic lands.",
              "  * Everything runs comfortably on CPU; no GPU is required to serve.", ""]
    (results_dir / "week6_api_benchmark.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_examples(results_dir: Path, service) -> None:
    lines = [_bar(), "WEEK 6: API EXAMPLE REQUESTS & JSON RESPONSES",
             "Project 3: Personalized Search Ranking System", _bar(), "",
             "Real responses produced by the SearchService (top_k=5). The same",
             "JSON is returned over HTTP by the FastAPI endpoints.", ""]
    demos = [
        ("ltr", "wireless noise cancelling headphones", "/ltr-search"),
        ("rerank", "iphone charger cable", "/rerank"),
        ("hybrid", "holiday pajamas", "/hybrid-search"),
        ("bm25", "stainless steel water bottle", "/search"),
    ]
    for method, query, path in demos:
        result = service.search(query, top_k=5, method=method)
        lines.append("-" * 78)
        lines.append(f"POST {path}")
        curl_body = {"query": query, "top_k": 5}
        if path == "/search":
            curl_body["method"] = method
        lines.append("curl example:")
        lines.append(f"  curl -X POST http://127.0.0.1:8000{path} \\")
        lines.append(f"       -H 'Content-Type: application/json' \\")
        lines.append(f"       -d '{json.dumps(curl_body)}'")
        lines.append("")
        lines.append("response:")
        lines.append(json.dumps(result, indent=2))
        lines.append("")
    (results_dir / "week6_api_examples.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
