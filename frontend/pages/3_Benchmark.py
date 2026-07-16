"""
Benchmark page  --  visualize the Week 6 performance reports.
============================================================
Reads (read-only) the existing results/week6_*.txt reports produced by
`python -m api.benchmark` and displays latency, memory, CPU, the per-stage
averages, and the cache speed-up as charts + metrics. Never re-runs anything.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

import components
import utils

components.configure_page("Benchmark · Search Ranking", "📊")
components.hero("Performance Benchmark",
                "Latency, memory, and CPU measured on the Week 6 backend "
                "(read from results/week6_*).")

present = utils.report_files_present()
if not any(present.values()):
    components.error_box(
        "No Week 6 benchmark reports found in results/. Generate them with: "
        "python -m api.benchmark")
    components.footer()
    st.stop()

latency = utils.parse_latency_report()
memory = utils.parse_memory_report()
summary = utils.parse_benchmark_summary()

# ---- Per-stage averages (metrics) ----------------------------------------
avgs = latency.get("averages", {})
if avgs:
    components.section_title("Average time per stage")
    cols = st.columns(4)
    cols[0].metric("Avg search", utils.fmt_ms(avgs.get("search", 0)))
    cols[1].metric("Avg rerank", utils.fmt_ms(avgs.get("rerank", 0)))
    cols[2].metric("Avg LTR", utils.fmt_ms(avgs.get("ltr", 0)))
    cols[3].metric("Avg cached", utils.fmt_ms(avgs.get("cached", 0)))

# ---- Latency per method (cold) chart -------------------------------------
per_method = latency.get("per_method", {})
if per_method:
    components.section_title("Cold latency by method (ms)")
    order = ["tfidf", "bm25", "embedding", "hybrid", "rerank", "ltr"]
    data = {m: per_method[m].get("cold", 0.0) for m in order if m in per_method}
    if data:
        df = pd.DataFrame({"cold latency (ms)": data})
        st.bar_chart(df)
        st.caption("rerank/ltr are dominated by the cross-encoder transformer "
                   "pass over the candidate shortlist — the expected cost.")

    # Cold vs warm (cache) comparison table.
    rows = []
    for m in order:
        if m in per_method:
            rows.append({"method": m,
                         "cold (ms)": round(per_method[m].get("cold", 0), 2),
                         "warm/cached (ms)": round(per_method[m].get("warm", 0), 4)})
    if rows:
        components.section_title("Cold vs cached")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---- Cache speedup --------------------------------------------------------
speed = summary.get("cache_speedup_x")
if speed:
    components.section_title("Cache effectiveness")
    st.metric("Warm (cached) vs cold LTR", f"~{speed:,}× faster")
    st.caption("A repeated popular query is served from the in-memory query "
               "cache in microseconds.")

# ---- Memory ---------------------------------------------------------------
if memory:
    components.section_title("Memory (RSS, MB)")
    mem_keys = [("rss_before_mb", "before load"),
                ("rss_after_mb", "after load"),
                ("rss_peak_mb", "peak")]
    mem_data = {label: memory[k] for k, label in mem_keys if k in memory}
    if mem_data:
        st.bar_chart(pd.DataFrame({"RSS (MB)": mem_data}))
    mcols = st.columns(3)
    if "load_footprint_mb" in memory:
        mcols[0].metric("Load footprint", f"{memory['load_footprint_mb']:.0f} MB")
    if "startup_s" in memory:
        mcols[1].metric("Startup time", f"{memory['startup_s']:.0f} s")
    if "corpus" in memory:
        mcols[2].metric("Corpus", f"{memory['corpus']:,}")

# ---- CPU ------------------------------------------------------------------
if "logical_cpus" in memory or "physical_cpus" in memory:
    components.section_title("CPU")
    ccols = st.columns(3)
    ccols[0].metric("Logical CPUs", memory.get("logical_cpus", "-"))
    ccols[1].metric("Physical CPUs", memory.get("physical_cpus", "-"))
    ccols[2].metric("Device", "cpu")

# ---- Raw examples ---------------------------------------------------------
examples = utils.benchmark_examples_text()
if examples:
    with st.expander("Week 6 request/response examples (raw)"):
        st.code(examples, language="text")

components.footer()
