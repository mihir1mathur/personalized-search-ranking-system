"""
Health page  --  live backend health & version.
================================================
Calls GET /health and GET /version and renders backend status, version, the
pipeline stages, corpus size, loaded models/components, cache status, and API
uptime. All data is live from the running Week 6 backend.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import components
import utils

components.configure_page("Health · Search Ranking", "🩺")
components.hero("Backend Health", "Live status of the FastAPI service and its "
                                  "loaded pipeline components.")

col_refresh, _ = st.columns([1, 5])
if col_refresh.button("🔄 Refresh"):
    st.rerun()

try:
    health = utils.get_health()
except utils.ApiError as exc:
    components.error_box(str(exc))
    components.footer()
    st.stop()

ready = bool(health.get("ready"))
top = st.columns([1, 1, 1, 1])
top[0].markdown("**Status**")
top[0].markdown(components.status_pill(ready, "OK", "DEGRADED"), unsafe_allow_html=True)
top[1].metric("Version", health.get("version", "-"))
top[2].metric("Environment", health.get("environment", "-"))
top[3].metric("Uptime", f"{health.get('uptime_seconds', 0):.0f} s")

st.metric("Corpus size (unique products)", f"{health.get('corpus_size', 0):,}")

# ---- Version / pipeline ---------------------------------------------------
try:
    version = utils.get_version()
except utils.ApiError:
    version = {}

if version:
    components.section_title("Pipeline stages")
    stages = version.get("pipeline", [])
    st.markdown("  →  ".join(f"`{s}`" for s in stages))

    components.section_title("Models")
    mcols = st.columns(3)
    mcols[0].metric("Embedding model", version.get("embedding_model", "-").split("/")[-1])
    mcols[1].metric("Cross-encoder", os.path.basename(version.get("cross_encoder_model", "-")))
    mcols[2].metric("LTR model", version.get("ltr_model", "-"))
    st.caption("Supported methods: " +
               ", ".join(f"`{m}`" for m in version.get("supported_methods", [])))

# ---- Components / loaded models ------------------------------------------
components.section_title("Loaded components")
comps = health.get("components", [])
if comps:
    for c in comps:
        ok = bool(c.get("ready"))
        mark = "✅" if ok else "❌"
        detail = c.get("detail") or ""
        st.markdown(f"{mark}  **{c.get('name')}** — {detail}")
else:
    st.info("No component detail reported by the backend.")

# ---- Cache status ---------------------------------------------------------
components.section_title("Cache")
cache_comp = next((c for c in comps if c.get("name") == "ce_warm_cache"), None)
if cache_comp:
    st.markdown(f"Warm cross-encoder cache: **{cache_comp.get('detail', 'n/a')}**")
st.caption("The in-memory query cache serves repeated queries in microseconds; "
           "see the Benchmark page for the measured speed-up.")

components.footer()
