"""
Health page  --  live backend health & version.
================================================
Calls GET /health and GET /version and renders backend status, version, the
pipeline stages, corpus size, loaded models/components, cache status, and API
uptime. All data is live from the running backend.
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

# ---- Prominent overall status --------------------------------------------
components.status_banner(
    ready,
    ok_text="All systems operational — backend ready",
    bad_text="Service degraded — one or more required components are not ready",
)

# ---- Key facts as consistent metric cards --------------------------------
components.metric_cards([
    ("Version", health.get("version", "-")),
    ("Environment", health.get("environment", "-")),
    ("Uptime", f"{health.get('uptime_seconds', 0):.0f} s"),
    ("Corpus (unique products)", f"{health.get('corpus_size', 0):,}"),
])

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
    components.metric_cards([
        ("Embedding model", version.get("embedding_model", "-").split("/")[-1]),
        ("Cross-encoder", os.path.basename(version.get("cross_encoder_model", "-"))),
        ("LTR model", version.get("ltr_model", "-")),
    ])
    st.caption("Supported methods: " +
               ", ".join(f"`{m}`" for m in version.get("supported_methods", [])))

# ---- Components / loaded models ------------------------------------------
# Components that the app runs fine without. When one of these is not loaded we
# show it as OPTIONAL rather than failed, so the health view isn't misread as
# broken. This is display-only; the backend health semantics are unchanged.
OPTIONAL_COMPONENTS = {"ce_warm_cache"}

components.section_title("Loaded components")
comps = health.get("components", [])
if comps:
    for c in comps:
        name = str(c.get("name", "component"))
        ok = bool(c.get("ready"))
        detail = c.get("detail") or ""
        if ok:
            mark, tag = "✅", "Ready"
        elif name in OPTIONAL_COMPONENTS:
            mark, tag = "◽", "Optional — not loaded"
        else:
            mark, tag = "❌", "Not ready"
        suffix = f" · {detail}" if detail else ""
        st.markdown(f"{mark}  **{name}** — {tag}{suffix}")
else:
    st.info("No component detail reported by the backend.")

# ---- Cache status ---------------------------------------------------------
components.section_title("Cache")
cache_comp = next((c for c in comps if c.get("name") == "ce_warm_cache"), None)
if cache_comp:
    loaded = bool(cache_comp.get("ready"))
    state = cache_comp.get("detail") or ("loaded" if loaded else "not loaded")
    label = "Warm cross-encoder cache (optional)"
    st.markdown(f"{label}: **{state}**")
st.caption("The in-memory query cache serves repeated queries in microseconds; "
           "see the Benchmark page for the measured speed-up.")

components.footer()
