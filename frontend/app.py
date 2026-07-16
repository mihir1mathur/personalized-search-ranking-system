"""
app.py  --  Personalized Search Ranking System :: Streamlit frontend (home).
============================================================================

This is the SEARCH page (the app's home). It renders a polished search UI and,
on submit, calls the existing Week 6 FastAPI backend over HTTP -- it never
reimplements any retrieval or ranking logic.

Run it with (backend must be running too):

    # terminal 1 -- backend
    HF_HUB_OFFLINE=1 uvicorn api.main:app
    # terminal 2 -- frontend
    streamlit run frontend/app.py

Other pages (Health, API Docs, Benchmark, Pipeline, Settings) live in
frontend/pages/ and are auto-discovered by Streamlit's multipage navigation.
"""

from __future__ import annotations

import os
import sys

# Make sibling modules (utils, components) importable no matter how launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import components
import utils

# Session defaults shared with the Settings page.
DEFAULTS = {"top_k": 10, "alpha": 0.4, "beta": 0.6, "candidate_depth": 50,
            "searching": False}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    components.configure_page("Search Ranking", "🔎")
    init_state()

    components.hero(
        "Personalized Search Ranking System",
        "Search 48K products through a retrieve → re-rank → learn-to-rank "
        "pipeline served by a FastAPI backend.",
    )

    # ---- Sidebar: backend status ----------------------------------------
    with st.sidebar:
        st.markdown("### Backend")
        reachable = utils.backend_reachable()
        st.markdown(components.status_pill(reachable), unsafe_allow_html=True)
        st.caption(f"URL: `{utils.backend_url()}`")
        if not reachable:
            st.caption("Start it with `uvicorn api.main:app`")
        st.markdown("---")
        st.caption("Adjust α/β, candidate depth and defaults on the "
                   "**Settings** page. Explore **Health**, **Benchmark**, "
                   "**Pipeline** and **API Docs** from the nav above.")

    # ---- Search controls -------------------------------------------------
    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Search query",
            placeholder="e.g. wireless noise cancelling headphones",
            help="Type what a shopper would search for.",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            method_label = st.selectbox(
                "Ranking method", utils.METHOD_LABELS,
                index=utils.METHOD_LABELS.index("LTR"),
                help="TF-IDF / BM25 / Embeddings / Hybrid / CrossEncoder / LTR",
            )
        with c2:
            top_k = st.slider("Top-K results", min_value=1, max_value=25,
                              value=int(st.session_state["top_k"]))
        submitted = st.form_submit_button(
            "🔎  Search", use_container_width=True,
            disabled=st.session_state["searching"],
        )

    st.caption(
        f"Method **{method_label}** → {utils.METHODS[method_label]['stage']}. "
        "Hybrid uses α/β; CrossEncoder & LTR use candidate depth (Settings page)."
    )

    # ---- Run the search --------------------------------------------------
    if submitted:
        if not query or not query.strip():
            components.error_box("Please enter a search query.")
            return

        st.session_state["searching"] = True
        progress = st.progress(0, text="Contacting backend…")
        try:
            progress.progress(30, text=f"Running {method_label} ranking…")
            with st.spinner(f"Searching with {method_label}…"):
                response = utils.search(
                    query=query.strip(),
                    method_label=method_label,
                    top_k=int(top_k),
                    alpha=float(st.session_state["alpha"]),
                    beta=float(st.session_state["beta"]),
                    candidate_depth=int(st.session_state["candidate_depth"]),
                )
            progress.progress(100, text="Done")
        except utils.ApiError as exc:
            progress.empty()
            components.error_box(str(exc))
            return
        finally:
            st.session_state["searching"] = False
        progress.empty()

        # ---- Results header + cached badge ------------------------------
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            components.section_title(f"Results for “{response.get('query', query)}”")
        with header_cols[1]:
            if response.get("cached"):
                components.cached_badge()

        components.performance_row(response, method_label)
        st.markdown("")
        components.results_grid(response.get("results", []), method_label)

    components.footer()


# Streamlit executes this file top-to-bottom on every run.
main()
