"""
Settings page  --  tune the parameters sent to the backend.
===========================================================
Lets the user change Top-K, hybrid alpha/beta, and candidate depth. Values are
stored in st.session_state and passed to the backend on the next search:
  * Top-K            -> every method
  * alpha / beta     -> /hybrid-search
  * candidate_depth  -> /rerank and /ltr-search
Nothing here computes anything; it only configures the API request parameters.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import components

components.configure_page("Settings · Search Ranking", "⚙️")
components.hero("Search Settings",
                "These values are sent to the FastAPI backend as request "
                "parameters on your next search.")

# Defaults (mirror the backend's configured defaults).
DEFAULTS = {"top_k": 10, "alpha": 0.4, "beta": 0.6, "candidate_depth": 50}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

components.section_title("Result count")
st.session_state["top_k"] = st.slider(
    "Top-K (results returned)", 1, 25, int(st.session_state["top_k"]),
    help="Applies to every ranking method (sent as top_k).")

components.section_title("Hybrid weights  (→ /hybrid-search)")
st.caption("How much to trust keyword (α, BM25) vs meaning (β, embeddings). "
           "The Week 3/5 best mix is α=0.4, β=0.6.")
c1, c2 = st.columns(2)
st.session_state["alpha"] = c1.slider(
    "α — BM25 (keyword) weight", 0.0, 1.0, float(st.session_state["alpha"]), 0.05)
st.session_state["beta"] = c2.slider(
    "β — embedding (semantic) weight", 0.0, 1.0, float(st.session_state["beta"]), 0.05)
if abs((st.session_state["alpha"] + st.session_state["beta"]) - 1.0) > 1e-6:
    st.info("Tip: α + β is usually 1.0, but the backend normalizes each signal "
            "independently, so any non-negative values are valid.")

components.section_title("Candidate depth  (→ /rerank and /ltr-search)")
st.session_state["candidate_depth"] = st.slider(
    "Candidates re-scored by the cross-encoder / LTR", 10, 100,
    int(st.session_state["candidate_depth"]), 5,
    help="Deeper = potentially better recall into the Top-K, but slower "
         "(more cross-encoder work). Must be ≤ the backend candidate pool (100).")

st.markdown("---")
colA, colB = st.columns([1, 3])
if colA.button("↩️ Reset to defaults", use_container_width=True):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

components.section_title("Current settings (sent to the backend)")
st.json({
    "top_k": st.session_state["top_k"],
    "alpha": st.session_state["alpha"],
    "beta": st.session_state["beta"],
    "candidate_depth": st.session_state["candidate_depth"],
})
st.caption("Go back to the **Search** page to run a query with these settings.")

components.footer()
