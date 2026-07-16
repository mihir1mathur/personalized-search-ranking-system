"""
Pipeline page  --  architecture & user flow.
=============================================
Illustrates how a request flows from the user, through the frontend and the
FastAPI backend, down the ranking pipeline, and back as results. Shows the
generated architecture/user-flow images (frontend/assets) and, if present, the
backend architecture diagram from Week 6 (results/week6_architecture.png).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import streamlit as st

import components

components.configure_page("Pipeline · Search Ranking", "🧭")
components.hero("System Architecture",
                "From a user's keystroke to a ranked list of products.")

# ---- Textual flow (chips) -------------------------------------------------
FLOW = ["User", "Frontend (Streamlit)", "FastAPI", "SearchService",
        "Hybrid Retriever", "CrossEncoder", "LTR", "Results"]
components.section_title("Request flow")
for i, step in enumerate(FLOW):
    st.markdown(f'<div class="flow-chip">{step}</div>', unsafe_allow_html=True)
    if i < len(FLOW) - 1:
        st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)

# ---- Generated images -----------------------------------------------------
components.section_title("User flow diagram")
components.show_asset("user_flow.png", "How a search request travels end to end")

components.section_title("Frontend ↔ backend architecture")
components.show_asset("architecture.png",
                      "Streamlit frontend calls the FastAPI backend over HTTP")

# ---- Backend (Week 6) architecture, if available --------------------------
backend_diagram = Path(__file__).resolve().parents[2] / "results" / "week6_architecture.png"
if backend_diagram.exists():
    components.section_title("Backend pipeline (Week 6)")
    st.image(str(backend_diagram),
             caption="The retrieve → re-rank → learn-to-rank backend pipeline",
             use_container_width=True)

st.caption("The frontend is a thin HTTP client: it never runs any retrieval or "
           "ranking itself. All ranking happens in the backend SearchService, "
           "which reuses the Week 0–5 models.")

components.footer()
