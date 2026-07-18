"""
Pipeline page  --  architecture & user flow.
=============================================
Illustrates how a request flows from the user, through the frontend and the
FastAPI backend, down the ranking pipeline, and back as results. Shows the
generated architecture/user-flow images (frontend/assets) and, if present, the
backend architecture diagram (results/week6_architecture.png).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import streamlit as st

import components
import utils

components.configure_page("Pipeline · Search Ranking", "🧭")
components.hero("System Architecture",
                "From a user's keystroke to a ranked list of products.")

# ---- Full ranking pipeline (shared visual) --------------------------------
# The complete funnel. Not every ranking method runs every stage -- the table
# below shows exactly which stages each method uses.
components.section_title("Ranking pipeline")
components.pipeline_flow()

# ---- Which stages each method actually runs -------------------------------
# Derived from the same method→endpoint map the Search page uses, so it stays
# accurate: single-stage retrievers stop early; only rerank/LTR run later stages.
components.section_title("What each ranking method runs")
_PATHS = {
    "TF-IDF": "Sparse retrieval only (lexical)",
    "BM25": "Sparse retrieval only (lexical)",
    "Embeddings": "Dense retrieval only (semantic)",
    "Hybrid": "Sparse + dense fusion → hybrid candidates",
    "CrossEncoder": "Hybrid candidates → cross-encoder reranking",
    "LTR": "Hybrid candidates → cross-encoder → learning-to-rank",
}
st.table({
    "Method": list(_PATHS.keys()),
    "Pipeline stages used": list(_PATHS.values()),
    "Endpoint": [utils.METHODS[m]["endpoint"] for m in _PATHS],
})
st.caption("Every method shares the same retrieval foundation; re-ranking and "
           "learning-to-rank stages run only for the methods listed above.")

# ---- Generated images -----------------------------------------------------
components.section_title("User flow diagram")
components.show_asset("user_flow.png", "How a search request travels end to end")

components.section_title("Frontend ↔ backend architecture")
components.show_asset("architecture.png",
                      "Streamlit frontend calls the FastAPI backend over HTTP")

# ---- Backend architecture diagram, if available ---------------------------
backend_diagram = Path(__file__).resolve().parents[2] / "results" / "week6_architecture.png"
if backend_diagram.exists():
    components.section_title("Backend pipeline")
    st.image(str(backend_diagram),
             caption="The retrieve → re-rank → learn-to-rank backend pipeline",
             use_container_width=True)

st.caption("The frontend is a thin HTTP client: it never runs any retrieval or "
           "ranking itself. All ranking happens in the backend SearchService, "
           "which reuses the trained retrieval and ranking models.")

components.footer()
