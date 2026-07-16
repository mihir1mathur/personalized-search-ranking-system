"""
API Docs page  --  links to the backend's auto-generated documentation.
=======================================================================
FastAPI serves Swagger UI (/docs), ReDoc (/redoc), and the raw OpenAPI schema
(/openapi.json). This page links to all three and lists the endpoints.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import components
import utils

components.configure_page("API Docs · Search Ranking", "📚")
components.hero("API Documentation",
                "Interactive, auto-generated docs from the FastAPI backend.")

base = utils.backend_url()
reachable = utils.backend_reachable()
st.markdown(components.status_pill(reachable), unsafe_allow_html=True)
st.caption(f"Backend base URL: `{base}`")

components.section_title("Documentation endpoints")
c1, c2, c3 = st.columns(3)
c1.link_button("🧩 Swagger UI", f"{base}/docs", use_container_width=True)
c2.link_button("📖 ReDoc", f"{base}/redoc", use_container_width=True)
c3.link_button("🗂️ OpenAPI JSON", f"{base}/openapi.json", use_container_width=True)

st.caption("Swagger UI lets you try each endpoint in the browser. These links "
           "open the backend directly; make sure it is running.")

components.section_title("Endpoints")
rows = [
    ("POST", "/search", "Generic search; choose method: tfidf|bm25|embedding|hybrid|rerank|ltr"),
    ("POST", "/hybrid-search", "Hybrid BM25 + embedding fusion (alpha/beta)"),
    ("POST", "/rerank", "Hybrid retrieval → cross-encoder re-ranking (candidate_depth)"),
    ("POST", "/ltr-search", "Full pipeline → Learning-to-Rank (candidate_depth)"),
    ("GET", "/health", "Liveness/readiness + loaded components"),
    ("GET", "/version", "Build metadata: version, pipeline, models"),
]
st.table({"Method": [r[0] for r in rows],
          "Path": [r[1] for r in rows],
          "Description": [r[2] for r in rows]})

# Show the live OpenAPI summary if reachable.
if reachable:
    try:
        schema = utils._request("GET", "/openapi.json", timeout=5.0)
        components.section_title("Live OpenAPI summary")
        st.write(f"**{schema.get('info', {}).get('title', 'API')}** "
                 f"v{schema.get('info', {}).get('version', '?')}")
        st.caption("Paths: " + ", ".join(f"`{p}`" for p in sorted(schema.get("paths", {}))))
    except utils.ApiError:
        pass

components.footer()
