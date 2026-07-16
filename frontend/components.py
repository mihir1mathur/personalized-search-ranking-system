"""
components.py  --  reusable Streamlit UI building blocks.
========================================================

Keeps app.py and the pages/ files small and declarative. Everything visual and
repeated lives here: the CSS loader, the page hero, result cards, status pills,
the cached badge, and the friendly error box. Result cards are rendered as our
own HTML (with classes from styles.css) so the look is consistent and polished.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

import utils

_CSS_PATH = Path(__file__).resolve().parent / "styles.css"
_ASSETS = Path(__file__).resolve().parent / "assets"


def configure_page(title: str = "Search Ranking", icon: str = "🔎") -> None:
    """Standard page config + CSS. Call once at the top of every page."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    load_css()


def load_css() -> None:
    try:
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        pass  # styling is a nicety, never a hard failure


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div class="app-hero">
              <h1>{html.escape(title)}</h1>
              <p>{html.escape(subtitle)}</p>
            </div>""",
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def error_box(message: str) -> None:
    """Render a friendly error (never a Python traceback)."""
    st.markdown(
        f'<div class="error-box">⚠️ {html.escape(str(message))}</div>',
        unsafe_allow_html=True,
    )


def cached_badge() -> None:
    st.markdown('<span class="cached-badge">⚡ Cached Result</span>',
                unsafe_allow_html=True)


def status_pill(ok: bool, label_ok: str = "ONLINE", label_bad: str = "OFFLINE") -> str:
    cls = "status-ok" if ok else "status-bad"
    label = label_ok if ok else label_bad
    return f'<span class="status-pill {cls}">{html.escape(label)}</span>'


def result_card(item: Dict[str, Any], method_label: str) -> None:
    """Render one ranked product as a polished card + an expandable detail."""
    rank = int(item.get("rank", 0))
    title = str(item.get("title") or "(untitled product)")
    pid = str(item.get("product_id", ""))
    score = float(item.get("score", 0.0))
    conf = float(item.get("confidence", 0.0))
    conf_pct = max(0.0, min(100.0, conf * 100.0))

    top_cls = " top" if rank == 1 else ""
    st.markdown(
        f"""<div class="result-card">
              <div class="rank-badge{top_cls}">#{rank}</div>
              <div class="card-body">
                <div class="card-title">{html.escape(title)}</div>
                <div class="card-meta">
                  <span class="pill method">{html.escape(method_label)}</span>
                  <span class="pill pid">ID: {html.escape(pid)}</span>
                  <span class="pill conf">Confidence {conf_pct:.1f}%</span>
                  <span class="pill score">Raw score {score:.4f}</span>
                </div>
                <div class="conf-bar"><div class="conf-fill" style="width:{conf_pct:.1f}%"></div></div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Details"):
        st.json({
            "rank": rank,
            "product_id": pid,
            "title": title,
            "score": score,
            "confidence": conf,
            "ranking_method": method_label,
        })


def results_grid(results: List[Dict[str, Any]], method_label: str) -> None:
    if not results:
        st.info("No results returned for this query.")
        return
    for item in results:
        result_card(item, method_label)


def performance_row(response: Dict[str, Any], method_label: str) -> None:
    """Show latency / processing time / method / model in a metric row."""
    cols = st.columns(4)
    cols[0].metric("Latency (processing)", utils.fmt_ms(response.get("processing_time_ms", 0)))
    cols[1].metric("Results returned", response.get("count", 0))
    cols[2].metric("Ranking method", method_label)
    model = utils.METHODS.get(method_label, {}).get("stage", "-")
    cols[3].metric("Pipeline stage", model)


def asset_path(name: str) -> Path:
    return _ASSETS / name


def show_asset(name: str, caption: str = "") -> None:
    path = asset_path(name)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image '{name}' not found. Generate it with: "
                "python frontend/generate_assets.py")


def footer() -> None:
    st.markdown(
        '<div class="app-footer">Personalized Search Ranking System · '
        'Week 7 frontend (Streamlit) · talks to the Week 6 FastAPI backend</div>',
        unsafe_allow_html=True,
    )
