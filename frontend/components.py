"""
components.py  --  reusable Streamlit UI building blocks.
========================================================

Keeps app.py and the pages/ files small and declarative. Everything visual and
repeated lives here: the CSS loader, the page hero, section headings, metric
cards, result cards, status pills/banners, the sidebar backend badge, empty and
error states, and the footer. Rendering our own small HTML blocks (with classes
from styles.css) keeps the look consistent and stable across Streamlit versions.

No business logic lives here -- these helpers only present data the caller
already has; they never call the backend or compute rankings.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import streamlit as st

import utils

_CSS_PATH = Path(__file__).resolve().parent / "styles.css"
_ASSETS = Path(__file__).resolve().parent / "assets"


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Headings
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Metric cards  (consistent, non-truncating summary tiles)
# --------------------------------------------------------------------------
def metric_cards(items: Sequence[Tuple[str, Any]]) -> None:
    """Render a responsive grid of label/value tiles.

    Values are shown in full (they wrap instead of truncating), so long labels
    like "Full learned pipeline" stay readable. Callers pass already-formatted
    values -- this helper never rounds, recomputes, or alters them.
    """
    cells = "".join(
        f'<div class="metric-card">'
        f'<div class="m-label">{html.escape(str(label))}</div>'
        f'<div class="m-value">{html.escape(str(value))}</div>'
        f'</div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-grid">{cells}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Status: pills, banners, sidebar backend card
# --------------------------------------------------------------------------
def status_pill(ok: bool, label_ok: str = "ONLINE", label_bad: str = "OFFLINE") -> str:
    """Return an inline status pill. Uses a dot + text (never colour alone)."""
    cls = "status-ok" if ok else "status-bad"
    label = label_ok if ok else label_bad
    return (f'<span class="status-pill {cls}"><span class="dot"></span>'
            f'{html.escape(label)}</span>')


def status_banner(ok: bool,
                  ok_text: str = "All systems operational",
                  bad_text: str = "Service degraded") -> None:
    """A prominent operational banner (icon + text, not colour alone)."""
    cls = "banner-ok" if ok else "banner-bad"
    icon = "✅" if ok else "⚠️"
    text = ok_text if ok else bad_text
    st.markdown(
        f'<div class="status-banner {cls}"><span class="sb-ico">{icon}</span>'
        f'<span>{html.escape(text)}</span></div>',
        unsafe_allow_html=True,
    )


def sidebar_backend_status(reachable: bool) -> None:
    """Compact, professional backend badge for the sidebar.

    Deliberately shows NO internal URL, IP, port, or process command -- only a
    human-readable connection state suitable for a production UI.
    """
    ok = bool(reachable)
    dot = "#2f7a52" if ok else "#b0343a"
    label = "ONLINE" if ok else "OFFLINE"
    sub = "Connected through Nginx" if ok else "Service currently unreachable"
    st.markdown(
        f"""<div class="sb-status">
              <div class="sb-head">Backend</div>
              <div class="sb-badge"><span class="sb-dot" style="background:{dot}"></span>{label}</div>
              <div class="sb-sub">{html.escape(sub)}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def cached_badge() -> None:
    st.markdown('<span class="cached-badge">⚡ Cached Result</span>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Empty / error states
# --------------------------------------------------------------------------
def empty_state(title: str, subtitle: str, icon: str = "🔍") -> None:
    st.markdown(
        f"""<div class="empty-state">
              <div class="es-icon">{icon}</div>
              <div class="es-title">{html.escape(title)}</div>
              <div class="es-sub">{html.escape(subtitle)}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def error_box(message: str) -> None:
    """Render a friendly error (never a Python traceback)."""
    st.markdown(
        f'<div class="error-box">⚠️ {html.escape(str(message))}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Result cards
# --------------------------------------------------------------------------
# Shown when a search succeeds but returns no products. Neutral wording -- an
# empty result set is a valid answer, not an error.
NO_RESULTS_MESSAGE = (
    "No matching products were found. Try a broader query or another "
    "ranking method."
)

# Confidence is a method-specific, within-response transform of the raw score
# (see the API's SearchResultItem contract) -- NOT a calibrated probability and
# NOT comparable across ranking methods. This note is surfaced as a tooltip on
# the confidence bar so the visual indicator can't be misread.
_CONF_BAR_HINT = ("Within-result confidence indicator (0–100%). Derived from the "
                  "raw score; not a calibrated probability and not comparable "
                  "across ranking methods.")

# Hover-tooltip text for the "What is Confidence?" helper next to the label.
_CONF_HELP = ("Confidence reflects the ranking model's relative confidence among "
              "returned results and should not be interpreted as an absolute "
              "probability of relevance.")

# Caveat shown inside the "Why was this ranked?" section (plain-text version).
_CONF_NOTE = ("This confidence indicates the model's relative confidence within "
              "this ranked response. It is NOT a calibrated probability.")

# The exact set of fields the search API returns per result item. The details
# expander shows only these (plus the method used) -- no invented component
# scores, and no placeholders for fields that aren't present.
_DETAIL_FIELDS = ("rank", "product_id", "title", "score", "confidence")

# The ACTUAL pipeline stages each ranking method runs, keyed by the internal
# method key from utils.METHODS. Used to explain a result truthfully -- a
# single-stage retriever is never described as having been cross-encoded or
# LTR-scored. These describe the served path (see api/services/search_service),
# not invented data.
_STAGE_BULLETS = {
    "tfidf":     ("Retrieved during candidate generation (sparse TF-IDF retrieval)",),
    "bm25":      ("Retrieved during candidate generation (sparse BM25 retrieval)",),
    "embedding": ("Retrieved during candidate generation (dense embedding retrieval)",),
    "hybrid":    ("Retrieved during candidate generation (sparse + dense)",
                  "Merged with hybrid fusion (BM25 + embeddings)"),
    "rerank":    ("Retrieved during candidate generation (hybrid)",
                  "Re-ranked using Cross Encoder"),
    "ltr":       ("Retrieved during candidate generation (hybrid)",
                  "Re-ranked using Cross Encoder",
                  "Final ranking produced by Learning-to-Rank"),
}

# Optional per-result component scores. The current API does NOT return these,
# so nothing is shown; this table only defines how they WOULD be labelled if a
# future response included them. Absent keys are simply skipped -- never faked.
_COMPONENT_SCORE_FIELDS = (
    ("bm25_score", "BM25 score"),
    ("sparse_score", "Sparse score"),
    ("dense_score", "Dense score"),
    ("cross_encoder_score", "Cross Encoder score"),
    ("ltr_score", "LTR score"),
)


def _num(value: Any) -> bool:
    """True only for a real numeric value (bool is intentionally excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _method_stage_bullets(method_label: str) -> List[str]:
    """Truthful list of stages the chosen method actually ran for this result."""
    key = utils.METHODS.get(method_label, {}).get("key")
    bullets = _STAGE_BULLETS.get(key)
    if bullets:
        return list(bullets)
    return [f"Ranked using the {method_label} method"]


def _component_score_lines(item: Dict[str, Any]) -> List[str]:
    """Return 'Label: value' lines for component scores that are ACTUALLY present.

    Returns [] for the current API (which omits these) -- never invents a value.
    """
    lines: List[str] = []
    for field, label in _COMPONENT_SCORE_FIELDS:
        value = item.get(field)
        if _num(value):
            lines.append(f"{label}: {float(value):.4f}")
    return lines


def _why_ranked_html(item: Dict[str, Any], method_label: str,
                     conf_pct: Any) -> str:
    """Collapsible 'Why was this ranked?' explanation built only from real data."""
    bullets = [f'<li>✔ {html.escape(b)}</li>'
               for b in _method_stage_bullets(method_label)]
    if conf_pct is not None:
        bullets.append(f'<li>✔ Confidence: {conf_pct:.1f}%</li>')

    comp_lines = _component_score_lines(item)
    comp_html = ""
    if comp_lines:
        comp_html = ('<div class="why-scores"><div class="why-sub">Component scores</div>'
                     + "".join(f'<div class="why-score">{html.escape(l)}</div>'
                               for l in comp_lines)
                     + '</div>')

    note_html = (f'<p class="why-note">{html.escape(_CONF_NOTE)}</p>'
                 if conf_pct is not None else "")

    return (
        '<details class="why-details"><summary>Why was this ranked?</summary>'
        '<div class="why-body"><ul class="why-list">'
        f'{"".join(bullets)}</ul>{comp_html}{note_html}</div></details>'
    )


def _result_card_html(item: Dict[str, Any], method_label: str) -> str:
    """Build the result-card HTML for one product. Pure (no Streamlit calls).

    Displays ONLY fields already present in the item: rank, product_id, title,
    and -- when actually returned -- raw score and confidence, plus a collapsible
    "Why was this ranked?" explanation derived from the method's real pipeline
    path. Absent optional fields are simply omitted; nothing is fabricated or
    defaulted to a fake 0. Values keep their existing formatting, unchanged.
    """
    rank = int(item.get("rank", 0))
    title = str(item.get("title") or "(untitled product)")
    pid = str(item.get("product_id", ""))

    # Restrained premium tiers for the top 3; #4+ share one consistent style.
    # The numeric rank is ALWAYS shown, so rank never depends on colour alone.
    rank_cls = f" r{rank}" if rank in (1, 2, 3) else ""

    badges = [
        f'<span class="pill method"><span class="pill-k">Method</span>'
        f'{html.escape(method_label)}</span>',
        f'<span class="pill pid"><span class="pill-k">ID</span>'
        f'{html.escape(pid)}</span>',
    ]
    score = item.get("score")
    if _num(score):
        badges.append(
            f'<span class="pill score"><span class="pill-k">Raw score</span>'
            f'{float(score):.4f}</span>')

    conf = item.get("confidence")
    conf_pct = max(0.0, min(100.0, float(conf) * 100.0)) if _num(conf) else None

    conf_block = ""
    if conf_pct is not None:
        conf_block = (
            '<div class="conf-head"><span class="conf-lab">Confidence '
            f'<span class="conf-help" title="{html.escape(_CONF_HELP)}">'
            'What is Confidence?</span></span>'
            f'<span class="conf-val">{conf_pct:.1f}%</span></div>'
            f'<div class="conf-bar" title="{html.escape(_CONF_BAR_HINT)}" '
            f'aria-label="Confidence {conf_pct:.1f} percent within these results">'
            f'<div class="conf-fill" style="width:{conf_pct:.1f}%"></div></div>'
        )

    why_block = _why_ranked_html(item, method_label, conf_pct)

    return (
        f'<div class="result-card">'
        f'<div class="rank-badge{rank_cls}" title="Rank {rank}">#{rank}</div>'
        f'<div class="card-body">'
        f'<div class="card-title" title="{html.escape(title)}">{html.escape(title)}</div>'
        f'<div class="card-meta">{"".join(badges)}</div>'
        f'{conf_block}'
        f'{why_block}'
        f'</div>'
        f'</div>'
    )


def result_card(item: Dict[str, Any], method_label: str) -> None:
    """Render one ranked product as a polished card + a details expander."""
    st.markdown(_result_card_html(item, method_label), unsafe_allow_html=True)
    # Only echo fields that are genuinely present in the response, plus the
    # method that produced this ranking. No component scores are invented.
    details: Dict[str, Any] = {k: item[k] for k in _DETAIL_FIELDS if k in item}
    details["ranking_method"] = method_label
    with st.expander("View ranking details"):
        st.json(details)


def results_grid(results: List[Dict[str, Any]], method_label: str) -> None:
    """Render results in the exact order received. Zero results is not an error."""
    if not results:
        empty_state("No matching products found", NO_RESULTS_MESSAGE, icon="🔎")
        return
    for item in results:
        result_card(item, method_label)


def performance_row(response: Dict[str, Any], method_label: str) -> None:
    """Show latency / result count / method / pipeline stage as metric cards.

    Values are passed through unchanged: latency uses the same fmt_ms helper and
    the count comes straight from the backend response -- no recomputation.
    """
    stage = utils.METHODS.get(method_label, {}).get("stage", "-")
    metric_cards([
        ("Processing latency", utils.fmt_ms(response.get("processing_time_ms", 0))),
        ("Results returned", response.get("count", 0)),
        ("Ranking method", method_label),
        ("Pipeline stage", stage),
    ])


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------
def asset_path(name: str) -> Path:
    return _ASSETS / name


def show_asset(name: str, caption: str = "") -> None:
    path = asset_path(name)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image '{name}' not found. Generate it with: "
                "python frontend/generate_assets.py")


# --------------------------------------------------------------------------
# Pipeline flow visual  (shared by the Search overview + the Pipeline page)
# --------------------------------------------------------------------------
# The full served ranking pipeline, top to bottom. This is an architecture
# overview of the whole system; an individual result only traverses the stages
# its chosen method runs (shown truthfully in each card's "Why was this ranked?").
_PIPELINE_STAGES = (
    ("🔎", "Query"),
    ("📚", "Sparse Retrieval"),
    ("🧠", "Dense Retrieval"),
    ("🔀", "Hybrid Merge"),
    ("🎯", "Cross Encoder"),
    ("🏆", "Learning-to-Rank"),
    ("📋", "Final Results"),
)


def pipeline_flow() -> None:
    """Render the polished vertical pipeline visual (icons + labels + arrows)."""
    parts: List[str] = []
    for i, (icon, label) in enumerate(_PIPELINE_STAGES):
        parts.append(
            f'<div class="pf-step"><span class="pf-icon">{icon}</span>'
            f'<span class="pf-label">{html.escape(label)}</span></div>'
        )
        if i < len(_PIPELINE_STAGES) - 1:
            parts.append('<div class="pf-arrow" aria-hidden="true">↓</div>')
    st.markdown(f'<div class="pipeline-flow">{"".join(parts)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
def footer() -> None:
    st.markdown(
        '<div class="app-footer">Personalized Search Ranking System · '
        'Streamlit interface · FastAPI search service</div>',
        unsafe_allow_html=True,
    )
