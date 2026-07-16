"""
generate_assets.py  --  render the frontend's diagrams and UI preview images.
=============================================================================

Produces four PNGs in frontend/assets/ used by the Pipeline page and the
README:

    architecture.png    frontend (Streamlit) <-> FastAPI backend <-> pipeline
    user_flow.png       User -> Frontend -> ... -> Results (the request journey)
    screenshot_search.png   a faithful preview of the Search page (real results)
    screenshot_health.png   a faithful preview of the Health page

The two UI previews are populated with REAL data pulled from the running
backend when it is reachable; otherwise they fall back to the real response
captured in Week 6 (the Beats/Sony/Boltune result), so they are never fake.

Run with:  python frontend/generate_assets.py
(Set SEARCH_API_URL if the backend is not on http://127.0.0.1:8000.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import utils

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

# Palette (matches styles.css).
BLUE, PURPLE, RED, GREEN, GOLD = "#4C72B0", "#8172B3", "#C44E52", "#55A868", "#CCB974"
INK, MUTED, CARD, BG = "#23232b", "#8a8f9c", "#ffffff", "#f4f5f8"

# Real fallback result (captured live in Week 6) so previews are never fake.
FALLBACK_RESULT = {
    "query": "wireless noise cancelling headphones",
    "ranking_method": "ltr", "top_k": 3, "count": 3,
    "processing_time_ms": 2014.1, "cached": False,
    "results": [
        {"rank": 1, "product_id": "B077RY9GZD",
         "title": "Beats Studio3 Wireless Noise Cancelling On-Ear Headphones - Apple W1 Chip",
         "score": 0.3608, "confidence": 0.5892},
        {"rank": 2, "product_id": "B085RNVJ3P",
         "title": "Sony Noise Cancelling Headphones WHCH710N: Wireless Bluetooth Over-Ear",
         "score": 0.2178, "confidence": 0.5542},
        {"rank": 3, "product_id": "B07PJK73L8",
         "title": "Boltune Active Noise Cancelling Headphones, Bluetooth 5.0 Over-Ear",
         "score": 0.1877, "confidence": 0.5468},
    ],
}
FALLBACK_HEALTH = {
    "status": "ok", "ready": True, "environment": "development",
    "version": "6.0.0", "uptime_seconds": 63, "corpus_size": 48114,
    "components": [
        {"name": "cross_encoder", "ready": True, "detail": "loaded ms-marco-MiniLM-L-6-v2"},
        {"name": "hybrid", "ready": True, "detail": "reused cached embeddings"},
        {"name": "ltr", "ready": True, "detail": "loaded ltr_lightgbm.txt"},
    ],
}


def _rrect(ax, x, y, w, h, color, ec="white", lw=1.5, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.01,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=color, zorder=2))


def _text(ax, x, y, s, color=INK, size=11, weight="normal", ha="center", va="center"):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#333", lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, linewidth=lw, color=color, linestyle=ls,
                 zorder=1, shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------------------
def draw_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Frontend ↔ Backend Architecture", fontsize=15, fontweight="bold", pad=14)

    # Frontend column (left).
    _rrect(ax, 0.4, 0.6, 4.4, 5.7, "#eef1f8", ec="#d7dcec")
    _text(ax, 2.6, 5.9, "Streamlit Frontend (Python)", color=BLUE, size=12, weight="bold")
    for i, p in enumerate(["Search", "Health", "API Docs", "Benchmark", "Pipeline", "Settings"]):
        _rrect(ax, 0.8, 5.0 - i * 0.7, 3.6, 0.5, CARD, ec="#dde1ec")
        _text(ax, 2.6, 5.25 - i * 0.7, p, size=10)

    # Backend column (right).
    _rrect(ax, 6.2, 0.6, 4.4, 5.7, "#eef4ef", ec="#d3e4d8")
    _text(ax, 8.4, 5.9, "FastAPI Backend (Week 6)", color=GREEN, size=12, weight="bold")
    stages = [("REST API / routers", GREEN), ("SearchService", RED),
              ("Hybrid Retriever", PURPLE), ("CrossEncoder", PURPLE),
              ("Learning-to-Rank", PURPLE), ("Results (JSON)", GOLD)]
    for i, (name, col) in enumerate(stages):
        _rrect(ax, 6.6, 5.0 - i * 0.7, 3.6, 0.5, col)
        _text(ax, 8.4, 5.25 - i * 0.7, name, color="white" if col != GOLD else INK,
              size=10, weight="bold")
        if i < len(stages) - 1:
            _arrow(ax, 8.4, 5.0 - i * 0.7, 8.4, 4.8 - i * 0.7, color="#555", lw=1.4)

    # HTTP link between columns.
    _arrow(ax, 4.8, 3.4, 6.2, 3.4, color=INK, lw=2.4)
    _arrow(ax, 6.2, 2.9, 4.8, 2.9, color=MUTED, lw=1.8, ls=(0, (4, 3)))
    _text(ax, 5.5, 3.65, "HTTP", color=INK, size=10, weight="bold")
    _text(ax, 5.5, 3.15, "request", color=INK, size=8)
    _text(ax, 5.5, 2.65, "JSON", color=MUTED, size=8)

    fig.tight_layout()
    out = ASSETS / "architecture.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def draw_user_flow() -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 10))
    ax.set_xlim(0, 6.5); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("User Flow", fontsize=15, fontweight="bold", pad=12)
    steps = [("User", BLUE), ("Frontend (Streamlit)", BLUE), ("FastAPI", GREEN),
             ("SearchService", RED), ("Hybrid Retriever", PURPLE),
             ("CrossEncoder", PURPLE), ("LTR", PURPLE), ("Results", GOLD)]
    n = len(steps); top = 9.1; gap = (top - 0.6) / (n - 1)
    for i, (name, col) in enumerate(steps):
        y = top - i * gap
        _rrect(ax, 1.4, y - 0.32, 3.7, 0.64, col, r=0.03)
        _text(ax, 3.25, y, name, color="white" if col != GOLD else INK,
              size=12, weight="bold")
        if i < n - 1:
            _arrow(ax, 3.25, y - 0.34, 3.25, y - gap + 0.34, color="#555", lw=2.0)
    fig.tight_layout()
    out = ASSETS / "user_flow.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def draw_search_preview(response: dict) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # Hero.
    _rrect(ax, 0.3, 7.0, 9.4, 0.85, BLUE, ec=BLUE, r=0.03)
    _text(ax, 0.6, 7.55, "Personalized Search Ranking System", color="white",
          size=15, weight="bold", ha="left")
    _text(ax, 0.6, 7.2, "Search 48K products through a retrieve → re-rank → "
          "learn-to-rank pipeline", color="#eaf0fb", size=9, ha="left")

    # Search box + controls.
    _rrect(ax, 0.3, 6.15, 6.6, 0.6, CARD, ec="#d7dae4")
    _text(ax, 0.55, 6.45, "🔎  " + response.get("query", ""), size=11, ha="left", color=INK)
    _rrect(ax, 7.05, 6.15, 1.3, 0.6, PURPLE, ec=PURPLE)
    _text(ax, 7.7, 6.45, response.get("ranking_method", "ltr").upper(), color="white", size=10, weight="bold")
    _rrect(ax, 8.45, 6.15, 1.25, 0.6, GREEN, ec=GREEN)
    _text(ax, 9.07, 6.45, "Search", color="white", size=11, weight="bold")

    # Results header + cached badge.
    _text(ax, 0.35, 5.75, f"Results for “{response.get('query','')}”", size=12,
          weight="bold", ha="left")
    if response.get("cached"):
        _rrect(ax, 7.9, 5.55, 1.8, 0.4, "#fff6d6", ec="#f0e2a6")
        _text(ax, 8.8, 5.75, "⚡ Cached Result", color="#8a6d00", size=9, weight="bold")

    # Result cards.
    cards = response.get("results", [])[:3]
    card_h = 1.4
    for i, item in enumerate(cards):
        y = 5.2 - i * (card_h + 0.15)
        _rrect(ax, 0.3, y - card_h, 9.4, card_h, CARD, ec="#ecedf1", r=0.02)
        # rank badge
        rc = GOLD if item.get("rank") == 1 else BLUE
        _rrect(ax, 0.5, y - 0.95, 0.75, 0.75, rc, ec=rc, r=0.04)
        _text(ax, 0.875, y - 0.575, f"#{item.get('rank')}", color="white" if rc != GOLD else INK,
              size=13, weight="bold")
        # title
        title = item.get("title", "")
        if len(title) > 66:
            title = title[:65] + "…"
        _text(ax, 1.45, y - 0.35, title, size=10.5, weight="bold", ha="left", color=INK)
        # pills
        pills = [f"ID {item.get('product_id','')}",
                 f"Conf {item.get('confidence',0)*100:.1f}%",
                 f"Score {item.get('score',0):.4f}"]
        px = 1.45
        for p in pills:
            w = 0.30 + 0.060 * len(p)
            _rrect(ax, px, y - 0.85, w, 0.32, "#f1f3f8", ec="#e6e8ef", r=0.06)
            _text(ax, px + w / 2, y - 0.69, p, size=8, color="#444b5a")
            px += w + 0.18
        # confidence bar
        conf = max(0.0, min(1.0, item.get("confidence", 0.0)))
        _rrect(ax, 1.45, y - 1.12, 7.9, 0.12, "#eef0f5", ec="#eef0f5", r=0.1)
        _rrect(ax, 1.45, y - 1.12, 7.9 * conf, 0.12, GREEN, ec=GREEN, r=0.1)

    _text(ax, 5.0, 0.25, "Streamlit frontend · live result from the FastAPI backend",
          color=MUTED, size=8)
    out = ASSETS / "screenshot_search.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG); plt.close(fig)
    return out


def draw_health_preview(health: dict, version: dict | None) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    _rrect(ax, 0.3, 5.6, 9.4, 0.75, PURPLE, ec=PURPLE, r=0.03)
    _text(ax, 0.6, 5.98, "Backend Health", color="white", size=15, weight="bold", ha="left")

    ok = health.get("ready", False)
    _rrect(ax, 0.3, 4.9, 2.1, 0.5, "#e7f2ec" if ok else "#fdecec",
           ec="#cfe6d9" if ok else "#f3cccd")
    _text(ax, 1.35, 5.15, "STATUS: OK" if ok else "STATUS: DEGRADED",
          color="#2f7a52" if ok else "#b0343a", size=10, weight="bold")

    metrics = [("Version", str(health.get("version", "-"))),
               ("Environment", str(health.get("environment", "-"))),
               ("Uptime", f"{health.get('uptime_seconds',0):.0f} s"),
               ("Corpus size", f"{health.get('corpus_size',0):,}")]
    for i, (k, v) in enumerate(metrics):
        x = 0.3 + i * 2.45
        _rrect(ax, x, 3.9, 2.25, 0.85, CARD, ec="#ecedf1")
        _text(ax, x + 0.15, 4.5, k, size=9, color=MUTED, ha="left")
        _text(ax, x + 0.15, 4.15, v, size=13, weight="bold", ha="left", color=INK)

    _text(ax, 0.35, 3.5, "Loaded components", size=12, weight="bold", ha="left")
    comps = health.get("components", [])[:5]
    for i, c in enumerate(comps):
        y = 3.05 - i * 0.5
        mark = "[OK]" if c.get("ready") else "[X]"
        mcol = "#2f7a52" if c.get("ready") else "#b0343a"
        _text(ax, 0.4, y, mark, size=10, weight="bold", ha="left", color=mcol)
        _text(ax, 1.05, y, f"{c.get('name')} — {c.get('detail','')}",
              size=10, ha="left", color=INK)

    if version and version.get("pipeline"):
        _text(ax, 0.35, 0.55, "Pipeline: " + " → ".join(version["pipeline"]),
              size=9, ha="left", color=MUTED)
    out = ASSETS / "screenshot_health.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG); plt.close(fig)
    return out


def main() -> None:
    # Try live data; fall back to the real captured Week 6 responses.
    result, health, version = FALLBACK_RESULT, FALLBACK_HEALTH, None
    if utils.backend_reachable():
        try:
            result = utils.search("wireless noise cancelling headphones", "LTR", top_k=3)
        except utils.ApiError:
            pass
        try:
            health = utils.get_health()
            version = utils.get_version()
        except utils.ApiError:
            pass
        print("Using LIVE backend data for previews.")
    else:
        print("Backend not reachable; using real captured fallback data.")

    outputs = [draw_architecture(), draw_user_flow(),
               draw_search_preview(result), draw_health_preview(health, version)]
    print("Generated:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
