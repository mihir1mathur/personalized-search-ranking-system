"""
architecture_diagram.py  --  render the Week 6 backend architecture diagram.
============================================================================

Draws a clean, presentation-quality diagram of the request flow through the
FastAPI backend and the reused Week 0-5 pipeline, and saves it to
``results/week6_architecture.png``.

    Client -> FastAPI -> SearchService -> Hybrid -> CrossEncoder -> LTR -> Results

with a side panel showing the reused artifacts (Embeddings, FAISS, Caches,
Models). Run with::

    python -m api.architecture_diagram
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from api.config.settings import get_settings

# Colour palette (colour-blind friendly, prints well in grayscale too).
C_CLIENT = "#4C72B0"
C_API = "#55A868"
C_SERVICE = "#C44E52"
C_STAGE = "#8172B3"
C_RESULT = "#CCB974"
C_ARTIFACT = "#4C4C4C"
C_TEXT = "#FFFFFF"


def _box(ax, x, y, w, h, label, color, sub="", fontsize=11, text_color=C_TEXT):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.5, edgecolor="white", facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.09 if sub else 0), label,
            ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.12, sub, ha="center", va="center",
                color=text_color, fontsize=fontsize - 3, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#333333", style="-|>", lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
        shrinkA=2, shrinkB=2))


def build() -> Path:
    settings = get_settings()
    out = settings.project_root / "results" / "week6_architecture.png"

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_title("Week 6 -- Production Search Backend Architecture\n"
                 "Client -> FastAPI -> Search Service -> Ranking Pipeline -> JSON",
                 fontsize=14, fontweight="bold", pad=16)

    cx, w = 3.4, 5.2  # main column x and box width

    # Main vertical flow.
    _box(ax, cx, 11.6, w, 0.9, "CLIENT", C_CLIENT,
         "browser / mobile / service (HTTP + JSON)")
    _box(ax, cx, 9.9, w, 1.2, "FastAPI  (REST API)", C_API,
         "routers - validation - logging - error handling - /docs Swagger")
    _box(ax, cx, 8.5, w, 0.95, "Search Service", C_SERVICE,
         "orchestration - query cache - confidence")

    _box(ax, cx, 6.9, w, 0.9, "Hybrid Retrieval", C_STAGE,
         "BM25 + Embeddings (min-max fusion)")
    _box(ax, cx, 5.4, w, 0.9, "Cross-Encoder", C_STAGE,
         "re-rank Top-50 candidates")
    _box(ax, cx, 3.9, w, 0.9, "Learning-to-Rank", C_STAGE,
         "LambdaMART / LightGBM (final order)")
    _box(ax, cx, 2.4, w, 0.9, "Results (Top-K JSON)", C_RESULT,
         "rank - product_id - title - score - confidence",
         text_color="#333333")

    # Vertical arrows (down the main flow).
    xmid = cx + w / 2
    ys = [(11.6, 11.1), (9.9, 9.45), (8.5, 7.8), (6.9, 6.3),
          (5.4, 4.8), (3.9, 3.3)]
    for y1, y2 in ys:
        _arrow(ax, xmid, y1, xmid, y2)
    # request/response label on the client<->api link
    ax.text(xmid + 0.15, 11.35, "request", ha="left", va="center", fontsize=8,
            color="#333333")

    # Side panel: reused Week 0-5 artifacts feeding the stages.
    ax_x = 9.2
    _box(ax, ax_x, 6.75, 2.4, 0.7, "Embeddings", C_ARTIFACT,
         "cached .npy (48k)", fontsize=9)
    _box(ax, ax_x, 5.9, 2.4, 0.7, "FAISS Index", C_ARTIFACT,
         "IndexFlatIP", fontsize=9)
    _box(ax, ax_x, 5.05, 2.4, 0.7, "Models", C_ARTIFACT,
         "MiniLM CE + LTR", fontsize=9)
    _box(ax, ax_x, 4.2, 2.4, 0.7, "Caches", C_ARTIFACT,
         "query LRU + CE cache", fontsize=9)

    # Dashed links from artifacts into the stages that use them (reuse).
    _arrow(ax, ax_x, 7.1, cx + w, 7.35, color="#777777", style="-|>",
           lw=1.4, ls=(0, (4, 3)))
    _arrow(ax, ax_x, 6.25, cx + w, 7.15, color="#777777", style="-|>",
           lw=1.4, ls=(0, (4, 3)))
    _arrow(ax, ax_x, 5.4, cx + w, 5.85, color="#777777", style="-|>",
           lw=1.4, ls=(0, (4, 3)))
    _arrow(ax, ax_x, 4.55, cx + w, 4.35, color="#777777", style="-|>",
           lw=1.4, ls=(0, (4, 3)))

    ax.text(ax_x + 1.2, 7.75, "Reused Week 0-5 artifacts\n(no retraining)",
            ha="center", va="bottom", fontsize=9, style="italic", color="#333333")

    # Config / logging rail on the left, spanning the app.
    _box(ax, 0.25, 3.9, 2.5, 5.0, "", "#EDEDED", text_color="#333333")
    ax.text(1.5, 8.6, "Cross-cutting", ha="center", fontsize=10,
            fontweight="bold", color="#333333")
    for i, item in enumerate([
        "Configuration", "Logging (rotating)", "Validation", "Error handling",
        "Middleware", "Dependency\ninjection", "Unit tests"]):
        ax.text(1.5, 8.0 - i * 0.62, item, ha="center", va="center",
                fontsize=8.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved architecture diagram to: {out}")
    return out


if __name__ == "__main__":
    build()
