"""
generate_architecture.py  --  render the production deployment architecture PNG.
================================================================================

Produces ``deployment/aws_deployment_architecture.png`` -- a clean, README-quality
diagram of the ACTUAL AWS EC2 deployment:

    Internet -> Elastic IP -> Nginx (:80, public) -> { FastAPI :8000, Streamlit
    :8501 } (both loopback, systemd-managed) -> SearchService -> Hybrid Retrieval
    (TF-IDF + Embeddings) -> Cross Encoder -> LightGBM LambdaMART -> 48K ESCI
    Product Dataset.

It draws only the deployed topology; it does not touch any model, dataset,
metric, or ranking code. Regenerate with:

    python deployment/generate_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "aws_deployment_architecture.png"

# Palette (kept consistent with the app's other diagrams).
INK, MUTED, LINE = "#23232b", "#6b7180", "#d7dae4"
AWS = "#FF9900"
NGINX = "#009639"
FASTAPI = "#009688"
STREAMLIT = "#E23B3B"
BLUE, VIOLET, GOLD = "#4C72B0", "#8172B3", "#CCB974"
LGBM = "#1F6FB2"
CLOUD = "#eef1f8"
EC2_FILL = "#f7f9fc"
NOTE_FILL = "#f4f6fb"


def _box(ax, cx, cy, w, h, face, text, *, sub=None, tcolor="#ffffff",
         ec="none", fs=11, sub_color=None, radius=0.035, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=face, zorder=3))
    if sub:
        ax.text(cx, cy + h * 0.16, text, ha="center", va="center",
                color=tcolor, fontsize=fs, fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.26, sub, ha="center", va="center",
                color=sub_color or tcolor, fontsize=fs - 3.0, zorder=4)
    else:
        ax.text(cx, cy, text, ha="center", va="center", color=tcolor,
                fontsize=fs, fontweight="bold", zorder=4)


def _arrow(ax, x1, y1, x2, y2, color="#4a4f5c", lw=2.0, ls="-", scale=15):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=scale,
        linewidth=lw, color=color, linestyle=ls, zorder=2,
        shrinkA=1, shrinkB=1))


def main() -> Path:
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(5, 13.62, "Personalized Search Ranking System",
            ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(5, 13.24, "AWS EC2 Production Deployment",
            ha="center", va="center", fontsize=11.5, color=MUTED)

    # ---- Public tier (outside the instance) -------------------------------
    _box(ax, 5, 12.5, 2.6, 0.66, CLOUD, "Internet", tcolor=INK, ec=LINE, fs=11)
    _box(ax, 5, 11.5, 3.0, 0.72, AWS, "Elastic IP", sub="stable public address",
         tcolor="#ffffff", sub_color="#fff3e0", fs=11.5)

    # ---- EC2 instance boundary -------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.45, 0.55), 9.1, 9.95,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        linewidth=1.6, edgecolor="#c3ccdb", facecolor=EC2_FILL,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(0.85, 10.18, "AWS EC2  ·  Ubuntu  ·  single instance (CPU-only)",
            ha="left", va="center", fontsize=10, fontweight="bold", color="#5a6478")

    # ---- Nginx (the only public surface) ----------------------------------
    _box(ax, 5, 9.45, 5.4, 0.86, NGINX,
         "Nginx Reverse Proxy  ·  Port 80",
         sub="PUBLIC  ·  0.0.0.0  ·  routes /  ,  /api/  ,  /healthz",
         tcolor="#ffffff", sub_color="#e5f6ea", fs=12)

    # ---- FastAPI + Streamlit (loopback, systemd) --------------------------
    _box(ax, 2.85, 7.95, 3.15, 0.94, FASTAPI, "FastAPI  ·  Port 8000",
         sub="127.0.0.1  ·  search-api.service", tcolor="#ffffff",
         sub_color="#e2f3f1", fs=11.5)
    _box(ax, 7.15, 7.95, 3.15, 0.94, STREAMLIT, "Streamlit  ·  Port 8501",
         sub="127.0.0.1  ·  search-frontend.service", tcolor="#ffffff",
         sub_color="#fde4e4", fs=11.5)

    # ---- Ranking pipeline (under FastAPI) ---------------------------------
    _box(ax, 2.85, 6.55, 3.15, 0.74, BLUE, "SearchService", tcolor="#ffffff", fs=11)
    _box(ax, 2.85, 5.35, 3.5, 0.86, VIOLET, "Hybrid Retrieval",
         sub="TF-IDF  +  Embeddings", tcolor="#ffffff", sub_color="#efe9f7", fs=11)
    _box(ax, 2.85, 4.15, 3.15, 0.74, "#6C63A6", "Cross Encoder", tcolor="#ffffff", fs=11)
    _box(ax, 2.85, 2.95, 3.35, 0.78, LGBM, "LightGBM LambdaMART",
         tcolor="#ffffff", fs=10.8)
    _box(ax, 2.85, 1.7, 3.5, 0.82, GOLD, "48K ESCI Product Dataset",
         tcolor=INK, fs=10.8)

    # ---- Notes panel (right column) ---------------------------------------
    ax.add_patch(FancyBboxPatch(
        (5.55, 1.35), 3.9, 5.0, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, edgecolor="#dfe3ee", facecolor=NOTE_FILL, zorder=2))
    ax.text(5.78, 6.02, "Deployment notes", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color="#3a4150", zorder=4)
    notes = [
        "Only Nginx (:80) is publicly exposed.",
        "FastAPI (:8000) & Streamlit (:8501)\nstay bound to 127.0.0.1.",
        "systemd manages & auto-starts\nboth services on boot.",
        "Elastic IP routes public traffic\nto the instance.",
        "Health endpoint: GET /health\n(public via Nginx /healthz).",
    ]
    y = 5.5
    for note in notes:
        ax.text(5.78, y, "▸", ha="left", va="top", fontsize=9,
                color=NGINX, zorder=4)
        ax.text(6.02, y, note, ha="left", va="top", fontsize=8.7,
                color="#3d4351", zorder=4, linespacing=1.25)
        y -= 0.6 + 0.28 * note.count("\n")

    # ---- Arrows -----------------------------------------------------------
    _arrow(ax, 5, 12.17, 5, 11.9)                      # Internet -> Elastic IP
    _arrow(ax, 5, 11.14, 5, 9.9)                       # Elastic IP -> Nginx
    _arrow(ax, 4.1, 9.02, 3.1, 8.45)                   # Nginx -> FastAPI
    _arrow(ax, 5.9, 9.02, 6.9, 8.45)                   # Nginx -> Streamlit
    # Streamlit -> FastAPI over loopback (it is a pure HTTP client).
    _arrow(ax, 5.55, 7.95, 4.45, 7.95, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.text(5.0, 8.2, "SEARCH_API_URL\n127.0.0.1", ha="center", va="center",
            fontsize=7.2, color=MUTED, linespacing=1.1, zorder=4)
    # FastAPI -> pipeline.
    _arrow(ax, 2.85, 7.46, 2.85, 6.94)
    _arrow(ax, 2.85, 6.16, 2.85, 5.80)
    _arrow(ax, 2.85, 4.90, 2.85, 4.54)
    _arrow(ax, 2.85, 3.76, 2.85, 3.36)
    _arrow(ax, 2.85, 2.54, 2.85, 2.13)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    path = main()
    print(f"Wrote {path}")
