# Frontend — Streamlit Search App

A polished, **Python-only** frontend for the Personalized Search Ranking System.
It is a **pure HTTP client** of the FastAPI backend — it does not import
any backend code and never reimplements retrieval or ranking. Built with
Streamlit (no Node.js, npm, React, or TypeScript).

## Run it

```bash
pip install -r frontend/requirements.txt

# backend (terminal 1)
HF_HUB_OFFLINE=1 uvicorn api.main:app

# frontend (terminal 2)
streamlit run frontend/app.py        # http://localhost:8501
```

Point at a different backend with `SEARCH_API_URL` (default
`http://127.0.0.1:8000`).

## Folder layout

| Path | What it is |
| --- | --- |
| `app.py` | The **Search** page (home). Query box, Top-K slider, ranking-method dropdown, loading spinner + progress, cached badge, result cards. |
| `components.py` | Reusable UI building blocks: page config + CSS loader, hero, result cards, status pills, cached badge, friendly error box, performance row, footer. |
| `utils.py` | The **only** place that talks to the backend: base URL, method→endpoint routing, `search()` / `get_health()` / `get_version()`, friendly `ApiError` handling, and parsers for the backend benchmark reports. |
| `styles.css` | Custom CSS (cards, hero gradient, pills, confidence bars) injected once per page. |
| `generate_assets.py` | Renders `assets/architecture.png`, `user_flow.png`, and the `screenshot_*.png` UI previews (populated from live backend data when available). |
| `pages/` | The multipage nav: `1_Health`, `2_API_Docs`, `3_Benchmark`, `4_Pipeline`, `5_Settings`. |
| `assets/` | Generated diagrams and UI screenshots (used by the Pipeline page and the README). |
| `static/` | Static assets. |
| `tests/` | Smoke tests: `utils` logic with mocked HTTP + Streamlit `AppTest` render checks for every page. |
| `requirements.txt` | Frontend-only dependencies (streamlit, requests, pandas, matplotlib, pytest). |

## How it talks to the backend

The six UI methods map onto backend endpoints:

| UI method | Endpoint |
| --- | --- |
| TF-IDF / BM25 / Embeddings | `POST /search` (with `method`) |
| Hybrid | `POST /hybrid-search` (`alpha`, `beta`) |
| CrossEncoder | `POST /rerank` (`candidate_depth`) |
| LTR | `POST /ltr-search` (`candidate_depth`) |

Health/version/docs use `GET /health`, `GET /version`, and the Swagger/ReDoc/
OpenAPI links; the Benchmark page reads the reports in `results/` (read-only).

## Tests

```bash
python -m pytest frontend/tests -q
```

Covers method routing + payloads, friendly error handling, report parsing,
formatting, clean imports, and — via Streamlit `AppTest` — that every page
renders without an uncaught exception even when the backend is down.
