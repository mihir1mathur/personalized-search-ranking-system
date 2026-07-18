# Deployment Architecture — Personalized Search Ranking System

Single EC2 instance running both tiers (backend + frontend) behind nginx.
This matches a portfolio-grade, cost-efficient deployment. The pipeline holds
~2 GB resident and is single-process, so we scale **out** (more instances behind
a load balancer) rather than **up** with more workers on one box.

```
                        Browser
                           │  HTTPS (custom domain)
                           ▼
                     Elastic IP  (stable public IP; survives stop/start)
                           │
                    ┌──────▼───────────┐
                    │  Security Group  │  22/tcp  ← your IP only
                    │                  │  80/tcp  ← 0.0.0.0/0  (redirect + ACME)
                    │                  │  443/tcp ← 0.0.0.0/0  (HTTPS)
                    └────────┬─────────┘  (8000/8501 NOT exposed)
                             │
        ┌────────────────────────────────────────────────┐
        │  EC2  t3.medium (2 vCPU / 4 GB) · Ubuntu LTS    │
        │  20 GB gp3 root volume                          │
        │                                                 │
        │   ┌──────────────────────────────────────────┐ │
        │   │  nginx  :80 → :443  (TLS terminates here) │ │
        │   │    :80 redirects → :443 (HTTP → HTTPS)     │ │
        │   │    /        → 127.0.0.1:8501 (Streamlit)  │ │
        │   │    /api/     → 127.0.0.1:8000 (FastAPI)    │ │
        │   │    /healthz  → 127.0.0.1:8000/health       │ │
        │   └──────┬───────────────────────┬────────────┘ │
        │          │ ws + http             │ http          │
        │   ┌──────▼──────────┐    ┌───────▼─────────────┐ │
        │   │ search-frontend  │    │ search-api          │ │
        │   │ systemd unit     │    │ systemd unit        │ │
        │   │ Streamlit :8501  │───▶│ uvicorn :8000       │ │
        │   │ (pure HTTP       │ SEARCH_API_URL          │ │
        │   │  client)         │ =http://127.0.0.1:8000  │ │
        │   └──────────────────┘    │  1 worker           │ │
        │                           │  SearchService      │ │
        │                           │   ├ TF-IDF / BM25    │ │
        │                           │   ├ Embeddings+FAISS │ │
        │                           │   ├ Hybrid fusion    │ │
        │                           │   ├ Cross-encoder    │ │
        │                           │   └ LTR (LambdaMART) │ │
        │                           └──────┬──────────────┘ │
        │                                  │ reads once at   │
        │                                  ▼ startup (~47s)  │
        │   /opt/search-ranking/                            │
        │     data/processed/sample_esci_50k.parquet        │
        │     results/product_embeddings_*.npy              │
        │     models/ms-marco-MiniLM-L-6-v2/  (cross-enc)    │
        │     models/ltr_lightgbm.txt         (LTR)          │
        │     results/week5_ce_cache.pkl      (warm cache)   │
        │   ~/.cache/huggingface/...all-MiniLM-L6-v2 (embed) │
        │                                                   │
        │   logs/api.log ──▶ CloudWatch agent (optional)    │
        └────────────────────────────────────────────────┘
```

## Component summary

| Layer          | Choice                          | Why |
| -------------- | ------------------------------- | --- |
| Compute        | EC2 `t3.medium` (2 vCPU, 4 GB)  | Backend RSS ≈ 2.0 GB steady / 2.1 GB peak (measured) + Streamlit + OS. 2 GB instances OOM. |
| OS             | Recent Ubuntu LTS (22.04+)      | Supported on recent Ubuntu LTS releases, preferably 22.04 or newer. Uses the distribution-provided `python3` inside a virtual environment. Dependencies were primarily validated on Python 3.10/3.11, so newer Python releases may require compatibility verification. |
| Storage        | 20 GB gp3                       | OS ~2.5 GB + venv (torch CPU) ~3 GB + artifacts ~0.4 GB + headroom. |
| Process mgmt   | systemd (2 units)               | Auto-restart, boot-on-start, journald logs, generous start timeout for the 47 s model load. |
| Reverse proxy  | nginx                           | Public 80/443; TLS terminates here; WebSocket upgrade for Streamlit; hides 8000/8501. |
| Observability  | CloudWatch agent (optional)     | Ships api.log + mem/disk/cpu; alarm on mem_used_percent. |
| Network        | Security Group                  | 22 from your IP, 80 from world; app ports stay loopback. |

## Production request flow (custom domain + HTTPS)

```
Browser
   └─ HTTPS  →  custom domain (A record → Elastic IP)
        └─ nginx  (TLS terminated here; HTTP :80 redirects to HTTPS :443)
             ├─ /        → Streamlit frontend  (127.0.0.1:8501)
             ├─ /api/     → FastAPI backend     (127.0.0.1:8000, /api stripped)
             └─ /healthz  → FastAPI /health     (127.0.0.1:8000/health)
```

- **TLS is terminated at nginx** (Let's Encrypt cert via Certbot's nginx plugin).
  The FastAPI and Streamlit backends speak plain HTTP over loopback, unchanged —
  **backend ports 8000/8501 remain private** and are never exposed publicly.
- **HTTP redirects to HTTPS**: Certbot adds the 80 → 443 redirect, so every
  request ends up encrypted while port 80 stays open for the redirect and for
  ACME renewals.
- **An Elastic IP** gives the box a stable public address. Point the domain's A
  record at the Elastic IP so a stop/start (which rotates the default EC2 public
  IP) does not break DNS or the certificate.
- Setup is a separate, optional step (`scripts/setup_https.sh`); the base
  deployment still works over plain HTTP by public IP before any of this. See
  `README.md` → "Custom Domain and HTTPS".

## Scaling / hardening path (beyond this single box)
- Put an **ALB** in front, health check `/healthz`, then run **N identical
  instances** in an Auto Scaling Group (each is stateless bar the in-process cache).
- Move artifacts to **S3** and pull at boot (see `stage_artifacts.sh` Strategy B).
- Add **TLS** (ACM cert on the ALB, or certbot on the box) and open 443.
