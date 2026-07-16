# `deployment/` — AWS/EC2 deployment kit

Everything needed to run the Personalized Search Ranking System on a single
Ubuntu EC2 instance behind nginx. **No source code outside this folder was
modified.** Nothing here creates or touches AWS resources on its own.

## Read first
- **`DEPLOYMENT_READINESS_REPORT.md`** — the full audit: inventory, risks,
  RAM/disk/cost estimates, security, instance sizing, go/no-go checklist.
- **`architecture.md`** — the deployment diagram + component rationale.

## Files
```
deployment/
├── DEPLOYMENT_READINESS_REPORT.md   # the audit + report (start here)
├── architecture.md                  # ASCII architecture diagram
├── env/
│   └── production.env.example        # -> /etc/search-ranking/search.env
├── systemd/
│   ├── search-api.service            # uvicorn backend unit
│   └── search-frontend.service       # streamlit frontend unit
├── nginx/
│   └── search-app.conf               # reverse proxy (:80 -> UI + /api)
├── cloudwatch/
│   └── amazon-cloudwatch-agent.json  # optional logs + mem/disk/cpu metrics
└── scripts/
    ├── setup_ec2.sh                  # provision the instance (run as root)
    ├── verify_artifacts.sh           # pre-flight: are all runtime files present?
    ├── stage_artifacts.sh            # 3 strategies to get artifacts onto EC2
    └── aws_cli_commands.sh           # SG + AMI + instance (reference; run later)
```

## Happy path (once you approve deployment)
1. Provision the instance (Console or `scripts/aws_cli_commands.sh`).
2. `git clone` the repo to `/opt/search-ranking`.
3. Stage runtime artifacts — `scripts/stage_artifacts.sh` (models, embeddings,
   HF cache). See report §2b/§2c for why some aren't in git.
4. `sudo bash deployment/scripts/setup_ec2.sh` (venv, deps, units, nginx).
5. `bash deployment/scripts/verify_artifacts.sh` → must exit 0.
6. Edit `/etc/search-ranking/search.env` (confirm `SEARCH_HOST=0.0.0.0`).
7. `sudo systemctl start search-api` → wait ~60 s → check `/health`.
8. `sudo systemctl start search-frontend` → `sudo systemctl reload nginx`.
9. Open `http://<public-dns>/` and `http://<public-dns>/api/docs`.

## ⚠️ Two things that will bite you if skipped
- The **embedding model** (`all-MiniLM-L6-v2`) is **not** in the repo — it loads
  from the HF cache. Stage it, or do the first boot with `HF_HUB_OFFLINE=0`.
- `models/` and `results/*.npy` are **untracked in git** — a plain `git clone`
  won't bring them. Commit them or scp/S3 them over.

`verify_artifacts.sh` catches both before you start the service.
