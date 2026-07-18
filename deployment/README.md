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
    ├── setup_https.sh                # attach custom domain + Let's Encrypt TLS
    ├── verify_https_readiness.sh     # read-only: is the box ready for HTTPS?
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

## Custom Domain and HTTPS

The base deployment serves over **plain HTTP by public IP** — that keeps working
and is the right starting point. This section is a **separate, optional** step to
put a custom domain and a Let's Encrypt TLS certificate in front of it. It does
**not** change the app, and `setup_ec2.sh` never runs Certbot for you.

TLS is terminated at nginx; the FastAPI (8000) and Streamlit (8501) backends stay
on loopback exactly as before. Certbot adds the `:443` listener and an automatic
**HTTP → HTTPS redirect**; the `/`, `/api/`, and `/healthz` routes are preserved.

### Prerequisites (manual AWS + registrar steps — you do these)
1. **Attach an Elastic IP** to the instance *before* configuring DNS. An EC2
   instance's public IP changes every stop/start; an Elastic IP is stable, so
   your A record won't silently break. (Associate the EIP in the EC2 console.)
2. **Point your domain (or subdomain) A record at the Elastic IP.** Use your
   registrar / DNS host — no vendor-specific steps are given here. Examples:
   `search.example.com  A  <ELASTIC_IP>`  (subdomain) or `example.com A <EIP>`.
3. **Wait for DNS propagation** (typically minutes; can be up to ~an hour).
4. **Security group inbound rules** (confirm in the console):
   ```
   22   from your administrator IP only
   80   from 0.0.0.0/0        (Let's Encrypt HTTP-01 + the redirect)
   443  from 0.0.0.0/0        (HTTPS)
   #    NO public rule for 8000 or 8501 — they stay on loopback.
   ```

### Verify DNS points at the box (run anywhere)
```bash
# Should print your Elastic IP:
dig +short A search.example.com
# or:  getent ahostsv4 search.example.com
# or:  nslookup search.example.com
```

### One-time HTTPS setup (run on EC2, once DNS resolves to the EIP)
```bash
# Read-only readiness check first (changes nothing, installs nothing):
bash deployment/scripts/verify_https_readiness.sh search.example.com

# Then obtain + install the certificate and enable the 80->443 redirect:
sudo bash deployment/scripts/setup_https.sh search.example.com you@example.com
```
`setup_https.sh` validates the args, checks the current nginx config, installs
`certbot` + `python3-certbot-nginx` only if missing, **timestamped-backs-up** the
active nginx site before editing, sets `server_name`, runs Certbot with the nginx
plugin, dry-run tests renewal, and reloads nginx only after `nginx -t` passes.

### Verify HTTPS works
```bash
curl -I  https://search.example.com/          # expect HTTP/2 200
curl -I  http://search.example.com/           # expect 301 -> https (redirect)
curl -s  https://search.example.com/healthz    # expect the health JSON
# full re-check (now that the cert exists):
bash deployment/scripts/verify_https_readiness.sh search.example.com
```

### Verify automatic renewal
Certbot installs a systemd timer that renews before expiry. Confirm it works
without actually issuing anything:
```bash
sudo certbot renew --dry-run
systemctl list-timers | grep -i certbot     # the renewal timer is scheduled
```

### Rollback (revert to plain HTTP)
`setup_https.sh` prints the exact backup path it created, e.g.
`/etc/nginx/sites-available/search-app.bak.YYYYMMDD-HHMMSS`. To undo the change:
```bash
sudo cp -a /etc/nginx/sites-available/search-app.bak.YYYYMMDD-HHMMSS \
           /etc/nginx/sites-available/search-app
sudo nginx -t && sudo systemctl reload nginx
```
The certificate under `/etc/letsencrypt/` is harmless to leave in place; the
restored config simply stops using it and the site serves over HTTP again.

> **Security note:** never expose port 8000 (or 8501) to the internet. HTTPS
> changes nothing about that — the backends remain loopback-only behind nginx.
