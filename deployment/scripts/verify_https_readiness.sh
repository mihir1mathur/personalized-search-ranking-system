#!/usr/bin/env bash
# =============================================================================
# verify_https_readiness.sh  --  READ-ONLY pre/post checks for custom-domain TLS.
# -----------------------------------------------------------------------------
# Diagnoses whether the box is ready for (or already serving) HTTPS on a domain.
# It CHANGES NOTHING: no files edited, no packages installed, no certificate
# requested, no AWS resource created or modified. Safe to run any time.
#
#   bash deployment/scripts/verify_https_readiness.sh search.example.com
#
# Exit code is non-zero if any REQUIRED readiness check fails, so it can gate a
# run of setup_https.sh. Checks that only apply once a cert exists are treated
# as informational until the certificate is present.
# =============================================================================
set -uo pipefail

DOMAIN="${1:-}"
NGINX_SITE="/etc/nginx/sites-available/search-app"
fail=0

ok()   { printf "  [ OK ]  %s\n" "$*"; }
warn() { printf "  [WARN]  %s\n" "$*"; }
bad()  { printf "  [FAIL]  %s\n" "$*"; fail=1; }

echo "HTTPS readiness check (READ-ONLY -- nothing is modified)"
echo "-------------------------------------------------------------------------"

# ---- 1) domain argument supplied -------------------------------------------
if [[ -z "$DOMAIN" ]]; then
    bad "no domain argument. Usage: bash $0 <domain>  (e.g. search.example.com)"
    echo "-------------------------------------------------------------------------"
    echo "RESULT: cannot continue without a domain."
    exit 1
fi
ok "domain argument: ${DOMAIN}"

# ---- 2) DNS A record resolves + show the resolved IP -----------------------
RESOLVED_IP=""
if command -v getent >/dev/null 2>&1; then
    RESOLVED_IP="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk '{print $1; exit}')"
fi
if [[ -z "$RESOLVED_IP" ]] && command -v dig >/dev/null 2>&1; then
    RESOLVED_IP="$(dig +short A "$DOMAIN" 2>/dev/null | grep -E '^[0-9.]+$' | head -n1)"
fi
if [[ -z "$RESOLVED_IP" ]] && command -v host >/dev/null 2>&1; then
    RESOLVED_IP="$(host -t A "$DOMAIN" 2>/dev/null | awk '/has address/{print $NF; exit}')"
fi
if [[ -n "$RESOLVED_IP" ]]; then
    ok "DNS A record resolves: ${DOMAIN} -> ${RESOLVED_IP}"
    # Best-effort: compare against this host's public IP (informational only).
    THIS_IP="$(curl -s --max-time 5 https://checkip.amazonaws.com 2>/dev/null || true)"
    if [[ -n "$THIS_IP" ]]; then
        if [[ "$THIS_IP" == "$RESOLVED_IP" ]]; then
            ok "resolved IP matches this host's public IP (${THIS_IP})"
        else
            warn "resolved IP (${RESOLVED_IP}) != this host's public IP (${THIS_IP}).
             If DNS is still propagating this is expected; certbot needs them equal.
             Tip: attach an Elastic IP and point the A record at it."
        fi
    fi
else
    bad "DNS A record for ${DOMAIN} does not resolve yet.
         Point the A record at your Elastic IP and wait for propagation."
fi

# ---- 3) ports 80 and 443 expected ------------------------------------------
warn "ports 80 and 443 must be open to the internet in the EC2 security group.
         This script cannot see the SG; confirm in the AWS console. Do NOT open
         8000/8501 (they stay on loopback behind nginx)."

# ---- 4) active nginx config exists -----------------------------------------
if [[ -f "$NGINX_SITE" ]]; then
    ok "active nginx site present: ${NGINX_SITE}"
else
    bad "active nginx site NOT found at ${NGINX_SITE}. Run setup_ec2.sh first."
fi

# ---- 5) server_name contains the requested domain --------------------------
if [[ -f "$NGINX_SITE" ]]; then
    if grep -E '^\s*server_name\b' "$NGINX_SITE" | grep -qw "$DOMAIN"; then
        ok "server_name in ${NGINX_SITE} includes ${DOMAIN}"
    else
        warn "server_name in ${NGINX_SITE} does NOT include ${DOMAIN} yet.
             setup_https.sh sets this for you, or edit it manually before certbot."
    fi
fi

# ---- 6) nginx syntax valid --------------------------------------------------
# nginx -t is read-only (validation only). Needs root to read some includes;
# if not root we note that rather than failing hard.
if command -v nginx >/dev/null 2>&1; then
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
        if nginx -t >/dev/null 2>&1; then
            ok "nginx configuration syntax is valid (nginx -t)"
        else
            bad "nginx -t reports an INVALID config. Fix before running certbot."
        fi
    else
        warn "skipped 'nginx -t' (needs root). Re-run with sudo to validate syntax."
    fi
else
    bad "nginx is not installed / not on PATH."
fi

# ---- 7) certbot availability ------------------------------------------------
if command -v certbot >/dev/null 2>&1; then
    ok "certbot is installed ($(certbot --version 2>/dev/null | head -n1))"
    if certbot plugins 2>/dev/null | grep -q '\bnginx\b'; then
        ok "certbot nginx plugin is available"
    else
        warn "certbot present but the nginx plugin was not detected.
             setup_https.sh installs python3-certbot-nginx if missing."
    fi
else
    warn "certbot not installed yet. setup_https.sh installs it on demand."
fi

# ---- 8) certificate presence (if already issued) ---------------------------
CERT_LIVE="/etc/letsencrypt/live/${DOMAIN}"
CERT_ISSUED=0
if [[ -d "$CERT_LIVE" && -f "${CERT_LIVE}/fullchain.pem" ]]; then
    CERT_ISSUED=1
    ok "certificate already issued for ${DOMAIN} (${CERT_LIVE}/fullchain.pem)"
else
    warn "no certificate found for ${DOMAIN} yet (expected before first run)."
fi

# ---- 9) HTTPS endpoint reachable (only if a cert exists) -------------------
if [[ "$CERT_ISSUED" -eq 1 ]]; then
    if curl -sSI --max-time 10 "https://${DOMAIN}/" >/dev/null 2>&1; then
        ok "HTTPS endpoint reachable: https://${DOMAIN}/"
    else
        bad "certificate exists but https://${DOMAIN}/ is not reachable.
             Check nginx is reloaded and 443 is open in the security group."
    fi

    # ---- 10) /healthz over HTTPS (only if a cert exists) -------------------
    if curl -sS --max-time 10 "https://${DOMAIN}/healthz" >/dev/null 2>&1; then
        ok "https://${DOMAIN}/healthz responded"
    else
        warn "https://${DOMAIN}/healthz did not respond cleanly.
             Confirm search-api is running and /healthz -> 127.0.0.1:8000/health."
    fi
else
    warn "skipped HTTPS reachability + /healthz checks (no certificate yet)."
fi

echo "-------------------------------------------------------------------------"
echo "Note: this script created/modified NO AWS resources and changed NO files."
if [[ "$fail" -eq 0 ]]; then
    echo "RESULT: no blocking issues. Ready to run setup_https.sh (review WARNs)."
else
    echo "RESULT: blocking issue(s) found above. Resolve them before certbot."
fi
exit "$fail"
